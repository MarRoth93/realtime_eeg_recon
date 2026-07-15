#!/usr/bin/env python3
"""Fit a real MVNN whitener from a subject's raw recording (pre-apply_mvnn).

Runs the custom EEGPreprocessor up to but not including apply_mvnn, then computes
cov**-0.5 exactly like the offline pipeline (rank-safe). The resulting 31x31
matrix is in Starstim31 channel order (the EEGPreprocessor's native order, before
the joint-dataset reorder), which matches the live preprocessor, so it can be
passed to --lab-whitening-matrix / lab_demo.json "whitening_matrix".
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
HIER = PROJECT_ROOT.parent / "Hierarchical_EEG2Image_Reconstruction"


def _load_eeg_preprocessor():
    spec = importlib.util.spec_from_file_location(
        "custom_eeg_preprocessing", HIER / "atms_pipeline" / "eeg_preprocessing.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epoch-tmin", type=float, default=-0.2)
    ap.add_argument("--epoch-tmax", type=float, default=1.0)
    ap.add_argument("--first-block", type=int, default=None,
                    help="Use only the first N epochs (calibration block) for the whitener.")
    args = ap.parse_args()

    from maryam_rt.integration.lab_calibration import compute_mvnn_whitener_custom, save_operator
    from maryam_rt.integration.montage import STARSTIM_31_CHANNELS

    ceeg = _load_eeg_preprocessor()
    pre = ceeg.EEGPreprocessor(
        xdf_path=str(args.xdf), target_sfreq=250,
        epoch_tmin=args.epoch_tmin, epoch_tmax=args.epoch_tmax,
    )
    pre.load_data()
    print("loaded; channels:", pre.raw.ch_names)
    pre.set_montage()
    pre.crop_task(mode="train")
    pre.apply_ica()
    pre.preprocess_signal()
    epochs = pre.create_epochs(mode="train")     # UNWHITENED, pre-apply_mvnn
    data = np.asarray(epochs.get_data(), dtype=np.float64)   # (N, 31, T)
    ch = list(epochs.info["ch_names"])
    print("pre-MVNN epochs:", data.shape, "| channel order matches STARSTIM_31:",
          ch == list(STARSTIM_31_CHANNELS))

    calib = data[: args.first_block] if args.first_block else data
    W = compute_mvnn_whitener_custom(calib, rank_safe=True)

    # prove it is a real whitener: whitened pooled cov ~ identity on the subspace
    white = np.einsum("ij,tjk->tik", W, calib)
    cov = np.cov(white.transpose(1, 0, 2).reshape(31, -1))
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    print(f"whitener: ||W - I|| = {np.linalg.norm(W - np.eye(31)):.2f}  "
          f"(large => real, not a no-op)")
    print(f"whitened cov eig: top={ev[0]:.3f}  28th={ev[27]:.3f}  null={ev[-1]:.1e}")

    save_operator(W, args.output, kind="mvnn", ch_names=ch, extra={
        "source_xdf": str(args.xdf), "method": "custom_rank_safe_pre_mvnn",
        "n_calibration_epochs": int(calib.shape[0]),
        "epoch_window_s": [args.epoch_tmin, args.epoch_tmax],
        "channel_order": ch,
    })
    print("saved ->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
