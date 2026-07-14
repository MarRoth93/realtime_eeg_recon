#!/usr/bin/env python3
"""Fit the offline calibration operators for the lab live/replay preprocessing.

The realtime path applies ICA eye-blink cleaning and MVNN whitening as static
31x31 matrices (see :mod:`maryam_rt.integration.lab_calibration`). This script
produces those matrices once per subject from a calibration recording, so the
live stream never has to fit ICA in real time (plan option a).

Typical use — fit the ICA cleaning operator from a resting-state block in a raw
``.easy`` recording, then reuse it for the whole session::

    python scripts/calibrate_lab_operators.py \
        --easy data/P04/<recording>.easy \
        --info data/P04/<recording>.info \
        --start-sample 0 --n-samples 60000 \
        --ica-output checkpoints/lab_calibration/P04_ica_operator.npy

The resulting ``.npy`` is passed to the live GUI via ``--lab-ica-operator`` and
to lab replay via ``--lab-ica-operator``. The MVNN whitener can be fit here from
an epochs ``.npz`` too, but the replay path can also fit it automatically from
the first image block (``--lab-whitening-first-block``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--easy", type=Path, default=None, help="Raw .easy recording for ICA calibration.")
    parser.add_argument("--info", type=Path, default=None, help="Matching .info file (channel names + sampling rate).")
    parser.add_argument("--start-sample", type=int, default=0, help="First sample of the calibration window.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Length of the calibration window in samples. Defaults to the rest of the recording.",
    )
    parser.add_argument("--sfreq", type=float, default=None, help="Override sampling rate. Defaults to .info, then 500 Hz.")
    parser.add_argument("--ica-output", type=Path, default=None, help="Where to save the 31x31 ICA cleaning matrix (.npy).")
    parser.add_argument("--ica-threshold", type=float, default=0.8, help="ICLabel probability threshold for exclusion.")
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        default=["eye blink"],
        help="ICLabel component labels to remove.",
    )
    parser.add_argument(
        "--exclude-indices",
        type=int,
        nargs="+",
        default=None,
        help="Manually exclude these component indices instead of running ICLabel (skips the mne-icalabel dependency).",
    )
    parser.add_argument(
        "--epochs-npz",
        type=Path,
        default=None,
        help="Optional (N, 31, 250) epochs .npz (key 'eeg') to fit the MVNN whitener.",
    )
    parser.add_argument("--whitener-output", type=Path, default=None, help="Where to save the 31x31 MVNN whitener (.npy).")
    parser.add_argument(
        "--whitener-first-block",
        type=int,
        default=None,
        help="Use only the first N epochs from --epochs-npz as the whitener calibration block.",
    )
    return parser.parse_args()


def _load_easy_calibration(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    from maryam_rt.integration.lab_replay import _load_easy_segment_uv, _parse_starstim_info
    from maryam_rt.integration.montage import STARSTIM_32_CHANNELS
    from maryam_rt.integration.starstim_preprocessing import average_reference, select_starstim31_channels

    info_sfreq, info_channels = _parse_starstim_info(args.info)
    sfreq = float(args.sfreq or info_sfreq or 500.0)
    ch_names = tuple(info_channels or STARSTIM_32_CHANNELS)

    n_samples = args.n_samples
    if n_samples is None:
        with args.easy.open("r", errors="ignore") as handle:
            total = sum(1 for _ in handle)
        n_samples = total - args.start_sample
    segment_uv = _load_easy_segment_uv(args.easy, start_sample=args.start_sample, samples=n_samples)

    continuous = select_starstim31_channels(segment_uv, ch_names)
    continuous = average_reference(continuous)
    return continuous, sfreq


def main() -> int:
    args = parse_args()
    from maryam_rt.integration.lab_calibration import (
        compute_mvnn_whitener_custom,
        fit_ica_cleaning_operator,
        save_operator,
    )
    from maryam_rt.integration.montage import STARSTIM_31_CHANNELS

    if args.easy is not None:
        if args.ica_output is None:
            raise SystemExit("--ica-output is required when fitting an ICA operator from --easy.")
        continuous, sfreq = _load_easy_calibration(args)
        print(f"Fitting ICA on {continuous.shape[1]} samples at {sfreq:g} Hz from {args.easy}.")
        result = fit_ica_cleaning_operator(
            continuous,
            sfreq,
            STARSTIM_31_CHANNELS,
            ica_threshold=args.ica_threshold,
            exclude_labels=tuple(args.exclude_labels),
            exclude_indices=args.exclude_indices,
        )
        save_operator(
            result.operator,
            args.ica_output,
            kind="ica_cleaning",
            ch_names=STARSTIM_31_CHANNELS,
            extra={
                "excluded_components": list(result.excluded_components),
                "excluded_labels": list(result.excluded_labels),
                "excluded_probabilities": list(result.excluded_probabilities),
                "n_components": result.n_components,
                "sfreq": sfreq,
                "source_easy": str(args.easy),
            },
        )
        print(
            f"Saved ICA cleaning operator -> {args.ica_output} "
            f"(excluded components {result.excluded_components})."
        )

    if args.epochs_npz is not None:
        if args.whitener_output is None:
            raise SystemExit("--whitener-output is required when fitting a whitener from --epochs-npz.")
        eeg = np.asarray(np.load(args.epochs_npz, allow_pickle=True)["eeg"], dtype=np.float32)
        if args.whitener_first_block is not None:
            eeg = eeg[: args.whitener_first_block]
        print(f"Fitting MVNN whitener on {eeg.shape[0]} epochs from {args.epochs_npz}.")
        whitener = compute_mvnn_whitener_custom(eeg)
        save_operator(
            whitener,
            args.whitener_output,
            kind="mvnn",
            ch_names=STARSTIM_31_CHANNELS,
            extra={"n_calibration_epochs": int(eeg.shape[0]), "source_npz": str(args.epochs_npz)},
        )
        print(f"Saved MVNN whitener -> {args.whitener_output}.")

    if args.easy is None and args.epochs_npz is None:
        raise SystemExit("Nothing to do: pass --easy (ICA) and/or --epochs-npz (MVNN whitener).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
