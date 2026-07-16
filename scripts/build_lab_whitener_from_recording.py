#!/usr/bin/env python3
"""Fit an MVNN whitener from a subject's raw recording (.easy / .edf / .xdf).

Unlike ``build_lab_whitener_from_raw.py`` (which needs an XDF and the sibling
Hierarchical ``EEGPreprocessor``), this script reads the NIC-native ``.easy``
export directly -- plus ``.edf`` / ``.xdf`` -- and builds ``(31, 250)`` model
epochs with the *same* live preprocessing path (``StarstimLivePreprocessor``)
that lab-live and lab-replay use. It then computes the MVNN whitener
(``cov ** -0.5`` pooled over trials and time) exactly like the offline
``apply_mvnn`` and the lab-replay whitener fit.

The result is a 31x31 ``.npy`` in Starstim31 channel order that can be handed to
the live GUI as the session's fixed covariance whitener::

    python scripts/run_realtime_gui.py --mode lab-live \\
        --lab-whitening mvnn --lab-whitening-cache <out.npy> ...

(or as ``--lab-whitening-matrix <out.npy>``).

Example (P13, .easy)::

    python scripts/build_lab_whitener_from_recording.py \\
        --recording data/P13/20260713150824_realtime_eeg_recon_test_jose_eeg-recording.easy \\
        --output outputs/gui_lab_live/cache/lab_raw_P13_mvnn_-200ms_1000ms.npy \\
        --bandpass-l-freq 0.01 --bandpass-h-freq 30.0 --filter-order 4 --reject-uv 250
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maryam_rt.integration.lab_calibration import (  # noqa: E402
    compute_mvnn_whitener_custom,
    load_operator,
    save_operator,
)
from maryam_rt.integration.lab_replay import (  # noqa: E402
    _load_easy_segment_uv,
    _parse_starstim_info,
)
from maryam_rt.integration.montage import (  # noqa: E402
    STARSTIM_31_CHANNELS,
    STARSTIM_32_CHANNELS,
)
from maryam_rt.integration.starstim_preprocessing import StarstimLivePreprocessor  # noqa: E402

# A continuous-segment reader: (start_sample, n_samples) -> (n_channels, n_samples) in microvolts.
SegmentReader = Callable[[int, int], np.ndarray]


def _sibling(recording: Path, suffix: str) -> Path | None:
    match = sorted(recording.parent.glob(f"*{suffix}"))
    return match[0] if match else None


def _onsets_from_trigger_column(trig: np.ndarray, lo: int, hi: int) -> list[int]:
    """First sample of each contiguous non-zero trigger run whose code is in [lo, hi]."""
    nonzero = np.flatnonzero(trig != 0)
    if nonzero.size == 0:
        return []
    starts = nonzero[np.r_[True, np.diff(nonzero) > 1]]
    codes = trig[starts].astype(int)
    return [int(s) for s, c in zip(starts, codes) if lo <= c <= hi]


def _load_easy(recording: Path, info: Path | None, trig_col: int, lo: int, hi: int):
    import pandas as pd

    sfreq, channels = _parse_starstim_info(info or _sibling(recording, ".info"))
    ch_names = list(channels or STARSTIM_32_CHANNELS)
    input_sfreq = float(sfreq or 500.0)

    trig = pd.read_csv(recording, sep="\t", header=None, usecols=[trig_col]).iloc[:, 0].to_numpy()
    events = _onsets_from_trigger_column(trig, lo, hi)

    def read_segment(start: int, n: int) -> np.ndarray:
        return _load_easy_segment_uv(recording, start_sample=start, samples=n)

    return read_segment, events, input_sfreq, ch_names


def _reorder_to_starstim32(data_uv: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Reorder an in-memory (n_ch, n_samples) array into STARSTIM_32 channel order."""
    by_name = {name: idx for idx, name in enumerate(ch_names)}
    missing = [name for name in STARSTIM_32_CHANNELS if name not in by_name]
    if missing:
        raise SystemExit(f"Recording is missing Starstim32 channels: {missing}")
    return np.ascontiguousarray(
        [data_uv[by_name[name]] for name in STARSTIM_32_CHANNELS], dtype=np.float32
    )


def _events_from_annotations(raw, sfreq: float, lo: int, hi: int) -> list[int]:
    import re

    events: list[int] = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        match = re.search(r"\d+", str(desc))
        if match and lo <= int(match.group()) <= hi:
            events.append(int(round(float(onset) * sfreq)))
    return events


def _load_edf(recording: Path, lo: int, hi: int):
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(recording, preload=True, encoding="latin1", misc=["X", "Y", "Z"])
    input_sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    # MNE returns SI volts; the live path works in microvolts.
    data_uv = _reorder_to_starstim32(raw.get_data() * 1e6, ch_names)
    events = _events_from_annotations(raw, input_sfreq, lo, hi)

    def read_segment(start: int, n: int) -> np.ndarray:
        if start < 0 or start + n > data_uv.shape[1]:
            raise ValueError(f"segment [{start}, {start + n}) out of bounds")
        return data_uv[:, start : start + n]

    return read_segment, events, input_sfreq, list(STARSTIM_32_CHANNELS)


def _load_xdf(recording: Path, lo: int, hi: int):
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(recording))
    eeg = next(s for s in streams if s["info"]["type"][0] == "EEG")
    markers = next(
        (s for s in streams if "Markers" in s["info"]["name"][0]),
        next(s for s in streams if "pyexp" in s["info"]["name"][0]),
    )
    input_sfreq = float(eeg["info"]["effective_srate"])
    # xdf EEG time_series is already in microvolts.
    data_uv = _reorder_to_starstim32(
        np.asarray(eeg["time_series"], dtype=np.float32).T, list(STARSTIM_32_CHANNELS)
    )
    eeg_t0 = float(eeg["time_stamps"][0])
    events = []
    for stamp, value in zip(markers["time_stamps"], markers["time_series"]):
        code = int(value[0])
        if lo <= code <= hi:
            events.append(int(round((float(stamp) - eeg_t0) * input_sfreq)))

    def read_segment(start: int, n: int) -> np.ndarray:
        if start < 0 or start + n > data_uv.shape[1]:
            raise ValueError(f"segment [{start}, {start + n}) out of bounds")
        return data_uv[:, start : start + n]

    return read_segment, events, input_sfreq, list(STARSTIM_32_CHANNELS)


def build_recording_source(args: argparse.Namespace):
    suffix = args.recording.suffix.lower()
    if suffix == ".easy":
        return _load_easy(args.recording, args.info, args.easy_trigger_col, args.trigger_lo, args.trigger_hi)
    if suffix == ".edf":
        return _load_edf(args.recording, args.trigger_lo, args.trigger_hi)
    if suffix == ".xdf":
        return _load_xdf(args.recording, args.trigger_lo, args.trigger_hi)
    raise SystemExit(f"Unsupported recording format: {suffix!r} (use .easy, .edf, or .xdf).")


def collect_model_epochs(
    read_segment: SegmentReader,
    events: list[int],
    preprocessor: StarstimLivePreprocessor,
) -> np.ndarray:
    input_sfreq = preprocessor.input_sfreq
    raw_samples = int(round((preprocessor.tmax - preprocessor.tmin) * input_sfreq))
    kept: list[np.ndarray] = []
    rejected = out_of_bounds = 0
    for event_sample in events:
        start = int(round(event_sample + preprocessor.tmin * input_sfreq))
        if start < 0:
            out_of_bounds += 1
            continue
        try:
            raw_epoch = read_segment(start, raw_samples)
        except ValueError:
            out_of_bounds += 1
            continue
        result = preprocessor.process(raw_epoch)
        if result.rejected:
            rejected += 1
            continue
        if result.epoch.shape != (31, 250):
            out_of_bounds += 1
            continue
        kept.append(result.epoch)
    print(
        f"epochs: {len(events)} triggers -> {len(kept)} usable "
        f"({rejected} artifact-rejected, {out_of_bounds} out-of-bounds/bad-shape)"
    )
    if len(kept) < 2:
        raise SystemExit("MVNN whitening needs at least two usable 31x250 epochs.")
    return np.stack(kept).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", type=Path, required=True, help="Raw .easy / .edf / .xdf recording.")
    ap.add_argument("--output", type=Path, required=True, help="Where to save the 31x31 MVNN whitener (.npy).")
    ap.add_argument("--info", type=Path, default=None, help="Optional .info override (.easy: sfreq + channels).")
    ap.add_argument("--tmin", type=float, default=-0.2, help="Raw epoch start (s) relative to trigger.")
    ap.add_argument("--tmax", type=float, default=1.0, help="Raw epoch end (s) relative to trigger.")
    ap.add_argument("--bandpass-l-freq", type=float, default=0.01, help="Highpass cutoff (Hz). Match the offline 0.01.")
    ap.add_argument("--bandpass-h-freq", type=float, default=30.0, help="Lowpass cutoff (Hz). Match the offline 30.")
    ap.add_argument("--filter-order", type=int, default=4, help="Butterworth order for the epoch bandpass.")
    ap.add_argument("--reject-uv", type=float, default=250.0, help="Peak-to-peak reject threshold (uV); epochs above are dropped.")
    ap.add_argument("--ica-operator", type=Path, default=None, help="Optional 31x31 ICA cleaning matrix (.npy) applied before whitening.")
    ap.add_argument("--trigger-lo", type=int, default=1001, help="Lowest image-onset trigger code.")
    ap.add_argument("--trigger-hi", type=int, default=1200, help="Highest image-onset trigger code.")
    ap.add_argument("--easy-trigger-col", type=int, default=35, help=".easy trigger column index (0-based; 32 EEG + 3 accel).")
    ap.add_argument("--first-block", type=int, default=None, help="Use only the first N usable epochs as the calibration block.")
    ap.add_argument("--method", choices=["custom", "shrinkage"], default="custom", help="Whitener estimator.")
    args = ap.parse_args()

    if not args.recording.exists():
        raise SystemExit(f"Recording not found: {args.recording}")

    ica_operator = load_operator(args.ica_operator, expected_channels=31) if args.ica_operator else None

    read_segment, events, input_sfreq, ch_names = build_recording_source(args)
    print(f"recording: {args.recording.name} | sfreq={input_sfreq:g} Hz | {len(ch_names)} channels")
    if not events:
        raise SystemExit(
            f"No image-onset triggers in [{args.trigger_lo}, {args.trigger_hi}] found in the recording."
        )

    preprocessor = StarstimLivePreprocessor(
        input_sfreq=input_sfreq,
        tmin=args.tmin,
        tmax=args.tmax,
        ch_names=tuple(ch_names),
        l_freq=args.bandpass_l_freq,
        h_freq=args.bandpass_h_freq,
        filter_order=args.filter_order,
        ica_operator=ica_operator,
        reject_peak_to_peak_uv=args.reject_uv,
    )

    epochs = collect_model_epochs(read_segment, events, preprocessor)
    if args.first_block is not None:
        epochs = epochs[: args.first_block]
        print(f"calibration block: first {epochs.shape[0]} epochs")

    if args.method == "custom":
        whitener = compute_mvnn_whitener_custom(epochs, rank_safe=True)
    else:
        from maryam_rt.integration.lab_replay import _compute_lab_mvnn_whitener_shrinkage

        whitener = _compute_lab_mvnn_whitener_shrinkage(epochs)

    # Sanity: whitened pooled covariance should be ~identity on the signal subspace.
    whitened = np.einsum("ij,tjk->tik", whitener, epochs)
    cov = np.cov(whitened.transpose(1, 0, 2).reshape(31, -1))
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    print(
        f"whitener: ||W - I|| = {np.linalg.norm(whitener - np.eye(31)):.2f} "
        f"(large => real, not a no-op)"
    )
    print(f"whitened cov eig: top={eig[0]:.3f}  28th={eig[27]:.3f}  null={eig[-1]:.1e}")

    save_operator(
        whitener,
        args.output,
        kind="mvnn",
        ch_names=list(STARSTIM_31_CHANNELS),
        extra={
            "source_recording": str(args.recording),
            "method": f"live_model_epoch_{args.method}",
            "n_calibration_epochs": int(epochs.shape[0]),
            "epoch_window_s": [args.tmin, args.tmax],
            "bandpass_hz": [args.bandpass_l_freq, args.bandpass_h_freq],
            "reject_peak_to_peak_uv": args.reject_uv,
            "input_sfreq": input_sfreq,
        },
    )
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
