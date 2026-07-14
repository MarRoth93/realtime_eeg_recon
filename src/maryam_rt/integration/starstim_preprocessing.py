from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import signal

from maryam_rt.integration.montage import STARSTIM_31_CHANNELS, STARSTIM_32_CHANNELS

# Highpass cutoffs below this fraction of Nyquist cannot be realized on a short
# epoch (the custom offline pipeline runs its 0.01 Hz highpass on minutes of
# continuous data). We skip the highpass in that regime and rely on average
# reference + baseline correction for DC/drift removal instead.
_MIN_HIGHPASS_NORM = 1e-3


def select_starstim31_channels(
    epoch: np.ndarray,
    ch_names: Sequence[str] = STARSTIM_32_CHANNELS,
) -> np.ndarray:
    """Select/reorder Starstim channels and drop Fz to match lab model artifacts."""
    data = np.asarray(epoch, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if data.shape[0] != len(ch_names):
        raise ValueError(f"Epoch has {data.shape[0]} channels but ch_names has {len(ch_names)} entries.")

    by_name = {name: idx for idx, name in enumerate(ch_names)}
    missing = [name for name in STARSTIM_31_CHANNELS if name not in by_name]
    if missing:
        raise ValueError(f"Epoch is missing expected Starstim31 channels: {missing}")
    return np.ascontiguousarray([data[by_name[name]] for name in STARSTIM_31_CHANNELS], dtype=np.float32)


def average_reference(epoch: np.ndarray) -> np.ndarray:
    """Apply per-sample average reference across channels."""
    data = np.asarray(epoch, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    mean = np.nanmean(data, axis=0, dtype=np.float64, keepdims=True)
    return np.nan_to_num(data.astype(np.float64) - mean).astype(np.float32)


def resample_epoch(epoch: np.ndarray, input_sfreq: float, target_sfreq: float, duration_s: float) -> np.ndarray:
    """Resample an epoch to `target_sfreq` for a known epoch duration."""
    data = np.asarray(epoch, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if input_sfreq <= 0 or target_sfreq <= 0 or duration_s <= 0:
        raise ValueError("input_sfreq, target_sfreq, and duration_s must be positive.")
    expected_in = int(round(duration_s * input_sfreq))
    if data.shape[1] != expected_in:
        raise ValueError(f"Expected {expected_in} samples at {input_sfreq:g} Hz, got {data.shape[1]}.")
    n_out = int(round(duration_s * target_sfreq))
    if data.shape[1] == n_out:
        return data.astype(np.float32)
    return signal.resample(data, n_out, axis=-1).astype(np.float32)


def baseline_correct(epoch: np.ndarray, times: np.ndarray, baseline: tuple[float, float] = (-0.2, 0.0)) -> np.ndarray:
    """Subtract per-channel baseline mean over the half-open baseline interval."""
    data = np.asarray(epoch, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if times.shape[0] != data.shape[1]:
        raise ValueError(f"times has {times.shape[0]} samples but epoch has {data.shape[1]}.")
    start, stop = baseline
    mask = (times >= start) & (times < stop)
    if not mask.any():
        return data.astype(np.float32)
    return (data - data[:, mask].mean(axis=1, keepdims=True)).astype(np.float32)


def crop_model_window(
    epoch: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float] = (0.0, 1.0),
    target_samples: int = 250,
) -> np.ndarray:
    """Crop the model-facing 0..1 s window used by the lab artifacts."""
    data = np.asarray(epoch, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if times.shape[0] != data.shape[1]:
        raise ValueError(f"times has {times.shape[0]} samples but epoch has {data.shape[1]}.")
    start, stop = window
    indices = np.flatnonzero((times >= start) & (times < stop))
    if indices.shape[0] < target_samples:
        raise ValueError(
            f"Model window {window} contains {indices.shape[0]} samples, expected at least {target_samples}."
        )
    return np.ascontiguousarray(data[:, indices[:target_samples]], dtype=np.float32)


def bandpass_filter(
    epoch: np.ndarray,
    sfreq: float,
    l_freq: Optional[float],
    h_freq: Optional[float],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass matching the offline 0.01-30 Hz cleaning.

    Highpass and lowpass are applied independently so either can be omitted
    (pass ``None``). A highpass whose cutoff is below ``_MIN_HIGHPASS_NORM`` of
    Nyquist is skipped: it is not realizable on a short epoch, and DC/drift is
    already handled by the average reference and baseline steps.
    """
    data = np.asarray(epoch, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")
    nyquist = 0.5 * sfreq
    x = data.astype(np.float64)

    if h_freq is not None:
        if not 0 < h_freq < nyquist:
            raise ValueError(f"Lowpass {h_freq} Hz must be in (0, {nyquist}).")
        sos = signal.butter(order, h_freq / nyquist, btype="low", output="sos")
        x = signal.sosfiltfilt(sos, x, axis=-1)

    if l_freq is not None:
        norm = l_freq / nyquist
        if not 0 < norm < 1:
            raise ValueError(f"Highpass {l_freq} Hz must be in (0, {nyquist}).")
        if norm >= _MIN_HIGHPASS_NORM:
            sos = signal.butter(order, norm, btype="high", output="sos")
            x = signal.sosfiltfilt(sos, x, axis=-1)

    return np.ascontiguousarray(x, dtype=np.float32)


def apply_linear_operator(epoch: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Apply a per-channel linear operator (ICA cleaning or MVNN whitening)."""
    data = np.asarray(epoch, dtype=np.float32)
    op = np.asarray(operator, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected epoch shaped (channels, samples), got {data.shape}.")
    if op.shape != (data.shape[0], data.shape[0]):
        raise ValueError(
            f"Linear operator must be ({data.shape[0]}, {data.shape[0]}) to match "
            f"{data.shape[0]} channels, got {op.shape}."
        )
    if not np.isfinite(op).all():
        raise ValueError("Linear operator contains non-finite values.")
    return np.asarray(op @ data, dtype=np.float32)


def peak_to_peak(epoch: np.ndarray) -> float:
    """Maximum per-channel peak-to-peak amplitude of an epoch."""
    data = np.asarray(epoch, dtype=np.float32)
    if data.size == 0:
        return 0.0
    return float((data.max(axis=1) - data.min(axis=1)).max())


@dataclass(frozen=True)
class PreprocessResult:
    """Model epoch plus per-epoch quality diagnostics."""

    epoch: np.ndarray
    peak_to_peak_uv: float
    rejected: bool


@dataclass(frozen=True)
class StarstimLivePreprocessor:
    """Convert raw Starstim live epochs into the offline lab model contract.

    The baseline pipeline (channel drop, average reference, resample, baseline,
    crop) reproduces the geometric contract of the derived epochs. The optional
    steps close the remaining gap to the custom offline preprocessing:

    - ``l_freq`` / ``h_freq``: bandpass to match the offline 0.01-30 Hz filter.
    - ``ica_operator``: a calibrated (31, 31) ICA cleaning matrix.
    - ``whitening_matrix``: a calibrated (31, 31) MVNN whitener applied last.
    - ``reject_peak_to_peak_uv``: peak-to-peak artifact threshold (µV).

    The ICA and MVNN operators are produced offline by
    :mod:`maryam_rt.integration.lab_calibration`; applying them here is a plain
    matrix multiply, so the live path needs no MNE dependency.
    """

    input_sfreq: float = 500.0
    target_sfreq: float = 250.0
    tmin: float = -0.2
    tmax: float = 1.0
    model_window: tuple[float, float] = (0.0, 1.0)
    model_samples: int = 250
    ch_names: tuple[str, ...] = tuple(STARSTIM_32_CHANNELS)
    l_freq: Optional[float] = None
    h_freq: Optional[float] = None
    filter_order: int = 4
    ica_operator: Optional[np.ndarray] = None
    whitening_matrix: Optional[np.ndarray] = None
    reject_peak_to_peak_uv: Optional[float] = None
    apply_baseline: bool = True

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        return self.process(epoch).epoch

    def process(self, epoch: np.ndarray) -> PreprocessResult:
        """Run the full pipeline and return the model epoch plus diagnostics.

        Order mirrors the offline custom pipeline: average reference -> ICA
        cleaning -> bandpass -> resample -> baseline -> crop -> MVNN whitening.
        Artifact rejection is measured on the cropped, pre-whitening epoch,
        matching where the offline pipeline rejects trials.
        """
        duration = self.tmax - self.tmin
        x = select_starstim31_channels(epoch, self.ch_names)
        x = average_reference(x)
        if self.ica_operator is not None:
            x = apply_linear_operator(x, self.ica_operator)
        if self.l_freq is not None or self.h_freq is not None:
            x = bandpass_filter(x, self.input_sfreq, self.l_freq, self.h_freq, self.filter_order)
        x = resample_epoch(x, input_sfreq=self.input_sfreq, target_sfreq=self.target_sfreq, duration_s=duration)
        times = self.tmin + np.arange(x.shape[1], dtype=np.float32) / float(self.target_sfreq)
        if self.apply_baseline:
            x = baseline_correct(x, times, baseline=(self.tmin, 0.0))
        x = crop_model_window(x, times, window=self.model_window, target_samples=self.model_samples)

        ptp = peak_to_peak(x)
        rejected = self.reject_peak_to_peak_uv is not None and ptp > self.reject_peak_to_peak_uv

        if self.whitening_matrix is not None:
            x = apply_linear_operator(x, self.whitening_matrix)

        return PreprocessResult(epoch=x, peak_to_peak_uv=ptp, rejected=rejected)
