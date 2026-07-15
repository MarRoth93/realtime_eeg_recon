from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import signal

from maryam_rt.integration.montage import STARSTIM_31_CHANNELS, STARSTIM_32_CHANNELS


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


@dataclass(frozen=True)
class StarstimLivePreprocessor:
    """Convert raw Starstim live epochs into the offline lab model contract."""

    input_sfreq: float = 500.0
    target_sfreq: float = 250.0
    tmin: float = -0.2
    tmax: float = 1.0
    model_window: tuple[float, float] = (0.0, 1.0)
    model_samples: int = 250
    ch_names: tuple[str, ...] = tuple(STARSTIM_32_CHANNELS)

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        duration = self.tmax - self.tmin
        x = select_starstim31_channels(epoch, self.ch_names)
        x = average_reference(x)
        x = resample_epoch(x, input_sfreq=self.input_sfreq, target_sfreq=self.target_sfreq, duration_s=duration)
        times = self.tmin + np.arange(x.shape[1], dtype=np.float32) / float(self.target_sfreq)
        x = baseline_correct(x, times, baseline=(self.tmin, 0.0))
        return crop_model_window(x, times, window=self.model_window, target_samples=self.model_samples)


@dataclass(frozen=True)
class SpatialWhiteningPreprocessor:
    """Apply a fixed channel-space whitening matrix after base preprocessing."""

    base_preprocessor: Callable[[np.ndarray], np.ndarray]
    whitener: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.whitener, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Spatial whitener must be a square matrix, got {matrix.shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError("Spatial whitener contains non-finite values.")
        object.__setattr__(self, "whitener", matrix)

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        processed = np.asarray(self.base_preprocessor(epoch), dtype=np.float32)
        expected_channels = self.whitener.shape[1]
        if processed.ndim != 2 or processed.shape[0] != expected_channels:
            raise ValueError(
                f"Spatial whitening expects {expected_channels} channels, got {processed.shape}."
            )
        return np.asarray(self.whitener @ processed, dtype=np.float32)
