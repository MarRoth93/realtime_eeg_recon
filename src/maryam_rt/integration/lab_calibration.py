"""Offline calibration operators for the lab live/replay preprocessing.

Both the ICA eye-blink cleaning and the MVNN whitening reduce to a single
per-channel linear operator (a ``(n_ch, n_ch)`` matrix) applied to each model
epoch. This module produces those operators once, offline, so the realtime
apply path (:class:`maryam_rt.integration.starstim_preprocessing.StarstimLivePreprocessor`)
only ever does a matrix multiply and never needs MNE at inference time.

- ``compute_mvnn_whitener_custom`` mirrors ``apply_mvnn`` from the custom
  ``Hierarchical_EEG2Image_Reconstruction`` preprocessing: pool all calibration
  trials and timepoints into one covariance, then take ``cov ** -0.5``.
- ``fit_ica_cleaning_operator`` fits a subject ICA on a calibration recording
  (e.g. the resting-state block or the first image block), auto-labels
  components with ICLabel, excludes eye-blink components, and returns the exact
  linear cleaning operator implied by that ICA solution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.linalg


def compute_mvnn_whitener_custom(
    epochs: np.ndarray,
    rank_safe: bool = True,
    tol: float = 1e-10,
) -> np.ndarray:
    """MVNN whitener ``cov ** -0.5`` pooled over trials and time.

    Matches ``apply_mvnn`` in the custom offline pipeline: stack the calibration
    trials into ``(n_ch, n_trials * n_time)`` and whiten with the inverse matrix
    square root of the resulting channel covariance.

    ``rank_safe`` (the default, matching the offline pipeline's rank-aware path)
    whitens only the real signal subspace: the covariance of average-referenced
    EEG is rank-deficient (average reference removes one rank, each removed ICA
    component another), so the naive ``fractional_matrix_power(cov, -0.5)``
    divides by the ~0 null eigenvalues and blows up. Dropping those null
    directions yields the exact identity-covariance whitening the lab checkpoints
    were trained on. Set ``rank_safe=False`` for the original unstable formula.

    Parameters
    ----------
    epochs:
        Calibration epochs shaped ``(n_trials, n_ch, n_time)``.
    """
    data = np.asarray(epochs, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected calibration epochs shaped (n_trials, n_ch, n_time), got {data.shape}.")
    n_trials, n_ch, _ = data.shape
    if n_trials < 2:
        raise ValueError("MVNN whitening needs at least two calibration trials.")

    pooled = data.transpose(1, 0, 2).reshape(n_ch, -1)
    cov = np.cov(pooled)
    if rank_safe:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        keep = eigenvalues > eigenvalues.max() * tol
        inv_sqrt = np.zeros_like(eigenvalues)
        inv_sqrt[keep] = eigenvalues[keep] ** -0.5
        whitener = (eigenvectors * inv_sqrt) @ eigenvectors.T
    else:
        whitener = scipy.linalg.fractional_matrix_power(cov, -0.5)
    whitener = np.asarray(np.real_if_close(whitener), dtype=np.float32)
    if whitener.shape != (n_ch, n_ch):
        raise ValueError(f"Expected MVNN whitener shaped ({n_ch}, {n_ch}), got {whitener.shape}.")
    if not np.isfinite(whitener).all():
        raise ValueError("MVNN whitener contains non-finite values.")
    return whitener


@dataclass(frozen=True)
class ICACleaningResult:
    """A calibrated ICA cleaning operator and the metadata behind it."""

    operator: np.ndarray
    ch_names: tuple[str, ...]
    excluded_components: tuple[int, ...]
    excluded_labels: tuple[str, ...]
    excluded_probabilities: tuple[float, ...]
    n_components: int


def _ica_linear_operator(ica, ch_names: Sequence[str], sfreq: float) -> np.ndarray:
    """Recover the exact ``(n_ch, n_ch)`` linear map implied by ``ica.apply``.

    ``ICA.apply`` is affine in the data (a fixed cleaning matrix plus a constant
    offset from the stored PCA mean), so we recover it by finite differences:
    probe the operator with the zero signal and with each channel unit vector.
    This is exact regardless of MNE internals and needs no assumptions about the
    PCA/whitening bookkeeping.
    """
    import mne

    names = list(ch_names)
    n = len(names)
    info = mne.create_info(names, float(sfreq), ch_types="eeg")

    # Columns: [zeros, e_0, e_1, ..., e_{n-1}]; two samples so RawArray is happy.
    probe = np.concatenate([np.zeros((n, 1)), np.eye(n)], axis=1)
    probe = np.repeat(probe, 2, axis=1).astype(np.float64)
    raw = mne.io.RawArray(probe, info, verbose="ERROR")
    ica.apply(raw, verbose="ERROR")
    out = raw.get_data()[:, ::2]  # one column per probe vector

    base = out[:, 0]
    operator = out[:, 1:] - base[:, None]
    operator = np.asarray(operator, dtype=np.float32)
    if operator.shape != (n, n):
        raise ValueError(f"Recovered ICA operator shaped {operator.shape}, expected ({n}, {n}).")
    if not np.isfinite(operator).all():
        raise ValueError("Recovered ICA operator contains non-finite values.")
    return operator


def fit_ica_cleaning_operator(
    calibration: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    *,
    ica_threshold: float = 0.8,
    exclude_labels: Sequence[str] = ("eye blink",),
    fit_l_freq: float = 1.0,
    fit_h_freq: float = 100.0,
    n_components: float | int | None = 0.95,
    random_state: int = 42,
    exclude_indices: Optional[Sequence[int]] = None,
) -> ICACleaningResult:
    """Fit a subject ICA on a calibration recording and return its cleaning map.

    ``calibration`` is either continuous data ``(n_ch, n_samples)`` (e.g. the
    resting-state block) or epoched data ``(n_trials, n_ch, n_time)`` (e.g. the
    first image block). Components labelled in ``exclude_labels`` by ICLabel with
    probability >= ``ica_threshold`` are removed; pass ``exclude_indices`` to
    override the automatic labelling (and skip the ICLabel dependency).
    """
    import mne
    from mne.preprocessing import ICA

    names = list(ch_names)
    data = np.asarray(calibration, dtype=np.float64)
    info = mne.create_info(names, float(sfreq), ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")

    if data.ndim == 2:
        inst = mne.io.RawArray(data, info, verbose="ERROR")
    elif data.ndim == 3:
        events = np.column_stack(
            [np.arange(data.shape[0]), np.zeros(data.shape[0], int), np.ones(data.shape[0], int)]
        )
        inst = mne.EpochsArray(data, info, events=events, verbose="ERROR")
    else:
        raise ValueError(
            f"Calibration must be (n_ch, n_samples) or (n_trials, n_ch, n_time), got {data.shape}."
        )

    inst_for_fit = inst.copy().filter(
        l_freq=fit_l_freq,
        h_freq=fit_h_freq,
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    ica = ICA(n_components=n_components, method="infomax", random_state=random_state)
    ica.fit(inst_for_fit, verbose="ERROR")

    labels: list[str] = []
    probabilities: list[float] = []
    if exclude_indices is not None:
        exclude = sorted(int(i) for i in exclude_indices)
        labels = ["manual"] * len(exclude)
        probabilities = [1.0] * len(exclude)
    else:
        from mne_icalabel import label_components

        result = label_components(inst_for_fit, ica, method="iclabel")
        component_labels = list(result["labels"])
        component_probs = list(result["y_pred_proba"])
        exclude = [
            i
            for i, (label, prob) in enumerate(zip(component_labels, component_probs))
            if label in exclude_labels and float(prob) >= ica_threshold
        ]
        labels = [component_labels[i] for i in exclude]
        probabilities = [float(component_probs[i]) for i in exclude]

    ica.exclude = list(exclude)
    operator = _ica_linear_operator(ica, names, sfreq)

    return ICACleaningResult(
        operator=operator,
        ch_names=tuple(names),
        excluded_components=tuple(exclude),
        excluded_labels=tuple(labels),
        excluded_probabilities=tuple(probabilities),
        n_components=int(ica.n_components_),
    )


def save_operator(
    matrix: np.ndarray,
    path: Path,
    *,
    kind: str,
    ch_names: Sequence[str],
    extra: Optional[dict] = None,
) -> Path:
    """Persist a calibration operator as ``.npy`` plus a sidecar ``.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    np.save(path, matrix)
    metadata = {
        "kind": kind,
        "shape": list(matrix.shape),
        "ch_names": list(ch_names),
    }
    if extra:
        metadata.update(extra)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def load_operator(path: Path, expected_channels: Optional[int] = None) -> np.ndarray:
    """Load a saved calibration operator and validate its shape."""
    path = Path(path)
    matrix = np.load(path).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Operator at {path} must be square, got {matrix.shape}.")
    if expected_channels is not None and matrix.shape[0] != expected_channels:
        raise ValueError(
            f"Operator at {path} has {matrix.shape[0]} channels, expected {expected_channels}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Operator at {path} contains non-finite values.")
    return matrix
