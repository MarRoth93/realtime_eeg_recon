from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from maryam_rt.integration.lab_calibration import (
    compute_mvnn_whitener_custom,
    load_operator,
    save_operator,
)
from maryam_rt.integration.montage import STARSTIM_31_CHANNELS, STARSTIM_32_CHANNELS
from maryam_rt.integration.starstim_preprocessing import (
    StarstimLivePreprocessor,
    apply_linear_operator,
    bandpass_filter,
    peak_to_peak,
)


def _sine(freq: float, sfreq: float, n: float, n_channels: int = 31) -> np.ndarray:
    t = np.arange(int(n)) / sfreq
    return np.tile(np.sin(2 * np.pi * freq * t), (n_channels, 1)).astype(np.float32)


# --------------------------------------------------------------------------
# bandpass
# --------------------------------------------------------------------------
def test_bandpass_lowpass_attenuates_above_cutoff() -> None:
    sfreq = 500.0
    high = _sine(80.0, sfreq, 1500)
    low = _sine(10.0, sfreq, 1500)
    signal = high + low

    filtered = bandpass_filter(signal, sfreq, l_freq=None, h_freq=30.0)

    # 80 Hz component is strongly attenuated; 10 Hz component is preserved.
    assert peak_to_peak(filtered) < peak_to_peak(signal)
    assert np.corrcoef(filtered[0], low[0])[0, 1] > 0.95


def test_bandpass_skips_infeasible_highpass_on_short_epoch() -> None:
    sfreq = 500.0
    x = _sine(10.0, sfreq, 600)
    # 0.01 Hz highpass is below the realizable range on a short epoch -> highpass
    # skipped, so a pure in-band sine passes through nearly unchanged.
    out = bandpass_filter(x, sfreq, l_freq=0.01, h_freq=None)
    assert np.allclose(out, x, atol=1e-4)


def test_bandpass_rejects_out_of_range_cutoff() -> None:
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros((31, 100), np.float32), 500.0, l_freq=None, h_freq=300.0)


# --------------------------------------------------------------------------
# linear operators (ICA / MVNN apply)
# --------------------------------------------------------------------------
def test_apply_identity_operator_is_noop() -> None:
    x = np.random.default_rng(0).standard_normal((31, 250)).astype(np.float32)
    assert np.allclose(apply_linear_operator(x, np.eye(31, dtype=np.float32)), x)


def test_apply_known_operator_matches_matmul() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((31, 250)).astype(np.float32)
    op = rng.standard_normal((31, 31)).astype(np.float32)
    assert np.allclose(apply_linear_operator(x, op), op @ x, atol=1e-4)


def test_apply_operator_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        apply_linear_operator(np.zeros((31, 250), np.float32), np.eye(30, dtype=np.float32))


# --------------------------------------------------------------------------
# custom MVNN whitener
# --------------------------------------------------------------------------
def test_custom_mvnn_matches_reference_apply_mvnn() -> None:
    rng = np.random.default_rng(2)
    epochs = rng.standard_normal((20, 31, 250)).astype(np.float32)

    # Reference implementation copied from the offline apply_mvnn.
    pooled = epochs.transpose(1, 0, 2).reshape(31, -1)
    cov = np.cov(pooled)
    reference = np.real_if_close(scipy.linalg.fractional_matrix_power(cov, -0.5))

    whitener = compute_mvnn_whitener_custom(epochs)
    assert whitener.shape == (31, 31)
    assert np.allclose(whitener, reference, atol=1e-4)


def test_custom_mvnn_actually_whitens() -> None:
    rng = np.random.default_rng(3)
    # Correlated channels via a random mixing of white sources.
    mixing = rng.standard_normal((31, 31))
    epochs = np.einsum("ij,tjk->tik", mixing, rng.standard_normal((40, 31, 500)))
    whitener = compute_mvnn_whitener_custom(epochs.astype(np.float32))

    whitened = np.einsum("ij,tjk->tik", whitener, epochs)
    pooled = whitened.transpose(1, 0, 2).reshape(31, -1)
    cov = np.cov(pooled)
    assert np.allclose(cov, np.eye(31), atol=0.05)


def test_custom_mvnn_needs_enough_trials() -> None:
    with pytest.raises(ValueError):
        compute_mvnn_whitener_custom(np.zeros((1, 31, 250), np.float32))


# --------------------------------------------------------------------------
# rejection
# --------------------------------------------------------------------------
def test_preprocessor_flags_peak_to_peak_rejection() -> None:
    pre = StarstimLivePreprocessor(input_sfreq=500.0, reject_peak_to_peak_uv=5.0)
    raw = np.zeros((len(STARSTIM_32_CHANNELS), 600), np.float32)
    # Put a large deflection on one channel so the model epoch exceeds threshold.
    raw[0, 300:] = 1000.0
    result = pre.process(raw)
    assert result.rejected is True
    assert result.peak_to_peak_uv > 5.0


def test_preprocessor_accepts_quiet_epoch() -> None:
    pre = StarstimLivePreprocessor(input_sfreq=500.0, reject_peak_to_peak_uv=1e6)
    raw = np.random.default_rng(4).standard_normal((len(STARSTIM_32_CHANNELS), 600)).astype(np.float32)
    result = pre.process(raw)
    assert result.rejected is False
    assert result.epoch.shape == (31, 250)


# --------------------------------------------------------------------------
# preprocessor ordering / operator plumbing
# --------------------------------------------------------------------------
def test_ica_and_whitening_operators_are_applied() -> None:
    rng = np.random.default_rng(5)
    raw = rng.standard_normal((len(STARSTIM_32_CHANNELS), 600)).astype(np.float32)

    base = StarstimLivePreprocessor(input_sfreq=500.0)
    baseline_epoch = base(raw)

    # Whitening is the final step, so a known whitener multiplies the baseline epoch.
    whitener = rng.standard_normal((31, 31)).astype(np.float32)
    whitened = StarstimLivePreprocessor(input_sfreq=500.0, whitening_matrix=whitener)(raw)
    assert np.allclose(whitened, whitener @ baseline_epoch, atol=1e-3)

    # Identity ICA operator leaves the pipeline output unchanged.
    identity_ica = StarstimLivePreprocessor(input_sfreq=500.0, ica_operator=np.eye(31, dtype=np.float32))(raw)
    assert np.allclose(identity_ica, baseline_epoch, atol=1e-4)


def test_legacy_call_unchanged_when_no_options_set() -> None:
    raw = np.random.default_rng(6).standard_normal((len(STARSTIM_32_CHANNELS), 600)).astype(np.float32)
    pre = StarstimLivePreprocessor(input_sfreq=500.0)
    out = pre(raw)
    assert out.shape == (31, 250)
    assert out.dtype == np.float32


# --------------------------------------------------------------------------
# ICA cleaning operator: the extracted linear map must equal ica.apply
# --------------------------------------------------------------------------
def test_ica_operator_reproduces_ica_apply() -> None:
    mne = pytest.importorskip("mne")
    from mne.preprocessing import ICA

    from maryam_rt.integration.lab_calibration import fit_ica_cleaning_operator

    mne.set_log_level("ERROR")
    rng = np.random.default_rng(0)
    n_ch = len(STARSTIM_31_CHANNELS)
    data = (rng.standard_normal((n_ch, n_ch)) @ rng.standard_normal((n_ch, 5000))) * 1e-6

    result = fit_ica_cleaning_operator(
        data, 250.0, STARSTIM_31_CHANNELS, exclude_indices=[0, 2], n_components=n_ch
    )

    info = mne.create_info(list(STARSTIM_31_CHANNELS), 250.0, "eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw_fit = raw.copy().filter(1.0, 100.0, phase="zero", fir_design="firwin", verbose="ERROR")
    ica = ICA(n_components=n_ch, method="infomax", random_state=42)
    ica.fit(raw_fit, verbose="ERROR")
    ica.exclude = [0, 2]
    cleaned = raw.copy()
    ica.apply(cleaned, verbose="ERROR")

    assert np.abs(result.operator @ data - cleaned.get_data()).max() < 1e-5


# --------------------------------------------------------------------------
# operator save / load round-trip
# --------------------------------------------------------------------------
def _lab_trial(epoch_model: np.ndarray, *, usable=True, rejected=False, index=0):
    from maryam_rt.integration.lab_replay import LabTrial

    return LabTrial(
        trial_index=index,
        epoch_index=index,
        target_index=index,
        trigger=1001 + index,
        subject="P04",
        lab_subject_id=0,
        category="cat",
        split="train",
        epoch_lab31=epoch_model,
        epoch_model=epoch_model,
        image_path=None,
        usable_for_finetune=usable,
        artifact_rejected=rejected,
    )


def test_first_block_and_rejection_gate_whitener_calibration() -> None:
    from maryam_rt.integration.lab_replay import _calibration_epochs, _compute_lab_mvnn_whitener

    rng = np.random.default_rng(8)
    trials = [_lab_trial(rng.standard_normal((31, 250)).astype(np.float32), index=i) for i in range(10)]
    trials[1] = _lab_trial(trials[1].epoch_model, rejected=True, index=1)
    trials[2] = _lab_trial(trials[2].epoch_model, usable=False, index=2)

    # first_block=5 with two of the first five gated out -> 3 calibration epochs.
    stacked = _calibration_epochs(trials, first_block=5)
    assert stacked.shape == (3, 31, 250)

    whitener = _compute_lab_mvnn_whitener(trials, method="custom", first_block=5)
    assert whitener.shape == (31, 31)
    assert np.isfinite(whitener).all()


def test_operator_round_trip(tmp_path) -> None:
    op = np.random.default_rng(7).standard_normal((31, 31)).astype(np.float32)
    path = tmp_path / "whitener.npy"
    save_operator(op, path, kind="mvnn", ch_names=STARSTIM_31_CHANNELS, extra={"note": "test"})
    loaded = load_operator(path, expected_channels=31)
    assert np.allclose(loaded, op)
    with pytest.raises(ValueError):
        load_operator(path, expected_channels=30)
