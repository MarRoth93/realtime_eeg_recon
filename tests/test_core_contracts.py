from __future__ import annotations

import numpy as np
import pytest
import torch

from maryam_rt.integration.lab_replay import (
    STARSTIM_31_CHANNELS as LAB_STARSTIM_31_CHANNELS,
    adapt_starstim31_to_things63,
)
from maryam_rt.integration.marker_payload import parse_marker_payload
from maryam_rt.integration.resample import (
    STARSTIM_31_CHANNELS,
    STARSTIM_32_CHANNELS,
    THINGS_63_CHANNELS,
    drop_fz_starstim32_torch,
    prepare_realtime_window_torch,
    prepare_starstim32_native_window_torch,
    starstim31_to_things63_torch,
)
from maryam_rt.integration.starstim31_atms import Starstim31ATMSEmbedder
from maryam_rt.integration.starstim_preprocessing import (
    StarstimLivePreprocessor,
    average_reference,
    baseline_correct,
    crop_model_window,
    resample_epoch,
    select_starstim31_channels,
)


def test_marker_payload_accepts_plain_and_json() -> None:
    plain = parse_marker_payload("stim_onset")
    assert plain.event_name == "stim_onset"
    assert plain.image_id is None

    parsed = parse_marker_payload(
        '{"event": "stim_onset", "image_id": 42, "image_path": "/tmp/img.png", "block": 3}'
    )
    assert parsed.event_name == "stim_onset"
    assert parsed.image_id == "42"
    assert parsed.image_path == "/tmp/img.png"
    assert parsed.metadata == {"block": 3}


def test_realtime_window_keeps_first_63_channels_and_downsamples() -> None:
    x = torch.arange(1 * 64 * 8, dtype=torch.float32).reshape(1, 64, 8)
    out = prepare_realtime_window_torch(x, keep_channels=63, downsample_factor=4)
    assert tuple(out.shape) == (1, 63, 2)
    expected = x[:, :63, :].reshape(1, 63, 2, 4).mean(dim=3)
    torch.testing.assert_close(out, expected)


def test_starstim31_mapping_matches_lab_replay_constants() -> None:
    assert STARSTIM_31_CHANNELS == LAB_STARSTIM_31_CHANNELS
    x = torch.arange(1 * 31 * 4, dtype=torch.float32).reshape(1, 31, 4)
    out = starstim31_to_things63_torch(x)
    assert tuple(out.shape) == (1, 63, 4)

    things_index = {name: idx for idx, name in enumerate(THINGS_63_CHANNELS)}
    for lab_idx, channel in enumerate(STARSTIM_31_CHANNELS):
        torch.testing.assert_close(out[:, things_index[channel], :], x[:, lab_idx, :])

    missing_channels = set(THINGS_63_CHANNELS) - set(STARSTIM_31_CHANNELS)
    for channel in missing_channels:
        assert torch.count_nonzero(out[:, things_index[channel], :]).item() == 0


def test_lab_numpy_adapter_places_channels_and_rejects_missing_names() -> None:
    epoch = np.arange(31 * 5, dtype=np.float32).reshape(31, 5)
    adapted = adapt_starstim31_to_things63(epoch, list(STARSTIM_31_CHANNELS))
    assert adapted.shape == (63, 5)
    things_index = {name: idx for idx, name in enumerate(THINGS_63_CHANNELS)}
    for lab_idx, channel in enumerate(STARSTIM_31_CHANNELS):
        np.testing.assert_allclose(adapted[things_index[channel]], epoch[lab_idx])

    with pytest.raises(ValueError, match="missing expected Starstim31"):
        adapt_starstim31_to_things63(epoch, ["bad"] * 31)


def test_starstim32_native_drop_fz_downsamples_to_31_channels() -> None:
    x = torch.arange(1 * 32 * 8, dtype=torch.float32).reshape(1, 32, 8)
    dropped = drop_fz_starstim32_torch(x)
    assert tuple(dropped.shape) == (1, 31, 8)
    assert STARSTIM_32_CHANNELS[12] == "Fz"
    torch.testing.assert_close(dropped[:, 12, :], x[:, 13, :])

    out = prepare_starstim32_native_window_torch(x, downsample_factor=4, drop_fz=True)
    expected = dropped.reshape(1, 31, 2, 4).mean(dim=3)
    torch.testing.assert_close(out, expected)


def test_starstim_live_preprocessor_converts_500hz_starstim32_to_31x250() -> None:
    samples = 600
    common = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    channel_offsets = np.arange(32, dtype=np.float32)[:, None]
    raw = channel_offsets + common[None, :]

    preprocessor = StarstimLivePreprocessor(input_sfreq=500.0, tmin=-0.2, tmax=1.0)
    processed = preprocessor(raw)

    assert processed.shape == (31, 250)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(processed.mean(axis=0, dtype=np.float64), 0.0, atol=1e-4)


def test_starstim_preprocessing_steps_match_offline_contract() -> None:
    common = np.linspace(-0.5, 0.5, 600, dtype=np.float32)
    raw = np.arange(32, dtype=np.float32)[:, None] * 0.01 + common[None, :]
    selected = select_starstim31_channels(raw)
    assert selected.shape == (31, 600)
    np.testing.assert_allclose(selected[12], raw[13])

    referenced = average_reference(selected)
    np.testing.assert_allclose(referenced.mean(axis=0, dtype=np.float64), 0.0, atol=1e-4)

    resampled = resample_epoch(referenced, input_sfreq=500.0, target_sfreq=250.0, duration_s=1.2)
    assert resampled.shape == (31, 300)
    times = -0.2 + np.arange(300, dtype=np.float32) / 250.0
    corrected = baseline_correct(resampled, times, baseline=(-0.2, 0.0))
    np.testing.assert_allclose(corrected[:, times < 0].mean(axis=1), 0.0, atol=1e-5)

    cropped = crop_model_window(corrected, times)
    assert cropped.shape == (31, 250)


class _FakeATMSModel:
    def __call__(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        assert tuple(x.shape[1:]) == (31, 250)
        assert tuple(subject_ids.shape) == (x.shape[0],)
        return torch.ones((x.shape[0], 1024), device=x.device)


def test_starstim31_atms_embedder_shape_contract_without_checkpoint() -> None:
    embedder = Starstim31ATMSEmbedder.__new__(Starstim31ATMSEmbedder)
    embedder.device_name = "cpu"
    embedder.num_subjects = 11
    embedder.model = _FakeATMSModel()

    out = embedder.embed(np.zeros((31, 250), dtype=np.float32), subject_id=4)
    assert tuple(out.shape) == (1, 1024)

    with pytest.raises(ValueError, match="outside checkpoint range"):
        embedder.embed(np.zeros((31, 250), dtype=np.float32), subject_id=11)
    with pytest.raises(ValueError, match="Expected Starstim31 epoch"):
        embedder.embed(np.zeros((32, 250), dtype=np.float32), subject_id=4)
