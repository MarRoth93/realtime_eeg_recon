from __future__ import annotations

import numpy as np
import pytest

from maryam_rt.integration.lab_replay import _load_easy_segment_uv
from maryam_rt.realtime.streaming.lsl_inlet import LSLInletConfig, LSLInletWrapper


class _EmptyNode:
    def child(self, _name: str) -> _EmptyNode:
        return self

    def child_value(self, _name: str) -> str:
        return ""

    def next_sibling(self) -> _EmptyNode:
        return self


class _ChannelNode:
    def __init__(self, channels: list[dict[str, str]], index: int = 0) -> None:
        self.channels = channels
        self.index = index

    def child_value(self, name: str) -> str:
        return self.channels[self.index].get(name, "")

    def next_sibling(self) -> _ChannelNode | _EmptyNode:
        next_index = self.index + 1
        if next_index >= len(self.channels):
            return _EmptyNode()
        return _ChannelNode(self.channels, next_index)


class _ChannelsNode:
    def __init__(self, channels: list[dict[str, str]]) -> None:
        self.channels = channels

    def child(self, name: str) -> _ChannelNode | _EmptyNode:
        if name == "channel" and self.channels:
            return _ChannelNode(self.channels)
        return _EmptyNode()


class _DescriptionNode:
    def __init__(self, channels: list[dict[str, str]]) -> None:
        self.channels = channels

    def child(self, name: str) -> _ChannelsNode | _ChannelNode | _EmptyNode:
        if name == "channels":
            return _ChannelsNode(self.channels)
        if name == "channel" and self.channels:
            return _ChannelNode(self.channels)
        return _EmptyNode()


class _StreamInfo:
    def __init__(self, units: list[str]) -> None:
        self.channels = [
            {"label": f"Ch{index + 1}", "unit": unit}
            for index, unit in enumerate(units)
        ]

    def channel_count(self) -> int:
        return len(self.channels)

    def nominal_srate(self) -> float:
        return 500.0

    def name(self) -> str:
        return "StarstimEEG"

    def type(self) -> str:
        return "EEG"

    def source_id(self) -> str:
        return "starstim-test"

    def desc(self) -> _DescriptionNode:
        return _DescriptionNode(self.channels)


def _microvolt_inlet(units: list[str], *, fallback: str = "microvolts") -> LSLInletWrapper:
    inlet = LSLInletWrapper(
        LSLInletConfig(
            stream_name="StarstimEEG",
            stream_type=None,
            channel_count=len(units),
            sampling_rate=500.0,
            convert_to_microvolts=True,
            unit_fallback=fallback,
        )
    )
    inlet._configure_from_info(_StreamInfo(units))
    return inlet


def test_lsl_nanovolts_match_raw_easy_microvolt_conversion(tmp_path) -> None:
    samples_nv = np.asarray(
        [
            [(channel + 1) * 1000 + sample * 250 for channel in range(32)]
            for sample in range(3)
        ],
        dtype=np.float32,
    )
    easy_path = tmp_path / "recording.easy"
    easy_path.write_text(
        "\n".join("\t".join(str(value) for value in row) for row in samples_nv) + "\n"
    )

    replay_microvolts = _load_easy_segment_uv(easy_path, start_sample=0, samples=3)
    inlet = _microvolt_inlet(["nanovolts"] * 32)
    inlet._append_samples(samples_nv.tolist())
    live_microvolts = inlet.get_window(3, timeout=0.0)

    assert inlet.input_unit == "nanovolts"
    assert inlet.output_unit == "microvolts"
    np.testing.assert_allclose(live_microvolts, replay_microvolts)


def test_lsl_uses_configured_unit_fallback_when_metadata_is_missing() -> None:
    inlet = _microvolt_inlet([""] * 2, fallback="nanovolts")
    inlet._append_samples([[1000.0, -2000.0]])

    assert inlet.input_unit == "nanovolts"
    np.testing.assert_allclose(inlet.get_window(1, timeout=0.0), [[1.0], [-2.0]])


def test_lsl_rejects_mixed_channel_units() -> None:
    with pytest.raises(ValueError, match="mixes channel units"):
        _microvolt_inlet(["microvolts", "nanovolts"])


def test_lsl_rejects_partially_declared_channel_units() -> None:
    with pytest.raises(ValueError, match="only some channels"):
        _microvolt_inlet(["", "microvolts"])
