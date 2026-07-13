from __future__ import annotations

from typing import Any

from pylsl import resolve_streams


def _channel_labels(info: object) -> list[str]:
    labels: list[str] = []
    try:
        desc = info.desc()
        channel = desc.child("channels").child("channel")
        if not (channel.child_value("label") or channel.child_value("name")):
            channel = desc.child("channel")
        for _ in range(int(info.channel_count())):
            label = channel.child_value("label") or channel.child_value("name")
            if label:
                labels.append(str(label))
            channel = channel.next_sibling()
    except Exception:
        return []
    return labels


def discover_lsl_streams(wait_time: float = 0.5) -> list[dict[str, Any]]:
    """Return browser-safe metadata for every currently visible LSL stream."""
    streams: list[dict[str, Any]] = []
    for info in resolve_streams(wait_time=wait_time):
        streams.append(
            {
                "name": str(info.name()),
                "type": str(info.type()),
                "source_id": str(info.source_id()),
                "channel_count": int(info.channel_count()),
                "nominal_sampling_rate": float(info.nominal_srate()),
                "channel_format": str(info.channel_format()),
                "channel_labels": _channel_labels(info),
            }
        )
    return streams


def recommend_streams(
    streams: list[dict[str, Any]],
    *,
    expected_channels: int = 32,
    expected_sampling_rate: float = 500.0,
    preferred_eeg_name: str | None = None,
    preferred_marker_name: str | None = None,
) -> dict[str, dict[str, Any] | None]:
    def eeg_score(stream: dict[str, Any]) -> tuple[int, int, float]:
        return (
            int(stream["name"] == preferred_eeg_name),
            int(stream["type"].lower() == "eeg" and stream["channel_count"] == expected_channels),
            -abs(float(stream["nominal_sampling_rate"]) - expected_sampling_rate),
        )

    def marker_score(stream: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(stream["name"] == preferred_marker_name),
            int(stream["type"].lower() in {"markers", "marker"}),
            int(stream["channel_count"] == 1),
        )

    eeg_candidates = [stream for stream in streams if stream["type"].lower() == "eeg"]
    marker_candidates = [
        stream for stream in streams if stream["type"].lower() in {"markers", "marker"}
    ]
    return {
        "eeg": max(eeg_candidates, key=eeg_score) if eeg_candidates else None,
        "markers": max(marker_candidates, key=marker_score) if marker_candidates else None,
    }
