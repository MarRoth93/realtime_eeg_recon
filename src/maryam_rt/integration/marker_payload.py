from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedMarkerPayload:
    raw_value: str
    event_name: str
    image_id: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_marker_payload(value: str) -> ParsedMarkerPayload:
    raw = str(value)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedMarkerPayload(raw_value=raw, event_name=raw)

    if not isinstance(parsed, dict):
        return ParsedMarkerPayload(raw_value=raw, event_name=raw)

    event_name = str(parsed.get("event") or parsed.get("marker") or raw)
    image_id = parsed.get("image_id")
    image_path = parsed.get("image_path")
    metadata = {
        key: val
        for key, val in parsed.items()
        if key not in {"event", "marker", "image_id", "image_path"}
    }
    return ParsedMarkerPayload(
        raw_value=raw,
        event_name=event_name,
        image_id=None if image_id is None else str(image_id),
        image_path=None if image_path is None else str(image_path),
        metadata=metadata,
    )

