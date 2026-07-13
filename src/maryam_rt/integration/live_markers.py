from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _normalise_target(trigger: object, row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    trigger_value = "" if trigger is None else str(trigger).strip()
    if not trigger_value:
        raise ValueError("Live trigger map contains an empty trigger value.")

    raw_path = str(
        row.get("exact_image_path")
        or row.get("image_path")
        or row.get("target_image_path")
        or ""
    ).strip()
    image_path = Path(raw_path).expanduser() if raw_path else None
    image_id = str(row.get("image_id") or "").strip()
    if not image_id and image_path is not None:
        image_id = image_path.stem

    return trigger_value, {
        "event": "stim_onset",
        "image_id": image_id,
        "image_path": str(image_path) if image_path is not None and image_path.exists() else "",
        "category": str(row.get("category") or ""),
        "target_index": str(row.get("target_index") or ""),
        "legacy_trigger": trigger_value,
    }


def load_live_trigger_map(path: str | Path) -> dict[str, dict[str, str]]:
    """Load the authoritative whitelist for legacy numeric stimulus markers."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Live trigger map not found: {source}")

    mapped: dict[str, dict[str, str]] = {}
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, dict):
            rows = payload.get("trials") or payload.get("mapping") or payload.get("targets")
            if rows is None:
                rows = [dict(value, trigger=key) if isinstance(value, dict) else {"trigger": key, "image_id": value} for key, value in payload.items()]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported JSON trigger-map structure in {source}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            trigger = row.get("trigger") or row.get("code") or row.get("marker")
            key, value = _normalise_target(trigger, row)
            mapped[key] = value
    else:
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "trigger" not in reader.fieldnames:
                raise ValueError(f"TSV trigger map needs a 'trigger' column: {source}")
            for row in reader:
                key, value = _normalise_target(row.get("trigger"), row)
                mapped[key] = value

    if not mapped:
        raise ValueError(f"Live trigger map is empty: {source}")
    return mapped
