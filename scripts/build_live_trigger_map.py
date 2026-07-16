#!/usr/bin/env python3
"""Build the live-pipeline trigger map from a session image-mapping JSON.

The stimulus PC writes an ``*_image_mapping.json`` describing which numeric
marker (trigger) was assigned to each target image for a session. The live
pipeline needs the reverse lookup: trigger -> target image path, so an incoming
LSL marker can be resolved to the image to reconstruct. This script emits that
whitelist as a TSV in the format ``load_live_trigger_map`` expects.

Only the ``sampled`` group is exported: those 160 entries are the unique target
images (triggers 1001-1160). ``block_trials`` is just the presentation order
drawn from that same set, and ``recall_test`` (301-310) are post-block probes,
so neither adds new trigger->image pairs.

sample_id (the ``sampled`` key, 1..N) relates to the other ids as:
    trigger      = 1000 + sample_id
    target_index = sample_id - 1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "target_index",
    "sample_id",
    "trigger",
    "category",
    "mapped_path",
    "exact_image_path",
]


def build_rows(mapping_path: Path, data_root: Path) -> list[dict[str, str]]:
    payload = json.loads(mapping_path.read_text())
    sampled = payload.get("sampled")
    if not isinstance(sampled, dict) or not sampled:
        raise SystemExit(f"No 'sampled' target images found in {mapping_path}")

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for sample_id, entry in sampled.items():
        # Mapping paths are already rooted at "images_THINGS/object_images/...",
        # so they resolve directly under the data root.
        rel_path = str(entry["path"])
        candidate = data_root / rel_path
        if not candidate.exists():
            missing.append(rel_path)
        rows.append(
            {
                "target_index": str(int(sample_id) - 1),
                "sample_id": str(int(sample_id)),
                "trigger": str(int(entry["trigger"])),
                "category": str(entry.get("category") or ""),
                "mapped_path": rel_path,
                "exact_image_path": str(candidate.resolve()),
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"{len(missing)} target image(s) not found under {data_root} (e.g. {preview}). "
            "Pass the correct --data-root."
        )
    rows.sort(key=lambda r: int(r["trigger"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mapping_json", type=Path, help="Session *_image_mapping.json from the stimulus PC.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Base the mapping's relative image paths resolve against (default: data/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "finetune" / "lab_target_manifest.tsv",
        help="Where to write the trigger-map TSV.",
    )
    args = parser.parse_args()

    mapping_path = args.mapping_json if args.mapping_json.is_absolute() else PROJECT_ROOT / args.mapping_json
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    rows = build_rows(mapping_path, data_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} target triggers -> {args.output}")
    print(f"trigger range: {rows[0]['trigger']}..{rows[-1]['trigger']}  data_root: {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
