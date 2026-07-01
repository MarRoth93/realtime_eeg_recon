#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side figure of target, low-level, and high-level reconstructions."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gui_things_replay",
        help="Output root containing targets/, low_level/, high_level/, and optional metadata/.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of examples to include. Uses the latest common examples by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to <root>/comparisons/reconstruction_comparison_<count>.png",
    )
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Use the oldest matching examples instead of the latest ones.",
    )
    return parser.parse_args()


def stem_without_suffix(path: Path, suffix: str) -> str | None:
    name = path.name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def indexed_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.exists():
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stem = stem_without_suffix(path, suffix)
        if stem is not None:
            indexed[stem] = path
    return indexed


def load_label(metadata_dir: Path, stem: str) -> str | None:
    metadata_path = metadata_dir / f"{stem}.json"
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text())
    except Exception:
        return None
    image_path = data.get("image_path")
    if not image_path:
        return None
    parent_name = Path(image_path).parent.name
    if "_" in parent_name:
        return parent_name.split("_", 1)[1]
    return parent_name


def build_examples(root: Path) -> list[dict[str, object]]:
    target_dir = root / "targets"
    low_dir = root / "low_level"
    high_dir = root / "high_level"
    metadata_dir = root / "metadata"

    targets = indexed_files(target_dir, "_target.jpg")
    targets.update(indexed_files(target_dir, "_target.jpeg"))
    targets.update(indexed_files(target_dir, "_target.png"))
    lows = indexed_files(low_dir, "_low.png")
    highs = indexed_files(high_dir, "_refined.png")

    common_stems = sorted(set(targets) & set(lows) & set(highs))
    examples: list[dict[str, object]] = []
    for stem in common_stems:
        examples.append(
            {
                "stem": stem,
                "target": targets[stem],
                "low": lows[stem],
                "high": highs[stem],
                "label": load_label(metadata_dir, stem),
            }
        )
    return examples


def default_output_path(root: Path, count: int) -> Path:
    out_dir = root / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"reconstruction_comparison_{count}.png"


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    examples = build_examples(args.root)
    if not examples:
        raise SystemExit(f"No matching target/low_level/high_level triplets found under {args.root}")

    if args.oldest_first:
        selected = examples[: args.count]
    else:
        selected = examples[-args.count :]

    output_path = args.output or default_output_path(args.root, len(selected))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, max(3 * n_rows, 3)))
    if n_rows == 1:
        axes = [axes]

    column_titles = ["Target", "Low-Level", "High-Level"]

    for row_idx, example in enumerate(selected):
        row_axes = axes[row_idx]
        images = [
            Image.open(example["target"]).convert("RGB"),
            Image.open(example["low"]).convert("RGB"),
            Image.open(example["high"]).convert("RGB"),
        ]
        for col_idx, (ax, image, title) in enumerate(zip(row_axes, images, column_titles)):
            ax.imshow(image)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=12)
            if col_idx == 0:
                label = example["label"] or str(example["stem"])
                ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=48)

    fig.suptitle(f"Reconstruction Comparison ({len(selected)} example{'s' if len(selected) != 1 else ''})", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
