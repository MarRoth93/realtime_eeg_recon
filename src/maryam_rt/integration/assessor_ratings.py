from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSESSOR_DIR = PROJECT_ROOT / "checkpoints" / "assessor"
DEFAULT_VA_BUNDLE = DEFAULT_ASSESSOR_DIR / "assessor_va_mixed_v4_bundle.pt"
DEFAULT_SIX_BUNDLE = DEFAULT_ASSESSOR_DIR / "assessor_six_clip_v3_bundle.pt"
SUMMARY_DIMS = ("valence", "arousal", "approach", "attention", "control", "dominance")


class AssessorBundle:
    """Frozen assessor head and calibration metadata for one output family."""

    def __init__(self, bundle_path: str | Path, device: torch.device) -> None:
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Assessor bundle not found: {self.bundle_path}")
        bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=True)
        if bundle.get("head") != "linear":
            raise ValueError(f"Only linear CLIP assessor bundles are supported: {self.bundle_path}")

        self.targets = [str(target) for target in bundle["targets"]]
        self.backbone_config = bundle["backbone_config"]
        self.scaler_x_mean = bundle["scaler_x_mean"].to(device)
        self.scaler_x_scale = bundle["scaler_x_scale"].to(device)
        self.scaler_y_mean = bundle["scaler_y_mean"].to(device)
        self.scaler_y_scale = bundle["scaler_y_scale"].to(device)
        self.feature_indices = bundle["feature_indices"].to(device)
        self.calibration = bundle.get("calibration")
        self.conformal = bundle.get("conformal")
        self.ood_stats = bundle.get("ood")
        if self.calibration is not None:
            self.calibration = {key: value.to(device) for key, value in self.calibration.items()}
        if self.ood_stats is not None:
            self.ood_stats = {key: value.to(device) for key, value in self.ood_stats.items()}

        self.head = nn.Linear(len(self.feature_indices), len(self.targets))
        self.head.load_state_dict(bundle["head_state_dict"])
        self.head.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def predict_features(self, features: torch.Tensor) -> np.ndarray:
        x = (features.to(self.device) - self.scaler_x_mean) / self.scaler_x_scale
        x = x[:, self.feature_indices]
        y = self.head(x)
        y = y * self.scaler_y_scale + self.scaler_y_mean
        if self.calibration is not None:
            y = y * self.calibration["slope"] + self.calibration["intercept"]
        return y.cpu().numpy()

    def interval_halfwidths(self, level: float) -> np.ndarray | None:
        if self.conformal is None:
            return None
        key = f"{level:.2f}"
        if key not in self.conformal:
            raise ValueError(
                f"Interval level {level} is not available in {self.bundle_path}; "
                f"choose from {sorted(self.conformal)}."
            )
        return self.conformal[key].cpu().numpy()

    @torch.inference_mode()
    def ood_percentile(self, features: torch.Tensor) -> np.ndarray | None:
        if self.ood_stats is None:
            return None
        x = (features.to(self.device) - self.scaler_x_mean) / self.scaler_x_scale
        diff = x - self.ood_stats["mean"]
        d2 = (diff @ self.ood_stats["precision"] * diff).sum(dim=1)
        dist = torch.sqrt(torch.clamp(d2, min=0)).cpu().numpy()
        quantiles = self.ood_stats["train_quantiles"].cpu().numpy()
        return np.interp(dist, quantiles, np.linspace(0, 100, len(quantiles)))


class AssessorRatingEngine:
    """Rate reconstructed images with the local VA and six-dimension assessors."""

    def __init__(
        self,
        va_bundle: str | Path = DEFAULT_VA_BUNDLE,
        six_bundle: str | Path = DEFAULT_SIX_BUNDLE,
        device: str | None = None,
        interval_level: float | None = 0.9,
        include_ood: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.interval_level = interval_level
        self.include_ood = include_ood
        self.bundles = {
            "va": AssessorBundle(va_bundle, self.device),
            "six": AssessorBundle(six_bundle, self.device),
        }
        self.backbones = self._build_backbones(self.bundles["va"].backbone_config)
        for name, bundle in self.bundles.items():
            if bundle.backbone_config != self.bundles["va"].backbone_config:
                raise ValueError(f"Assessor bundle {name} uses a different backbone config.")

    def _build_backbones(self, configs: list[dict[str, Any]]) -> list[tuple[nn.Module, Any]]:
        try:
            from timm import create_model
            from timm.data import create_transform
        except ImportError as exc:
            raise RuntimeError("Assessor ratings require timm: pip install timm") from exc

        pairs = []
        for cfg in configs:
            model = create_model(cfg["model_id"], pretrained=True, num_classes=0)
            transform = create_transform(**cfg["data_config"], is_training=False)
            pairs.append((model.eval().to(self.device), transform))
        return pairs

    @torch.inference_mode()
    def extract_features(self, images: list[Image.Image]) -> torch.Tensor:
        parts = []
        for model, transform in self.backbones:
            batch = torch.stack([transform(image.convert("RGB")) for image in images])
            parts.append(model(batch.to(self.device)).float())
        return torch.cat(parts, dim=1)

    def rate_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        features = self.extract_features(images)
        rows = [{"models": {}} for _ in images]
        for model_name, bundle in self.bundles.items():
            predictions = bundle.predict_features(features)
            halfwidths = (
                bundle.interval_halfwidths(self.interval_level)
                if self.interval_level is not None
                else None
            )
            ood = bundle.ood_percentile(features) if self.include_ood else None
            for idx, prediction in enumerate(predictions):
                values: dict[str, float] = {}
                for target, value in zip(bundle.targets, prediction):
                    target_key = target.lower()
                    values[target_key] = float(value)
                if halfwidths is not None:
                    for target, value, width in zip(bundle.targets, prediction, halfwidths):
                        target_key = target.lower()
                        values[f"{target_key}_low"] = float(value - width)
                        values[f"{target_key}_high"] = float(value + width)
                if ood is not None:
                    values["ood_percentile"] = float(ood[idx])
                rows[idx]["models"][model_name] = values
        return rows

    def rate_path(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        with Image.open(path) as raw:
            raw.load()
            image = ImageOps.exif_transpose(raw)
        return self.rate_images([image])[0]


class AssessorRatingWriter:
    """Write assessor ratings for reconstructed images as JSON sidecars."""

    def __init__(self, engine: AssessorRatingEngine, output_dir: str | Path) -> None:
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def assess_path(self, image_path: str | Path, stem: str, image_role: str) -> dict[str, Any]:
        return self.assess_paths({image_role: image_path}, stem)[image_role]

    def assess_paths(self, image_paths: dict[str, str | Path], stem: str) -> dict[str, Any]:
        roles = list(image_paths)
        paths = {role: Path(path) for role, path in image_paths.items()}
        images = []
        for role in roles:
            with Image.open(paths[role]) as raw:
                raw.load()
                images.append(ImageOps.exif_transpose(raw))
        with self._lock:
            ratings_by_role = self.engine.rate_images(images)

        results: dict[str, Any] = {}
        for role, ratings in zip(roles, ratings_by_role):
            output_path = self.output_dir / f"{stem}_{role}_assessor.json"
            payload = {
                "image_path": str(paths[role]),
                "image_role": role,
                "ratings": ratings,
            }
            output_path.write_text(json.dumps(payload, indent=2))
            results[role] = {
                "path": str(output_path),
                "image_role": role,
                "ratings": ratings,
            }
        return results

    def write_triplet_summary(
        self,
        *,
        stem: str,
        target_path: str | Path,
        low_level_path: str | Path,
        high_level_path: str | Path,
        title: str | None = None,
    ) -> dict[str, Any]:
        records = self.assess_paths(
            {
                "target": target_path,
                "low_level": low_level_path,
                "high_level": high_level_path,
            },
            stem,
        )
        figure_path = self.output_dir / f"{stem}_assessor_summary.png"
        summary_path = self.output_dir / f"{stem}_assessor_summary.json"
        self._write_summary_figure(
            figure_path=figure_path,
            stem=stem,
            title=title or stem,
            image_paths={
                "target": Path(target_path),
                "low_level": Path(low_level_path),
                "high_level": Path(high_level_path),
            },
            records=records,
        )
        payload = {
            "stem": stem,
            "figure_path": str(figure_path),
            "records": records,
        }
        summary_path.write_text(json.dumps(payload, indent=2))
        return {
            "path": str(figure_path),
            "summary_path": str(summary_path),
            "records": records,
        }

    def _write_summary_figure(
        self,
        *,
        figure_path: Path,
        stem: str,
        title: str,
        image_paths: dict[str, Path],
        records: dict[str, Any],
    ) -> None:
        width, height = 1800, 1420
        margin = 56
        image_size = 430
        gap = 44
        card_width = image_size + 38
        top = 130
        figure = Image.new("RGB", (width, height), "#f7f4ec")
        draw = ImageDraw.Draw(figure)

        title_font = _font(42, bold=True)
        subtitle_font = _font(22)
        heading_font = _font(25, bold=True)
        body_font = _font(21)
        small_font = _font(18)

        draw.text((margin, 38), "Realtime EEG Reconstruction Summary", fill="#18231f", font=title_font)
        draw.text((margin, 88), title, fill="#526057", font=subtitle_font)

        role_labels = {
            "target": "Original target",
            "low_level": "Low-level recon",
            "high_level": "High-level recon",
        }
        starts = [margin, margin + card_width + gap, margin + 2 * (card_width + gap)]
        for x, role in zip(starts, ("target", "low_level", "high_level")):
            _round_rect(draw, (x, top, x + card_width, top + 610), radius=14, fill="#ffffff", outline="#d9d4c8")
            _draw_centered(draw, role_labels[role], (x, top + 18, x + card_width, top + 54), heading_font, "#1f2a25")
            thumb = _image_tile(image_paths[role], image_size)
            figure.paste(thumb, (x + 19, top + 72))
            summary = _image_score_summary(records[role])
            y = top + 522
            draw.text((x + 24, y), summary[0], fill="#1f2a25", font=body_font)
            draw.text((x + 24, y + 34), summary[1], fill="#69716c", font=small_font)

        table_top = top + 660
        _draw_score_table(draw, records, (margin, table_top), width - 2 * margin, body_font, heading_font, small_font)

        distance_top = table_top + 405
        _draw_distance_panel(draw, records, (margin, distance_top), width - 2 * margin, heading_font, body_font, small_font)

        draw.text(
            (margin, height - 46),
            f"Generated for {stem}. Scores are assessor predictions on the original rating scale.",
            fill="#69716c",
            font=small_font,
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.save(figure_path)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _round_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: str | None = None,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        text,
        fill=fill,
        font=font,
    )


def _image_tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as raw:
        raw.load()
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "#ede8de")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def _model_values(record: dict[str, Any]) -> dict[str, float]:
    models = record["ratings"]["models"]
    va = models.get("va", {})
    six = models.get("six", {})
    values: dict[str, float] = {}
    for dim in SUMMARY_DIMS:
        source = va if dim in {"valence", "arousal"} and dim in va else six
        if dim in source:
            values[dim] = float(source[dim])
    return values


def _ood_value(record: dict[str, Any]) -> float | None:
    models = record["ratings"]["models"]
    if "va" in models and "ood_percentile" in models["va"]:
        return float(models["va"]["ood_percentile"])
    if "six" in models and "ood_percentile" in models["six"]:
        return float(models["six"]["ood_percentile"])
    return None


def _image_score_summary(record: dict[str, Any]) -> tuple[str, str]:
    values = _model_values(record)
    valence = values.get("valence")
    arousal = values.get("arousal")
    first = "Valence --  Arousal --"
    if valence is not None and arousal is not None:
        first = f"Valence {valence:.2f}  |  Arousal {arousal:.2f}"
    ood = _ood_value(record)
    second = "OOD percentile --"
    if ood is not None:
        second = f"OOD percentile {ood:.1f}"
    return first, second


def _score_color(value: float) -> tuple[int, int, int]:
    low = np.array([98, 127, 176], dtype=np.float32)
    mid = np.array([242, 239, 230], dtype=np.float32)
    high = np.array([217, 143, 67], dtype=np.float32)
    value = float(np.clip(value, 1.0, 9.0))
    if value <= 5.0:
        t = (value - 1.0) / 4.0
        rgb = low * (1 - t) + mid * t
    else:
        t = (value - 5.0) / 4.0
        rgb = mid * (1 - t) + high * t
    return tuple(int(v) for v in rgb)


def _draw_score_table(
    draw: ImageDraw.ImageDraw,
    records: dict[str, Any],
    origin: tuple[int, int],
    width: int,
    body_font: ImageFont.ImageFont,
    heading_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x, y = origin
    _round_rect(draw, (x, y, x + width, y + 370), radius=14, fill="#ffffff", outline="#d9d4c8")
    draw.text((x + 24, y + 20), "Assessor score matrix", fill="#1f2a25", font=heading_font)
    draw.text(
        (x + 24, y + 56),
        "Primary valence/arousal from VA model; remaining dimensions from six-dimension model.",
        fill="#69716c",
        font=small_font,
    )
    role_labels = [("target", "Target"), ("low_level", "Low-level"), ("high_level", "High-level")]
    label_w = 250
    cell_w = (width - label_w - 48) // 3
    row_h = 36
    start_y = y + 102
    for col, (_, label) in enumerate(role_labels):
        _draw_centered(
            draw,
            label,
            (x + label_w + col * cell_w, start_y, x + label_w + (col + 1) * cell_w, start_y + row_h),
            body_font,
            "#1f2a25",
        )
    for row, dim in enumerate(SUMMARY_DIMS):
        row_y = start_y + (row + 1) * row_h
        draw.text((x + 24, row_y + 6), dim.capitalize(), fill="#1f2a25", font=body_font)
        for col, (role, _) in enumerate(role_labels):
            values = _model_values(records[role])
            value = values.get(dim)
            cell = (
                x + label_w + col * cell_w + 8,
                row_y + 3,
                x + label_w + (col + 1) * cell_w - 8,
                row_y + row_h - 3,
            )
            fill = "#f0eee8" if value is None else _score_color(value)
            _round_rect(draw, cell, radius=8, fill=fill, outline=None)
            text = "--" if value is None else f"{value:.2f}"
            _draw_centered(draw, text, cell, body_font, "#17201c")


def _mean_abs_distance(records: dict[str, Any], role: str) -> float | None:
    target_values = _model_values(records["target"])
    role_values = _model_values(records[role])
    diffs = []
    for dim in SUMMARY_DIMS:
        if dim in target_values and dim in role_values:
            diffs.append(abs(role_values[dim] - target_values[dim]))
    if not diffs:
        return None
    return float(np.mean(diffs))


def _draw_distance_panel(
    draw: ImageDraw.ImageDraw,
    records: dict[str, Any],
    origin: tuple[int, int],
    width: int,
    heading_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x, y = origin
    _round_rect(draw, (x, y, x + width, y + 170), radius=14, fill="#ffffff", outline="#d9d4c8")
    draw.text((x + 24, y + 20), "Affect distance to target", fill="#1f2a25", font=heading_font)
    draw.text((x + 24, y + 56), "Mean absolute score difference across available dimensions.", fill="#69716c", font=small_font)
    low = _mean_abs_distance(records, "low_level")
    high = _mean_abs_distance(records, "high_level")
    bar_x = x + 470
    bar_w = width - 540
    for idx, (label, value, fill) in enumerate(
        [
            ("Low-level", low, "#9f7161"),
            ("High-level", high, "#4f8d69"),
        ]
    ):
        row_y = y + 82 + idx * 46
        draw.text((x + 270, row_y + 6), label, fill="#1f2a25", font=body_font)
        _round_rect(draw, (bar_x, row_y + 8, bar_x + bar_w, row_y + 28), radius=10, fill="#eeeae0", outline=None)
        if value is not None:
            length = int(bar_w * min(value / 4.0, 1.0))
            _round_rect(draw, (bar_x, row_y + 8, bar_x + length, row_y + 28), radius=10, fill=fill, outline=None)
            draw.text((bar_x + bar_w + 18, row_y + 1), f"{value:.2f}", fill="#1f2a25", font=body_font)
        else:
            draw.text((bar_x + bar_w + 18, row_y + 1), "--", fill="#1f2a25", font=body_font)
