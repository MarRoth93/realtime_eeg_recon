from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from maryam_rt.integration.assessor_ratings import AssessorRatingWriter


class _FakeAssessorEngine:
    def rate_images(self, images):
        rows = []
        for idx, _image in enumerate(images):
            rows.append(
                {
                    "models": {
                        "va": {
                            "valence": 6.0 + idx,
                            "valence_low": 5.0 + idx,
                            "valence_high": 7.0 + idx,
                            "arousal": 4.0,
                            "ood_percentile": 12.0,
                        },
                        "six": {
                            "approach": 6.5,
                            "arousal": 4.2,
                            "valence": 6.1,
                            "attention": 5.5,
                            "control": 4.8,
                            "dominance": 3.2,
                            "ood_percentile": 14.0,
                        },
                    }
                }
            )
        return rows


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_assessor_writer_writes_sidecar_schema(tmp_path: Path) -> None:
    writer = AssessorRatingWriter(_FakeAssessorEngine(), tmp_path / "assessments")
    image_path = _image(tmp_path / "low.png", (120, 130, 140))

    record = writer.assess_path(image_path, stem="trial001", image_role="low_level")

    sidecar = Path(record["path"])
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["image_role"] == "low_level"
    assert payload["image_path"] == str(image_path)
    assert payload["ratings"]["models"]["va"]["valence"] == 6.0
    assert payload["ratings"]["models"]["six"]["dominance"] == 3.2


def test_assessor_triplet_summary_writes_figure_and_json(tmp_path: Path) -> None:
    writer = AssessorRatingWriter(_FakeAssessorEngine(), tmp_path / "assessments")
    target = _image(tmp_path / "target.png", (200, 60, 60))
    low = _image(tmp_path / "low.png", (60, 120, 180))
    high = _image(tmp_path / "high.png", (80, 180, 100))

    summary = writer.write_triplet_summary(
        stem="trial001",
        target_path=target,
        low_level_path=low,
        high_level_path=high,
        title="P07 trial001",
    )

    assert Path(summary["path"]).exists()
    assert Path(summary["summary_path"]).exists()
    payload = json.loads(Path(summary["summary_path"]).read_text())
    assert set(payload["records"]) == {"target", "low_level", "high_level"}
    assert payload["records"]["target"]["ratings"]["models"]["va"]["valence"] == 6.0
