from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_demo import build_command


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_live_demo_command_can_match_verified_p07_replay_assets(tmp_path: Path) -> None:
    low_level = tmp_path / "vae_custom_data.pth"
    trigger_map = tmp_path / "lab_target_manifest.tsv"
    whitening_cache = tmp_path / "p07_train_mvnn.npy"
    for path in (low_level, trigger_map, whitening_cache):
        path.touch()
    config_path = tmp_path / "lab_demo.json"
    config_path.write_text(
        json.dumps(
            {
                "participant_code": "P07-live",
                "participant_mode": "known",
                "subject_id": 4,
                "low_level_checkpoint": str(low_level),
                "low_level_arch": "transformed",
                "low_level_subject_id": 4,
                "low_level_latent_scaling": "sdxl",
                "lab_whitening": "mvnn",
                "lab_whitening_cache": str(whitening_cache),
                "legacy_trigger_map": str(trigger_map),
                "enable_experimental_unseen_high_level": False,
            }
        )
    )

    command = build_command(config_path)

    assert _argument(command, "--mode") == "lab-live"
    assert _argument(command, "--participant-mode") == "known"
    assert _argument(command, "--subject-id") == "4"
    assert _argument(command, "--lab-low-level-checkpoint") == str(low_level)
    assert _argument(command, "--lab-low-level-arch") == "transformed"
    assert _argument(command, "--lab-low-level-subject-id") == "4"
    assert _argument(command, "--low-level-latent-scaling") == "sdxl"
    assert _argument(command, "--lab-whitening") == "mvnn"
    assert _argument(command, "--lab-whitening-cache") == str(whitening_cache)
    assert "--disable-high-level" in command


def test_live_demo_rejects_transformed_checkpoint_without_subject_row(tmp_path: Path) -> None:
    low_level = tmp_path / "vae_custom_data.pth"
    trigger_map = tmp_path / "lab_target_manifest.tsv"
    low_level.touch()
    trigger_map.touch()
    config_path = tmp_path / "lab_demo.json"
    config_path.write_text(
        json.dumps(
            {
                "participant_mode": "known",
                "subject_id": 4,
                "low_level_checkpoint": str(low_level),
                "low_level_arch": "transformed",
                "legacy_trigger_map": str(trigger_map),
            }
        )
    )

    with pytest.raises(SystemExit, match="low_level_subject_id"):
        build_command(config_path)
