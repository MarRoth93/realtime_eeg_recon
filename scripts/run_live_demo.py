#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab_demo.json"
PARTICIPANT_MODES = {"known", "unseen"}
LOW_LEVEL_ARCHITECTURES = {"plain", "transformed"}
LOW_LEVEL_LATENT_SCALINGS = {"auto", "direct", "sdxl"}
LAB_WHITENING_MODES = {"none", "mvnn"}


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _required_path(config: dict[str, object], key: str) -> Path:
    value = str(config.get(key) or "").strip()
    if not value:
        raise SystemExit(f"Demo config is missing {key!r}.")
    path = _project_path(value)
    if not path.exists():
        raise SystemExit(f"Demo asset does not exist for {key!r}: {path}")
    return path


def _choice(config: dict[str, object], key: str, default: str, choices: set[str]) -> str:
    value = str(config.get(key) or default).strip()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise SystemExit(f"Demo config {key!r} must be one of: {expected}.")
    return value


def _optional_int(config: dict[str, object], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Demo config {key!r} must be an integer.") from exc


def build_command(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise SystemExit(f"Demo config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"Demo config must contain a JSON object: {config_path}")

    participant_mode = _choice(config, "participant_mode", "unseen", PARTICIPANT_MODES)
    subject_id = _optional_int(config, "subject_id")
    low_level_arch = _choice(config, "low_level_arch", "plain", LOW_LEVEL_ARCHITECTURES)
    low_level_subject_id = _optional_int(config, "low_level_subject_id")
    latent_scaling = _choice(
        config,
        "low_level_latent_scaling",
        "auto",
        LOW_LEVEL_LATENT_SCALINGS,
    )
    lab_whitening = _choice(config, "lab_whitening", "none", LAB_WHITENING_MODES)
    if participant_mode == "known" and subject_id is None:
        raise SystemExit("Demo config needs 'subject_id' when participant_mode is 'known'.")
    if low_level_arch == "transformed" and low_level_subject_id is None:
        raise SystemExit("Demo config needs 'low_level_subject_id' for a transformed low-level checkpoint.")

    low_level = _required_path(config, "low_level_checkpoint")
    trigger_map = _required_path(config, "legacy_trigger_map")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_realtime_gui.py"),
        "--mode",
        "lab-live",
        "--participant-mode",
        participant_mode,
        "--participant-code",
        str(config.get("participant_code") or "unseen-demo"),
        "--host",
        str(config.get("host") or "127.0.0.1"),
        "--port",
        str(int(config.get("port") or 8010)),
        "--device",
        str(config.get("device") or "cuda"),
        "--eeg-stream-name",
        str(config.get("preferred_eeg_stream") or "eeg_recon_pyexp-EEG"),
        "--marker-stream-name",
        str(config.get("preferred_marker_stream") or "eeg_recon_pyexp"),
        "--lab-low-level-checkpoint",
        str(low_level),
        "--lab-model-channels",
        "31",
        "--lab-low-level-arch",
        low_level_arch,
        "--low-level-latent-scaling",
        latent_scaling,
        "--live-trigger-map",
        str(trigger_map),
        "--no-auto-connect",
    ]
    if subject_id is not None:
        command.extend(["--subject-id", str(subject_id)])
    if low_level_subject_id is not None:
        command.extend(["--lab-low-level-subject-id", str(low_level_subject_id)])
    if lab_whitening == "mvnn":
        command.extend(
            [
                "--lab-whitening",
                "mvnn",
                "--lab-whitening-cache",
                str(_required_path(config, "lab_whitening_cache")),
            ]
        )

    image_root = str(config.get("image_root") or "").strip()
    if image_root:
        command.extend(["--image-root", str(_project_path(image_root))])
    command.append(
        "--legacy-start-on-first-trigger"
        if bool(config.get("legacy_start_on_first_trigger", True))
        else "--no-legacy-start-on-first-trigger"
    )
    if bool(config.get("marker_test_required", False)):
        command.append("--marker-test-required")
    heartbeat_timeout = config.get("marker_heartbeat_timeout_seconds")
    if heartbeat_timeout is not None:
        command.extend(["--marker-heartbeat-timeout-seconds", str(float(heartbeat_timeout))])
    if bool(config.get("open_browser", True)):
        command.append("--open-browser")

    if bool(config.get("enable_experimental_unseen_high_level", False)):
        command.extend(
            [
                "--enable-experimental-unseen-high-level",
                "--lab-atms-checkpoint",
                str(_required_path(config, "atms_checkpoint")),
                "--lab-prior-checkpoint",
                str(_required_path(config, "prior_checkpoint")),
            ]
        )
    else:
        command.append("--disable-high-level")
    return command


def main() -> int:
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_CONFIG
    command = build_command(config_path)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
