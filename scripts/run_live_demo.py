#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab_demo.json"


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


def build_command(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise SystemExit(f"Demo config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"Demo config must contain a JSON object: {config_path}")

    low_level = _required_path(config, "low_level_checkpoint")
    trigger_map = _required_path(config, "legacy_trigger_map")
    eeg_unit_fallback = str(config.get("eeg_unit_fallback") or "microvolts")
    if eeg_unit_fallback not in {"volts", "millivolts", "microvolts", "nanovolts"}:
        raise SystemExit(f"Unsupported eeg_unit_fallback in demo config: {eeg_unit_fallback!r}")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_realtime_gui.py"),
        "--mode",
        "lab-live",
        "--participant-mode",
        "unseen",
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
        "--eeg-unit-fallback",
        eeg_unit_fallback,
        "--lab-low-level-checkpoint",
        str(low_level),
        "--lab-model-channels",
        "31",
        "--lab-low-level-arch",
        "plain",
        "--live-trigger-map",
        str(trigger_map),
        "--no-auto-connect",
    ]

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
