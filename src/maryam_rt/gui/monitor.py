from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LIVE_PHASES = {
    "loading_models",
    "waiting_for_equipment",
    "checking_signal",
    "ready",
    "armed",
    "running",
    "finished",
    "error",
    "stopped",
}

_UNSET = object()


@dataclass(frozen=True)
class StreamStatus:
    engine_running: bool = False
    eeg_connected: bool = False
    marker_connected: bool = False
    high_level_enabled: bool = False
    mode: str = "live"
    phase: str = "loading_models"
    models_loaded: bool = False
    message: str = ""
    error: str | None = None
    eeg_stream_name: str | None = None
    marker_stream_name: str | None = None
    eeg_channel_count: int | None = None
    eeg_sampling_rate: float | None = None
    expected_eeg_channel_count: int | None = None
    expected_eeg_sampling_rate: float | None = None
    eeg_format_valid: bool = False
    eeg_data_fresh: bool = False
    buffer_seconds: float = 0.0
    buffer_ready: bool = False
    marker_test_required: bool = False
    marker_tested: bool = False
    channel_order_confirmation_required: bool = False
    channel_order_confirmed: bool = True
    output_writable: bool = True
    output_root: str | None = None
    session_configured: bool = True
    armed: bool = False
    experiment_running: bool = False
    session_id: str | None = None
    participant_code: str | None = None
    participant_mode: str = "known"
    high_level_reason: str | None = None
    trials_received: int = 0
    trials_processed: int = 0
    trials_dropped: int = 0
    marker_events_dropped: int = 0
    eeg_timestamp_gaps: int = 0
    signal_quality_status: str = "unknown"
    signal_quality_warnings: tuple[str, ...] = ()
    signal_median_peak_to_peak_uv: float | None = None


class RuntimeMonitorState:
    """Thread-safe, authoritative runtime state shared by the runner and GUI."""

    def __init__(
        self,
        mode: str,
        high_level_enabled: bool,
        max_markers: int = 128,
        *,
        session_id: str | None = None,
        participant_code: str | None = None,
        participant_mode: str = "known",
        high_level_reason: str | None = None,
        marker_test_required: bool = False,
        expected_eeg_channel_count: int | None = None,
        expected_eeg_sampling_rate: float | None = None,
        channel_order_confirmation_required: bool = False,
        output_writable: bool = True,
        output_root: str | Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        initial_phase = "loading_models" if mode in {"live", "lab-live"} else "ready"
        self._status = StreamStatus(
            mode=mode,
            high_level_enabled=high_level_enabled,
            phase=initial_phase,
            session_id=session_id,
            participant_code=participant_code,
            participant_mode=participant_mode,
            high_level_reason=high_level_reason,
            marker_test_required=marker_test_required,
            expected_eeg_channel_count=expected_eeg_channel_count,
            expected_eeg_sampling_rate=expected_eeg_sampling_rate,
            channel_order_confirmation_required=channel_order_confirmation_required,
            channel_order_confirmed=not channel_order_confirmation_required,
            output_writable=output_writable,
            output_root=None if output_root is None else str(output_root),
            session_configured=bool(session_id and participant_code)
            if mode in {"live", "lab-live"}
            else True,
        )
        self._recent_markers: deque[dict[str, Any]] = deque(maxlen=max_markers)
        self._latest_epoch: np.ndarray | None = None
        self._latest_epoch_sampling_rate: float | None = None
        self._latest_epoch_start_seconds: float = 0.0
        self._latest_low_level_path: str | None = None
        self._latest_target_path: str | None = None
        self._latest_high_level_path: str | None = None
        self._latest_low_level_info: dict[str, Any] | None = None
        self._latest_target_info: dict[str, Any] | None = None
        self._latest_high_level_info: dict[str, Any] | None = None
        self._last_updated = time.time()

    @property
    def status(self) -> StreamStatus:
        with self._lock:
            return self._status

    @property
    def is_ready(self) -> bool:
        with self._lock:
            readiness = self._readiness_unlocked()
            return all(item["ok"] for item in readiness.values())

    def _replace_status(self, **updates: Any) -> None:
        current = self._status
        values = dict(current.__dict__)
        values.update(updates)
        self._status = StreamStatus(**values)
        self._last_updated = time.time()

    def _readiness_unlocked(self) -> dict[str, dict[str, Any]]:
        status = self._status
        expected_format = "EEG format matches the configured model"
        if status.expected_eeg_channel_count is not None and status.expected_eeg_sampling_rate is not None:
            expected_format = (
                f"EEG format matches {status.expected_eeg_channel_count} channels at "
                f"{status.expected_eeg_sampling_rate:g} Hz"
            )
        return {
            "session": {
                "ok": status.session_configured,
                "label": "Anonymous session is configured",
            },
            "output": {
                "ok": status.output_writable,
                "label": "Session output is writable",
            },
            "models": {
                "ok": status.models_loaded,
                "label": "Models loaded",
            },
            "eeg_stream": {
                "ok": status.eeg_connected,
                "label": "EEG stream connected",
            },
            "eeg_format": {
                "ok": status.eeg_format_valid,
                "label": expected_format,
            },
            "eeg_fresh": {
                "ok": status.eeg_data_fresh,
                "label": "EEG samples are arriving now",
            },
            "channel_order": {
                "ok": (not status.channel_order_confirmation_required)
                or status.channel_order_confirmed,
                "label": (
                    "Starstim channel order confirmed"
                    if status.channel_order_confirmation_required
                    else "Channel order check not required"
                ),
            },
            "buffer": {
                "ok": status.buffer_ready,
                "label": "EEG buffer warmed",
            },
            "marker_stream": {
                "ok": status.marker_connected,
                "label": "Marker stream connected",
            },
            "marker_test": {
                "ok": (not status.marker_test_required) or status.marker_tested,
                "label": "Marker test received" if status.marker_test_required else "Marker test optional",
            },
        }

    def set_status(
        self,
        *,
        engine_running: bool | None = None,
        eeg_connected: bool | None = None,
        marker_connected: bool | None = None,
        high_level_enabled: bool | None = None,
        mode: str | None = None,
        message: str | None = None,
        error: str | None | object = _UNSET,
        phase: str | None = None,
        models_loaded: bool | None = None,
    ) -> None:
        """Compatibility update used by both replay and live controllers."""
        if phase is not None and phase not in LIVE_PHASES:
            raise ValueError(f"Unsupported runtime phase: {phase}")
        with self._lock:
            updates: dict[str, Any] = {}
            if error is not _UNSET:
                updates["error"] = error
            optional = {
                "engine_running": engine_running,
                "eeg_connected": eeg_connected,
                "marker_connected": marker_connected,
                "high_level_enabled": high_level_enabled,
                "mode": mode,
                "message": message,
                "phase": phase,
                "models_loaded": models_loaded,
            }
            updates.update({key: value for key, value in optional.items() if value is not None})
            self._replace_status(**updates)

    def set_models_loaded(self, loaded: bool = True) -> None:
        with self._lock:
            phase = self._status.phase
            if loaded and phase == "loading_models":
                phase = "waiting_for_equipment"
            self._replace_status(models_loaded=loaded, phase=phase)

    def set_stream_health(
        self,
        *,
        eeg_connected: bool,
        marker_connected: bool,
        eeg_stream_name: str | None,
        marker_stream_name: str | None,
        eeg_channel_count: int | None,
        eeg_sampling_rate: float | None,
        eeg_format_valid: bool,
        eeg_data_fresh: bool,
        buffer_seconds: float,
        buffer_ready: bool,
        marker_events_dropped: int = 0,
        eeg_timestamp_gaps: int = 0,
    ) -> None:
        with self._lock:
            self._replace_status(
                eeg_connected=eeg_connected,
                marker_connected=marker_connected,
                eeg_stream_name=eeg_stream_name,
                marker_stream_name=marker_stream_name,
                eeg_channel_count=eeg_channel_count,
                eeg_sampling_rate=eeg_sampling_rate,
                eeg_format_valid=eeg_format_valid,
                eeg_data_fresh=eeg_data_fresh,
                buffer_seconds=max(float(buffer_seconds), 0.0),
                buffer_ready=buffer_ready,
                marker_events_dropped=max(int(marker_events_dropped), 0),
                eeg_timestamp_gaps=max(int(eeg_timestamp_gaps), 0),
            )

    def confirm_channel_order(self, confirmed: bool = True) -> None:
        with self._lock:
            self._replace_status(channel_order_confirmed=confirmed)

    def set_signal_quality(
        self,
        *,
        status: str,
        warnings: list[str],
        median_peak_to_peak_uv: float | None,
    ) -> None:
        with self._lock:
            self._replace_status(
                signal_quality_status=status,
                signal_quality_warnings=tuple(warnings),
                signal_median_peak_to_peak_uv=median_peak_to_peak_uv,
            )

    def configure_session(self, participant_code: str, session_id: str | None = None) -> None:
        participant_code = participant_code.strip()
        if not participant_code:
            raise ValueError("participant_code must not be empty")
        with self._lock:
            self._replace_status(
                participant_code=participant_code,
                session_id=session_id or self._status.session_id,
                session_configured=bool(session_id or self._status.session_id),
            )

    def set_waiting_or_ready(self) -> bool:
        """Move an idle live session to READY only when every readiness gate passes."""
        with self._lock:
            ready = all(item["ok"] for item in self._readiness_unlocked().values())
            if self._status.armed or self._status.experiment_running:
                return ready
            if self._status.phase in {"finished", "stopped"}:
                return ready
            if ready:
                phase = "ready"
                message = "Equipment ready. Arm the session when the participant is ready."
            elif self._status.eeg_connected and self._status.marker_connected:
                phase = "checking_signal"
                message = "Streams found. Checking EEG format, live samples, and buffer readiness."
            else:
                phase = "waiting_for_equipment"
                message = "Waiting for valid EEG and marker streams."
            self._replace_status(phase=phase, message=message, error=None)
            return ready

    def arm(self) -> tuple[bool, str]:
        with self._lock:
            readiness = self._readiness_unlocked()
            missing = [item["label"] for item in readiness.values() if not item["ok"]]
            if missing:
                message = "Cannot arm: " + "; ".join(missing)
                self._replace_status(armed=False, experiment_running=False, message=message)
                return False, message
            self._replace_status(
                phase="armed",
                armed=True,
                experiment_running=False,
                message="Armed. Waiting for experiment_start or the first stimulus trigger.",
                error=None,
            )
            return True, self._status.message

    def disarm(self, message: str = "Session disarmed.") -> None:
        with self._lock:
            ready = all(item["ok"] for item in self._readiness_unlocked().values())
            self._replace_status(
                phase="ready" if ready else "waiting_for_equipment",
                armed=False,
                experiment_running=False,
                message=message,
                error=None,
            )

    def start_experiment(self, message: str = "Experiment running.") -> bool:
        with self._lock:
            if not self._status.armed:
                return False
            self._replace_status(
                phase="running",
                armed=True,
                experiment_running=True,
                message=message,
                error=None,
            )
            return True

    def pause_experiment(self, message: str = "Experiment paused.") -> bool:
        with self._lock:
            if not self._status.armed:
                return False
            self._replace_status(
                phase="armed",
                armed=True,
                experiment_running=False,
                message=message,
                error=None,
            )
            return True

    def finish(self, message: str = "Experiment finished.") -> None:
        with self._lock:
            self._replace_status(
                phase="finished",
                armed=False,
                experiment_running=False,
                message=message,
                error=None,
            )

    def fail(self, message: str) -> None:
        with self._lock:
            self._replace_status(
                phase="error",
                armed=False,
                experiment_running=False,
                message="Live session needs attention.",
                error=message,
            )

    def mark_marker_tested(self) -> None:
        with self._lock:
            self._replace_status(marker_tested=True, message="Marker test received.")

    def reset_marker_test(self) -> None:
        with self._lock:
            if self._status.marker_test_required:
                self._replace_status(marker_tested=False)

    def record_trial_received(self) -> None:
        with self._lock:
            self._replace_status(trials_received=self._status.trials_received + 1)

    def record_trial_processed(self) -> None:
        with self._lock:
            self._replace_status(trials_processed=self._status.trials_processed + 1)

    def record_trial_dropped(self) -> None:
        with self._lock:
            self._replace_status(trials_dropped=self._status.trials_dropped + 1)

    def add_marker(self, marker: dict[str, Any]) -> None:
        with self._lock:
            self._recent_markers.append(marker)
            self._last_updated = time.time()

    def set_latest_epoch(
        self,
        epoch: np.ndarray,
        sampling_rate: float,
        start_seconds: float,
    ) -> None:
        with self._lock:
            self._latest_epoch = np.asarray(epoch, dtype=np.float32).copy()
            self._latest_epoch_sampling_rate = float(sampling_rate)
            self._latest_epoch_start_seconds = float(start_seconds)
            self._last_updated = time.time()

    def set_latest_low_level(self, path: str | Path, info: dict[str, Any]) -> None:
        with self._lock:
            self._latest_low_level_path = str(path)
            self._latest_low_level_info = dict(info)
            self._last_updated = time.time()

    def set_latest_target(self, path: str | Path, info: dict[str, Any]) -> None:
        with self._lock:
            self._latest_target_path = str(path)
            self._latest_target_info = dict(info)
            self._last_updated = time.time()

    def set_latest_high_level(self, path: str | Path, info: dict[str, Any]) -> None:
        with self._lock:
            self._latest_high_level_path = str(path)
            self._latest_high_level_info = dict(info)
            self._last_updated = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = self._status
            readiness = self._readiness_unlocked()
            return {
                "status": {
                    **status.__dict__,
                    "ready": all(item["ok"] for item in readiness.values()),
                },
                "readiness": readiness,
                "recent_markers": list(self._recent_markers),
                "latest_low_level": {
                    "path": self._latest_low_level_path,
                    "info": self._latest_low_level_info,
                },
                "latest_target": {
                    "path": self._latest_target_path,
                    "info": self._latest_target_info,
                },
                "latest_high_level": {
                    "path": self._latest_high_level_path,
                    "info": self._latest_high_level_info,
                },
                "latest_epoch": {
                    "available": self._latest_epoch is not None,
                    "sampling_rate": self._latest_epoch_sampling_rate,
                    "start_seconds": self._latest_epoch_start_seconds,
                },
                "updated_at": self._last_updated,
            }

    def latest_epoch_plot(self, max_channels: int = 8) -> dict[str, Any]:
        with self._lock:
            if self._latest_epoch is None or self._latest_epoch_sampling_rate is None:
                return {"sampling_rate": None, "window_seconds": None, "traces": [], "markers": []}
            data = self._latest_epoch[:max_channels]
            sampling_rate = self._latest_epoch_sampling_rate
            start_seconds = self._latest_epoch_start_seconds

        n_samples = int(data.shape[1])
        times = (np.arange(n_samples, dtype=np.float32) / sampling_rate) + start_seconds
        scale = max(float(np.nanmax(np.abs(data))) * 2.5, 1.0)
        traces = []
        for idx, channel in enumerate(data):
            traces.append(
                {
                    "name": f"Ch {idx + 1}",
                    "x": times.tolist(),
                    "y": (channel + idx * scale).astype(np.float32).tolist(),
                }
            )
        return {
            "sampling_rate": sampling_rate,
            "window_seconds": n_samples / sampling_rate,
            "traces": traces,
            "markers": [{"x": 0.0, "label": "event", "status": "accepted"}],
        }
