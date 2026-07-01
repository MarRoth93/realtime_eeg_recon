from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StreamStatus:
    engine_running: bool = False
    eeg_connected: bool = False
    marker_connected: bool = False
    high_level_enabled: bool = False
    mode: str = "live"
    message: str = ""
    error: str | None = None


class RuntimeMonitorState:
    def __init__(self, mode: str, high_level_enabled: bool, max_markers: int = 128) -> None:
        self._lock = threading.Lock()
        self._status = StreamStatus(mode=mode, high_level_enabled=high_level_enabled)
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

    def set_status(
        self,
        *,
        engine_running: bool | None = None,
        eeg_connected: bool | None = None,
        marker_connected: bool | None = None,
        high_level_enabled: bool | None = None,
        mode: str | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            current = self._status
            self._status = StreamStatus(
                engine_running=current.engine_running if engine_running is None else engine_running,
                eeg_connected=current.eeg_connected if eeg_connected is None else eeg_connected,
                marker_connected=current.marker_connected if marker_connected is None else marker_connected,
                high_level_enabled=current.high_level_enabled if high_level_enabled is None else high_level_enabled,
                mode=current.mode if mode is None else mode,
                message=current.message if message is None else message,
                error=error,
            )
            self._last_updated = time.time()

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
            return {
                "status": {
                    "engine_running": self._status.engine_running,
                    "eeg_connected": self._status.eeg_connected,
                    "marker_connected": self._status.marker_connected,
                    "high_level_enabled": self._status.high_level_enabled,
                    "mode": self._status.mode,
                    "message": self._status.message,
                    "error": self._status.error,
                },
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

