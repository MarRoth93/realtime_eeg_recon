"""LSL marker inlet wrapper for task-triggered epoch collection."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from pylsl import StreamInfo, StreamInlet, resolve_streams


@dataclass(frozen=True)
class MarkerEvent:
    """Single marker sample from an LSL marker stream."""

    value: str
    timestamp: float


@dataclass
class MarkerInletConfig:
    """Configuration for marker stream acquisition."""

    stream_name: Optional[str] = None
    stream_type: Optional[str] = "Markers"
    source_id: Optional[str] = None
    resolve_timeout: float = 5.0
    reconnect_interval: float = 1.0
    pull_timeout: float = 0.1
    chunk_size: int = 64
    max_buflen_seconds: float = 60.0
    max_queue_size: int = 2048

    def __post_init__(self) -> None:
        if self.stream_name is None and self.stream_type is None:
            raise ValueError("stream_name or stream_type must be provided.")


class MarkerInletWrapper:
    """Background reader for irregular LSL marker streams."""

    def __init__(self, config: Optional[MarkerInletConfig] = None, **kwargs: object) -> None:
        if config is None:
            config = MarkerInletConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self._config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inlet: Optional[StreamInlet] = None
        self._info: Optional[StreamInfo] = None
        self._last_error: Optional[Exception] = None
        self._events: Deque[MarkerEvent] = deque(maxlen=config.max_queue_size)

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    @property
    def last_error(self) -> Optional[Exception]:
        return self._last_error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="MarkerInletReader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=timeout)

    def wait_for_stream(self, timeout: Optional[float] = None) -> bool:
        return self._connected_event.wait(timeout)

    def pop_events(self) -> list[MarkerEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._inlet is None:
                info = self._resolve_stream_info()
                if info is None:
                    time.sleep(self._config.reconnect_interval)
                    continue
                try:
                    self._inlet = self._create_inlet(info)
                    self._info = info
                    self._last_error = None
                    self._connected_event.set()
                except Exception as exc:
                    self._last_error = exc
                    self._disconnect()
                    time.sleep(self._config.reconnect_interval)
                continue

            try:
                self._pull_events()
            except Exception as exc:
                self._last_error = exc
                self._disconnect()
                time.sleep(self._config.reconnect_interval)

    def _resolve_stream_info(self) -> Optional[StreamInfo]:
        infos = resolve_streams(wait_time=self._config.resolve_timeout)
        if self._config.stream_name is not None:
            infos = [info for info in infos if info.name() == self._config.stream_name]
        if self._config.stream_type is not None:
            infos = [info for info in infos if info.type() == self._config.stream_type]
        if self._config.source_id is not None:
            infos = [info for info in infos if info.source_id() == self._config.source_id]
        return infos[0] if infos else None

    def _create_inlet(self, info: StreamInfo) -> StreamInlet:
        inlet = StreamInlet(info, max_buflen=int(self._config.max_buflen_seconds))
        try:
            inlet.open_stream(timeout=self._config.resolve_timeout)
        except Exception:
            pass
        try:
            inlet.flush()
        except Exception:
            pass
        return inlet

    def _pull_events(self) -> None:
        if self._inlet is None:
            return
        samples, timestamps = self._inlet.pull_chunk(
            timeout=self._config.pull_timeout,
            max_samples=self._config.chunk_size,
        )
        if not samples:
            return
        events: list[MarkerEvent] = []
        for sample, timestamp in zip(samples, timestamps):
            if isinstance(sample, (list, tuple)):
                value = sample[0]
            else:
                value = sample
            events.append(MarkerEvent(value=str(value), timestamp=float(timestamp)))
        with self._lock:
            self._events.extend(events)

    def _disconnect(self) -> None:
        inlet = self._inlet
        self._inlet = None
        self._info = None
        self._connected_event.clear()
        if inlet is not None:
            try:
                inlet.close_stream()
            except Exception:
                pass
