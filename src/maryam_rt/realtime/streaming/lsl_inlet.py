"""Real-time LSL inlet wrapper with chunk pulling and ring buffering."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pylsl import StreamInfo, StreamInlet, resolve_streams

from .timestamp_sync import TimestampSyncConfig, TimestampSynchronizer


@dataclass
class LSLInletConfig:
    """Configuration settings for the LSL inlet wrapper."""

    stream_name: Optional[str] = None
    stream_type: Optional[str] = "EEG"
    source_id: Optional[str] = None
    resolve_timeout: float = 5.0
    reconnect_interval: float = 1.0
    pull_timeout: float = 0.1
    chunk_size: int = 128
    ring_buffer_seconds: float = 5.0
    max_idle_seconds: float = 2.0
    max_lsl_buffer_seconds: float = 60.0
    channel_count: Optional[int] = None
    sampling_rate: Optional[float] = None
    dtype: np.dtype = np.float32
    enable_time_sync: bool = True
    time_sync_interval: float = 0.5
    timestamp_sync_config: Optional[TimestampSyncConfig] = None

    def __post_init__(self) -> None:
        if self.stream_name is None and self.stream_type is None:
            raise ValueError("stream_name or stream_type must be provided.")
        if self.resolve_timeout < 0:
            raise ValueError("resolve_timeout must be >= 0.")
        if self.reconnect_interval < 0:
            raise ValueError("reconnect_interval must be >= 0.")
        if self.pull_timeout < 0:
            raise ValueError("pull_timeout must be >= 0.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if self.ring_buffer_seconds <= 0:
            raise ValueError("ring_buffer_seconds must be > 0.")
        if self.max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be > 0.")
        if self.max_lsl_buffer_seconds <= 0:
            raise ValueError("max_lsl_buffer_seconds must be > 0.")
        if self.time_sync_interval < 0:
            raise ValueError("time_sync_interval must be >= 0.")


class RingBuffer:
    """Fixed-size ring buffer for samples shaped (n_channels, n_samples)."""

    def __init__(self, n_channels: int, capacity: int, dtype: np.dtype = np.float32) -> None:
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._n_channels = int(n_channels)
        self._capacity = int(capacity)
        self._buffer = np.zeros((self._n_channels, self._capacity), dtype=dtype)
        self._write_index = 0
        self._filled = 0

    @property
    def capacity(self) -> int:
        """Maximum number of samples stored by the buffer."""
        return self._capacity

    @property
    def channel_count(self) -> int:
        """Number of channels stored by the buffer."""
        return self._n_channels

    @property
    def available_samples(self) -> int:
        """Number of valid samples currently stored."""
        return self._filled

    def clear(self) -> None:
        """Reset the buffer to an empty state."""
        self._write_index = 0
        self._filled = 0

    def append(self, data: np.ndarray) -> int:
        """Append samples shaped (n_channels, n_samples) to the buffer."""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}.")
        if data.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {data.shape[0]} in {data.shape}."
            )
        n_samples = data.shape[1]
        if n_samples == 0:
            return 0
        if n_samples >= self._capacity:
            data = data[:, -self._capacity :]
            n_samples = self._capacity

        end_index = self._write_index + n_samples
        if end_index <= self._capacity:
            self._buffer[:, self._write_index:end_index] = data
        else:
            first = self._capacity - self._write_index
            self._buffer[:, self._write_index:] = data[:, :first]
            self._buffer[:, : end_index % self._capacity] = data[:, first:]

        self._write_index = end_index % self._capacity
        self._filled = min(self._capacity, self._filled + n_samples)
        return n_samples

    def get_latest(self, n_samples: int, fill_value: float = 0.0) -> np.ndarray:
        """Return the latest n_samples as (n_channels, n_samples)."""
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0.")
        if n_samples > self._capacity:
            raise ValueError(
                f"n_samples ({n_samples}) exceeds buffer capacity ({self._capacity})."
            )
        if self._filled == 0:
            return np.full((self._n_channels, n_samples), fill_value, dtype=self._buffer.dtype)
        if self._filled < n_samples:
            pad = n_samples - self._filled
            data = self._slice_latest(self._filled)
            pad_block = np.full((self._n_channels, pad), fill_value, dtype=self._buffer.dtype)
            return np.ascontiguousarray(np.concatenate((pad_block, data), axis=1))
        return np.ascontiguousarray(self._slice_latest(n_samples))

    def _slice_latest(self, n_samples: int) -> np.ndarray:
        start = (self._write_index - n_samples) % self._capacity
        if start < self._write_index:
            return self._buffer[:, start:self._write_index].copy()
        return np.concatenate(
            (self._buffer[:, start:], self._buffer[:, : self._write_index]), axis=1
        ).copy()


class LSLInletWrapper:
    """Thread-safe wrapper around pylsl.StreamInlet with a ring buffer."""

    def __init__(self, config: Optional[LSLInletConfig] = None, **kwargs: object) -> None:
        if config is None:
            config = LSLInletConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self._config = config
        self._lock = threading.Lock()
        self._data_ready = threading.Condition(self._lock)
        self._connected_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inlet: Optional[StreamInlet] = None
        self._info: Optional[StreamInfo] = None
        self._ring: Optional[RingBuffer] = None
        self._channel_count: Optional[int] = config.channel_count
        self._sampling_rate: Optional[float] = config.sampling_rate
        self._last_lsl_timestamp: Optional[float] = None
        self._last_error: Optional[Exception] = None
        self._timestamp_sync: Optional[TimestampSynchronizer] = None
        self._last_time_sync_update = 0.0

        if self._config.enable_time_sync:
            sync_config = self._config.timestamp_sync_config or TimestampSyncConfig()
            self._timestamp_sync = TimestampSynchronizer(sync_config)

        self._initialize_ring_if_possible()

    def __enter__(self) -> "LSLInletWrapper":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def channel_count(self) -> Optional[int]:
        """Number of channels expected from the stream."""
        return self._channel_count

    @property
    def sampling_rate(self) -> Optional[float]:
        """Nominal sampling rate for the stream."""
        return self._sampling_rate

    @property
    def is_connected(self) -> bool:
        """True when an LSL stream is currently connected."""
        return self._connected_event.is_set()

    @property
    def last_error(self) -> Optional[Exception]:
        """Last error encountered by the reader thread, if any."""
        return self._last_error

    @property
    def last_lsl_timestamp(self) -> Optional[float]:
        """Most recent raw LSL timestamp observed for this stream."""
        with self._lock:
            return self._last_lsl_timestamp

    @property
    def available_samples(self) -> int:
        """Number of samples currently buffered."""
        with self._data_ready:
            if self._ring is None:
                return 0
            return self._ring.available_samples

    @property
    def timestamp_sync(self) -> Optional[TimestampSynchronizer]:
        """Timestamp synchronizer for aligning LSL timestamps to local time."""
        return self._timestamp_sync

    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="LSLInletReader", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        """Stop the background reader thread."""
        self._stop_event.set()
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=timeout)

    def wait_for_stream(self, timeout: Optional[float] = None) -> bool:
        """Block until a stream is connected or timeout expires."""
        return self._connected_event.wait(timeout)

    def get_window(
        self, n_samples: int, timeout: Optional[float] = None, fill_value: float = 0.0
    ) -> np.ndarray:
        """Return the latest samples as (n_channels, n_samples)."""
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0.")

        deadline = None if timeout is None else time.monotonic() + timeout

        if self._ring is None:
            if deadline is None:
                self._connected_event.wait()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._connected_event.wait(remaining):
                    raise RuntimeError(
                        "Ring buffer not initialized; provide channel_count/sampling_rate "
                        "or wait for stream connection."
                    )

        with self._data_ready:
            if self._ring is None:
                raise RuntimeError("Ring buffer not initialized.")
            while self._ring.available_samples < n_samples:
                if deadline is None:
                    self._data_ready.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._data_ready.wait(remaining)
            return self._ring.get_latest(n_samples, fill_value=fill_value)

    def get_time_correction(self, lsl_timestamp: float) -> float:
        """Return the correction value for mapping an LSL timestamp to local time."""
        if self._timestamp_sync is None:
            raise RuntimeError("Timestamp synchronization is disabled.")
        return self._timestamp_sync.correction_for_lsl(lsl_timestamp)

    def get_segment_by_lsl_times(
        self,
        start_lsl: float,
        end_lsl: float,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Return a buffered EEG segment spanning [start_lsl, end_lsl)."""
        if end_lsl <= start_lsl:
            raise ValueError("end_lsl must be greater than start_lsl.")
        if self._sampling_rate is None:
            raise RuntimeError("Sampling rate is unknown; cannot extract timestamped segments.")

        n_samples = int(round((end_lsl - start_lsl) * self._sampling_rate))
        if n_samples <= 0:
            raise ValueError("Requested timestamp range resolves to zero samples.")

        with self._data_ready:
            if self._ring is None:
                raise RuntimeError("Ring buffer not initialized.")
            if self._last_lsl_timestamp is None:
                raise RuntimeError("No EEG timestamps available yet.")

            end_offset = int(round((self._last_lsl_timestamp - end_lsl) * self._sampling_rate))
            if end_offset < 0:
                raise RuntimeError("Requested segment ends in the future relative to buffered EEG.")

            total = n_samples + end_offset
            if total > self._ring.capacity:
                raise ValueError(
                    f"Requested segment needs {total} samples but ring capacity is {self._ring.capacity}. "
                    "Increase ring_buffer_seconds."
                )
            if self._ring.available_samples < total:
                raise RuntimeError("Not enough buffered EEG samples to satisfy requested segment.")

            window = self._ring.get_latest(total, fill_value=fill_value)

        return np.ascontiguousarray(window[:, :n_samples])

    def _initialize_ring_if_possible(self) -> None:
        if self._channel_count is None or self._sampling_rate is None:
            return
        capacity = max(
            self._config.chunk_size,
            int(round(self._sampling_rate * self._config.ring_buffer_seconds)),
        )
        self._ring = RingBuffer(self._channel_count, capacity, dtype=self._config.dtype)

    def _run(self) -> None:
        last_sample_time = time.monotonic()
        while not self._stop_event.is_set():
            if self._inlet is None:
                info = self._resolve_stream_info()
                if info is None:
                    time.sleep(self._config.reconnect_interval)
                    continue
                try:
                    self._configure_from_info(info)
                    self._inlet = self._create_inlet(info)
                    self._info = info
                    self._last_error = None
                    self._connected_event.set()
                    last_sample_time = time.monotonic()
                except Exception as exc:
                    self._last_error = exc
                    self._disconnect()
                    time.sleep(self._config.reconnect_interval)
                continue

            try:
                received = self._pull_and_buffer()
            except Exception as exc:
                self._last_error = exc
                self._disconnect()
                time.sleep(self._config.reconnect_interval)
                continue

            if received:
                last_sample_time = time.monotonic()
            elif time.monotonic() - last_sample_time > self._config.max_idle_seconds:
                self._disconnect()
                time.sleep(self._config.reconnect_interval)

    def _resolve_stream_info(self) -> Optional[StreamInfo]:
        infos = resolve_streams(wait_time=self._config.resolve_timeout)

        if not infos:
            return None

        # Filter by stream name
        if self._config.stream_name is not None:
            infos = [info for info in infos if info.name() == self._config.stream_name]

        # Filter by stream type
        if self._config.stream_type is not None:
            infos = [info for info in infos if info.type() == self._config.stream_type]

        # Filter by source ID
        if self._config.source_id:
            infos = [info for info in infos if info.source_id() == self._config.source_id]

        return infos[0] if infos else None

    def _configure_from_info(self, info: StreamInfo) -> None:
        channel_count = int(info.channel_count())
        if self._channel_count is None:
            self._channel_count = channel_count
        elif self._channel_count != channel_count:
            raise ValueError(
                f"Stream has {channel_count} channels but {self._channel_count} expected."
            )

        sampling_rate = self._sampling_rate or float(info.nominal_srate())
        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be provided for irregular streams with nominal_srate=0."
            )
        self._sampling_rate = sampling_rate
        capacity = max(
            self._config.chunk_size, int(round(self._sampling_rate * self._config.ring_buffer_seconds))
        )
        with self._data_ready:
            self._ring = RingBuffer(channel_count, capacity, dtype=self._config.dtype)
            self._data_ready.notify_all()

    def _create_inlet(self, info: StreamInfo) -> StreamInlet:
        kwargs = {
            "max_buflen": int(self._config.max_lsl_buffer_seconds),
            "max_chunklen": self._config.chunk_size,
        }
        try:
            inlet = StreamInlet(info, recover=True, **kwargs)
        except TypeError:
            inlet = StreamInlet(info, **kwargs)

        try:
            inlet.open_stream(timeout=self._config.resolve_timeout)
        except Exception:
            pass
        try:
            inlet.flush()
        except Exception:
            pass
        return inlet

    def _pull_and_buffer(self) -> bool:
        samples, timestamps = self._pull_chunk(self._config.pull_timeout)
        if not samples:
            return False
        self._append_samples(samples)
        self._update_time_sync(timestamps)
        while True:
            extra_samples, extra_timestamps = self._pull_chunk(0.0)
            if not extra_samples:
                break
            self._append_samples(extra_samples)
            self._update_time_sync(extra_timestamps)
        return True

    def _pull_chunk(
        self, timeout: float
    ) -> tuple[Sequence[Sequence[float]], Sequence[float]]:
        if self._inlet is None:
            return [], []
        samples, _timestamps = self._inlet.pull_chunk(
            timeout=timeout, max_samples=self._config.chunk_size
        )
        return samples or [], _timestamps or []

    def _update_time_sync(self, timestamps: Sequence[float]) -> None:
        if timestamps:
            with self._lock:
                self._last_lsl_timestamp = float(timestamps[-1])
        if self._timestamp_sync is None:
            return
        self._timestamp_sync.update_local_alignment()
        if self._inlet is None:
            return
        now = self._timestamp_sync.local_time()
        if (
            self._config.time_sync_interval == 0.0
            or now - self._last_time_sync_update >= self._config.time_sync_interval
        ):
            try:
                correction = self._inlet.time_correction()
            except Exception:
                correction = None
            if correction is not None:
                self._timestamp_sync.update_time_correction(correction)
                self._last_time_sync_update = now
        if timestamps:
            self._timestamp_sync.note_lsl_timestamp(float(timestamps[-1]))

    def _append_samples(self, samples: Sequence[Sequence[float]]) -> None:
        if self._ring is None or self._channel_count is None:
            return
        data = np.asarray(samples, dtype=self._config.dtype)
        if data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] == self._channel_count:
            data_samples = data
        elif data.shape[0] == self._channel_count:
            data_samples = data.T
        else:
            raise ValueError(
                f"Unexpected sample shape {data.shape} for {self._channel_count} channels."
            )

        data_channels = np.ascontiguousarray(data_samples.T)
        with self._data_ready:
            if self._ring is None:
                return
            self._ring.append(data_channels)
            self._data_ready.notify_all()

    def _disconnect(self) -> None:
        inlet = self._inlet
        self._inlet = None
        self._info = None
        self._connected_event.clear()
        with self._lock:
            self._last_lsl_timestamp = None
        if self._timestamp_sync is not None:
            self._timestamp_sync.reset()
            self._last_time_sync_update = 0.0
        if inlet is not None:
            try:
                inlet.close_stream()
            except Exception:
                pass
        with self._data_ready:
            if self._ring is not None:
                self._ring.clear()
            self._data_ready.notify_all()
