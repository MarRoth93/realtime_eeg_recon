"""Fixed-size ring buffer for preprocessing windows."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Optional

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer for samples shaped (n_channels, n_samples)."""

    def __init__(
        self,
        n_channels: int,
        capacity: int,
        dtype: np.dtype = np.float32,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._n_channels = int(n_channels)
        self._capacity = int(capacity)
        self._buffer = np.zeros((self._n_channels, self._capacity), dtype=dtype)
        self._write_index = 0
        self._filled = 0
        self._guard = lock if lock is not None else nullcontext()

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
        with self._guard:
            return self._filled

    def clear(self) -> None:
        """Reset the buffer to an empty state."""
        with self._guard:
            self._write_index = 0
            self._filled = 0

    def append(self, data: np.ndarray) -> int:
        """Append samples shaped (n_channels, n_samples) to the buffer."""
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}.")
        if data.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {data.shape[0]} in {data.shape}."
            )

        n_samples = data.shape[1]
        if n_samples == 0:
            return 0

        with self._guard:
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
        with self._guard:
            return self._get_window_locked(self._filled - n_samples, self._filled, fill_value)

    def get_window(self, start: int, end: int, fill_value: float = 0.0) -> np.ndarray:
        """Return samples from [start, end) as (n_channels, n_samples)."""
        length = end - start
        if length <= 0:
            raise ValueError("end must be greater than start.")
        if length > self._capacity:
            raise ValueError(
                f"Requested window length ({length}) exceeds buffer capacity ({self._capacity})."
            )
        with self._guard:
            return self._get_window_locked(start, end, fill_value)

    def _get_window_locked(self, start: int, end: int, fill_value: float) -> np.ndarray:
        if self._filled == 0:
            return np.full((self._n_channels, end - start), fill_value, dtype=self._buffer.dtype)

        available_start = max(start, 0)
        available_end = min(end, self._filled)

        if available_end <= available_start:
            return np.full((self._n_channels, end - start), fill_value, dtype=self._buffer.dtype)

        if start >= 0 and end <= self._filled:
            return np.ascontiguousarray(self._read_window(available_start, available_end))

        output = np.full((self._n_channels, end - start), fill_value, dtype=self._buffer.dtype)
        data = self._read_window(available_start, available_end)
        offset = available_start - start
        output[:, offset : offset + data.shape[1]] = data
        return output

    def _read_window(self, start: int, end: int) -> np.ndarray:
        length = end - start
        if length <= 0:
            return np.empty((self._n_channels, 0), dtype=self._buffer.dtype)

        start_index = (self._write_index - self._filled + start) % self._capacity
        end_index = start_index + length

        if end_index <= self._capacity:
            return self._buffer[:, start_index:end_index].copy()

        first = self._capacity - start_index
        return np.concatenate(
            (self._buffer[:, start_index:], self._buffer[:, : end_index % self._capacity]),
            axis=1,
        ).copy()
