"""Timestamp synchronization utilities for aligning LSL time with local clocks."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

try:
    from pylsl import local_clock as lsl_local_clock
except Exception:  # Fallback for environments without pylsl available.
    def lsl_local_clock() -> float:
        return time.time()


@dataclass(frozen=True)
class RunningStatsSnapshot:
    """Immutable snapshot of running statistics."""

    count: int
    mean: float
    variance: float
    stddev: float
    min_value: Optional[float]
    max_value: Optional[float]


class RunningStats:
    """Online mean/variance tracker using Welford's algorithm."""

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    @property
    def count(self) -> int:
        """Number of samples seen so far."""
        return self._count

    @property
    def mean(self) -> float:
        """Running mean of the samples."""
        return self._mean if self._count else 0.0

    @property
    def variance(self) -> float:
        """Unbiased sample variance of the samples."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def stddev(self) -> float:
        """Unbiased sample standard deviation of the samples."""
        return math.sqrt(self.variance)

    @property
    def min_value(self) -> Optional[float]:
        """Minimum sample value observed."""
        return self._min

    @property
    def max_value(self) -> Optional[float]:
        """Maximum sample value observed."""
        return self._max

    def update(self, value: float) -> None:
        """Update statistics with a new sample value."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if self._min is None or value < self._min:
            self._min = value
        if self._max is None or value > self._max:
            self._max = value

    def reset(self) -> None:
        """Reset all statistics."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = None
        self._max = None

    def snapshot(self) -> RunningStatsSnapshot:
        """Return an immutable snapshot of the current statistics."""
        return RunningStatsSnapshot(
            count=self.count,
            mean=self.mean,
            variance=self.variance,
            stddev=self.stddev,
            min_value=self.min_value,
            max_value=self.max_value,
        )


class LinearTimeMapper:
    """Sliding-window linear regression for clock alignment."""

    def __init__(self, window_size: int, min_samples: int = 10) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if min_samples > window_size:
            raise ValueError("min_samples must be <= window_size.")
        self._window_size = window_size
        self._min_samples = min_samples
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_x2 = 0.0
        self._sum_xy = 0.0
        self._slope = 1.0
        self._intercept = 0.0
        self._residual_stats = RunningStats()

    @property
    def slope(self) -> float:
        """Current slope estimate (drift correction)."""
        return self._slope

    @property
    def intercept(self) -> float:
        """Current intercept estimate (offset correction)."""
        return self._intercept

    @property
    def sample_count(self) -> int:
        """Number of samples in the current window."""
        return len(self._samples)

    @property
    def residual_stats(self) -> RunningStats:
        """Running statistics of alignment residuals."""
        return self._residual_stats

    def reset(self) -> None:
        """Clear the regression window and statistics."""
        self._samples.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_x2 = 0.0
        self._sum_xy = 0.0
        self._slope = 1.0
        self._intercept = 0.0
        self._residual_stats.reset()

    def update(self, source_time: float, target_time: float) -> None:
        """Add a new (source, target) time pair to the regression window."""
        if len(self._samples) == self._window_size:
            old_x, old_y = self._samples.popleft()
            self._sum_x -= old_x
            self._sum_y -= old_y
            self._sum_x2 -= old_x * old_x
            self._sum_xy -= old_x * old_y

        self._samples.append((source_time, target_time))
        self._sum_x += source_time
        self._sum_y += target_time
        self._sum_x2 += source_time * source_time
        self._sum_xy += source_time * target_time
        self._recompute_fit()
        residual = target_time - self.predict(source_time)
        self._residual_stats.update(residual)

    def predict(self, source_time: float) -> float:
        """Predict target time from source time."""
        return (self._slope * source_time) + self._intercept

    def invert(self, target_time: float) -> float:
        """Predict source time from target time."""
        if abs(self._slope) < 1e-12:
            return target_time
        return (target_time - self._intercept) / self._slope

    def _recompute_fit(self) -> None:
        n = len(self._samples)
        if n == 0:
            self._slope = 1.0
            self._intercept = 0.0
            return

        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        sxx = self._sum_x2 - (n * mean_x * mean_x)
        sxy = self._sum_xy - (n * mean_x * mean_y)

        if n >= self._min_samples and abs(sxx) > 1e-12:
            self._slope = sxy / sxx
            self._intercept = mean_y - (self._slope * mean_x)
        else:
            self._slope = 1.0
            self._intercept = mean_y - mean_x


@dataclass(frozen=True)
class TimestampSyncConfig:
    """Configuration for timestamp synchronization."""

    window_size: int = 256
    min_samples: int = 10
    time_correction_alpha: float = 0.1
    local_time_fn: Callable[[], float] = time.perf_counter
    lsl_time_fn: Callable[[], float] = lsl_local_clock

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if self.min_samples > self.window_size:
            raise ValueError("min_samples must be <= window_size.")
        if not (0.0 < self.time_correction_alpha <= 1.0):
            raise ValueError("time_correction_alpha must be in (0, 1].")


class TimestampSynchronizer:
    """Align LSL timestamps to the local clock with drift correction."""

    def __init__(self, config: Optional[TimestampSyncConfig] = None, **kwargs: object) -> None:
        if config is None:
            config = TimestampSyncConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")
        self._config = config
        self._aligner = LinearTimeMapper(config.window_size, config.min_samples)
        self._time_correction: Optional[float] = None
        self._time_correction_stats = RunningStats()
        self._correction_stats = RunningStats()
        self._last_lsl_timestamp: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def config(self) -> TimestampSyncConfig:
        """Access the configuration used for synchronization."""
        return self._config

    @property
    def last_lsl_timestamp(self) -> Optional[float]:
        """Most recent raw LSL timestamp observed."""
        with self._lock:
            return self._last_lsl_timestamp

    def reset(self) -> None:
        """Reset the synchronizer state and statistics."""
        with self._lock:
            self._aligner.reset()
            self._time_correction = None
            self._time_correction_stats.reset()
            self._correction_stats.reset()
            self._last_lsl_timestamp = None

    def local_time(self) -> float:
        """Return the current local clock time."""
        return self._config.local_time_fn()

    def lsl_time(self) -> float:
        """Return the current local LSL clock time."""
        return self._config.lsl_time_fn()

    def update_local_alignment(
        self, local_time: Optional[float] = None, lsl_time: Optional[float] = None
    ) -> Tuple[float, float]:
        """Update the local clock alignment with a sampled time pair."""
        if local_time is None:
            local_time = self.local_time()
        if lsl_time is None:
            lsl_time = self.lsl_time()
        with self._lock:
            self._aligner.update(lsl_time, local_time)
        return lsl_time, local_time

    def update_time_correction(self, time_correction: float) -> None:
        """Update the smoothed time correction from StreamInlet.time_correction()."""
        with self._lock:
            if self._time_correction is None:
                self._time_correction = time_correction
            else:
                alpha = self._config.time_correction_alpha
                self._time_correction = ((1.0 - alpha) * self._time_correction) + (
                    alpha * time_correction
                )
            self._time_correction_stats.update(time_correction)

    def note_lsl_timestamp(self, lsl_timestamp: float) -> None:
        """Record the latest LSL timestamp and update correction statistics."""
        with self._lock:
            self._last_lsl_timestamp = lsl_timestamp
            if self._time_correction is None:
                return
            correction = self._predict_local_time(lsl_timestamp) - lsl_timestamp
            self._correction_stats.update(correction)

    def lsl_to_local_time(self, lsl_timestamp: float) -> float:
        """Convert an LSL timestamp to the local clock domain."""
        with self._lock:
            return self._predict_local_time(lsl_timestamp)

    def local_to_lsl_time(self, local_time: float) -> float:
        """Convert a local clock time back to the LSL timestamp domain."""
        with self._lock:
            lsl_local_time = self._aligner.invert(local_time)
            if self._time_correction is None:
                return lsl_local_time
            return lsl_local_time - self._time_correction

    def correction_for_lsl(self, lsl_timestamp: float) -> float:
        """Return the correction that maps an LSL timestamp to local time."""
        return self.lsl_to_local_time(lsl_timestamp) - lsl_timestamp

    def time_correction(self) -> Optional[float]:
        """Return the current smoothed time correction (stream to local LSL clock)."""
        with self._lock:
            return self._time_correction

    def alignment_stats(self) -> RunningStatsSnapshot:
        """Return residual statistics for local clock alignment."""
        with self._lock:
            return self._aligner.residual_stats.snapshot()

    def correction_stats(self) -> RunningStatsSnapshot:
        """Return running statistics for applied LSL-to-local corrections."""
        with self._lock:
            return self._correction_stats.snapshot()

    def time_correction_stats(self) -> RunningStatsSnapshot:
        """Return running statistics of StreamInlet time corrections."""
        with self._lock:
            return self._time_correction_stats.snapshot()

    def drift_ppm(self) -> float:
        """Return estimated drift in parts per million."""
        with self._lock:
            return (self._aligner.slope - 1.0) * 1_000_000.0

    def estimated_jitter_seconds(self) -> float:
        """Return a rough jitter estimate based on alignment residuals."""
        with self._lock:
            alignment_jitter = self._aligner.residual_stats.stddev
            correction_jitter = self._time_correction_stats.stddev
        return math.hypot(alignment_jitter, correction_jitter)

    def _predict_local_time(self, lsl_timestamp: float) -> float:
        lsl_local_time = lsl_timestamp + (self._time_correction or 0.0)
        return self._aligner.predict(lsl_local_time)
