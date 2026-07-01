"""Online normalization and standardization for real-time EEG preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class NormalizationMethod(Enum):
    """Available normalization methods."""

    ZSCORE = "zscore"
    BASELINE = "baseline"
    MINMAX = "minmax"
    ROBUST = "robust"


@dataclass
class NormalizationConfig:
    """Configuration for online normalization."""

    method: NormalizationMethod = NormalizationMethod.ZSCORE
    window_samples: int = 5000
    epsilon: float = 1e-8
    percentile_low: float = 5.0
    percentile_high: float = 95.0
    update_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.window_samples < 1:
            raise ValueError("window_samples must be >= 1.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not (0 <= self.percentile_low < self.percentile_high <= 100):
            raise ValueError("percentile_low must be < percentile_high, both in [0, 100].")
        if not (0 < self.update_rate <= 1):
            raise ValueError("update_rate must be in (0, 1].")


class RunningStatistics:
    """Efficient online computation of running mean and variance.

    Uses Welford's algorithm for numerical stability.
    """

    def __init__(self, n_channels: int, dtype: np.dtype = np.float64) -> None:
        """Initialize running statistics tracker.

        Args:
            n_channels: Number of channels to track.
            dtype: Data type for accumulators.
        """
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        self._n_channels = n_channels
        self._dtype = dtype
        self._count = 0
        self._mean = np.zeros(n_channels, dtype=dtype)
        self._m2 = np.zeros(n_channels, dtype=dtype)
        self._min = np.full(n_channels, np.inf, dtype=dtype)
        self._max = np.full(n_channels, -np.inf, dtype=dtype)

    @property
    def n_channels(self) -> int:
        """Number of channels tracked."""
        return self._n_channels

    @property
    def count(self) -> int:
        """Number of samples seen."""
        return self._count

    @property
    def mean(self) -> np.ndarray:
        """Current running mean per channel."""
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        """Current running variance per channel."""
        if self._count < 2:
            return np.zeros(self._n_channels, dtype=self._dtype)
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> np.ndarray:
        """Current running standard deviation per channel."""
        return np.sqrt(self.variance)

    @property
    def min(self) -> np.ndarray:
        """Minimum values seen per channel."""
        return self._min.copy()

    @property
    def max(self) -> np.ndarray:
        """Maximum values seen per channel."""
        return self._max.copy()

    def reset(self) -> None:
        """Reset all statistics."""
        self._count = 0
        self._mean.fill(0)
        self._m2.fill(0)
        self._min.fill(np.inf)
        self._max.fill(-np.inf)

    def update(self, data: np.ndarray) -> None:
        """Update statistics with new samples.

        Args:
            data: Input array of shape (n_channels, n_samples).
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}.")
        if data.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {data.shape[0]}."
            )

        for i in range(data.shape[1]):
            sample = data[:, i].astype(self._dtype)
            self._count += 1
            delta = sample - self._mean
            self._mean += delta / self._count
            delta2 = sample - self._mean
            self._m2 += delta * delta2
            np.minimum(self._min, sample, out=self._min)
            np.maximum(self._max, sample, out=self._max)

    def update_batch(self, data: np.ndarray) -> None:
        """Update statistics with batch of samples (faster for large chunks).

        Args:
            data: Input array of shape (n_channels, n_samples).
        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != self._n_channels:
            self.update(data)
            return

        n_samples = data.shape[1]
        if n_samples == 0:
            return

        batch_mean = np.mean(data, axis=1, dtype=self._dtype)
        batch_var = np.var(data, axis=1, dtype=self._dtype, ddof=0)
        batch_min = np.min(data, axis=1)
        batch_max = np.max(data, axis=1)

        if self._count == 0:
            self._count = n_samples
            self._mean = batch_mean
            self._m2 = batch_var * n_samples
            self._min = batch_min
            self._max = batch_max
        else:
            n_total = self._count + n_samples
            delta = batch_mean - self._mean
            self._mean = (self._count * self._mean + n_samples * batch_mean) / n_total
            self._m2 += batch_var * n_samples + delta**2 * self._count * n_samples / n_total
            self._count = n_total
            np.minimum(self._min, batch_min, out=self._min)
            np.maximum(self._max, batch_max, out=self._max)


class RollingBuffer:
    """Fixed-size rolling buffer for percentile computation."""

    def __init__(self, n_channels: int, capacity: int, dtype: np.dtype = np.float32) -> None:
        """Initialize rolling buffer.

        Args:
            n_channels: Number of channels.
            capacity: Maximum number of samples to store.
            dtype: Data type for storage.
        """
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._n_channels = n_channels
        self._capacity = capacity
        self._buffer = np.zeros((n_channels, capacity), dtype=dtype)
        self._write_idx = 0
        self._filled = 0

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._n_channels

    @property
    def filled(self) -> int:
        """Number of samples currently stored."""
        return self._filled

    def reset(self) -> None:
        """Clear the buffer."""
        self._write_idx = 0
        self._filled = 0

    def append(self, data: np.ndarray) -> None:
        """Append samples to the buffer.

        Args:
            data: Input array of shape (n_channels, n_samples).
        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != self._n_channels:
            raise ValueError(f"Expected shape ({self._n_channels}, n_samples), got {data.shape}.")

        n_samples = data.shape[1]
        if n_samples == 0:
            return

        if n_samples >= self._capacity:
            self._buffer[:] = data[:, -self._capacity:]
            self._write_idx = 0
            self._filled = self._capacity
            return

        end_idx = self._write_idx + n_samples
        if end_idx <= self._capacity:
            self._buffer[:, self._write_idx:end_idx] = data
        else:
            first = self._capacity - self._write_idx
            self._buffer[:, self._write_idx:] = data[:, :first]
            self._buffer[:, :end_idx % self._capacity] = data[:, first:]

        self._write_idx = end_idx % self._capacity
        self._filled = min(self._capacity, self._filled + n_samples)

    def get_data(self) -> np.ndarray:
        """Return all stored data.

        Returns:
            Array of shape (n_channels, filled_samples).
        """
        if self._filled < self._capacity:
            return self._buffer[:, :self._filled].copy()
        return self._buffer.copy()

    def percentile(self, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute percentiles across all stored samples.

        Args:
            low: Lower percentile (0-100).
            high: Upper percentile (0-100).

        Returns:
            Tuple of (low_percentile, high_percentile) arrays per channel.
        """
        if self._filled == 0:
            zeros = np.zeros(self._n_channels)
            ones = np.ones(self._n_channels)
            return zeros, ones

        data = self.get_data()
        p_low = np.percentile(data, low, axis=1)
        p_high = np.percentile(data, high, axis=1)
        return p_low, p_high


class OnlineNormalizer:
    """Real-time normalization for streaming EEG data.

    Supports multiple normalization methods with efficient online updates.
    """

    def __init__(
        self,
        n_channels: int,
        config: Optional[NormalizationConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize online normalizer.

        Args:
            n_channels: Number of EEG channels.
            config: NormalizationConfig instance or None for defaults.
            **kwargs: Keyword arguments passed to NormalizationConfig if config is None.
        """
        if config is None:
            config = NormalizationConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self._config = config
        self._n_channels = n_channels
        self._stats = RunningStatistics(n_channels)
        self._buffer = RollingBuffer(n_channels, config.window_samples)
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._samples_since_update = 0

    @property
    def config(self) -> NormalizationConfig:
        """Normalization configuration."""
        return self._config

    @property
    def n_channels(self) -> int:
        """Number of channels configured."""
        return self._n_channels

    @property
    def is_ready(self) -> bool:
        """True if enough samples have been collected for stable normalization."""
        return self._stats.count >= 100

    def reset(self) -> None:
        """Reset normalization state."""
        self._stats.reset()
        self._buffer.reset()
        self._baseline_mean = None
        self._baseline_std = None
        self._samples_since_update = 0

    def set_baseline(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set explicit baseline statistics.

        Args:
            mean: Baseline mean per channel.
            std: Baseline standard deviation per channel.
        """
        mean = np.asarray(mean).flatten()
        std = np.asarray(std).flatten()
        if mean.shape[0] != self._n_channels or std.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} values, got {mean.shape[0]}, {std.shape[0]}."
            )
        self._baseline_mean = mean.astype(np.float64)
        self._baseline_std = np.maximum(std.astype(np.float64), self._config.epsilon)

    def update(self, data: np.ndarray) -> None:
        """Update normalization statistics with new samples.

        Args:
            data: Input array of shape (n_channels, n_samples).
        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != self._n_channels:
            raise ValueError(f"Expected shape ({self._n_channels}, n_samples), got {data.shape}.")

        self._stats.update_batch(data)
        self._buffer.append(data)
        self._samples_since_update += data.shape[1]

    def normalize(self, data: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize data using the configured method.

        Args:
            data: Input array of shape (n_channels, n_samples).
            update: Whether to update statistics with this data.

        Returns:
            Normalized array of shape (n_channels, n_samples).
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] != self._n_channels:
            raise ValueError(f"Expected shape ({self._n_channels}, n_samples), got {data.shape}.")

        if update:
            self.update(data)

        method = self._config.method
        eps = self._config.epsilon

        if method == NormalizationMethod.ZSCORE:
            if self._baseline_mean is not None:
                mean = self._baseline_mean
                std = self._baseline_std
            else:
                mean = self._stats.mean
                std = np.maximum(self._stats.std, eps)
            return ((data.T - mean) / std).T.astype(np.float32)

        elif method == NormalizationMethod.BASELINE:
            if self._baseline_mean is not None:
                mean = self._baseline_mean
            else:
                mean = self._stats.mean
            return (data.T - mean).T.astype(np.float32)

        elif method == NormalizationMethod.MINMAX:
            min_val = self._stats.min
            max_val = self._stats.max
            range_val = np.maximum(max_val - min_val, eps)
            return ((data.T - min_val) / range_val).T.astype(np.float32)

        elif method == NormalizationMethod.ROBUST:
            p_low, p_high = self._buffer.percentile(
                self._config.percentile_low, self._config.percentile_high
            )
            median = (p_low + p_high) / 2
            iqr = np.maximum(p_high - p_low, eps)
            return ((data.T - median) / iqr).T.astype(np.float32)

        else:
            raise ValueError(f"Unknown normalization method: {method}")


class PreprocessingPipeline:
    """Complete preprocessing pipeline combining filtering, artifact rejection, and normalization."""

    def __init__(
        self,
        n_channels: int,
        sampling_rate: float,
        filter_config=None,
        artifact_config=None,
        normalization_config=None,
    ) -> None:
        """Initialize preprocessing pipeline.

        Args:
            n_channels: Number of EEG channels.
            sampling_rate: Sampling rate in Hz.
            filter_config: FilterConfig instance (optional).
            artifact_config: ArtifactConfig instance (optional).
            normalization_config: NormalizationConfig instance (optional).
        """
        from .filters import RealTimeFilter, FilterConfig
        from .artifact import ArtifactDetector, ArtifactConfig

        self._n_channels = n_channels
        self._sampling_rate = sampling_rate

        if filter_config is None:
            filter_config = FilterConfig(sampling_rate=sampling_rate)
        self._filter = RealTimeFilter(n_channels, filter_config)

        if artifact_config is None:
            artifact_config = ArtifactConfig()
        self._artifact_detector = ArtifactDetector(n_channels, artifact_config)

        if normalization_config is None:
            normalization_config = NormalizationConfig()
        self._normalizer = OnlineNormalizer(n_channels, normalization_config)

    @property
    def n_channels(self) -> int:
        """Number of channels configured."""
        return self._n_channels

    @property
    def sampling_rate(self) -> float:
        """Configured sampling rate."""
        return self._sampling_rate

    def reset(self) -> None:
        """Reset all pipeline components."""
        self._filter.reset()
        self._artifact_detector.reset()
        self._normalizer.reset()

    def process(self, data: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process data through the complete pipeline.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            Tuple of (processed_data, is_valid).
        """
        filtered = self._filter.process(data)

        cleaned, report = self._artifact_detector.process(filtered)

        normalized = self._normalizer.normalize(cleaned)

        is_valid = report.violation_ratio < 0.1

        return normalized, is_valid
