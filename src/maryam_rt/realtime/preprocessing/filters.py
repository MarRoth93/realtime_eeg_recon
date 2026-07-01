"""Real-time digital signal processing filters for EEG preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi


@dataclass
class FilterConfig:
    """Configuration for real-time EEG filtering."""

    sampling_rate: float = 1000.0
    bandpass_low: float = 0.1
    bandpass_high: float = 100.0
    bandpass_order: int = 4
    notch_freqs: Tuple[float, ...] = (50.0, 60.0)
    notch_q: float = 30.0

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if self.bandpass_low <= 0:
            raise ValueError("bandpass_low must be positive.")
        if self.bandpass_high >= self.sampling_rate / 2:
            raise ValueError("bandpass_high must be less than Nyquist frequency.")
        if self.bandpass_low >= self.bandpass_high:
            raise ValueError("bandpass_low must be less than bandpass_high.")
        if self.bandpass_order < 1:
            raise ValueError("bandpass_order must be >= 1.")
        if self.notch_q <= 0:
            raise ValueError("notch_q must be positive.")


def design_bandpass_filter(
    low_freq: float,
    high_freq: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Design a Butterworth bandpass filter in SOS format.

    Args:
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Second-order sections representation of the filter.
    """
    nyquist = sampling_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def design_notch_filter(
    notch_freq: float,
    sampling_rate: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Design an IIR notch filter in SOS format.

    Args:
        notch_freq: Center frequency to notch out in Hz.
        sampling_rate: Sampling rate in Hz.
        quality_factor: Quality factor (higher = narrower notch).

    Returns:
        Second-order sections representation of the filter.
    """
    nyquist = sampling_rate / 2.0
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])
    return sos


class StatefulFilter:
    """Stateful wrapper for scipy SOS filter with per-channel state."""

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        """Initialize stateful filter.

        Args:
            sos: Second-order sections filter coefficients.
            n_channels: Number of channels to filter independently.
        """
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        self._sos = np.asarray(sos)
        self._n_channels = n_channels
        self._zi_template = sosfilt_zi(self._sos)
        self._zi: Optional[np.ndarray] = None
        self.reset()

    @property
    def n_channels(self) -> int:
        """Number of channels this filter handles."""
        return self._n_channels

    @property
    def sos(self) -> np.ndarray:
        """Filter coefficients in SOS format."""
        return self._sos

    def reset(self) -> None:
        """Reset filter state to initial conditions."""
        self._zi = np.zeros((self._n_channels, self._zi_template.shape[0], 2))

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data with state preservation.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            Filtered array of shape (n_channels, n_samples).
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}.")
        if data.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {data.shape[0]}."
            )

        if data.shape[1] == 0:
            return data.copy()

        output = np.empty_like(data)
        for ch in range(self._n_channels):
            output[ch], self._zi[ch] = sosfilt(
                self._sos, data[ch], zi=self._zi[ch]
            )
        return output


class FilterChain:
    """Chain of stateful filters applied sequentially."""

    def __init__(self, filters: Optional[List[StatefulFilter]] = None) -> None:
        """Initialize filter chain.

        Args:
            filters: List of StatefulFilter instances to apply in order.
        """
        self._filters: List[StatefulFilter] = filters or []
        self._n_channels: Optional[int] = None
        if self._filters:
            self._n_channels = self._filters[0].n_channels
            for f in self._filters:
                if f.n_channels != self._n_channels:
                    raise ValueError("All filters must have the same n_channels.")

    @property
    def n_channels(self) -> Optional[int]:
        """Number of channels, if filters are configured."""
        return self._n_channels

    def add_filter(self, filt: StatefulFilter) -> None:
        """Add a filter to the chain.

        Args:
            filt: StatefulFilter instance to append.
        """
        if self._n_channels is None:
            self._n_channels = filt.n_channels
        elif filt.n_channels != self._n_channels:
            raise ValueError(
                f"Filter has {filt.n_channels} channels, expected {self._n_channels}."
            )
        self._filters.append(filt)

    def reset(self) -> None:
        """Reset all filter states."""
        for filt in self._filters:
            filt.reset()

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            Filtered array of shape (n_channels, n_samples).
        """
        output = data
        for filt in self._filters:
            output = filt.process(output)
        return output


class RealTimeFilter:
    """Complete real-time filtering pipeline for EEG data.

    Combines bandpass and notch filters with stateful processing
    for continuous streaming applications.
    """

    def __init__(
        self,
        n_channels: int,
        config: Optional[FilterConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize real-time filter.

        Args:
            n_channels: Number of EEG channels.
            config: FilterConfig instance or None to use defaults.
            **kwargs: Keyword arguments passed to FilterConfig if config is None.
        """
        if config is None:
            config = FilterConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self._config = config
        self._n_channels = n_channels
        self._chain = FilterChain()

        bandpass_sos = design_bandpass_filter(
            config.bandpass_low,
            config.bandpass_high,
            config.sampling_rate,
            config.bandpass_order,
        )
        self._chain.add_filter(StatefulFilter(bandpass_sos, n_channels))

        for notch_freq in config.notch_freqs:
            if 0 < notch_freq < config.sampling_rate / 2:
                notch_sos = design_notch_filter(
                    notch_freq, config.sampling_rate, config.notch_q
                )
                self._chain.add_filter(StatefulFilter(notch_sos, n_channels))

    @property
    def config(self) -> FilterConfig:
        """Filter configuration."""
        return self._config

    @property
    def n_channels(self) -> int:
        """Number of channels configured."""
        return self._n_channels

    def reset(self) -> None:
        """Reset all filter states."""
        self._chain.reset()

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply the complete filtering pipeline.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            Filtered array of shape (n_channels, n_samples).

        Note:
            Processing time is typically < 0.5ms per chunk on modern hardware.
        """
        return self._chain.process(data)
