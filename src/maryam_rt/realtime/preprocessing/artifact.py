"""Simple artifact rejection for real-time EEG preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class ArtifactAction(Enum):
    """Action to take when artifacts are detected."""

    REJECT = "reject"
    CLAMP = "clamp"
    INTERPOLATE = "interpolate"


@dataclass
class ArtifactConfig:
    """Configuration for artifact rejection.

    Default thresholds are based on THINGS-EEG2 dataset analysis:
    - amplitude_threshold_uv: 150 µV (covers 99.9th percentile)
    - gradient_threshold_uv: 35 µV/sample (covers 99.9th percentile)
    - flat_threshold_uv: 1.0 µV (below 2nd percentile of window std)
    """

    amplitude_threshold_uv: float = 150.0
    gradient_threshold_uv: float = 35.0
    flat_threshold_uv: float = 1.0
    flat_duration_samples: int = 50
    action: ArtifactAction = ArtifactAction.CLAMP
    clamp_value_uv: Optional[float] = None

    def __post_init__(self) -> None:
        if self.amplitude_threshold_uv <= 0:
            raise ValueError("amplitude_threshold_uv must be positive.")
        if self.gradient_threshold_uv <= 0:
            raise ValueError("gradient_threshold_uv must be positive.")
        if self.flat_threshold_uv < 0:
            raise ValueError("flat_threshold_uv must be non-negative.")
        if self.flat_duration_samples < 1:
            raise ValueError("flat_duration_samples must be >= 1.")
        if self.clamp_value_uv is None:
            self.clamp_value_uv = self.amplitude_threshold_uv

    @classmethod
    def for_things_eeg2(cls) -> "ArtifactConfig":
        """Create config optimized for THINGS-EEG2 dataset.

        Based on analysis of sub-01/ses-01 training data:
        - 99.9th percentile amplitude: 151.86 µV
        - 99.9th percentile gradient: 27.09 µV/sample
        - 2nd percentile window std: 1.38 µV
        """
        return cls(
            amplitude_threshold_uv=180.0,
            gradient_threshold_uv=35.0,
            flat_threshold_uv=1.2,
            flat_duration_samples=50,
            action=ArtifactAction.CLAMP,
        )

    @classmethod
    def conservative(cls) -> "ArtifactConfig":
        """Create conservative config that rejects more artifacts."""
        return cls(
            amplitude_threshold_uv=100.0,
            gradient_threshold_uv=25.0,
            flat_threshold_uv=0.5,
            action=ArtifactAction.REJECT,
        )

    @classmethod
    def permissive(cls) -> "ArtifactConfig":
        """Create permissive config that allows more signal through."""
        return cls(
            amplitude_threshold_uv=250.0,
            gradient_threshold_uv=75.0,
            flat_threshold_uv=0.3,
            action=ArtifactAction.CLAMP,
        )


@dataclass
class ArtifactReport:
    """Report of artifact detection results."""

    is_clean: bool
    amplitude_violations: int
    gradient_violations: int
    flat_channel_count: int
    affected_channels: Tuple[int, ...]
    violation_ratio: float

    def __bool__(self) -> bool:
        """Return True if the window is clean."""
        return self.is_clean


class ArtifactDetector:
    """Detect and handle artifacts in EEG windows.

    Implements simple threshold-based artifact detection:
    - Amplitude threshold: Flags samples exceeding ±threshold_uv.
    - Gradient threshold: Flags sample-to-sample changes exceeding threshold.
    - Flat signal detection: Flags channels with no variation.
    """

    def __init__(
        self,
        n_channels: int,
        config: Optional[ArtifactConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize artifact detector.

        Args:
            n_channels: Number of EEG channels.
            config: ArtifactConfig instance or None to use defaults.
            **kwargs: Keyword arguments passed to ArtifactConfig if config is None.
        """
        if config is None:
            config = ArtifactConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self._config = config
        self._n_channels = n_channels

    @property
    def config(self) -> ArtifactConfig:
        """Artifact detection configuration."""
        return self._config

    @property
    def n_channels(self) -> int:
        """Number of channels configured."""
        return self._n_channels

    def reset(self) -> None:
        """Reset detector state. Currently stateless, but provides API for future use."""
        pass

    def detect(self, data: np.ndarray) -> ArtifactReport:
        """Detect artifacts in an EEG window.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            ArtifactReport with detection results.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}.")
        if data.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {data.shape[0]}."
            )

        n_samples = data.shape[1]
        if n_samples == 0:
            return ArtifactReport(
                is_clean=True,
                amplitude_violations=0,
                gradient_violations=0,
                flat_channel_count=0,
                affected_channels=(),
                violation_ratio=0.0,
            )

        affected_channels = set()

        amplitude_mask = np.abs(data) > self._config.amplitude_threshold_uv
        amplitude_violations = int(np.sum(amplitude_mask))
        if amplitude_violations > 0:
            affected_channels.update(np.where(np.any(amplitude_mask, axis=1))[0])

        gradient = np.diff(data, axis=1)
        gradient_mask = np.abs(gradient) > self._config.gradient_threshold_uv
        gradient_violations = int(np.sum(gradient_mask))
        if gradient_violations > 0:
            affected_channels.update(np.where(np.any(gradient_mask, axis=1))[0])

        flat_channels = 0
        if n_samples >= self._config.flat_duration_samples:
            for ch in range(self._n_channels):
                channel_data = data[ch]
                window_std = np.std(channel_data[-self._config.flat_duration_samples:])
                if window_std < self._config.flat_threshold_uv:
                    flat_channels += 1
                    affected_channels.add(ch)

        total_violations = amplitude_violations + gradient_violations
        total_samples = self._n_channels * n_samples
        violation_ratio = total_violations / total_samples if total_samples > 0 else 0.0

        is_clean = (
            amplitude_violations == 0
            and gradient_violations == 0
            and flat_channels == 0
        )

        return ArtifactReport(
            is_clean=is_clean,
            amplitude_violations=amplitude_violations,
            gradient_violations=gradient_violations,
            flat_channel_count=flat_channels,
            affected_channels=tuple(sorted(affected_channels)),
            violation_ratio=violation_ratio,
        )

    def process(
        self, data: np.ndarray, report: Optional[ArtifactReport] = None
    ) -> Tuple[np.ndarray, ArtifactReport]:
        """Detect artifacts and apply configured action.

        Args:
            data: Input array of shape (n_channels, n_samples).
            report: Optional pre-computed ArtifactReport.

        Returns:
            Tuple of (processed_data, report).
        """
        data = np.asarray(data, dtype=np.float32)
        if report is None:
            report = self.detect(data)

        if report.is_clean:
            return data.copy(), report

        output = data.copy()

        if self._config.action == ArtifactAction.REJECT:
            pass

        elif self._config.action == ArtifactAction.CLAMP:
            clamp_val = self._config.clamp_value_uv
            np.clip(output, -clamp_val, clamp_val, out=output)

        elif self._config.action == ArtifactAction.INTERPOLATE:
            for ch in report.affected_channels:
                channel = output[ch]
                bad_mask = np.abs(channel) > self._config.amplitude_threshold_uv
                if np.any(bad_mask) and not np.all(bad_mask):
                    good_indices = np.where(~bad_mask)[0]
                    bad_indices = np.where(bad_mask)[0]
                    channel[bad_indices] = np.interp(
                        bad_indices, good_indices, channel[good_indices]
                    )

        return output, report


class WindowValidator:
    """Validate EEG windows for real-time processing."""

    def __init__(
        self,
        n_channels: int,
        config: Optional[ArtifactConfig] = None,
        max_violation_ratio: float = 0.1,
        max_bad_channels: int = 5,
    ) -> None:
        """Initialize window validator.

        Args:
            n_channels: Number of EEG channels.
            config: ArtifactConfig instance.
            max_violation_ratio: Maximum allowed violation ratio (0-1).
            max_bad_channels: Maximum number of bad channels allowed.
        """
        self._detector = ArtifactDetector(n_channels, config)
        self._max_violation_ratio = max_violation_ratio
        self._max_bad_channels = max_bad_channels

    @property
    def config(self) -> ArtifactConfig:
        """Artifact detection configuration."""
        return self._detector.config

    def is_valid(self, data: np.ndarray) -> bool:
        """Check if a window is valid for processing.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            True if the window passes validation criteria.
        """
        report = self._detector.detect(data)
        if report.is_clean:
            return True

        if report.violation_ratio > self._max_violation_ratio:
            return False

        if len(report.affected_channels) > self._max_bad_channels:
            return False

        return True

    def validate_and_clean(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, bool, ArtifactReport]:
        """Validate and clean a window.

        Args:
            data: Input array of shape (n_channels, n_samples).

        Returns:
            Tuple of (processed_data, is_valid, report).
        """
        processed, report = self._detector.process(data)
        is_valid = (
            report.violation_ratio <= self._max_violation_ratio
            and len(report.affected_channels) <= self._max_bad_channels
        )
        return processed, is_valid, report
