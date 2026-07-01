"""Preprocessing utilities for the real-time pipeline."""

from .buffer import RingBuffer
from .filters import (
    FilterConfig,
    RealTimeFilter,
    StatefulFilter,
    FilterChain,
    design_bandpass_filter,
    design_notch_filter,
)
from .artifact import (
    ArtifactAction,
    ArtifactConfig,
    ArtifactDetector,
    ArtifactReport,
    WindowValidator,
)
from .normalize import (
    NormalizationMethod,
    NormalizationConfig,
    OnlineNormalizer,
    RunningStatistics,
    PreprocessingPipeline,
)

__all__ = [
    "RingBuffer",
    "FilterConfig",
    "RealTimeFilter",
    "StatefulFilter",
    "FilterChain",
    "design_bandpass_filter",
    "design_notch_filter",
    "ArtifactAction",
    "ArtifactConfig",
    "ArtifactDetector",
    "ArtifactReport",
    "WindowValidator",
    "NormalizationMethod",
    "NormalizationConfig",
    "OnlineNormalizer",
    "RunningStatistics",
    "PreprocessingPipeline",
]
