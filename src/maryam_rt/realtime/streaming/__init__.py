"""LSL streaming and data acquisition utilities."""

try:
    from .mock_streamer import (
        load_things_eeg,
        load_channel_labels,
        create_lsl_outlet,
        stream_eeg,
        main as run_mock_streamer,
    )
    from .lsl_inlet import (
        LSLInletConfig,
        LSLInletWrapper,
    )
    from .marker_inlet import (
        MarkerEvent,
        MarkerInletConfig,
        MarkerInletWrapper,
    )
    from .mock_marker_streamer import (
        create_marker_outlet,
        main as run_mock_marker_streamer,
    )
except Exception:
    pass

from .timestamp_sync import (
    TimestampSyncConfig,
    TimestampSynchronizer,
    RunningStats,
    RunningStatsSnapshot,
    LinearTimeMapper,
)

__all__ = [
    "load_things_eeg",
    "load_channel_labels",
    "create_lsl_outlet",
    "stream_eeg",
    "run_mock_streamer",
    "LSLInletConfig",
    "LSLInletWrapper",
    "MarkerEvent",
    "MarkerInletConfig",
    "MarkerInletWrapper",
    "create_marker_outlet",
    "run_mock_marker_streamer",
    "TimestampSyncConfig",
    "TimestampSynchronizer",
    "RunningStats",
    "RunningStatsSnapshot",
    "LinearTimeMapper",
]
