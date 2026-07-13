from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from maryam_rt.gui.monitor import RuntimeMonitorState
from maryam_rt.integration.live_markers import load_live_trigger_map
from maryam_rt.integration.triggered_runner import TriggeredReconstructionRunner, TriggeredRunnerConfig
from maryam_rt.realtime.streaming.marker_inlet import MarkerEvent


class FakeEEGInlet:
    def __init__(self) -> None:
        self.is_connected = False
        self.channel_count = 32
        self.nominal_sampling_rate = 500.0
        self.sampling_rate = 500.0
        self.buffer_seconds = 0.0
        self.last_sample_age_seconds: float | None = None
        self.stream_name = "StarstimEEG"
        self.channel_labels: tuple[str, ...] = ()
        self.connection_generation = 1
        self.last_lsl_timestamp: float | None = None
        self.started = False

    @property
    def available_samples(self) -> int:
        return int(round(self.buffer_seconds * self.sampling_rate))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def get_segment_by_lsl_times(self, start_lsl: float, end_lsl: float) -> np.ndarray:
        samples = int(round((end_lsl - start_lsl) * self.sampling_rate))
        return np.zeros((32, samples), dtype=np.float32)

    def get_window(self, n_samples: int, timeout: float = 0.0) -> np.ndarray:
        return np.zeros((32, n_samples), dtype=np.float32)


class FakeMarkerInlet:
    def __init__(self) -> None:
        self.is_connected = False
        self.stream_name = "TaskMarkers"
        self.connection_generation = 1
        self.started = False
        self.events: list[MarkerEvent] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def pop_events(self) -> list[MarkerEvent]:
        events = list(self.events)
        self.events.clear()
        return events


class FakeEncoder:
    device_name = "cpu"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert tuple(x.shape[1:]) == (31, 250)
        return torch.zeros((1, 4, 8, 8), dtype=torch.float32)

    def snapshot_last_x250(self) -> torch.Tensor:
        return torch.zeros((1, 31, 250), dtype=torch.float32)


class FakeDecoder:
    def __call__(self, latent: torch.Tensor) -> np.ndarray:
        return np.zeros((16, 16, 3), dtype=np.uint8)


class FailOnceEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("expected first-trial failure")
        return super().__call__(x)


def make_monitor(*, marker_test_required: bool = False) -> RuntimeMonitorState:
    monitor = RuntimeMonitorState(
        mode="lab-live",
        high_level_enabled=False,
        session_id="demo-session",
        participant_code="unseen-001",
        participant_mode="unseen",
        marker_test_required=marker_test_required,
        expected_eeg_channel_count=32,
        expected_eeg_sampling_rate=500.0,
    )
    monitor.set_models_loaded(True)
    return monitor


def make_runner(
    tmp_path: Path,
    monitor: RuntimeMonitorState,
    *,
    eeg: FakeEEGInlet | None = None,
    markers: FakeMarkerInlet | None = None,
    config: TriggeredRunnerConfig | None = None,
) -> TriggeredReconstructionRunner:
    eeg = eeg or FakeEEGInlet()
    markers = markers or FakeMarkerInlet()
    config = config or TriggeredRunnerConfig(
        eeg_stream_name="StarstimEEG",
        marker_stream_name="TaskMarkers",
        pre_event_ms=200.0,
        post_event_ms=1000.0,
        eeg_sampling_rate=500.0,
        eeg_channels=32,
        expected_epoch_samples=None,
        min_buffer_seconds=1.0,
        post_event_timeout_s=0.02,
        require_arm=True,
        start_on_first_trigger=False,
        session_id="demo-session",
        participant_code="unseen-001",
    )
    return TriggeredReconstructionRunner(
        encoder=FakeEncoder(),
        decoder=FakeDecoder(),
        worker=None,
        output_low_dir=tmp_path / "low_level",
        output_meta_dir=tmp_path / "events",
        output_target_dir=tmp_path / "targets",
        config=config,
        epoch_preprocessor=lambda epoch: np.zeros((31, 250), dtype=np.float32),
        monitor=monitor,
        eeg_inlet=eeg,
        marker_inlet=markers,
    )


def make_ready(runner: TriggeredReconstructionRunner) -> None:
    runner.eeg_inlet.is_connected = True
    runner.eeg_inlet.buffer_seconds = 1.2
    runner.eeg_inlet.last_sample_age_seconds = 0.01
    runner.marker_inlet.is_connected = True
    assert runner._refresh_stream_health()


def test_readiness_waits_for_late_streams_and_valid_buffer(tmp_path: Path) -> None:
    monitor = make_monitor()
    runner = make_runner(tmp_path, monitor)

    assert not runner._refresh_stream_health()
    assert monitor.status.phase == "waiting_for_equipment"

    runner.eeg_inlet.is_connected = True
    runner.marker_inlet.is_connected = True
    runner.eeg_inlet.buffer_seconds = 0.5
    runner.eeg_inlet.last_sample_age_seconds = 0.01
    assert not runner._refresh_stream_health()
    assert not monitor.snapshot()["readiness"]["buffer"]["ok"]

    runner.eeg_inlet.buffer_seconds = 1.1
    assert runner._refresh_stream_health()
    assert monitor.status.phase == "ready"


def test_runner_does_not_exit_when_streams_are_late(tmp_path: Path) -> None:
    monitor = make_monitor()
    config = TriggeredRunnerConfig(
        eeg_stream_name="late-eeg",
        marker_stream_name="late-markers",
        eeg_sampling_rate=500.0,
        eeg_channels=32,
        expected_epoch_samples=None,
        connect_timeout_s=0.01,
        poll_interval_ms=1.0,
        session_id="demo-session",
        participant_code="unseen-001",
    )
    runner = make_runner(tmp_path, monitor, config=config)
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    time.sleep(0.03)

    assert thread.is_alive()
    assert monitor.status.phase == "waiting_for_equipment"

    runner.stop()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_wrong_actual_rate_never_becomes_ready(tmp_path: Path) -> None:
    monitor = make_monitor()
    eeg = FakeEEGInlet()
    eeg.is_connected = True
    eeg.nominal_sampling_rate = 250.0
    eeg.buffer_seconds = 2.0
    eeg.last_sample_age_seconds = 0.01
    markers = FakeMarkerInlet()
    markers.is_connected = True
    runner = make_runner(tmp_path, monitor, eeg=eeg, markers=markers)

    assert not runner._refresh_stream_health()
    state = monitor.snapshot()
    assert not state["readiness"]["eeg_format"]["ok"]
    assert state["status"]["eeg_sampling_rate"] == 250.0


def test_equipment_loss_disarms_and_requires_operator_rearm(tmp_path: Path) -> None:
    monitor = make_monitor()
    runner = make_runner(tmp_path, monitor)
    make_ready(runner)
    assert runner.arm()[0]

    runner.eeg_inlet.is_connected = False
    assert not runner._refresh_stream_health()
    assert not runner.is_armed
    assert monitor.status.phase == "waiting_for_equipment"

    runner.eeg_inlet.is_connected = True
    runner.eeg_inlet.buffer_seconds = 1.2
    runner.eeg_inlet.last_sample_age_seconds = 0.01
    assert runner._refresh_stream_health()
    assert monitor.status.phase == "ready"
    assert not runner.is_armed


def test_arm_start_stimulus_and_end_are_explicit(tmp_path: Path) -> None:
    monitor = make_monitor()
    runner = make_runner(tmp_path, monitor)
    make_ready(runner)
    armed, _ = runner.arm()
    assert armed

    stimulus = MarkerEvent(
        value=json.dumps(
            {
                "event": "stim_onset",
                "session_id": "demo-session",
                "trial_id": 1,
            }
        ),
        timestamp=10.0,
    )
    runner._handle_event(stimulus)
    assert monitor.status.trials_processed == 0
    assert monitor.status.trials_dropped == 0

    runner._handle_event(
        MarkerEvent(
            value=json.dumps({"event": "experiment_start", "session_id": "demo-session"}),
            timestamp=10.1,
        )
    )
    assert monitor.status.phase == "running"

    runner.eeg_inlet.last_lsl_timestamp = 12.0
    runner._handle_event(replace(stimulus, timestamp=10.2))
    assert monitor.status.trials_processed == 1
    assert len(list((tmp_path / "low_level").glob("*.png"))) == 1

    runner._handle_event(
        MarkerEvent(
            value=json.dumps({"event": "experiment_end", "session_id": "demo-session"}),
            timestamp=12.1,
        )
    )
    assert monitor.status.phase == "finished"
    assert not runner.is_armed


def test_post_event_timeout_drops_only_that_trial(tmp_path: Path) -> None:
    monitor = make_monitor()
    config = TriggeredRunnerConfig(
        eeg_stream_name="StarstimEEG",
        marker_stream_name="TaskMarkers",
        pre_event_ms=200.0,
        post_event_ms=1000.0,
        eeg_sampling_rate=500.0,
        eeg_channels=32,
        expected_epoch_samples=None,
        min_buffer_seconds=1.0,
        post_event_timeout_s=0.01,
        require_arm=True,
        start_on_first_trigger=True,
        session_id="demo-session",
        participant_code="unseen-001",
    )
    runner = make_runner(tmp_path, monitor, config=config)
    make_ready(runner)
    assert runner.arm()[0]
    runner.eeg_inlet.last_lsl_timestamp = 5.0

    runner._handle_event(MarkerEvent(value="stim_onset", timestamp=5.0))

    assert monitor.status.trials_dropped == 1
    assert monitor.status.phase == "running"
    metadata = json.loads(next((tmp_path / "events").glob("*.json")).read_text())
    assert metadata["reason"] == "post_eeg_timeout"


def test_legacy_trigger_map_whitelists_targets(tmp_path: Path) -> None:
    target = tmp_path / "beaver" / "beaver_13n.jpg"
    target.parent.mkdir()
    target.write_bytes(b"image")
    manifest = tmp_path / "targets.tsv"
    manifest.write_text(
        "target_index\ttrigger\tcategory\texact_image_path\n"
        f"95\t1096\tbeaver\t{target}\n"
    )

    mapped = load_live_trigger_map(manifest)

    assert set(mapped) == {"1096"}
    assert mapped["1096"]["event"] == "stim_onset"
    assert mapped["1096"]["image_id"] == "beaver_13n"
    assert mapped["1096"]["image_path"] == str(target)


def test_marker_test_gate_resets_on_marker_reconnect(tmp_path: Path) -> None:
    monitor = make_monitor(marker_test_required=True)
    runner = make_runner(tmp_path, monitor)
    runner.eeg_inlet.is_connected = True
    runner.eeg_inlet.buffer_seconds = 1.2
    runner.eeg_inlet.last_sample_age_seconds = 0.01
    runner.marker_inlet.is_connected = True
    assert not runner._refresh_stream_health()
    assert not monitor.is_ready

    runner._handle_event(MarkerEvent(value="marker_test", timestamp=1.0))
    assert monitor.is_ready

    runner.marker_inlet.connection_generation += 1
    runner._refresh_stream_health()
    assert not monitor.is_ready


def test_inference_failure_is_isolated_and_next_trial_succeeds(tmp_path: Path) -> None:
    monitor = make_monitor()
    config = TriggeredRunnerConfig(
        eeg_stream_name="StarstimEEG",
        marker_stream_name="TaskMarkers",
        pre_event_ms=200.0,
        post_event_ms=1000.0,
        eeg_sampling_rate=500.0,
        eeg_channels=32,
        expected_epoch_samples=None,
        min_buffer_seconds=1.0,
        require_arm=True,
        start_on_first_trigger=True,
        session_id="demo-session",
        participant_code="unseen-001",
    )
    runner = make_runner(tmp_path, monitor, config=config)
    runner.encoder = FailOnceEncoder()
    make_ready(runner)
    assert runner.arm()[0]
    runner.eeg_inlet.last_lsl_timestamp = 50.0

    runner._handle_event(MarkerEvent(value="stim_onset", timestamp=10.0))
    runner._handle_event(MarkerEvent(value="stim_onset", timestamp=20.0))

    assert monitor.status.trials_dropped == 1
    assert monitor.status.trials_processed == 1
    reasons = [json.loads(path.read_text()).get("reason") for path in (tmp_path / "events").glob("*.json")]
    assert "inference_error" in reasons


def test_session_mismatch_pause_resume_and_duplicate_trial(tmp_path: Path) -> None:
    monitor = make_monitor()
    runner = make_runner(tmp_path, monitor)
    make_ready(runner)
    assert runner.arm()[0]
    runner.eeg_inlet.last_lsl_timestamp = 100.0

    runner._handle_event(
        MarkerEvent(
            value=json.dumps({"event": "experiment_start", "session_id": "wrong"}),
            timestamp=1.0,
        )
    )
    assert monitor.status.phase == "armed"

    runner._handle_event(
        MarkerEvent(
            value=json.dumps({"event": "experiment_start", "session_id": "demo-session"}),
            timestamp=2.0,
        )
    )
    first = json.dumps(
        {"event": "stim_onset", "session_id": "demo-session", "trial_id": "trial-1"}
    )
    runner._handle_event(MarkerEvent(value=first, timestamp=3.0))
    runner._handle_event(MarkerEvent(value=first, timestamp=4.0))
    assert monitor.status.trials_processed == 1
    assert monitor.status.trials_dropped == 1

    runner._handle_event(MarkerEvent(value="experiment_pause", timestamp=5.0))
    assert monitor.status.phase == "armed"
    runner._handle_event(MarkerEvent(value="stim_onset", timestamp=6.0))
    assert monitor.status.trials_dropped == 1
    runner._handle_event(MarkerEvent(value="experiment_resume", timestamp=7.0))
    assert monitor.status.phase == "running"
