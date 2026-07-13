from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from PIL import Image

from maryam_rt.gui.monitor import RuntimeMonitorState
from maryam_rt.integration.marker_payload import ParsedMarkerPayload, parse_marker_payload
from maryam_rt.integration.target_resolver import ImageTargetResolver
from maryam_rt.realtime.preprocessing import PreprocessingPipeline
from maryam_rt.realtime.streaming.lsl_inlet import LSLInletConfig, LSLInletWrapper
from maryam_rt.realtime.streaming.marker_inlet import (
    MarkerEvent,
    MarkerInletConfig,
    MarkerInletWrapper,
)

if TYPE_CHECKING:
    from maryam_rt.integration.assessor_ratings import AssessorRatingWriter
    from maryam_rt.integration.low_level import LowLevelRealtimeEncoder, LowLevelVAEDecoder
    from maryam_rt.integration.worker import SemanticRefinementWorker


@dataclass(frozen=True)
class TriggeredRunnerConfig:
    eeg_stream_name: str = "MockEEG"
    marker_stream_name: str = "TaskMarkers"
    eeg_source_id: str | None = None
    marker_source_id: str | None = None
    pre_event_ms: float = 0.0
    post_event_ms: float = 1000.0
    trigger_values: tuple[str, ...] = ("stim_onset",)
    trigger_cooldown_ms: float = 0.0
    eeg_sampling_rate: float = 1000.0
    eeg_channels: int = 64
    ring_buffer_seconds: float = 10.0
    poll_interval_ms: float = 5.0
    connect_timeout_s: float | None = None
    image_root: str | None = None
    expected_epoch_samples: int | None = 1000
    min_buffer_seconds: float = 1.0
    post_event_timeout_s: float = 3.0
    sampling_rate_tolerance_hz: float = 2.0
    max_eeg_sample_age_s: float = 1.0
    expected_channel_labels: tuple[str, ...] | None = None
    require_arm: bool = False
    start_on_first_trigger: bool = False
    experiment_start_values: tuple[str, ...] = ("experiment_start",)
    experiment_pause_values: tuple[str, ...] = ("experiment_pause",)
    experiment_resume_values: tuple[str, ...] = ("experiment_resume",)
    experiment_end_values: tuple[str, ...] = ("experiment_end",)
    marker_test_values: tuple[str, ...] = ("marker_test",)
    heartbeat_values: tuple[str, ...] = ("heartbeat",)
    marker_heartbeat_timeout_s: float | None = None
    numeric_trigger_map: dict[str, dict[str, str]] | None = None
    session_id: str | None = None
    participant_code: str | None = None
    model_metadata: dict[str, str] | None = None


class TriggeredReconstructionRunner:
    """Run low-level and optional high-level reconstruction on task triggers."""

    def __init__(
        self,
        encoder: LowLevelRealtimeEncoder,
        decoder: LowLevelVAEDecoder,
        worker: Optional[SemanticRefinementWorker],
        output_low_dir: str | Path,
        output_meta_dir: str | Path,
        config: TriggeredRunnerConfig,
        output_target_dir: str | Path | None = None,
        epoch_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
        assessor_writer: AssessorRatingWriter | None = None,
        monitor: RuntimeMonitorState | None = None,
        eeg_inlet: LSLInletWrapper | None = None,
        marker_inlet: MarkerInletWrapper | None = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.worker = worker
        self.config = config
        self.output_low_dir = Path(output_low_dir)
        self.output_meta_dir = Path(output_meta_dir)
        self.output_target_dir = None if output_target_dir is None else Path(output_target_dir)
        self.epoch_preprocessor = epoch_preprocessor
        self.assessor_writer = assessor_writer
        self.monitor = monitor
        self.output_low_dir.mkdir(parents=True, exist_ok=True)
        self.output_meta_dir.mkdir(parents=True, exist_ok=True)
        if self.output_target_dir is not None:
            self.output_target_dir.mkdir(parents=True, exist_ok=True)
        self.target_resolver = ImageTargetResolver(config.image_root)

        self.eeg_inlet = eeg_inlet or LSLInletWrapper(
            LSLInletConfig(
                stream_name=config.eeg_stream_name,
                stream_type=None,
                source_id=config.eeg_source_id,
                channel_count=config.eeg_channels,
                sampling_rate=config.eeg_sampling_rate,
                ring_buffer_seconds=config.ring_buffer_seconds,
            )
        )
        self.marker_inlet = marker_inlet or MarkerInletWrapper(
            MarkerInletConfig(
                stream_name=config.marker_stream_name,
                stream_type=None,
                source_id=config.marker_source_id,
            )
        )
        self._last_trigger_lsl: Optional[float] = None
        self._stop_event = threading.Event()
        self._armed_event = threading.Event()
        self._experiment_running_event = threading.Event()
        self._equipment_was_ready = False
        self._marker_generation = self.marker_inlet.connection_generation
        self._eeg_generation = self.eeg_inlet.connection_generation
        self._seen_trial_ids: set[str] = set()
        self._experiment_started_at: float | None = None
        self._experiment_finished_at: float | None = None

    @property
    def epoch_samples(self) -> int:
        return int(round((self.config.pre_event_ms + self.config.post_event_ms) * self.config.eeg_sampling_rate / 1000.0))

    def stop(self) -> None:
        self._stop_event.set()

    def set_participant_code(self, participant_code: str) -> None:
        self.config = replace(self.config, participant_code=participant_code)
        self._write_session_summary("participant_configured")

    def arm(self) -> tuple[bool, str]:
        if self.monitor is None:
            self._armed_event.set()
            return True, "Armed."
        armed, message = self.monitor.arm()
        if armed:
            self._armed_event.set()
            self._experiment_running_event.clear()
            self._write_session_summary("armed")
        return armed, message

    def disarm(self, message: str = "Session disarmed.") -> None:
        self._armed_event.clear()
        self._experiment_running_event.clear()
        if self.monitor is not None:
            self.monitor.disarm(message)
        self._write_session_summary("disarmed")

    def finish_session(self, message: str = "Experiment finished.") -> None:
        self._armed_event.clear()
        self._experiment_running_event.clear()
        if self.monitor is not None:
            self.monitor.finish(message)
        self._experiment_finished_at = time.time()
        self._write_session_summary("finished")

    @property
    def is_armed(self) -> bool:
        return self._armed_event.is_set()

    @property
    def is_experiment_running(self) -> bool:
        return self._experiment_running_event.is_set()

    def run(self) -> int:
        if self.config.expected_epoch_samples is not None and self.epoch_samples != self.config.expected_epoch_samples:
            raise ValueError(
                f"Triggered epoch resolves to {self.epoch_samples} samples, "
                f"expected {self.config.expected_epoch_samples}."
            )

        self._stop_event.clear()
        self._armed_event.clear()
        self._experiment_running_event.clear()
        self._last_trigger_lsl = None
        self._equipment_was_ready = False
        self._seen_trial_ids.clear()
        self._experiment_started_at = None
        self._experiment_finished_at = None
        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=True,
                phase="waiting_for_equipment",
                message="Waiting for EEG and marker streams.",
            )
        self.eeg_inlet.start()
        self.marker_inlet.start()
        self._write_session_summary("waiting_for_equipment")
        try:
            while not self._stop_event.is_set():
                self._refresh_stream_health()
                events = self.marker_inlet.pop_events()
                for event in events:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._handle_event(event)
                    except Exception as exc:
                        self._drop_event(event, "event_error", str(exc))
                time.sleep(self.config.poll_interval_ms / 1000.0)
            return 0
        finally:
            self.marker_inlet.stop()
            self.eeg_inlet.stop()
            if self.monitor is not None:
                self.monitor.set_status(
                    engine_running=False,
                    eeg_connected=False,
                    marker_connected=False,
                    phase="stopped",
                    message="Live runner stopped.",
                )
            self._write_session_summary("stopped")

    def _refresh_stream_health(self) -> bool:
        eeg_connected = bool(self.eeg_inlet.is_connected)
        marker_connected = bool(self.marker_inlet.is_connected)
        if marker_connected and self.config.marker_heartbeat_timeout_s is not None:
            marker_age = self.marker_inlet.last_event_age_seconds
            marker_connected = bool(
                marker_age is not None and marker_age <= self.config.marker_heartbeat_timeout_s
            )
        eeg_generation = self.eeg_inlet.connection_generation
        if eeg_generation != self._eeg_generation:
            self._eeg_generation = eeg_generation
            if self.monitor is not None and self.monitor.status.channel_order_confirmation_required:
                self.monitor.confirm_channel_order(False)
        marker_generation = self.marker_inlet.connection_generation
        if marker_generation != self._marker_generation:
            self._marker_generation = marker_generation
            if self.monitor is not None:
                self.monitor.reset_marker_test()
        actual_channels = self.eeg_inlet.channel_count if eeg_connected else None
        actual_rate = self.eeg_inlet.nominal_sampling_rate if eeg_connected else None
        if actual_rate is None and eeg_connected:
            actual_rate = self.eeg_inlet.sampling_rate
        format_valid = bool(
            eeg_connected
            and actual_channels == self.config.eeg_channels
            and actual_rate is not None
            and abs(float(actual_rate) - self.config.eeg_sampling_rate)
            <= self.config.sampling_rate_tolerance_hz
        )
        buffer_seconds = self.eeg_inlet.buffer_seconds if eeg_connected else 0.0
        sample_age = self.eeg_inlet.last_sample_age_seconds if eeg_connected else None
        data_fresh = bool(sample_age is not None and sample_age <= self.config.max_eeg_sample_age_s)
        buffer_ready = bool(format_valid and data_fresh and buffer_seconds >= self.config.min_buffer_seconds)

        if self.monitor is not None:
            if (
                self.config.expected_channel_labels
                and self.eeg_inlet.channel_labels == self.config.expected_channel_labels
            ):
                self.monitor.confirm_channel_order(True)
            self.monitor.set_stream_health(
                eeg_connected=eeg_connected,
                marker_connected=marker_connected,
                eeg_stream_name=self.eeg_inlet.stream_name or self.config.eeg_stream_name,
                marker_stream_name=self.marker_inlet.stream_name or self.config.marker_stream_name,
                eeg_channel_count=actual_channels,
                eeg_sampling_rate=actual_rate,
                eeg_format_valid=format_valid,
                eeg_data_fresh=data_fresh,
                buffer_seconds=buffer_seconds,
                buffer_ready=buffer_ready,
                marker_events_dropped=getattr(self.marker_inlet, "dropped_event_count", 0),
                eeg_timestamp_gaps=getattr(self.eeg_inlet, "timestamp_gap_count", 0),
            )
            self._update_signal_quality(eeg_connected and data_fresh)
            ready = self.monitor.set_waiting_or_ready()
        else:
            ready = eeg_connected and marker_connected and format_valid and buffer_ready

        if self._equipment_was_ready and not ready and self.is_armed:
            self.disarm("Equipment connection or EEG readiness was lost. Check the setup and arm again.")
        self._equipment_was_ready = ready
        return ready

    def _update_signal_quality(self, can_measure: bool) -> None:
        if self.monitor is None:
            return
        if not can_measure or self.eeg_inlet.available_samples < 32:
            self.monitor.set_signal_quality(
                status="unknown",
                warnings=[],
                median_peak_to_peak_uv=None,
            )
            return
        rate = self.eeg_inlet.sampling_rate or self.config.eeg_sampling_rate
        samples = min(self.eeg_inlet.available_samples, max(int(round(rate * 0.5)), 32))
        try:
            window = self.eeg_inlet.get_window(samples, timeout=0.0)
        except Exception:
            return
        finite_fraction = float(np.isfinite(window).mean())
        clean = np.nan_to_num(window, copy=True)
        peak_to_peak = np.ptp(clean, axis=1)
        median_peak_to_peak = float(np.median(peak_to_peak))
        warnings: list[str] = []
        if finite_fraction < 0.99:
            warnings.append("EEG contains non-finite samples")
        if median_peak_to_peak < 0.1:
            warnings.append("EEG appears flat; check the electrodes and NIC2 stream")
        if median_peak_to_peak > 1000.0:
            warnings.append("EEG variation is unusually large; check contact and units")
        self.monitor.set_signal_quality(
            status="warning" if warnings else "good",
            warnings=warnings,
            median_peak_to_peak_uv=median_peak_to_peak,
        )

    @staticmethod
    def _matches(payload: ParsedMarkerPayload, values: tuple[str, ...]) -> bool:
        return payload.event_name in values or payload.raw_value in values

    def _mapped_payload(self, event: MarkerEvent) -> ParsedMarkerPayload:
        payload = parse_marker_payload(event.value)
        trigger_map = self.config.numeric_trigger_map or {}
        mapped = trigger_map.get(payload.raw_value)
        if mapped is None:
            return payload
        return ParsedMarkerPayload(
            raw_value=payload.raw_value,
            event_name=mapped.get("event", self.config.trigger_values[0]),
            image_id=mapped.get("image_id") or None,
            image_path=mapped.get("image_path") or None,
            metadata={**payload.metadata, **mapped, "numeric_trigger": payload.raw_value},
        )

    def _marker_record(
        self,
        event: MarkerEvent,
        payload: ParsedMarkerPayload,
        status: str,
        *,
        reason: str | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "timestamp": float(event.timestamp),
            "label": payload.event_name,
            "raw_value": payload.raw_value,
            "event_name": payload.event_name,
            "status": status,
            "image_id": payload.image_id,
            "image_path": payload.image_path,
        }
        if reason is not None:
            record["reason"] = reason
        return record

    def _handle_event(self, event: MarkerEvent) -> None:
        payload = self._mapped_payload(event)

        marker_session_id = payload.metadata.get("session_id")
        if (
            marker_session_id is not None
            and self.config.session_id is not None
            and str(marker_session_id) != self.config.session_id
        ):
            if self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(event, payload, "ignored", reason="session_id_mismatch")
                )
            return

        if self._matches(payload, self.config.heartbeat_values):
            if self.monitor is not None:
                self.monitor.add_marker(self._marker_record(event, payload, "heartbeat"))
            return

        if self._matches(payload, self.config.marker_test_values):
            if self.monitor is not None:
                self.monitor.mark_marker_tested()
                self.monitor.add_marker(self._marker_record(event, payload, "test_received"))
                self.monitor.set_waiting_or_ready()
            return

        if self._matches(payload, self.config.experiment_start_values):
            if self.is_armed:
                self._experiment_running_event.set()
                if self._experiment_started_at is None:
                    self._experiment_started_at = time.time()
                if self.monitor is not None:
                    self.monitor.start_experiment("Experiment running. Waiting for stimulus markers.")
                    self.monitor.add_marker(self._marker_record(event, payload, "experiment_started"))
                self._write_session_summary("running")
            elif self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(event, payload, "ignored", reason="session_not_armed")
                )
            return

        if self._matches(payload, self.config.experiment_end_values):
            if self.is_armed or self.is_experiment_running:
                if self.monitor is not None:
                    self.monitor.add_marker(self._marker_record(event, payload, "experiment_finished"))
                self.finish_session()
            elif self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(event, payload, "ignored", reason="session_not_running")
                )
            return

        if self._matches(payload, self.config.experiment_pause_values):
            if self.is_experiment_running:
                self._experiment_running_event.clear()
                if self.monitor is not None:
                    self.monitor.pause_experiment()
                    self.monitor.add_marker(self._marker_record(event, payload, "experiment_paused"))
            elif self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(event, payload, "ignored", reason="session_not_running")
                )
            return

        if self._matches(payload, self.config.experiment_resume_values):
            if self.is_armed and not self.is_experiment_running:
                self._experiment_running_event.set()
                if self.monitor is not None:
                    self.monitor.start_experiment("Experiment resumed.")
                    self.monitor.add_marker(self._marker_record(event, payload, "experiment_resumed"))
            elif self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(event, payload, "ignored", reason="session_not_paused")
                )
            return

        is_reconstruction_trigger = (
            payload.event_name in self.config.trigger_values
            or payload.raw_value in self.config.trigger_values
        )
        if not self._accept_event(event, payload):
            if self.monitor is not None:
                self.monitor.add_marker(
                    self._marker_record(
                        event,
                        payload,
                        "ignored",
                        reason="trigger_cooldown" if is_reconstruction_trigger else "unknown_marker",
                    )
                )
            return

        if self.monitor is not None:
            self.monitor.record_trial_received()
            self.monitor.add_marker(self._marker_record(event, payload, "received"))

        if self.monitor is not None and self.monitor.status.phase == "finished":
            self._ignore_event(event, payload, "session_finished")
            return
        if self.config.require_arm and not self.is_armed:
            self._ignore_event(event, payload, "session_not_armed")
            return
        if not self.config.require_arm and not self.is_armed:
            self._armed_event.set()
            self._experiment_running_event.set()
            if self._experiment_started_at is None:
                self._experiment_started_at = time.time()
        if not self.is_experiment_running:
            if self.config.start_on_first_trigger and self.is_armed:
                self._experiment_running_event.set()
                if self._experiment_started_at is None:
                    self._experiment_started_at = time.time()
                if self.monitor is not None:
                    self.monitor.start_experiment("Experiment running from first stimulus trigger.")
            else:
                self._ignore_event(event, payload, "experiment_not_started")
                return

        trial_id = payload.metadata.get("trial_id")
        if trial_id is not None:
            trial_key = str(trial_id)
            if trial_key in self._seen_trial_ids:
                self._drop_event(event, "duplicate_trial_id", payload=payload)
                return
            self._seen_trial_ids.add(trial_key)

        self._wait_and_process_event(event, payload)

    def _ignore_event(
        self,
        event: MarkerEvent,
        payload: ParsedMarkerPayload,
        reason: str,
    ) -> None:
        if self.monitor is not None:
            self.monitor.add_marker(self._marker_record(event, payload, "ignored", reason=reason))

    def _accept_event(self, event: MarkerEvent, payload: ParsedMarkerPayload | None = None) -> bool:
        payload = payload or self._mapped_payload(event)
        if payload.event_name not in self.config.trigger_values and payload.raw_value not in self.config.trigger_values:
            return False
        if self._last_trigger_lsl is None:
            return True
        delta_ms = (event.timestamp - self._last_trigger_lsl) * 1000.0
        return delta_ms >= self.config.trigger_cooldown_ms

    def _wait_and_process_event(
        self,
        event: MarkerEvent,
        payload: ParsedMarkerPayload | None = None,
    ) -> None:
        payload = payload or self._mapped_payload(event)
        self._last_trigger_lsl = event.timestamp
        eeg_generation = self.eeg_inlet.connection_generation
        target_end = event.timestamp + (self.config.post_event_ms / 1000.0)
        deadline = time.monotonic() + self.config.post_event_timeout_s
        while not self._stop_event.is_set():
            if not self.eeg_inlet.is_connected:
                self._drop_event(event, "eeg_disconnected", payload=payload)
                return
            if self.eeg_inlet.connection_generation != eeg_generation:
                self._drop_event(event, "eeg_stream_reset", payload=payload)
                return
            last_lsl = self.eeg_inlet.last_lsl_timestamp
            if last_lsl is not None and last_lsl >= target_end:
                break
            if time.monotonic() >= deadline:
                self._drop_event(event, "post_eeg_timeout", payload=payload)
                return
            time.sleep(self.config.poll_interval_ms / 1000.0)
        if self._stop_event.is_set():
            return

        start_lsl = event.timestamp - (self.config.pre_event_ms / 1000.0)
        end_lsl = target_end
        try:
            epoch = self.eeg_inlet.get_segment_by_lsl_times(start_lsl, end_lsl)
        except Exception as exc:
            self._drop_event(event, "epoch_unavailable", str(exc), payload=payload)
            return
        if self.monitor is not None:
            self.monitor.set_latest_epoch(epoch, sampling_rate=self.config.eeg_sampling_rate, start_seconds=-self.config.pre_event_ms / 1000.0)
        try:
            self._run_inference(event, payload, epoch, start_lsl, end_lsl)
        except Exception as exc:
            self._drop_event(event, "inference_error", str(exc), payload=payload)

    def _drop_event(
        self,
        event: MarkerEvent,
        reason: str,
        detail: str | None = None,
        *,
        payload: ParsedMarkerPayload | None = None,
    ) -> None:
        payload = payload or self._mapped_payload(event)
        if self.monitor is not None:
            self.monitor.record_trial_dropped()
            self.monitor.add_marker(self._marker_record(event, payload, "dropped", reason=reason))
        stem = self._build_stem(event)
        metadata = {
            "session_id": self.config.session_id,
            "participant_code": self.config.participant_code,
            "models": self.config.model_metadata or {},
            "marker_value": payload.raw_value,
            "marker_label": payload.event_name,
            "marker_lsl_timestamp": float(event.timestamp),
            "status": "dropped",
            "reason": reason,
            "detail": detail,
            "marker_metadata": payload.metadata,
        }
        (self.output_meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
        print(f"dropped marker '{event.value}' at {event.timestamp:.3f}: {reason}")
        self._write_session_summary("running" if self.is_experiment_running else "armed")

    def _write_session_summary(self, lifecycle_status: str) -> None:
        if self.monitor is None:
            return
        snapshot = self.monitor.snapshot()
        summary_path = self.output_meta_dir.parent / "session.json"
        summary: dict[str, object] = {}
        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text())
                if isinstance(existing, dict):
                    summary.update(existing)
            except (OSError, json.JSONDecodeError):
                pass
        summary.update({
            "session_id": self.config.session_id,
            "participant_code": self.config.participant_code,
            "models": self.config.model_metadata or {},
            "lifecycle_status": lifecycle_status,
            "updated_at": time.time(),
            "experiment_started_at": self._experiment_started_at,
            "experiment_finished_at": self._experiment_finished_at,
            "mode": snapshot["status"]["mode"],
            "participant_mode": snapshot["status"]["participant_mode"],
            "high_level_enabled": snapshot["status"]["high_level_enabled"],
            "high_level_reason": snapshot["status"]["high_level_reason"],
            "eeg_stream_name": snapshot["status"]["eeg_stream_name"],
            "eeg_source_id": self.config.eeg_source_id,
            "marker_stream_name": snapshot["status"]["marker_stream_name"],
            "marker_source_id": self.config.marker_source_id,
            "actual_eeg_channel_count": snapshot["status"]["eeg_channel_count"],
            "actual_eeg_sampling_rate": snapshot["status"]["eeg_sampling_rate"],
            "channel_labels": list(self.eeg_inlet.channel_labels),
            "channel_order_confirmed": snapshot["status"]["channel_order_confirmed"],
            "expected_eeg_channels": self.config.eeg_channels,
            "expected_eeg_sampling_rate": self.config.eeg_sampling_rate,
            "pre_event_ms": self.config.pre_event_ms,
            "post_event_ms": self.config.post_event_ms,
            "trials_received": snapshot["status"]["trials_received"],
            "trials_processed": snapshot["status"]["trials_processed"],
            "trials_dropped": snapshot["status"]["trials_dropped"],
        })
        summary_path.write_text(json.dumps(summary, indent=2))

    def _run_inference(
        self,
        event: MarkerEvent,
        payload: ParsedMarkerPayload,
        epoch: np.ndarray,
        start_lsl: float,
        end_lsl: float,
    ) -> None:
        if self.epoch_preprocessor is None:
            preprocessor = PreprocessingPipeline(
                n_channels=self.config.eeg_channels,
                sampling_rate=self.config.eeg_sampling_rate,
            )
            processed, is_valid = preprocessor.process(epoch)
        else:
            processed = self.epoch_preprocessor(epoch)
            is_valid = True
        stem = self._build_stem(event)
        parsed = payload
        target = self.target_resolver.resolve(parsed)
        text_prompt = self._text_prompt_for_target(target)

        metadata = {
            "session_id": self.config.session_id,
            "participant_code": self.config.participant_code,
            "models": self.config.model_metadata or {},
            "marker_value": parsed.raw_value,
            "marker_label": parsed.event_name,
            "marker_metadata": parsed.metadata,
            "trial_id": parsed.metadata.get("trial_id"),
            "block": parsed.metadata.get("block"),
            "marker_lsl_timestamp": event.timestamp,
            "epoch_start_lsl": start_lsl,
            "epoch_end_lsl": end_lsl,
            "epoch_samples": int(epoch.shape[1]),
            "processed_shape": list(processed.shape),
            "artifact_valid": bool(is_valid),
            "image_id": parsed.image_id,
            "image_path": None if target is None else str(target.image_path),
        }

        if not is_valid:
            metadata["status"] = "dropped"
            metadata["reason"] = "artifact_rejected"
            (self.output_meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
            print(f"rejected epoch for marker '{event.value}' at {event.timestamp:.3f}")
            if self.monitor is not None:
                self.monitor.record_trial_dropped()
                self.monitor.add_marker(self._marker_record(event, parsed, "dropped", reason="artifact_rejected"))
            self._write_session_summary("running" if self.is_experiment_running else "armed")
            return

        input_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.encoder.device_name)
        with torch.inference_mode():
            latent = self.encoder(input_tensor)
            image = self.decoder(latent)

        image = image.astype(np.uint8)
        low_path = self.output_low_dir / f"{stem}_low.png"
        Image.fromarray(image).save(low_path)
        target_copy: Path | None = None
        if target is not None and self.output_target_dir is not None:
            target_copy = self.output_target_dir / f"{stem}_target{target.image_path.suffix}"
            Image.open(target.image_path).save(target_copy)
            metadata["target_copy_path"] = str(target_copy)
        target_assessor_path = target_copy if target_copy is not None else (None if target is None else target.image_path)
        if self.assessor_writer is not None:
            assessor_records = {
                "low_level": self.assessor_writer.assess_path(low_path, stem, "low_level")
            }
            if target_assessor_path is not None:
                assessor_records["target"] = self.assessor_writer.assess_path(
                    target_assessor_path,
                    stem,
                    "target",
                )
            metadata["assessor"] = assessor_records
        metadata["status"] = "accepted"
        (self.output_meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
        print(f"saved low-level image for marker '{event.value}' at {event.timestamp:.3f}")
        if self.monitor is not None:
            self.monitor.record_trial_processed()
            self.monitor.add_marker(self._marker_record(event, parsed, "accepted"))
            self.monitor.set_latest_low_level(
                low_path,
                {
                    "marker_label": parsed.event_name,
                    "marker_lsl_timestamp": float(event.timestamp),
                    "event_time": time.strftime("%H:%M:%S"),
                    "stem": stem,
                },
            )
            if target is not None:
                self.monitor.set_latest_target(
                    target.image_path,
                    {
                        "image_id": target.image_id,
                        "label": target.label,
                        "marker_label": parsed.event_name,
                    },
                )

        x250 = self.encoder.snapshot_last_x250()
        if self.worker is not None and x250 is not None:
            self.worker.submit(
                stem=stem,
                x250=x250,
                low_level_image=image,
                text_prompt=text_prompt,
                low_level_path=low_path,
                target_image_path=target_assessor_path,
            )
        self._write_session_summary("running" if self.is_experiment_running else "armed")

    def _build_stem(self, event: MarkerEvent) -> str:
        parsed = self._mapped_payload(event)
        safe_value = re.sub(r"[^a-zA-Z0-9_-]+", "_", parsed.event_name).strip("_") or "marker"
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_value}_{int(event.timestamp * 1000)}"

    def _text_prompt_for_target(self, target: object) -> str | None:
        if target is None:
            return None
        label = getattr(target, "label", None)
        if not label:
            return None
        return label
