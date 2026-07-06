from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from maryam_rt.gui.monitor import RuntimeMonitorState
from maryam_rt.integration.high_level import HighLevelRefiner
from maryam_rt.integration.assessor_ratings import AssessorRatingWriter
from maryam_rt.integration.low_level import LowLevelRealtimeEncoder, LowLevelVAEDecoder
from maryam_rt.integration.marker_payload import parse_marker_payload
from maryam_rt.integration.target_resolver import ImageTargetResolver
from maryam_rt.integration.worker import SemanticRefinementWorker
from maryam_rt.realtime.preprocessing import PreprocessingPipeline
from maryam_rt.realtime.streaming.lsl_inlet import LSLInletConfig, LSLInletWrapper
from maryam_rt.realtime.streaming.marker_inlet import (
    MarkerEvent,
    MarkerInletConfig,
    MarkerInletWrapper,
)


@dataclass(frozen=True)
class TriggeredRunnerConfig:
    eeg_stream_name: str = "MockEEG"
    marker_stream_name: str = "TaskMarkers"
    pre_event_ms: float = 0.0
    post_event_ms: float = 1000.0
    trigger_values: tuple[str, ...] = ("stim_onset",)
    trigger_cooldown_ms: float = 0.0
    eeg_sampling_rate: float = 1000.0
    eeg_channels: int = 64
    ring_buffer_seconds: float = 10.0
    poll_interval_ms: float = 5.0
    connect_timeout_s: float = 10.0
    image_root: str | None = None
    expected_epoch_samples: int | None = 1000


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

        self.eeg_inlet = LSLInletWrapper(
            LSLInletConfig(
                stream_name=config.eeg_stream_name,
                stream_type=None,
                channel_count=config.eeg_channels,
                sampling_rate=config.eeg_sampling_rate,
                ring_buffer_seconds=config.ring_buffer_seconds,
            )
        )
        self.marker_inlet = MarkerInletWrapper(
            MarkerInletConfig(
                stream_name=config.marker_stream_name,
                stream_type=None,
            )
        )
        self._last_trigger_lsl: Optional[float] = None
        self._stop_event = threading.Event()

    @property
    def epoch_samples(self) -> int:
        return int(round((self.config.pre_event_ms + self.config.post_event_ms) * self.config.eeg_sampling_rate / 1000.0))

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> int:
        if self.config.expected_epoch_samples is not None and self.epoch_samples != self.config.expected_epoch_samples:
            raise ValueError(
                f"Triggered epoch resolves to {self.epoch_samples} samples, "
                f"expected {self.config.expected_epoch_samples}."
            )

        if self.monitor is not None:
            self.monitor.set_status(engine_running=True, mode="live", message="Connecting to EEG and marker streams.")
        self.eeg_inlet.start()
        self.marker_inlet.start()
        try:
            if not self.eeg_inlet.wait_for_stream(self.config.connect_timeout_s):
                raise RuntimeError(f"Failed to connect to EEG stream '{self.config.eeg_stream_name}'.")
            if not self.marker_inlet.wait_for_stream(self.config.connect_timeout_s):
                raise RuntimeError(f"Failed to connect to marker stream '{self.config.marker_stream_name}'.")

            print(f"Connected EEG stream: {self.config.eeg_stream_name}")
            print(f"Connected marker stream: {self.config.marker_stream_name}")
            if self.monitor is not None:
                self.monitor.set_status(
                    eeg_connected=True,
                    marker_connected=True,
                    message=f"Waiting for markers {self.config.trigger_values}.",
                )
            print(
                f"Waiting for markers {self.config.trigger_values} with epoch "
                f"[{-self.config.pre_event_ms:.0f}ms, +{self.config.post_event_ms:.0f}ms]."
            )

            while not self._stop_event.is_set():
                events = self.marker_inlet.pop_events()
                for event in events:
                    if self._accept_event(event):
                        self._wait_and_process_event(event)
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
                    message="Live runner stopped.",
                )

    def _accept_event(self, event: MarkerEvent) -> bool:
        payload = parse_marker_payload(event.value)
        if payload.event_name not in self.config.trigger_values and payload.raw_value not in self.config.trigger_values:
            return False
        if self._last_trigger_lsl is None:
            return True
        delta_ms = (event.timestamp - self._last_trigger_lsl) * 1000.0
        return delta_ms >= self.config.trigger_cooldown_ms

    def _wait_and_process_event(self, event: MarkerEvent) -> None:
        payload = parse_marker_payload(event.value)
        self._last_trigger_lsl = event.timestamp
        target_end = event.timestamp + (self.config.post_event_ms / 1000.0)
        while not self._stop_event.is_set():
            last_lsl = self.eeg_inlet.last_lsl_timestamp
            if last_lsl is not None and last_lsl >= target_end:
                break
            time.sleep(self.config.poll_interval_ms / 1000.0)
        if self._stop_event.is_set():
            return

        start_lsl = event.timestamp - (self.config.pre_event_ms / 1000.0)
        end_lsl = target_end
        epoch = self.eeg_inlet.get_segment_by_lsl_times(start_lsl, end_lsl)
        if self.monitor is not None:
            self.monitor.set_latest_epoch(epoch, sampling_rate=self.config.eeg_sampling_rate, start_seconds=-self.config.pre_event_ms / 1000.0)
            self.monitor.add_marker(
                {
                    "timestamp": float(event.timestamp),
                    "label": payload.event_name,
                    "raw_value": payload.raw_value,
                    "event_name": payload.event_name,
                    "status": "received",
                    "image_id": payload.image_id,
                    "image_path": payload.image_path,
                }
            )
        self._run_inference(event, payload, epoch, start_lsl, end_lsl)

    def _run_inference(
        self,
        event: MarkerEvent,
        payload: object,
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
        parsed = parse_marker_payload(event.value)
        target = self.target_resolver.resolve(parsed)
        text_prompt = self._text_prompt_for_target(target)

        metadata = {
            "marker_value": parsed.raw_value,
            "marker_label": parsed.event_name,
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
            (self.output_meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
            print(f"rejected epoch for marker '{event.value}' at {event.timestamp:.3f}")
            if self.monitor is not None:
                self.monitor.add_marker(
                    {
                        "timestamp": float(event.timestamp),
                        "label": parsed.event_name,
                        "raw_value": parsed.raw_value,
                        "event_name": parsed.event_name,
                        "status": "rejected",
                        "image_id": parsed.image_id,
                        "image_path": parsed.image_path,
                    }
                )
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
        (self.output_meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
        print(f"saved low-level image for marker '{event.value}' at {event.timestamp:.3f}")
        if self.monitor is not None:
            self.monitor.add_marker(
                {
                    "timestamp": float(event.timestamp),
                    "label": parsed.event_name,
                    "raw_value": parsed.raw_value,
                    "event_name": parsed.event_name,
                    "status": "accepted",
                    "image_id": parsed.image_id,
                    "image_path": parsed.image_path,
                }
            )
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

    def _build_stem(self, event: MarkerEvent) -> str:
        parsed = parse_marker_payload(event.value)
        safe_value = re.sub(r"[^a-zA-Z0-9_-]+", "_", parsed.event_name).strip("_") or "marker"
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_value}_{int(event.timestamp * 1000)}"

    def _text_prompt_for_target(self, target: object) -> str | None:
        if target is None:
            return None
        label = getattr(target, "label", None)
        if not label:
            return None
        return label
