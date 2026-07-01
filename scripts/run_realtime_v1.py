#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Maryam realtime EEG-to-image V1 pipeline.")
    parser.add_argument("--eeg-stream-name", default="MockEEG", help="EEG LSL stream name.")
    parser.add_argument("--marker-stream-name", default="TaskMarkers", help="Marker LSL stream name.")
    parser.add_argument(
        "--trigger-values",
        default="stim_onset",
        help="Comma-separated marker values that should trigger reconstruction.",
    )
    parser.add_argument("--pre-event-ms", type=float, default=0.0, help="Milliseconds of EEG before the trigger.")
    parser.add_argument("--post-event-ms", type=float, default=1000.0, help="Milliseconds of EEG after the trigger.")
    parser.add_argument("--trigger-cooldown-ms", type=float, default=0.0, help="Ignore triggers closer than this.")
    parser.add_argument("--subject-id", type=int, default=0, help="ATMS subject id.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--text-prompt", default="a photo of an object", help="Prompt for slow refinement.")
    parser.add_argument("--disable-high-level", action="store_true", help="Run only the low-level direct path.")
    parser.add_argument("--poll-interval-ms", type=float, default=5.0, help="Polling interval for marker processing.")
    parser.add_argument("--ring-buffer-seconds", type=float, default=10.0, help="EEG ring buffer duration.")
    parser.add_argument("--image-root", default=None, help="Optional image root used to resolve image_id markers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from maryam_rt.integration.high_level import HighLevelRefiner
    from maryam_rt.integration.low_level import LowLevelRealtimeEncoder, LowLevelVAEDecoder
    from maryam_rt.integration.triggered_runner import (
        TriggeredReconstructionRunner,
        TriggeredRunnerConfig,
    )
    from maryam_rt.integration.worker import SemanticRefinementWorker

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "hierarchical"
    output_low = PROJECT_ROOT / "outputs" / "low_level"
    output_high = PROJECT_ROOT / "outputs" / "high_level"
    output_meta = PROJECT_ROOT / "outputs" / "events"
    output_targets = PROJECT_ROOT / "outputs" / "targets"
    output_low.mkdir(parents=True, exist_ok=True)
    output_high.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)
    output_targets.mkdir(parents=True, exist_ok=True)

    encoder = LowLevelRealtimeEncoder(
        checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
        device=args.device,
    )
    decoder = LowLevelVAEDecoder(device=args.device)

    worker = None
    if not args.disable_high_level:
        refiner = HighLevelRefiner(
            atms_checkpoint=checkpoints_dir / "atms_sub01_40.pth",
            prior_checkpoint=checkpoints_dir / "prior_sub01_fdn.pt",
            subject_id=args.subject_id,
            text_prompt=args.text_prompt,
            device=args.device,
        )
        worker = SemanticRefinementWorker(refiner=refiner, output_dir=output_high)
        worker.start()

    try:
        runner = TriggeredReconstructionRunner(
            encoder=encoder,
            decoder=decoder,
            worker=worker,
            output_low_dir=output_low,
            output_meta_dir=output_meta,
            output_target_dir=output_targets,
            config=TriggeredRunnerConfig(
                eeg_stream_name=args.eeg_stream_name,
                marker_stream_name=args.marker_stream_name,
                pre_event_ms=args.pre_event_ms,
                post_event_ms=args.post_event_ms,
                trigger_values=tuple(v.strip() for v in args.trigger_values.split(",") if v.strip()),
                trigger_cooldown_ms=args.trigger_cooldown_ms,
                poll_interval_ms=args.poll_interval_ms,
                ring_buffer_seconds=args.ring_buffer_seconds,
                image_root=args.image_root,
            ),
        )
        return runner.run()
    finally:
        if worker is not None:
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
