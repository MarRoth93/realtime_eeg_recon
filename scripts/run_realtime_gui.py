#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Maryam realtime browser monitor.")
    parser.add_argument("--mode", choices=["live", "things-replay"], default="live", help="Source mode for the GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the local web app.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the local web app.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--disable-high-level", action="store_true", help="Run only the low-level direct path.")
    parser.add_argument("--subject-id", type=int, default=0, help="ATMS subject id.")
    parser.add_argument("--text-prompt", default="a photo of an object", help="Prompt for slow refinement.")

    parser.add_argument("--eeg-stream-name", default="MockEEG", help="EEG LSL stream name.")
    parser.add_argument("--marker-stream-name", default="TaskMarkers", help="Marker LSL stream name.")
    parser.add_argument("--trigger-values", default="stim_onset", help="Comma-separated trigger marker names.")
    parser.add_argument("--pre-event-ms", type=float, default=0.0, help="Milliseconds of EEG before the trigger.")
    parser.add_argument("--post-event-ms", type=float, default=1000.0, help="Milliseconds of EEG after the trigger.")
    parser.add_argument("--trigger-cooldown-ms", type=float, default=0.0, help="Ignore triggers closer than this.")
    parser.add_argument("--poll-interval-ms", type=float, default=5.0, help="Polling interval for marker processing.")
    parser.add_argument("--ring-buffer-seconds", type=float, default=10.0, help="EEG ring buffer duration.")
    parser.add_argument("--image-root", default=None, help="Optional image root used to resolve image_id markers.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data"),
        help="Root containing raw_eeg/, training_images/, and test_images/.",
    )
    parser.add_argument("--subject", default="sub-01", help="Dataset subject for THINGS replay mode.")
    parser.add_argument("--session", default="ses-01", help="Dataset session for THINGS replay mode.")
    parser.add_argument("--split", choices=["training", "test"], default="test", help="Dataset split for THINGS replay mode.")
    parser.add_argument("--max-trials", type=int, default=10, help="Maximum number of replay trials.")
    parser.add_argument("--start-trial", type=int, default=0, help="Start from this replay trial index.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between replay trials.")
    parser.add_argument("--epoch-tmin", type=float, default=-0.2, help="Replay epoch start in seconds relative to stimulus.")
    parser.add_argument("--epoch-tmax", type=float, default=1.0, help="Replay epoch end in seconds relative to stimulus.")
    parser.add_argument("--disable-whitening", action="store_true", help="Skip session whitening in THINGS replay mode.")
    parser.add_argument("--disable-offline-compare", action="store_true", help="Skip comparison against saved offline preprocessed tensors.")
    parser.add_argument("--average-by-event", action="store_true", help="Average repetitions by event in THINGS replay mode.")
    parser.add_argument("--calibration-block", action="store_true", help="Replay a training calibration block first and build/use a calibration whitener from it.")
    parser.add_argument("--calibration-split", choices=["training", "test"], default="training", help="Dataset split used for the calibration block.")
    parser.add_argument("--calibration-max-conditions", type=int, default=40, help="Number of image conditions to use in the calibration block.")
    parser.add_argument("--calibration-repetitions", type=int, default=3, help="Repetitions per condition to use in the calibration block.")
    parser.add_argument("--calibration-sleep-seconds", type=float, default=0.0, help="Optional delay between calibration trials.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import uvicorn

    from maryam_rt.gui.monitor import RuntimeMonitorState
    from maryam_rt.gui.server import LiveController, ReplayController, create_app
    from maryam_rt.integration.low_level import (
        LowLevelEpochEncoder,
        LowLevelRealtimeEncoder,
        LowLevelVAEDecoder,
    )
    from maryam_rt.integration.things_raw_demo import ThingsRawDemoConfig, ThingsRawDemoRunner
    from maryam_rt.integration.triggered_runner import (
        TriggeredReconstructionRunner,
        TriggeredRunnerConfig,
    )
    from maryam_rt.integration.worker import SemanticRefinementWorker

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "hierarchical"
    output_root = PROJECT_ROOT / "outputs" / ("gui_live" if args.mode == "live" else "gui_things_replay")
    output_low = output_root / "low_level"
    output_high = output_root / "high_level"
    output_meta = output_root / "events"
    output_targets = output_root / "targets"
    for directory in [output_low, output_high, output_meta, output_targets]:
        directory.mkdir(parents=True, exist_ok=True)

    monitor = RuntimeMonitorState(mode=args.mode, high_level_enabled=not args.disable_high_level)
    decoder = LowLevelVAEDecoder(device=args.device)

    worker = None
    if not args.disable_high_level:
        from maryam_rt.integration.high_level import HighLevelRefiner

        def _on_refined(path: Path) -> None:
            monitor.set_latest_high_level(path, {"path": str(path)})

        refiner = HighLevelRefiner(
            atms_checkpoint=checkpoints_dir / "atms_sub01_40.pth",
            prior_checkpoint=checkpoints_dir / "prior_sub01_fdn.pt",
            subject_id=args.subject_id,
            text_prompt=args.text_prompt,
            device=args.device,
        )
        worker = SemanticRefinementWorker(refiner=refiner, output_dir=output_high, on_result=_on_refined)
        worker.start()

    if args.mode == "live":
        encoder = LowLevelRealtimeEncoder(
            checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
            device=args.device,
        )
        runner = TriggeredReconstructionRunner(
            encoder=encoder,
            decoder=decoder,
            worker=worker,
            output_low_dir=output_low,
            output_meta_dir=output_meta,
            output_target_dir=output_targets,
            monitor=monitor,
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
        controller = LiveController(runner=runner, monitor=monitor)
    else:
        encoder = LowLevelEpochEncoder(
            checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
            device=args.device,
        )
        runner = ThingsRawDemoRunner(
            config=ThingsRawDemoConfig(
                data_root=args.data_root,
                subject=args.subject,
                session=args.session,
                split=args.split,
                output_root=output_root,
                max_trials=args.max_trials,
                start_trial=args.start_trial,
                sleep_seconds=args.sleep_seconds,
                tmin=args.epoch_tmin,
                tmax=args.epoch_tmax,
                apply_whitening=not args.disable_whitening,
                compare_to_offline_saved=not args.disable_offline_compare,
                average_by_event=args.average_by_event,
                calibration_enabled=args.calibration_block,
                calibration_split=args.calibration_split,
                calibration_max_conditions=args.calibration_max_conditions,
                calibration_repetitions_per_condition=args.calibration_repetitions,
                calibration_sleep_seconds=args.calibration_sleep_seconds,
            ),
            encoder=encoder,
            decoder=decoder,
            worker=worker,
            monitor=monitor,
        )
        controller = ReplayController(runner=runner, monitor=monitor)

    app = create_app(controller)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if worker is not None:
            worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
