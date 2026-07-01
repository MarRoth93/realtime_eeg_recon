#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay raw THINGS EEG using true stim-channel events and run reconstruction."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data"),
        help="Root containing raw_eeg/, training_images/, and test_images/.",
    )
    parser.add_argument("--subject", default="sub-01", help="Dataset subject.")
    parser.add_argument("--session", default="ses-01", help="Dataset session.")
    parser.add_argument("--split", choices=["training", "test"], default="test", help="Dataset split.")
    parser.add_argument("--subject-id", type=int, default=0, help="ATMS subject id.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--text-prompt", default="a photo of an object", help="Prompt for high-level refinement.")
    parser.add_argument("--disable-high-level", action="store_true", help="Run only low-level reconstruction.")
    parser.add_argument("--max-trials", type=int, default=10, help="Maximum number of raw trials to replay.")
    parser.add_argument("--start-trial", type=int, default=0, help="Start from this raw trial index.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between trials.")
    parser.add_argument("--epoch-tmin", type=float, default=-0.2, help="Epoch start in seconds relative to stimulus.")
    parser.add_argument("--epoch-tmax", type=float, default=1.0, help="Epoch end in seconds relative to stimulus.")
    parser.add_argument(
        "--disable-whitening",
        action="store_true",
        help="Skip session whitening. Mainly useful for debugging; normally keep whitening enabled.",
    )
    parser.add_argument(
        "--disable-offline-compare",
        action="store_true",
        help="Skip comparison against saved offline preprocessed tensors.",
    )
    parser.add_argument(
        "--calibration-block",
        action="store_true",
        help="Replay a training calibration block first and build/use a calibration whitener from it.",
    )
    parser.add_argument(
        "--calibration-split",
        choices=["training", "test"],
        default="training",
        help="Dataset split used for the calibration block.",
    )
    parser.add_argument(
        "--calibration-max-conditions",
        type=int,
        default=40,
        help="Number of image conditions to use in the calibration block.",
    )
    parser.add_argument(
        "--calibration-repetitions",
        type=int,
        default=3,
        help="Repetitions per condition to use in the calibration block.",
    )
    parser.add_argument(
        "--calibration-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between calibration trials.",
    )
    parser.add_argument(
        "--average-by-event",
        action="store_true",
        help="Average all repetitions of the same image condition before reconstruction.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "things_raw_demo",
        help="Output directory for reconstructions and target images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from maryam_rt.integration.low_level import LowLevelEpochEncoder, LowLevelVAEDecoder
    from maryam_rt.integration.things_raw_demo import ThingsRawDemoConfig, ThingsRawDemoRunner

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "hierarchical"
    encoder = LowLevelEpochEncoder(
        checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
        device=args.device,
    )
    decoder = LowLevelVAEDecoder(device=args.device)

    worker = None
    if not args.disable_high_level:
        from maryam_rt.integration.high_level import HighLevelRefiner
        from maryam_rt.integration.worker import SemanticRefinementWorker

        refiner = HighLevelRefiner(
            atms_checkpoint=checkpoints_dir / "atms_sub01_40.pth",
            prior_checkpoint=checkpoints_dir / "prior_sub01_fdn.pt",
            subject_id=args.subject_id,
            text_prompt=args.text_prompt,
            device=args.device,
        )
        worker = SemanticRefinementWorker(refiner=refiner, output_dir=args.output_root / "high_level")
        worker.start()

    try:
        runner = ThingsRawDemoRunner(
            config=ThingsRawDemoConfig(
                data_root=args.data_root,
                subject=args.subject,
                session=args.session,
                split=args.split,
                output_root=args.output_root,
                max_trials=args.max_trials,
                start_trial=args.start_trial,
                sleep_seconds=args.sleep_seconds,
                tmin=args.epoch_tmin,
                tmax=args.epoch_tmax,
                apply_whitening=not args.disable_whitening,
                average_by_event=args.average_by_event,
                compare_to_offline_saved=not args.disable_offline_compare,
                calibration_enabled=args.calibration_block,
                calibration_split=args.calibration_split,
                calibration_max_conditions=args.calibration_max_conditions,
                calibration_repetitions_per_condition=args.calibration_repetitions,
                calibration_sleep_seconds=args.calibration_sleep_seconds,
            ),
            encoder=encoder,
            decoder=decoder,
            worker=worker,
        )
        return runner.run()
    finally:
        if worker is not None:
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
