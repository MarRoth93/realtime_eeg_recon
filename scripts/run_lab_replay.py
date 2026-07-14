#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay manifest-aligned lab Starstim31 epochs through the realtime reconstruction stack."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root containing raw subject folders and derived/finetune manifests.",
    )
    parser.add_argument(
        "--replay-source",
        choices=["raw", "derived"],
        default="raw",
        help="raw re-extracts .easy epochs with the lab-live preprocessor; derived uses cached 31x250 epochs.",
    )
    parser.add_argument("--epoch-npz", type=Path, default=None, help="Optional lab epoch npz override.")
    parser.add_argument("--target-manifest", type=Path, default=None, help="Optional lab target manifest override.")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Optional lab split manifest override.")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all", help="Lab split to replay.")
    parser.add_argument("--subject", default=None, help="Optional subject folder to replay, e.g. P06 or zar.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--subject-id", type=int, default=0, help="ATMS subject id for optional high-level refinement.")
    parser.add_argument("--text-prompt", default="a photo of an object", help="Fallback high-level prompt.")
    parser.add_argument("--disable-high-level", action="store_true", help="Run only low-level reconstruction.")
    parser.add_argument("--max-trials", type=int, default=10, help="Maximum number of lab trials to replay.")
    parser.add_argument("--start-trial", type=int, default=0, help="Start from this filtered lab trial index.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between trials.")
    parser.add_argument("--raw-pre-event-ms", type=float, default=200.0, help="Raw replay milliseconds before trigger.")
    parser.add_argument("--raw-post-event-ms", type=float, default=1000.0, help="Raw replay milliseconds after trigger.")
    parser.add_argument(
        "--raw-input-sfreq",
        type=float,
        default=None,
        help="Override raw .easy sampling rate. Defaults to .info, then 500 Hz.",
    )
    parser.add_argument(
        "--lab-whitening",
        choices=["none", "mvnn"],
        default="none",
        help="Optional lab replay whitening applied to the model input. Use mvnn for train-split MVNN.",
    )
    parser.add_argument(
        "--lab-whitening-split",
        choices=["all", "train", "val"],
        default="train",
        help="Lab split used to compute the MVNN whitener.",
    )
    parser.add_argument(
        "--lab-whitening-cache",
        type=Path,
        default=None,
        help="Optional .npy path for loading/saving the lab MVNN whitening matrix.",
    )
    parser.add_argument(
        "--lab-whitening-subject",
        default=None,
        help="Optional subject folder used to compute the whitener. Defaults to --subject.",
    )
    parser.add_argument(
        "--lab-whitening-method",
        choices=["custom", "shrinkage"],
        default="custom",
        help="MVNN estimator. 'custom' pools trials+time like the offline apply_mvnn; "
        "'shrinkage' averages per-trial shrinkage covariances.",
    )
    parser.add_argument(
        "--lab-whitening-matrix",
        type=Path,
        default=None,
        help="Optional precomputed 31x31 whitening matrix (.npy). Overrides on-the-fly fitting and cache.",
    )
    parser.add_argument(
        "--lab-whitening-first-block",
        type=int,
        default=None,
        help="Use the first N trials (per subject, manifest order) as the calibration block "
        "to fit the whitener; the remaining trials reuse the fitted matrix.",
    )
    parser.add_argument(
        "--lab-bandpass-l-freq",
        type=float,
        default=None,
        help="Highpass cutoff (Hz) for the epoch bandpass. Use 0.01 to match the offline filter.",
    )
    parser.add_argument(
        "--lab-bandpass-h-freq",
        type=float,
        default=None,
        help="Lowpass cutoff (Hz) for the epoch bandpass. Use 30 to match the offline filter.",
    )
    parser.add_argument(
        "--lab-filter-order",
        type=int,
        default=4,
        help="Butterworth order for the epoch bandpass.",
    )
    parser.add_argument(
        "--lab-ica-operator",
        type=Path,
        default=None,
        help="Optional 31x31 calibrated ICA cleaning matrix (.npy) from lab_calibration. "
        "Applied to raw-source epochs only.",
    )
    parser.add_argument(
        "--lab-reject-uv",
        type=float,
        default=None,
        help="Peak-to-peak artifact threshold (µV) flagged per trial. Use 250 to match the offline reject.",
    )
    parser.add_argument(
        "--adapter",
        choices=["starstim31-to-things63", "none"],
        default="none",
        help="Use none for a native lab checkpoint, or starstim31-to-things63 for explicit THINGS checkpoint compatibility.",
    )
    parser.add_argument(
        "--lab-low-level-checkpoint",
        type=Path,
        default=None,
        help="Native lab low-level checkpoint. Required when --adapter none.",
    )
    parser.add_argument(
        "--lab-model-channels",
        type=int,
        choices=[31, 32],
        default=None,
        help="Native lab checkpoint channel count. Defaults to 31 for current lab replay epochs.",
    )
    parser.add_argument(
        "--lab-low-level-arch",
        choices=["plain", "transformed"],
        default="plain",
        help="Low-level checkpoint architecture. Use transformed for custom encoder_low_level_transformed checkpoints.",
    )
    parser.add_argument(
        "--lab-low-level-subject-id",
        type=int,
        default=None,
        help="Subject id for transformed low-level checkpoints. Defaults to each trial's manifest lab_subject_id.",
    )
    parser.add_argument(
        "--low-level-latent-scaling",
        choices=["auto", "direct", "sdxl"],
        default="auto",
        help="How to decode low-level latents. auto uses SDXL scaling for transformed lab checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lab_replay",
        help="Output directory for lab replay reconstructions and metadata.",
    )
    parser.add_argument(
        "--enable-lab-atms-embedding",
        action="store_true",
        help="Export Starstim31 ATMS EEG embeddings for each replayed lab trial.",
    )
    parser.add_argument(
        "--lab-atms-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hierarchical" / "atms_starstim31_10sub_best.pth",
        help="Starstim31 ATMS checkpoint used when --enable-lab-atms-embedding is set.",
    )
    parser.add_argument(
        "--lab-prior-checkpoint",
        type=Path,
        default=None,
        help="Native lab diffusion prior checkpoint for high-level refinement.",
    )
    parser.add_argument(
        "--lab-atms-subject-id",
        type=int,
        default=None,
        help="Override lab_subject_id from the split manifest for ATMS subject conditioning.",
    )
    parser.add_argument(
        "--enable-assessor-ratings",
        action="store_true",
        help="Rate reconstructed low/high images with the local VA and six-dimension assessors.",
    )
    parser.add_argument(
        "--assessor-va-bundle",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "assessor" / "assessor_va_mixed_v4_bundle.pt",
        help="Local valence/arousal assessor bundle.",
    )
    parser.add_argument(
        "--assessor-six-bundle",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "assessor" / "assessor_six_clip_v3_bundle.pt",
        help="Local six-dimension assessor bundle.",
    )
    parser.add_argument("--assessor-device", default=None, help="Torch device for assessor CLIP inference.")
    parser.add_argument("--assessor-interval-level", type=float, default=0.9, help="Conformal interval level.")
    parser.add_argument("--disable-assessor-ood", action="store_true", help="Skip assessor OOD percentiles.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.adapter == "none":
        if args.lab_low_level_checkpoint is None:
            raise SystemExit(
                "Lab replay is native Starstim by default, but no native lab low-level checkpoint was given. "
                "Raw and derived lab replay both produce 31-channel model epochs because preprocessing drops Fz. "
                "Pass --lab-low-level-checkpoint /path/to/checkpoint.pth, or explicitly pass "
                "--adapter starstim31-to-things63 to run the old THINGS63 compatibility path."
            )
        if not args.disable_high_level:
            if args.lab_prior_checkpoint is None:
                raise SystemExit(
                    "Native lab high-level replay needs a lab diffusion prior. Pass "
                    "--lab-prior-checkpoint /path/to/lab_prior_best_fdn.pt, or pass --disable-high-level."
                )
        if args.lab_low_level_arch == "transformed" and args.lab_model_channels == 32:
            raise SystemExit("Transformed lab low-level checkpoints are currently supported only for 31-channel epochs.")
    if args.lab_whitening != "none" and args.adapter != "none":
        raise SystemExit("--lab-whitening mvnn is only supported with --adapter none.")

    from maryam_rt.integration.lab_replay import LabReplayConfig, LabReplayRunner
    from maryam_rt.integration.low_level import (
        LowLevelEpochEncoder,
        LowLevelTransformedEpochEncoder,
        LowLevelVAEDecoder,
    )

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "hierarchical"
    low_level_checkpoint = args.lab_low_level_checkpoint
    model_num_channels = args.lab_model_channels or 31
    if args.adapter == "starstim31-to-things63":
        low_level_checkpoint = checkpoints_dir / "low_level_encoder_sub01_60.pth"
        model_num_channels = 63
    if args.lab_low_level_arch == "transformed":
        encoder = LowLevelTransformedEpochEncoder(
            checkpoint_path=low_level_checkpoint,
            device=args.device,
            model_num_channels=model_num_channels,
            subject_id=args.lab_low_level_subject_id,
        )
    else:
        encoder = LowLevelEpochEncoder(
            checkpoint_path=low_level_checkpoint,
            device=args.device,
            model_num_channels=model_num_channels,
        )
    scaled_low_level_latents = args.low_level_latent_scaling == "sdxl" or (
        args.low_level_latent_scaling == "auto"
        and args.adapter == "none"
    )
    decoder = LowLevelVAEDecoder(device=args.device, scaled_latents=scaled_low_level_latents)

    worker = None
    atms_embedder = None
    assessor_writer = None
    if args.enable_assessor_ratings:
        from maryam_rt.integration.assessor_ratings import AssessorRatingEngine, AssessorRatingWriter

        assessor_engine = AssessorRatingEngine(
            va_bundle=args.assessor_va_bundle,
            six_bundle=args.assessor_six_bundle,
            device=args.assessor_device or args.device,
            interval_level=args.assessor_interval_level,
            include_ood=not args.disable_assessor_ood,
        )
        assessor_writer = AssessorRatingWriter(assessor_engine, args.output_root / "assessments")
    if args.enable_lab_atms_embedding:
        from maryam_rt.integration.starstim31_atms import Starstim31ATMSEmbedder

        atms_embedder = Starstim31ATMSEmbedder(
            checkpoint_path=args.lab_atms_checkpoint,
            device=args.device,
        )
    if not args.disable_high_level:
        from maryam_rt.integration.high_level import HighLevelRefiner
        from maryam_rt.integration.worker import SemanticRefinementWorker

        refiner = HighLevelRefiner(
            atms_checkpoint=args.lab_atms_checkpoint if args.adapter == "none" else checkpoints_dir / "atms_sub01_40.pth",
            prior_checkpoint=args.lab_prior_checkpoint if args.adapter == "none" else checkpoints_dir / "prior_sub01_fdn.pt",
            subject_id=args.subject_id,
            text_prompt=args.text_prompt,
            device=args.device,
            atms_mode="starstim31" if args.adapter == "none" else "legacy",
        )
        worker = SemanticRefinementWorker(
            refiner=refiner,
            output_dir=args.output_root / "high_level",
            assessor_writer=assessor_writer,
        )
        worker.start()

    try:
        runner = LabReplayRunner(
            config=LabReplayConfig(
                data_root=args.data_root,
                replay_source=args.replay_source,
                epoch_npz=args.epoch_npz,
                target_manifest=args.target_manifest,
                split_manifest=args.split_manifest,
                output_root=args.output_root,
                split=args.split,
                subject=args.subject,
                max_trials=args.max_trials,
                start_trial=args.start_trial,
                sleep_seconds=args.sleep_seconds,
                adapter=args.adapter,
                atms_subject_id=args.lab_atms_subject_id,
                raw_tmin=-args.raw_pre_event_ms / 1000.0,
                raw_tmax=args.raw_post_event_ms / 1000.0,
                raw_input_sfreq=args.raw_input_sfreq,
                lab_whitening=args.lab_whitening,
                lab_whitening_split=args.lab_whitening_split,
                lab_whitening_cache=args.lab_whitening_cache,
                lab_whitening_subject=args.lab_whitening_subject,
                lab_whitening_method=args.lab_whitening_method,
                lab_whitening_matrix=args.lab_whitening_matrix,
                lab_whitening_first_block=args.lab_whitening_first_block,
                lab_bandpass_l_freq=args.lab_bandpass_l_freq,
                lab_bandpass_h_freq=args.lab_bandpass_h_freq,
                lab_filter_order=args.lab_filter_order,
                lab_ica_operator=args.lab_ica_operator,
                lab_reject_peak_to_peak_uv=args.lab_reject_uv,
            ),
            encoder=encoder,
            decoder=decoder,
            worker=worker,
            atms_embedder=atms_embedder,
            assessor_writer=assessor_writer,
        )
        return runner.run()
    finally:
        if worker is not None:
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
