#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Maryam realtime browser monitor.")
    parser.add_argument("--mode", choices=["live", "lab-live", "things-replay", "lab-replay"], default="live", help="Source mode for the GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the local web app.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the local web app.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--disable-high-level", action="store_true", help="Run only the low-level direct path.")
    parser.add_argument("--subject-id", type=int, default=None, help="Known-participant ATMS subject id.")
    parser.add_argument("--text-prompt", default="a photo of an object", help="Prompt for slow refinement.")
    parser.add_argument("--participant-code", default="unseen-demo", help="Anonymous code stored with live outputs.")
    parser.add_argument(
        "--participant-mode",
        choices=["auto", "known", "unseen"],
        default="auto",
        help="Use unseen for a participant who has no trained checkpoint row.",
    )
    parser.add_argument(
        "--enable-experimental-unseen-high-level",
        action="store_true",
        help="Use the ATMS checkpoint's trained shared token for experimental high-level output.",
    )
    parser.add_argument("--session-id", default=None, help="Live session identifier. Defaults to a timestamped ID.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local GUI after the server starts.")
    parser.add_argument(
        "--auto-connect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Connect immediately instead of showing the live setup screen first.",
    )

    parser.add_argument("--eeg-stream-name", default=None, help="Preferred EEG LSL stream name.")
    parser.add_argument("--marker-stream-name", default=None, help="Preferred marker LSL stream name.")
    parser.add_argument("--eeg-source-id", default=None, help="Optional exact EEG LSL source_id.")
    parser.add_argument("--marker-source-id", default=None, help="Optional exact marker LSL source_id.")
    parser.add_argument("--trigger-values", default="stim_onset", help="Comma-separated trigger marker names.")
    parser.add_argument("--experiment-start-values", default="experiment_start", help="Comma-separated experiment-start markers.")
    parser.add_argument("--experiment-pause-values", default="experiment_pause", help="Comma-separated experiment-pause markers.")
    parser.add_argument("--experiment-resume-values", default="experiment_resume", help="Comma-separated experiment-resume markers.")
    parser.add_argument("--experiment-end-values", default="experiment_end", help="Comma-separated experiment-end markers.")
    parser.add_argument("--marker-test-values", default="marker_test", help="Comma-separated setup marker-test values.")
    parser.add_argument("--heartbeat-values", default="heartbeat", help="Comma-separated marker heartbeat values.")
    parser.add_argument("--marker-heartbeat-timeout-seconds", type=float, default=None, help="Require periodic marker heartbeat samples within this interval.")
    parser.add_argument("--marker-test-required", action="store_true", help="Require marker_test before Arm is enabled.")
    parser.add_argument(
        "--legacy-start-on-first-trigger",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow the first whitelisted legacy numeric trigger to start an armed session.",
    )
    parser.add_argument(
        "--live-trigger-map",
        type=Path,
        default=None,
        help="TSV or JSON whitelist mapping legacy numeric markers to live targets.",
    )
    parser.add_argument("--pre-event-ms", type=float, default=None, help="Milliseconds of EEG before the trigger. Defaults to 200 for native lab-live, otherwise 0.")
    parser.add_argument("--post-event-ms", type=float, default=None, help="Milliseconds of EEG after the trigger. Defaults to 1000.")
    parser.add_argument("--min-buffer-seconds", type=float, default=1.0, help="EEG required before Arm becomes available.")
    parser.add_argument("--post-event-timeout-seconds", type=float, default=3.0, help="Drop a trial if post-event EEG does not arrive in time.")
    parser.add_argument("--trigger-cooldown-ms", type=float, default=0.0, help="Ignore triggers closer than this.")
    parser.add_argument("--poll-interval-ms", type=float, default=5.0, help="Polling interval for marker processing.")
    parser.add_argument("--ring-buffer-seconds", type=float, default=10.0, help="EEG ring buffer duration.")
    parser.add_argument("--image-root", default=None, help="Optional image root used to resolve image_id markers.")
    parser.add_argument("--eeg-sampling-rate", type=float, default=None, help="Incoming realtime EEG sampling rate. Defaults to 500 for native lab-live, otherwise 1000.")
    parser.add_argument(
        "--eeg-unit-fallback",
        choices=["volts", "millivolts", "microvolts", "nanovolts"],
        default=None,
        help="Input unit used by lab-live when the LSL stream has no channel-unit metadata. Defaults to microvolts.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parent / "Hierarchical_EEG2Image_Reconstruction" / "data",
        help="Root containing raw_eeg/, training_images/, and test_images/ for THINGS replay.",
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
    parser.add_argument(
        "--lab-data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root containing lab raw subject folders and derived artifacts.",
    )
    parser.add_argument(
        "--lab-replay-source",
        choices=["raw", "derived"],
        default="raw",
        help="raw re-extracts .easy epochs with the lab-live preprocessor; derived uses cached 31x250 epochs.",
    )
    parser.add_argument("--lab-epoch-npz", type=Path, default=None, help="Optional lab epoch npz override.")
    parser.add_argument("--lab-target-manifest", type=Path, default=None, help="Optional lab target manifest override.")
    parser.add_argument("--lab-split-manifest", type=Path, default=None, help="Optional lab split manifest override.")
    parser.add_argument("--lab-split", choices=["all", "train", "val"], default="all", help="Lab split to replay.")
    parser.add_argument("--lab-subject", default=None, help="Optional lab subject folder to replay, e.g. P06 or zar.")
    parser.add_argument(
        "--lab-bandpass-l-freq",
        type=float,
        default=None,
        help="lab-live: highpass cutoff (Hz) for the epoch bandpass. Use 0.01 to match the offline filter.",
    )
    parser.add_argument(
        "--lab-bandpass-h-freq",
        type=float,
        default=None,
        help="lab-live: lowpass cutoff (Hz) for the epoch bandpass. Use 30 to match the offline filter.",
    )
    parser.add_argument(
        "--lab-filter-order",
        type=int,
        default=4,
        help="lab-live: Butterworth order for the epoch bandpass.",
    )
    parser.add_argument(
        "--lab-ica-operator",
        type=Path,
        default=None,
        help="lab-live: 31x31 calibrated ICA cleaning matrix (.npy) from lab_calibration.",
    )
    parser.add_argument(
        "--lab-whitening-matrix",
        type=Path,
        default=None,
        help="lab-live: precomputed 31x31 MVNN whitening matrix (.npy) applied to each live epoch.",
    )
    parser.add_argument(
        "--lab-reject-uv",
        type=float,
        default=None,
        help="lab-live: peak-to-peak artifact threshold (µV) flagged per epoch. Use 250 to match the offline reject.",
    )
    parser.add_argument(
        "--lab-whitening",
        choices=["none", "mvnn"],
        default="none",
        help="Optional lab-live/replay whitening applied to the model input. Use mvnn for train-split MVNN.",
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
        help="Optional .npy path for loading/saving the lab MVNN matrix; required for lab-live MVNN.",
    )
    parser.add_argument(
        "--lab-whitening-subject",
        default=None,
        help="Optional subject folder used to compute the whitener. Defaults to --lab-subject.",
    )
    parser.add_argument(
        "--lab-low-level-checkpoint",
        type=Path,
        default=None,
        help="Native 31/32-channel lab low-level checkpoint. Required when --lab-adapter none.",
    )
    parser.add_argument(
        "--lab-model-channels",
        type=int,
        choices=[31, 32],
        default=None,
        help="Native lab checkpoint channel count. Defaults to 31 for native lab-live/lab-replay.",
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
        help="Subject id for transformed low-level checkpoints. Lab replay defaults to the manifest lab_subject_id.",
    )
    parser.add_argument(
        "--low-level-latent-scaling",
        choices=["auto", "direct", "sdxl"],
        default="auto",
        help="How to decode low-level latents. auto uses SDXL scaling for transformed lab checkpoints.",
    )
    parser.add_argument(
        "--lab-adapter",
        choices=["starstim31-to-things63", "none"],
        default="none",
        help="Use none for a native lab checkpoint, or starstim31-to-things63 for explicit THINGS checkpoint compatibility.",
    )
    parser.add_argument(
        "--enable-lab-atms-embedding",
        action="store_true",
        help="In lab-replay GUI mode, export Starstim31 ATMS EEG embeddings for replayed trials.",
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
        help="Native lab diffusion prior checkpoint for high-level refinement in lab-live/lab-replay.",
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


def apply_mode_defaults(args: argparse.Namespace) -> None:
    native_lab_live = args.mode == "lab-live" and args.lab_adapter == "none"
    raw_lab_replay = args.mode == "lab-replay" and args.lab_replay_source == "raw"
    if args.eeg_sampling_rate is None:
        args.eeg_sampling_rate = 500.0 if native_lab_live or raw_lab_replay else 1000.0
    if args.pre_event_ms is None:
        args.pre_event_ms = 200.0 if native_lab_live or raw_lab_replay else 0.0
    if args.post_event_ms is None:
        args.post_event_ms = 1000.0
    if args.eeg_unit_fallback is None and args.mode == "lab-live":
        args.eeg_unit_fallback = "microvolts"
    if args.participant_mode == "auto":
        args.participant_mode = "unseen" if args.mode == "lab-live" else "known"
    if args.subject_id is None and args.participant_mode == "known":
        args.subject_id = 0
    if args.eeg_stream_name is None:
        args.eeg_stream_name = "eeg_recon_pyexp-EEG" if args.mode == "lab-live" else "MockEEG"
    if args.marker_stream_name is None:
        args.marker_stream_name = "eeg_recon_pyexp" if args.mode == "lab-live" else "TaskMarkers"
    if args.auto_connect is None:
        args.auto_connect = args.mode != "lab-live"
    if args.legacy_start_on_first_trigger is None:
        args.legacy_start_on_first_trigger = args.mode == "lab-live"
    if args.live_trigger_map is None and args.mode == "lab-live":
        args.live_trigger_map = PROJECT_ROOT / "data" / "derived" / "finetune" / "lab_target_manifest.tsv"
    if args.session_id is None and args.mode in {"live", "lab-live"}:
        args.session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_live"

    if args.mode == "lab-live" and args.participant_mode == "unseen":
        if args.lab_low_level_arch == "transformed":
            raise SystemExit(
                "A transformed low-level checkpoint needs a trained participant row. "
                "Use --lab-low-level-arch plain for an unseen live participant."
            )
        if not args.enable_experimental_unseen_high_level:
            args.disable_high_level = True


def validate_lab_args(args: argparse.Namespace) -> None:
    if args.enable_lab_atms_embedding and args.mode != "lab-replay":
        raise SystemExit("--enable-lab-atms-embedding is currently wired for --mode lab-replay.")
    if args.lab_whitening != "none" and args.mode not in {"lab-live", "lab-replay"}:
        raise SystemExit("--lab-whitening is supported only for --mode lab-live or lab-replay.")
    if args.lab_whitening != "none" and args.lab_adapter != "none":
        raise SystemExit("--lab-whitening mvnn is only supported with --lab-adapter none.")
    if args.mode == "lab-live" and args.lab_whitening == "mvnn":
        if args.lab_whitening_cache is None:
            raise SystemExit("lab-live MVNN needs --lab-whitening-cache pointing to a fixed .npy matrix.")
        if not args.lab_whitening_cache.expanduser().is_file():
            raise SystemExit(f"Lab MVNN whitening cache does not exist: {args.lab_whitening_cache}")
    if args.mode not in {"lab-live", "lab-replay"} or args.lab_adapter == "starstim31-to-things63":
        return
    if args.mode == "lab-live" and args.lab_model_channels == 32:
        raise SystemExit(
            "Native lab-live uses the offline Starstim preprocessing path, which drops Fz and outputs "
            "31 channels. Use a 31-channel native lab checkpoint."
        )
    if args.lab_low_level_arch == "transformed" and args.lab_model_channels == 32:
        raise SystemExit("Transformed lab low-level checkpoints are currently supported only for 31-channel epochs.")
    if args.lab_low_level_checkpoint is None:
        if args.mode == "lab-live":
            raise SystemExit(
                "lab-live is native Starstim31 after offline-style preprocessing, but no native lab low-level "
                "checkpoint was given. Pass --lab-low-level-checkpoint /path/to/checkpoint.pth, "
                "or explicitly pass --lab-adapter starstim31-to-things63 to run the old THINGS63 compatibility path."
            )
        raise SystemExit(
            "lab-replay is native Starstim by default, but no native lab low-level checkpoint was given. "
            "Raw and derived lab replay both produce 31-channel model epochs because preprocessing drops Fz. "
            "Pass --lab-low-level-checkpoint /path/to/checkpoint.pth, or explicitly pass "
            "--lab-adapter starstim31-to-things63 to run the old THINGS63 compatibility path."
        )
    if not args.disable_high_level:
        if args.lab_prior_checkpoint is None:
            raise SystemExit(
                "Native lab high-level mode needs a lab diffusion prior. Pass "
                "--lab-prior-checkpoint /path/to/lab_prior_best_fdn.pt, or pass --disable-high-level."
            )


def _run_main() -> int:
    args = parse_args()
    apply_mode_defaults(args)
    validate_lab_args(args)

    import uvicorn

    from maryam_rt.gui.monitor import RuntimeMonitorState
    from maryam_rt.gui.server import LiveController, ReplayController, create_app
    from maryam_rt.integration.low_level import (
        LowLevelEpochEncoder,
        LowLevelRealtimeEncoder,
        LowLevelStarstimThingsAdapterRealtimeEncoder,
        LowLevelTransformedEpochEncoder,
        LowLevelVAEDecoder,
    )
    from maryam_rt.integration.lab_replay import LabReplayConfig, LabReplayRunner
    from maryam_rt.integration.things_raw_demo import ThingsRawDemoConfig, ThingsRawDemoRunner
    from maryam_rt.integration.triggered_runner import (
        TriggeredReconstructionRunner,
        TriggeredRunnerConfig,
    )
    from maryam_rt.integration.worker import SemanticRefinementWorker

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "hierarchical"
    output_name = {
        "live": "gui_live",
        "lab-live": "gui_lab_live",
        "things-replay": "gui_things_replay",
        "lab-replay": "gui_lab_replay",
    }[args.mode]
    if args.output_root is not None:
        output_root = args.output_root.expanduser()
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
    elif args.mode in {"live", "lab-live"}:
        output_root = PROJECT_ROOT / "outputs" / output_name / "sessions" / str(args.session_id)
    else:
        output_root = PROJECT_ROOT / "outputs" / output_name
    output_low = output_root / "low_level"
    output_high = output_root / "high_level"
    output_meta = output_root / "events"
    output_targets = output_root / "targets"
    for directory in [output_low, output_high, output_meta, output_targets]:
        directory.mkdir(parents=True, exist_ok=True)
    output_probe = output_root / ".write_probe"
    output_writable = False
    try:
        output_probe.write_text("ok")
        output_probe.unlink()
        output_writable = True
    except OSError:
        output_writable = False
    if args.mode in {"live", "lab-live"}:
        (output_root / "session.json").write_text(
            json.dumps(
                {
                    "session_id": args.session_id,
                    "participant_code": args.participant_code,
                    "participant_mode": args.participant_mode,
                    "mode": args.mode,
                    "lifecycle_status": "loading_models",
                    "created_at": time.time(),
                    "output_root": str(output_root),
                    "preferred_eeg_stream": args.eeg_stream_name,
                    "preferred_marker_stream": args.marker_stream_name,
                },
                indent=2,
            )
        )

    high_level_reason = None
    if args.mode == "lab-live" and args.participant_mode == "unseen":
        high_level_reason = (
            "experimental shared/unseen ATMS conditioning"
            if not args.disable_high_level
            else "disabled by default for an unseen participant"
        )
    expected_channels = 32 if args.mode == "lab-live" else (64 if args.mode == "live" else None)
    monitor = RuntimeMonitorState(
        mode=args.mode,
        high_level_enabled=not args.disable_high_level,
        session_id=args.session_id,
        participant_code=args.participant_code,
        participant_mode=args.participant_mode,
        high_level_reason=high_level_reason,
        marker_test_required=args.marker_test_required,
        expected_eeg_channel_count=expected_channels,
        expected_eeg_sampling_rate=args.eeg_sampling_rate if args.mode in {"live", "lab-live"} else None,
        channel_order_confirmation_required=args.mode == "lab-live",
        output_writable=output_writable,
        output_root=output_root,
    )
    scaled_low_level_latents = args.low_level_latent_scaling == "sdxl" or (
        args.low_level_latent_scaling == "auto"
        and args.mode in {"lab-live", "lab-replay"}
        and args.lab_adapter == "none"
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
        assessor_writer = AssessorRatingWriter(assessor_engine, output_root / "assessments")
    if args.enable_lab_atms_embedding:
        from maryam_rt.integration.starstim31_atms import Starstim31ATMSEmbedder

        atms_embedder = Starstim31ATMSEmbedder(
            checkpoint_path=args.lab_atms_checkpoint,
            device=args.device,
        )
    if not args.disable_high_level:
        from maryam_rt.integration.high_level import HighLevelRefiner

        def _on_refined(path: Path) -> None:
            monitor.set_latest_high_level(path, {"path": str(path)})

        refiner = HighLevelRefiner(
            atms_checkpoint=args.lab_atms_checkpoint
            if args.mode in {"lab-live", "lab-replay"} and args.lab_adapter == "none"
            else checkpoints_dir / "atms_sub01_40.pth",
            prior_checkpoint=args.lab_prior_checkpoint
            if args.mode in {"lab-live", "lab-replay"} and args.lab_adapter == "none"
            else checkpoints_dir / "prior_sub01_fdn.pt",
            subject_id=None
            if args.mode == "lab-live" and args.participant_mode == "unseen"
            else args.subject_id,
            text_prompt=args.text_prompt,
            device=args.device,
            atms_mode="starstim31" if args.mode in {"lab-live", "lab-replay"} and args.lab_adapter == "none" else "legacy",
        )
        worker = SemanticRefinementWorker(
            refiner=refiner,
            output_dir=output_high,
            on_result=_on_refined,
            assessor_writer=assessor_writer,
        )
        worker.start()

    if args.mode in {"live", "lab-live"}:
        epoch_preprocessor = None
        expected_epoch_samples = 1000
        expected_channel_labels = None
        if args.mode == "lab-live":
            if args.lab_adapter == "starstim31-to-things63":
                encoder = LowLevelStarstimThingsAdapterRealtimeEncoder(
                    checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
                    device=args.device,
                )
            else:
                from maryam_rt.integration.starstim_preprocessing import (
                    SpatialWhiteningPreprocessor,
                    StarstimLivePreprocessor,
                )
                from maryam_rt.integration.montage import STARSTIM_32_CHANNELS

                lab_model_channels = args.lab_model_channels or 31
                if args.lab_low_level_arch == "transformed":
                    encoder = LowLevelTransformedEpochEncoder(
                        checkpoint_path=args.lab_low_level_checkpoint,
                        device=args.device,
                        model_num_channels=lab_model_channels,
                        subject_id=args.lab_low_level_subject_id
                        if args.lab_low_level_subject_id is not None
                        else args.subject_id,
                    )
                else:
                    encoder = LowLevelEpochEncoder(
                        checkpoint_path=args.lab_low_level_checkpoint,
                        device=args.device,
                        model_num_channels=lab_model_channels,
                    )
                from maryam_rt.integration.lab_calibration import load_operator

                # The preprocessor drops Fz, so calibration operators are always 31x31
                # regardless of the checkpoint's declared channel count.
                lab_ica_operator = (
                    load_operator(args.lab_ica_operator, expected_channels=31)
                    if args.lab_ica_operator is not None
                    else None
                )
                lab_whitening_matrix = (
                    load_operator(args.lab_whitening_matrix, expected_channels=31)
                    if args.lab_whitening_matrix is not None
                    else None
                )
                epoch_preprocessor = StarstimLivePreprocessor(
                    input_sfreq=args.eeg_sampling_rate,
                    tmin=-args.pre_event_ms / 1000.0,
                    tmax=args.post_event_ms / 1000.0,
                    l_freq=args.lab_bandpass_l_freq,
                    h_freq=args.lab_bandpass_h_freq,
                    filter_order=args.lab_filter_order,
                    ica_operator=lab_ica_operator,
                    whitening_matrix=lab_whitening_matrix,
                    reject_peak_to_peak_uv=args.lab_reject_uv,
                )
                if args.lab_whitening == "mvnn":
                    import numpy as np

                    whitener = np.load(args.lab_whitening_cache.expanduser())
                    epoch_preprocessor = SpatialWhiteningPreprocessor(epoch_preprocessor, whitener)
                expected_epoch_samples = None
                expected_channel_labels = tuple(STARSTIM_32_CHANNELS)
            eeg_channels = 32
        else:
            encoder = LowLevelRealtimeEncoder(
                checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
                device=args.device,
            )
            eeg_channels = 64
        numeric_trigger_map = None
        if args.live_trigger_map is not None:
            from maryam_rt.integration.live_markers import load_live_trigger_map

            numeric_trigger_map = load_live_trigger_map(args.live_trigger_map)

        def _values(raw: str) -> tuple[str, ...]:
            return tuple(value.strip() for value in raw.split(",") if value.strip())

        def _build_live_runner(
            eeg_stream_name: str,
            marker_stream_name: str,
            eeg_source_id: str | None,
            marker_source_id: str | None,
        ) -> TriggeredReconstructionRunner:
            return TriggeredReconstructionRunner(
                encoder=encoder,
                decoder=decoder,
                worker=worker,
                output_low_dir=output_low,
                output_meta_dir=output_meta,
                output_target_dir=output_targets,
                assessor_writer=assessor_writer,
                monitor=monitor,
                config=TriggeredRunnerConfig(
                    eeg_stream_name=eeg_stream_name,
                    marker_stream_name=marker_stream_name,
                    eeg_source_id=eeg_source_id,
                    marker_source_id=marker_source_id,
                    pre_event_ms=args.pre_event_ms,
                    post_event_ms=args.post_event_ms,
                    trigger_values=_values(args.trigger_values),
                    experiment_start_values=_values(args.experiment_start_values),
                    experiment_pause_values=_values(args.experiment_pause_values),
                    experiment_resume_values=_values(args.experiment_resume_values),
                    experiment_end_values=_values(args.experiment_end_values),
                    marker_test_values=_values(args.marker_test_values),
                    heartbeat_values=_values(args.heartbeat_values),
                    marker_heartbeat_timeout_s=args.marker_heartbeat_timeout_seconds,
                    trigger_cooldown_ms=args.trigger_cooldown_ms,
                    eeg_sampling_rate=args.eeg_sampling_rate,
                    eeg_channels=eeg_channels,
                    eeg_convert_to_microvolts=args.mode == "lab-live",
                    eeg_unit_fallback=args.eeg_unit_fallback if args.mode == "lab-live" else None,
                    poll_interval_ms=args.poll_interval_ms,
                    ring_buffer_seconds=args.ring_buffer_seconds,
                    image_root=args.image_root,
                    expected_epoch_samples=expected_epoch_samples,
                    expected_channel_labels=expected_channel_labels,
                    min_buffer_seconds=args.min_buffer_seconds,
                    post_event_timeout_s=args.post_event_timeout_seconds,
                    require_arm=args.mode == "lab-live",
                    start_on_first_trigger=args.legacy_start_on_first_trigger,
                    numeric_trigger_map=numeric_trigger_map,
                    session_id=args.session_id,
                    participant_code=args.participant_code,
                    model_metadata={
                        "low_level_checkpoint": str(args.lab_low_level_checkpoint)
                        if args.mode == "lab-live"
                        else str(checkpoints_dir / "low_level_encoder_sub01_60.pth"),
                        "low_level_arch": args.lab_low_level_arch if args.mode == "lab-live" else "legacy",
                        "low_level_subject_id": str(args.lab_low_level_subject_id)
                        if args.mode == "lab-live" and args.lab_low_level_subject_id is not None
                        else "none",
                        "low_level_latent_scaling": args.low_level_latent_scaling
                        if args.mode == "lab-live"
                        else "legacy",
                        "lab_whitening": args.lab_whitening if args.mode == "lab-live" else "none",
                        "lab_whitening_cache": str(args.lab_whitening_cache)
                        if args.mode == "lab-live" and args.lab_whitening_cache is not None
                        else "none",
                        "atms_checkpoint": str(args.lab_atms_checkpoint) if not args.disable_high_level else "disabled",
                        "prior_checkpoint": str(args.lab_prior_checkpoint) if not args.disable_high_level else "disabled",
                        "participant_conditioning": "shared_unseen"
                        if args.participant_mode == "unseen"
                        else f"known:{args.subject_id}",
                    },
                ),
                epoch_preprocessor=epoch_preprocessor,
            )

        controller = LiveController(
            runner=None,
            monitor=monitor,
            runner_factory=_build_live_runner,
            auto_connect=args.auto_connect,
            default_eeg_stream_name=args.eeg_stream_name,
            default_marker_stream_name=args.marker_stream_name,
        )
    elif args.mode == "things-replay":
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
            assessor_writer=assessor_writer,
            monitor=monitor,
        )
        controller = ReplayController(runner=runner, monitor=monitor)
    else:
        if args.lab_adapter == "starstim31-to-things63":
            encoder = LowLevelEpochEncoder(
                checkpoint_path=checkpoints_dir / "low_level_encoder_sub01_60.pth",
                device=args.device,
            )
        else:
            lab_model_channels = args.lab_model_channels or 31
            if args.lab_low_level_arch == "transformed":
                encoder = LowLevelTransformedEpochEncoder(
                    checkpoint_path=args.lab_low_level_checkpoint,
                    device=args.device,
                    model_num_channels=lab_model_channels,
                    subject_id=args.lab_low_level_subject_id,
                )
            else:
                encoder = LowLevelEpochEncoder(
                    checkpoint_path=args.lab_low_level_checkpoint,
                    device=args.device,
                    model_num_channels=lab_model_channels,
                )
        runner = LabReplayRunner(
            config=LabReplayConfig(
                data_root=args.lab_data_root,
                replay_source=args.lab_replay_source,
                epoch_npz=args.lab_epoch_npz,
                target_manifest=args.lab_target_manifest,
                split_manifest=args.lab_split_manifest,
                output_root=output_root,
                split=args.lab_split,
                subject=args.lab_subject,
                max_trials=args.max_trials,
                start_trial=args.start_trial,
                sleep_seconds=args.sleep_seconds,
                adapter=args.lab_adapter,
                atms_subject_id=args.lab_atms_subject_id,
                raw_tmin=-args.pre_event_ms / 1000.0,
                raw_tmax=args.post_event_ms / 1000.0,
                raw_input_sfreq=args.eeg_sampling_rate,
                lab_whitening=args.lab_whitening,
                lab_whitening_split=args.lab_whitening_split,
                lab_whitening_cache=args.lab_whitening_cache,
                lab_whitening_subject=args.lab_whitening_subject,
            ),
            encoder=encoder,
            decoder=decoder,
            worker=worker,
            atms_embedder=atms_embedder,
            assessor_writer=assessor_writer,
            monitor=monitor,
        )
        controller = ReplayController(runner=runner, monitor=monitor)

    if args.mode in {"live", "lab-live"}:
        session_path = output_root / "session.json"
        session_payload = json.loads(session_path.read_text())
        session_payload.update(
            {
                "lifecycle_status": "waiting_for_equipment",
                "models_loaded": True,
                "low_level_checkpoint": str(args.lab_low_level_checkpoint)
                if args.mode == "lab-live"
                else str(checkpoints_dir / "low_level_encoder_sub01_60.pth"),
                "high_level_enabled": not args.disable_high_level,
                "high_level_reason": high_level_reason,
            }
        )
        session_path.write_text(json.dumps(session_payload, indent=2))

    app = create_app(controller)
    if args.open_browser:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{browser_host}:{args.port}")).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if worker is not None:
            worker.stop()
    return 0


def main() -> int:
    try:
        return _run_main()
    except Exception as exc:
        args = parse_args()
        apply_mode_defaults(args)
        import uvicorn

        from maryam_rt.gui.monitor import RuntimeMonitorState
        from maryam_rt.gui.server import ErrorController, create_app

        monitor = RuntimeMonitorState(
            mode=args.mode,
            high_level_enabled=False,
            session_id=args.session_id,
            participant_code=args.participant_code,
            participant_mode=args.participant_mode,
            high_level_reason="unavailable because startup failed",
        )
        monitor.fail(f"Startup failed: {exc}")
        controller = ErrorController(monitor)
        app = create_app(controller)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
