from __future__ import annotations

import csv
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import scipy.linalg
import torch
from PIL import Image
from sklearn.discriminant_analysis import _cov

from maryam_rt.gui.monitor import RuntimeMonitorState
from maryam_rt.integration.lab_calibration import compute_mvnn_whitener_custom, load_operator
from maryam_rt.integration.montage import STARSTIM_31_CHANNELS, STARSTIM_32_CHANNELS, THINGS_63_CHANNELS
from maryam_rt.integration.starstim_preprocessing import StarstimLivePreprocessor, peak_to_peak

if TYPE_CHECKING:
    from maryam_rt.integration.assessor_ratings import AssessorRatingWriter
    from maryam_rt.integration.low_level import LowLevelEpochEncoder, LowLevelVAEDecoder
    from maryam_rt.integration.starstim31_atms import Starstim31ATMSEmbedder
    from maryam_rt.integration.worker import SemanticRefinementWorker

@dataclass(frozen=True)
class LabReplayConfig:
    data_root: Path
    replay_source: str = "raw"
    epoch_npz: Optional[Path] = None
    target_manifest: Optional[Path] = None
    split_manifest: Optional[Path] = None
    output_root: Optional[Path] = None
    split: str = "all"
    subject: Optional[str] = None
    max_trials: Optional[int] = None
    start_trial: int = 0
    sleep_seconds: float = 0.0
    adapter: str = "none"
    copy_targets: bool = True
    atms_subject_id: Optional[int] = None
    raw_tmin: float = -0.2
    raw_tmax: float = 1.0
    raw_input_sfreq: Optional[float] = None
    lab_whitening: str = "none"
    lab_whitening_split: str = "train"
    lab_whitening_cache: Optional[Path] = None
    lab_whitening_subject: Optional[str] = None
    lab_whitening_method: str = "custom"
    lab_whitening_matrix: Optional[Path] = None
    lab_whitening_first_block: Optional[int] = None
    lab_bandpass_l_freq: Optional[float] = None
    lab_bandpass_h_freq: Optional[float] = None
    lab_filter_order: int = 4
    lab_ica_operator: Optional[Path] = None
    lab_reject_peak_to_peak_uv: Optional[float] = None


@dataclass(frozen=True)
class LabTrial:
    trial_index: int
    epoch_index: int
    target_index: int
    trigger: int
    subject: str
    lab_subject_id: Optional[int]
    category: str
    split: str
    epoch_lab31: np.ndarray
    epoch_model: np.ndarray
    image_path: Optional[Path]
    usable_for_finetune: bool = True
    replay_source: str = "derived"
    raw_easy_path: Optional[Path] = None
    raw_start_sample: Optional[int] = None
    raw_samples: Optional[int] = None
    whitening_mode: str = "none"
    whitening_cache_path: Optional[Path] = None
    whitening_source_split: Optional[str] = None
    artifact_rejected: bool = False
    peak_to_peak_uv: Optional[float] = None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _default_epoch_path(data_root: Path) -> Path:
    return data_root / "derived" / "lab_starstim31_epochs_250hz.npz"


def _default_target_manifest(data_root: Path) -> Path:
    return data_root / "derived" / "finetune" / "lab_target_manifest.tsv"


def _default_split_manifest(data_root: Path) -> Path:
    return data_root / "derived" / "finetune" / "lab_finetune_split_manifest.tsv"


def _find_subject_file(data_root: Path, subject: str, suffix: str) -> Optional[Path]:
    subject_dir = data_root / subject
    if not subject_dir.exists():
        return None
    matches = sorted(subject_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _parse_starstim_info(path: Path | None) -> tuple[Optional[float], Optional[list[str]]]:
    if path is None or not path.exists():
        return None, None

    sfreq: Optional[float] = None
    channels: list[str] = []
    channel_re = re.compile(r"Channel\s+\d+:\s+(.+)$")
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("Effective EEG sampling rate:"):
            sfreq = float(line.split(":", 1)[1].split()[0])
        match = channel_re.search(line.strip())
        if match:
            channels.append(match.group(1).strip())
    return sfreq, channels or None


def _load_easy_segment_uv(path: Path, start_sample: int, samples: int) -> np.ndarray:
    if start_sample < 0:
        raise ValueError(f"Raw replay epoch starts before the recording: sample {start_sample}.")
    if samples <= 0:
        raise ValueError("Raw replay sample count must be positive.")

    with path.open("r", errors="ignore") as handle:
        lines = list(islice(handle, start_sample, start_sample + samples))
    if len(lines) != samples:
        raise ValueError(
            f"Raw replay expected {samples} samples from {path} starting at {start_sample}, "
            f"but only read {len(lines)}."
        )

    data_nv = np.loadtxt(lines, dtype=np.float32, usecols=range(len(STARSTIM_32_CHANNELS)))
    if data_nv.ndim == 1:
        data_nv = data_nv[None, :]
    return (data_nv.T / 1000.0).astype(np.float32)


def _output_root(config: LabReplayConfig) -> Path:
    return config.output_root or Path.cwd() / "outputs" / "lab_replay"


def _lab_whitening_cache_path(config: LabReplayConfig) -> Path:
    if config.lab_whitening_cache is not None:
        config.lab_whitening_cache.parent.mkdir(parents=True, exist_ok=True)
        return config.lab_whitening_cache

    cache_dir = _output_root(config) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    subject = config.lab_whitening_subject or config.subject or "all_subjects"
    subject = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject)
    if config.replay_source == "raw":
        window = f"{int(round(config.raw_tmin * 1000))}ms_{int(round(config.raw_tmax * 1000))}ms"
    else:
        window = "derived250hz"
    scope = (
        f"first{config.lab_whitening_first_block}"
        if config.lab_whitening_first_block is not None
        else config.lab_whitening_split
    )
    stem = f"lab_{config.replay_source}_{subject}_{scope}_{config.lab_whitening_method}_mvnn_{window}"
    return cache_dir / f"{stem}.npy"


def adapt_starstim31_to_things63(epoch: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Place Starstim31 channels into the THINGS63 montage and zero-fill missing channels."""
    if epoch.ndim != 2:
        raise ValueError(f"Expected lab epoch shaped (channels, samples), got {epoch.shape}.")
    if epoch.shape[0] != len(ch_names):
        raise ValueError(f"Epoch has {epoch.shape[0]} channels but ch_names has {len(ch_names)} entries.")

    by_name = {name: idx for idx, name in enumerate(ch_names)}
    missing_lab = [name for name in STARSTIM_31_CHANNELS if name not in by_name]
    if missing_lab:
        raise ValueError(f"Lab epoch is missing expected Starstim31 channels: {missing_lab}")

    adapted = np.zeros((len(THINGS_63_CHANNELS), epoch.shape[1]), dtype=np.float32)
    things_index = {name: idx for idx, name in enumerate(THINGS_63_CHANNELS)}
    for name in STARSTIM_31_CHANNELS:
        adapted[things_index[name]] = epoch[by_name[name]]
    return adapted


def _resolve_image_path(row: dict[str, str]) -> Optional[Path]:
    for key in ("exact_image_path", "target_image_path", "image_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path
    return None


def _optional_int(value: str | None) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _load_split_rows(config: LabReplayConfig) -> tuple[list[dict[str, str]], dict[int, dict[str, str]], Path]:
    target_path = config.target_manifest or _default_target_manifest(config.data_root)
    split_path = config.split_manifest or _default_split_manifest(config.data_root)

    if not target_path.exists():
        raise FileNotFoundError(f"Lab target manifest not found: {target_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Lab split manifest not found: {split_path}")

    target_rows = _read_tsv(target_path)
    target_by_index = {int(row["target_index"]): row for row in target_rows}
    split_rows = _read_tsv(split_path)
    return split_rows, target_by_index, split_path


def _iter_filtered_split_rows(
    config: LabReplayConfig,
    split_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in split_rows:
        split = row.get("split", "")
        subject = row.get("subject_folder", "")
        if config.split != "all" and split != config.split:
            continue
        if config.subject is not None and subject != config.subject:
            continue
        rows.append(row)
    return rows


def _model_epoch(config: LabReplayConfig, epoch_lab31: np.ndarray, ch_names: list[str]) -> np.ndarray:
    if config.adapter == "starstim31-to-things63":
        return adapt_starstim31_to_things63(epoch_lab31, ch_names)
    if config.adapter == "none":
        return epoch_lab31
    raise ValueError(f"Unsupported lab adapter: {config.adapter}")


def _calibration_epochs(trials: Sequence[LabTrial], first_block: Optional[int]) -> np.ndarray:
    """Stack the usable, non-rejected 31x250 model epochs used to fit a whitener.

    ``first_block`` treats the leading N trials as this subject's calibration
    ("train") block; the rest of the session then reuses the fitted matrix.
    """
    candidates = list(trials)
    if first_block is not None:
        # The first block is an acquisition window; artifacts inside it are
        # dropped rather than back-filled from later trials.
        candidates = candidates[:first_block]
    usable = [
        trial
        for trial in candidates
        if trial.usable_for_finetune
        and not trial.artifact_rejected
        and trial.epoch_model.shape == (31, 250)
    ]
    if len(usable) < 2:
        raise ValueError(
            "Lab MVNN whitening needs at least two usable 31x250 calibration trials "
            f"(got {len(usable)})."
        )
    return np.stack([trial.epoch_model for trial in usable]).astype(np.float64)


def _compute_lab_mvnn_whitener_shrinkage(epochs: np.ndarray) -> np.ndarray:
    """Per-trial shrinkage covariances averaged, then ``sigma ** -0.5``."""
    covariances = np.empty((epochs.shape[0], 31, 31), dtype=np.float64)
    for index in range(epochs.shape[0]):
        covariances[index] = _cov(epochs[index].T, shrinkage="auto")
    sigma_tot = covariances.mean(axis=0)
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)
    sigma_inv = np.asarray(np.real_if_close(sigma_inv), dtype=np.float32)
    if sigma_inv.shape != (31, 31):
        raise ValueError(f"Expected lab MVNN whitener shaped (31, 31), got {sigma_inv.shape}.")
    if not np.isfinite(sigma_inv).all():
        raise ValueError("Lab MVNN whitener contains non-finite values.")
    return sigma_inv


def _compute_lab_mvnn_whitener(
    trials: Sequence[LabTrial],
    method: str = "custom",
    first_block: Optional[int] = None,
) -> np.ndarray:
    epochs = _calibration_epochs(trials, first_block)
    if method == "custom":
        return compute_mvnn_whitener_custom(epochs)
    if method == "shrinkage":
        return _compute_lab_mvnn_whitener_shrinkage(epochs)
    raise ValueError(f"Unsupported lab whitening method: {method}. Use 'custom' or 'shrinkage'.")


def _apply_lab_whitener(epoch: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    if epoch.shape != (31, 250):
        raise ValueError(f"Lab MVNN whitening expects a 31x250 epoch, got {epoch.shape}.")
    if sigma_inv.shape != (31, 31):
        raise ValueError(f"Lab MVNN whitener must be 31x31, got {sigma_inv.shape}.")
    return np.asarray(sigma_inv @ epoch, dtype=np.float32)


def _load_base_lab_trials(config: LabReplayConfig) -> list[LabTrial]:
    if config.replay_source == "raw":
        return load_lab_trials_from_raw(config)
    if config.replay_source == "derived":
        return load_lab_trials_from_derived(config)
    raise ValueError(f"Unsupported lab replay source: {config.replay_source}")


def _load_or_build_lab_whitener(config: LabReplayConfig) -> tuple[np.ndarray, Path]:
    # 1. An explicitly supplied whitening matrix always wins (frozen per subject).
    if config.lab_whitening_matrix is not None:
        sigma_inv = load_operator(config.lab_whitening_matrix, expected_channels=31)
        return sigma_inv, config.lab_whitening_matrix

    # 2. A previously cached matrix for this subject/scope is reused verbatim.
    cache_path = _lab_whitening_cache_path(config)
    if cache_path.exists():
        sigma_inv = np.load(cache_path).astype(np.float32)
        if sigma_inv.shape != (31, 31):
            raise ValueError(f"Cached lab MVNN whitener must be 31x31, got {sigma_inv.shape}: {cache_path}")
        return sigma_inv, cache_path

    # 3. Otherwise compute it from the calibration ("first block") trials. When a
    # first block is requested we take the leading trials in manifest order
    # regardless of split; otherwise we fit on the configured split.
    first_block = config.lab_whitening_first_block
    calibration_config = replace(
        config,
        split="all" if first_block is not None else config.lab_whitening_split,
        subject=config.lab_whitening_subject or config.subject,
        max_trials=first_block,
        start_trial=0,
        sleep_seconds=0.0,
        lab_whitening="none",
        lab_whitening_cache=None,
        lab_whitening_matrix=None,
    )
    calibration_trials = _load_base_lab_trials(calibration_config)
    sigma_inv = _compute_lab_mvnn_whitener(
        calibration_trials,
        method=config.lab_whitening_method,
        first_block=first_block,
    )
    np.save(cache_path, sigma_inv)

    metadata = {
        "mode": "lab_mvnn",
        "method": config.lab_whitening_method,
        "cache_path": str(cache_path),
        "replay_source": config.replay_source,
        "subject": calibration_config.subject,
        "split": calibration_config.split,
        "first_block": first_block,
        "usable_trial_count": int(
            sum(
                1
                for trial in calibration_trials
                if trial.usable_for_finetune
                and not trial.artifact_rejected
                and trial.epoch_model.shape == (31, 250)
            )
        ),
        "raw_window_seconds": [float(config.raw_tmin), float(config.raw_tmax)],
        "shape": list(sigma_inv.shape),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return sigma_inv, cache_path


def _apply_configured_lab_whitening(config: LabReplayConfig, trials: list[LabTrial]) -> list[LabTrial]:
    if config.lab_whitening == "none":
        return trials
    if config.lab_whitening != "mvnn":
        raise ValueError(f"Unsupported lab whitening mode: {config.lab_whitening}")
    if config.adapter != "none":
        raise ValueError("Lab MVNN whitening is only supported with --lab-adapter none / --adapter none.")

    sigma_inv, cache_path = _load_or_build_lab_whitener(config)
    return [
        replace(
            trial,
            epoch_model=_apply_lab_whitener(trial.epoch_model, sigma_inv),
            whitening_mode="mvnn",
            whitening_cache_path=cache_path,
            whitening_source_split=config.lab_whitening_split,
        )
        for trial in trials
    ]


def _trial_from_row(
    config: LabReplayConfig,
    row: dict[str, str],
    target_by_index: dict[int, dict[str, str]],
    epoch_lab31: np.ndarray,
    ch_names: list[str],
    trial_index: int,
    replay_source: str,
    raw_easy_path: Optional[Path] = None,
    raw_start_sample: Optional[int] = None,
    raw_samples: Optional[int] = None,
) -> LabTrial:
    target_index = int(row["target_index"])
    target_row = target_by_index.get(target_index, {})
    epoch_model = _model_epoch(config, epoch_lab31, ch_names)
    # Rejection is measured on the pre-whitening model epoch, matching where the
    # offline pipeline rejects trials (reject=250 µV at epoching, before MVNN).
    ptp = peak_to_peak(epoch_model)
    rejected = (
        config.lab_reject_peak_to_peak_uv is not None
        and ptp > config.lab_reject_peak_to_peak_uv
    )
    return LabTrial(
        trial_index=trial_index,
        epoch_index=int(row["epoch_index"]),
        target_index=target_index,
        trigger=int(row["trigger"]),
        subject=row.get("subject_folder", ""),
        lab_subject_id=_optional_int(row.get("lab_subject_id")),
        category=row.get("category") or target_row.get("category", ""),
        split=row.get("split", ""),
        epoch_lab31=epoch_lab31,
        epoch_model=epoch_model,
        image_path=_resolve_image_path({**target_row, **row}),
        usable_for_finetune=str(row.get("usable_for_finetune", "true")).lower() == "true",
        replay_source=replay_source,
        raw_easy_path=raw_easy_path,
        raw_start_sample=raw_start_sample,
        raw_samples=raw_samples,
        artifact_rejected=rejected,
        peak_to_peak_uv=float(ptp),
    )


def load_lab_trials_from_derived(config: LabReplayConfig) -> list[LabTrial]:
    epoch_path = config.epoch_npz or _default_epoch_path(config.data_root)
    split_rows, target_by_index, _split_path = _load_split_rows(config)

    if not epoch_path.exists():
        raise FileNotFoundError(f"Lab epoch file not found: {epoch_path}")

    npz = np.load(epoch_path, allow_pickle=True)
    eeg = np.asarray(npz["eeg"], dtype=np.float32)
    ch_names = npz["ch_names"].astype(str).tolist()
    if eeg.ndim != 3:
        raise ValueError(f"Expected lab EEG shaped (N, C, T), got {eeg.shape}.")
    if eeg.shape[1:] != (31, 250):
        raise ValueError(
            f"Expected model-ready lab epochs shaped (N, 31, 250), got {eeg.shape}."
        )
    if len(split_rows) != eeg.shape[0]:
        raise ValueError(f"Split manifest rows ({len(split_rows)}) do not match EEG epochs ({eeg.shape[0]}).")

    trials: list[LabTrial] = []
    for row in _iter_filtered_split_rows(config, split_rows):
        epoch_index = int(row["epoch_index"])
        if epoch_index < 0 or epoch_index >= eeg.shape[0]:
            raise ValueError(f"Epoch index {epoch_index} is outside EEG array with {eeg.shape[0]} rows.")

        trials.append(
            _trial_from_row(
                config,
                row,
                target_by_index,
                eeg[epoch_index],
                ch_names,
                len(trials),
                replay_source="derived",
            )
        )
    return trials


def _load_lab_ica_operator(config: LabReplayConfig) -> Optional[np.ndarray]:
    if config.lab_ica_operator is None:
        return None
    return load_operator(config.lab_ica_operator, expected_channels=31)


def load_lab_trials_from_raw(config: LabReplayConfig) -> list[LabTrial]:
    split_rows, target_by_index, _split_path = _load_split_rows(config)
    selected_rows = _iter_filtered_split_rows(config, split_rows)
    indexed_rows = list(enumerate(selected_rows))
    indexed_rows = indexed_rows[config.start_trial :]
    if config.max_trials is not None:
        indexed_rows = indexed_rows[: config.max_trials]
    trials: list[LabTrial] = []
    preprocessors: dict[str, StarstimLivePreprocessor] = {}
    easy_paths: dict[str, Path] = {}
    ica_operator = _load_lab_ica_operator(config)

    for source_trial_index, row in indexed_rows:
        subject = row.get("subject_folder", "")
        if not subject:
            raise ValueError("Raw lab replay requires subject_folder in the split manifest.")
        if not row.get("eeg_sample"):
            raise ValueError("Raw lab replay requires eeg_sample in the split manifest.")

        easy_path = easy_paths.get(subject)
        if easy_path is None:
            easy_path = _find_subject_file(config.data_root, subject, ".easy")
            if easy_path is None:
                raise FileNotFoundError(f"No .easy file found for raw lab replay subject {subject}.")
            easy_paths[subject] = easy_path

        preprocessor = preprocessors.get(subject)
        if preprocessor is None:
            info_sfreq, info_channels = _parse_starstim_info(_find_subject_file(config.data_root, subject, ".info"))
            input_sfreq = float(config.raw_input_sfreq or info_sfreq or 500.0)
            preprocessor = StarstimLivePreprocessor(
                input_sfreq=input_sfreq,
                tmin=config.raw_tmin,
                tmax=config.raw_tmax,
                ch_names=tuple(info_channels or STARSTIM_32_CHANNELS),
                l_freq=config.lab_bandpass_l_freq,
                h_freq=config.lab_bandpass_h_freq,
                filter_order=config.lab_filter_order,
                ica_operator=ica_operator,
                reject_peak_to_peak_uv=config.lab_reject_peak_to_peak_uv,
            )
            preprocessors[subject] = preprocessor

        event_sample = int(row["eeg_sample"])
        raw_samples = int(round((config.raw_tmax - config.raw_tmin) * preprocessor.input_sfreq))
        raw_start_sample = int(round(event_sample + config.raw_tmin * preprocessor.input_sfreq))
        raw_epoch = _load_easy_segment_uv(easy_path, start_sample=raw_start_sample, samples=raw_samples)
        epoch_lab31 = preprocessor(raw_epoch)
        trials.append(
            _trial_from_row(
                config,
                row,
                target_by_index,
                epoch_lab31,
                list(STARSTIM_31_CHANNELS),
                source_trial_index,
                replay_source="raw",
                raw_easy_path=easy_path,
                raw_start_sample=raw_start_sample,
                raw_samples=raw_samples,
            )
        )
    return trials


def load_lab_trials(config: LabReplayConfig) -> list[LabTrial]:
    return _apply_configured_lab_whitening(config, _load_base_lab_trials(config))


class LabReplayRunner:
    """Replay manifest-aligned native Starstim lab epochs through the realtime GUI pipeline."""

    def __init__(
        self,
        config: LabReplayConfig,
        encoder: LowLevelEpochEncoder,
        decoder: LowLevelVAEDecoder,
        worker: Optional[SemanticRefinementWorker],
        atms_embedder: Optional[Starstim31ATMSEmbedder] = None,
        assessor_writer: Optional[AssessorRatingWriter] = None,
        monitor: RuntimeMonitorState | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.worker = worker
        self.atms_embedder = atms_embedder
        self.assessor_writer = assessor_writer
        self.monitor = monitor
        self._stop_event = threading.Event()

        self.output_root = _output_root(config)
        self.low_dir = self.output_root / "low_level"
        self.high_dir = self.output_root / "high_level"
        self.embedding_dir = self.output_root / "embeddings"
        self.assessment_dir = self.output_root / "assessments"
        self.target_dir = self.output_root / "targets"
        self.meta_dir = self.output_root / "metadata"
        for directory in [self.low_dir, self.high_dir, self.target_dir, self.meta_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        if self.atms_embedder is not None:
            self.embedding_dir.mkdir(parents=True, exist_ok=True)
        if self.assessor_writer is not None:
            self.assessment_dir.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> int:
        trials = load_lab_trials(self.config)
        if self.config.replay_source == "raw":
            selected = trials
        else:
            selected = trials[self.config.start_trial :]
            if self.config.max_trials is not None:
                selected = selected[: self.config.max_trials]
        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=True,
                eeg_connected=True,
                marker_connected=True,
                mode="lab-replay",
                message="Starting lab epoch replay.",
            )

        print(
            f"Loaded {len(trials)} lab trial(s); running {len(selected)} trial(s) "
            f"with source={self.config.replay_source}, adapter={self.config.adapter}, "
            f"whitening={self.config.lab_whitening}."
        )
        for trial in selected:
            if self._stop_event.is_set():
                break
            self._run_trial(trial)
            if self.config.sleep_seconds > 0:
                time.sleep(self.config.sleep_seconds)

        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=False,
                eeg_connected=False,
                marker_connected=False,
                message="Lab replay complete.",
            )
        return 0

    def _run_trial(self, trial: LabTrial) -> None:
        stem = (
            f"lab_{trial.trial_index:05d}_epoch{trial.epoch_index:05d}_"
            f"trigger{trial.trigger}_{trial.subject}"
        )
        if self.monitor is not None:
            self.monitor.set_latest_epoch(trial.epoch_lab31, sampling_rate=250.0, start_seconds=0.0)
            self.monitor.add_marker(
                {
                    "timestamp": float(trial.trial_index),
                    "label": f"trigger {trial.trigger}",
                    "event_name": str(trial.trigger),
                    "raw_value": str(trial.trigger),
                    "status": "accepted"
                    if trial.usable_for_finetune and not trial.artifact_rejected
                    else "artifact_review",
                    "image_id": str(trial.target_index),
                    "subject": trial.subject,
                    "split": trial.split,
                    "replay_source": trial.replay_source,
                }
            )
            if trial.image_path is not None:
                self.monitor.set_latest_target(
                    trial.image_path,
                    {
                        "image_id": str(trial.target_index),
                        "label": trial.category,
                        "subject": trial.subject,
                        "split": trial.split,
                    },
                )

        input_tensor = torch.from_numpy(trial.epoch_model).unsqueeze(0).to(self.encoder.device_name)
        if hasattr(self.encoder, "set_subject_id"):
            self.encoder.set_subject_id(trial.lab_subject_id)
        with torch.inference_mode():
            latent = self.encoder(input_tensor)
            image = self.decoder(latent).astype(np.uint8)

        low_path = self.low_dir / f"{stem}_low.png"
        Image.fromarray(image).save(low_path)
        metadata = {
            "mode": "lab-replay",
            "replay_source": trial.replay_source,
            "adapter": self.config.adapter,
            "trial_index": trial.trial_index,
            "epoch_index": trial.epoch_index,
            "target_index": trial.target_index,
            "trigger": trial.trigger,
            "subject": trial.subject,
            "lab_subject_id": trial.lab_subject_id,
            "category": trial.category,
            "split": trial.split,
            "usable_for_finetune": trial.usable_for_finetune,
            "epoch_lab31_shape": list(trial.epoch_lab31.shape),
            "epoch_model_shape": list(trial.epoch_model.shape),
            "image_path": None if trial.image_path is None else str(trial.image_path),
            "low_level_path": str(low_path),
            "whitening_mode": trial.whitening_mode,
            "artifact_rejected": trial.artifact_rejected,
            "peak_to_peak_uv": trial.peak_to_peak_uv,
        }
        if trial.whitening_cache_path is not None:
            metadata["whitening_cache_path"] = str(trial.whitening_cache_path)
            metadata["whitening_source_split"] = trial.whitening_source_split
        if trial.raw_easy_path is not None:
            metadata["raw_easy_path"] = str(trial.raw_easy_path)
            metadata["raw_start_sample"] = trial.raw_start_sample
            metadata["raw_samples"] = trial.raw_samples
        target_copy: Path | None = None
        if trial.image_path is not None and self.config.copy_targets:
            target_copy = self.target_dir / f"{stem}_target{trial.image_path.suffix}"
            shutil.copy2(trial.image_path, target_copy)
            metadata["target_copy_path"] = str(target_copy)
        target_assessor_path = target_copy or trial.image_path
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
        if self.atms_embedder is not None:
            atms_subject_id = (
                self.config.atms_subject_id
                if self.config.atms_subject_id is not None
                else trial.lab_subject_id
            )
            if atms_subject_id is None:
                raise ValueError(
                    "ATMS embedding is enabled, but the lab replay manifest has no lab_subject_id. "
                    "Pass --lab-atms-subject-id explicitly."
                )
            embedding_path = self.embedding_dir / f"{stem}_starstim31_atms.pt"
            metadata["starstim31_atms_embedding"] = self.atms_embedder.save_embedding(
                trial.epoch_lab31,
                subject_id=atms_subject_id,
                output_path=embedding_path,
            )
        (self.meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))

        if self.monitor is not None:
            self.monitor.set_latest_low_level(
                low_path,
                {
                    "marker_label": str(trial.trigger),
                    "event_time": time.strftime("%H:%M:%S"),
                    "stem": stem,
                    "subject": trial.subject,
                    "split": trial.split,
                    "replay_source": trial.replay_source,
                },
            )

        x250 = self.encoder.snapshot_last_x250()
        if self.worker is not None and x250 is not None:
            prompt = trial.category.replace("_", " ") if trial.category else None
            self.worker.submit(
                stem=stem,
                x250=x250,
                low_level_image=image,
                text_prompt=prompt,
                low_level_path=low_path,
                target_image_path=target_assessor_path,
            )

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, object]:
        if self.monitor is None:
            return {"sampling_rate": None, "window_seconds": None, "traces": [], "markers": []}
        return self.monitor.latest_epoch_plot(max_channels=max_channels)
