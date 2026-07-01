from __future__ import annotations

import json
import pickle
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import scipy.linalg
import torch
from PIL import Image
from sklearn.discriminant_analysis import _cov
from sklearn.utils import shuffle

from maryam_rt.gui.monitor import RuntimeMonitorState
from maryam_rt.integration.high_level import HighLevelRefiner
from maryam_rt.integration.low_level import LowLevelEpochEncoder, LowLevelVAEDecoder
from maryam_rt.integration.worker import SemanticRefinementWorker


CHAN_ORDER = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]


OFFLINE_PREPROCESSING_SEED = 20200220


@dataclass(frozen=True)
class ThingsRawDemoConfig:
    data_root: Path
    subject: str = "sub-01"
    session: str = "ses-01"
    split: str = "test"
    output_root: Optional[Path] = None
    target_sfreq: int = 250
    tmin: float = -0.2
    tmax: float = 1.0
    crop_start_index: int = 50
    keep_samples: int = 250
    max_trials: Optional[int] = None
    start_trial: int = 0
    sleep_seconds: float = 0.0
    copy_targets: bool = True
    apply_whitening: bool = True
    average_by_event: bool = False
    compare_to_offline_saved: bool = True
    calibration_enabled: bool = False
    calibration_split: str = "training"
    calibration_max_conditions: int = 40
    calibration_repetitions_per_condition: int = 3
    calibration_sleep_seconds: float = 0.0


@dataclass(frozen=True)
class ThingsTrial:
    trial_index: int
    event_code: int
    epoch_full: np.ndarray
    epoch_x250: np.ndarray
    image_path: Optional[Path]
    event_sample: int
    repetition_count: int = 1


@dataclass(frozen=True)
class ThingsReplaySession:
    continuous_eeg: np.ndarray
    continuous_sfreq: float
    trials: list[ThingsTrial]
    offline_saved_data: np.ndarray | None = None


@dataclass(frozen=True)
class CalibrationBlock:
    trials: list[ThingsTrial]
    sigma_inv: np.ndarray
    cache_path: Path
    metadata_path: Path
    metadata: dict[str, object]
    from_cache: bool


def build_image_index(data_root: Path, split: str) -> dict[int, Path]:
    """Map THINGS event codes to image paths using numeric folder ordering."""
    if split == "training":
        root = data_root / "training_images"
        image_paths: list[Path] = []
        for concept_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for image_path in sorted([p for p in concept_dir.iterdir() if p.is_file()]):
                image_paths.append(image_path)
    elif split == "test":
        root = data_root / "test_images"
        image_paths = []
        for concept_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            files = sorted([p for p in concept_dir.iterdir() if p.is_file()])
            if not files:
                continue
            image_paths.append(files[0])
    else:
        raise ValueError(f"Unsupported split: {split}")
    return {idx + 1: path for idx, path in enumerate(image_paths)}


def _output_root(config: ThingsRawDemoConfig) -> Path:
    return config.output_root or Path.cwd() / "outputs" / "things_raw_demo"


def _pick_model_channels(raw: mne.io.BaseRaw) -> list[str]:
    picked_channels = [ch for ch in CHAN_ORDER if ch in raw.ch_names]
    if len(picked_channels) != len(CHAN_ORDER):
        missing = [ch for ch in CHAN_ORDER if ch not in raw.ch_names]
        raise ValueError(f"Raw EEG is missing required model channels: {missing}")
    return picked_channels


def _load_raw_split(config: ThingsRawDemoConfig, split: str) -> tuple[mne.io.BaseRaw, np.ndarray, np.ndarray, float]:
    raw_path = config.data_root / "raw_eeg" / config.subject / config.session / f"raw_eeg_{split}.npy"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw EEG file not found: {raw_path}")

    raw_dict = np.load(raw_path, allow_pickle=True).item()
    info = mne.create_info(raw_dict["ch_names"], raw_dict["sfreq"], raw_dict["ch_types"])
    raw = mne.io.RawArray(raw_dict["raw_eeg_data"], info, verbose=False)
    events = mne.find_events(raw, stim_channel="stim", verbose=False)
    events = events[events[:, 2] != 99999]

    picked_channels = _pick_model_channels(raw)
    try:
        raw.pick(picked_channels, ordered=True)
    except TypeError:
        raw.pick(picked_channels)
        raw.reorder_channels(picked_channels)

    continuous_eeg = np.asarray(raw.get_data(), dtype=np.float32).copy()
    return raw, events, continuous_eeg, float(raw.info["sfreq"])


def _epoch_raw(
    config: ThingsRawDemoConfig,
    raw: mne.io.BaseRaw,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = mne.Epochs(
        raw,
        events,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=(None, 0),
        preload=True,
        verbose=False,
    )
    if config.target_sfreq < raw.info["sfreq"]:
        epochs.resample(config.target_sfreq, verbose=False)

    full_epochs = np.asarray(epochs.get_data(copy=True), dtype=np.float32)
    if full_epochs.ndim != 3:
        raise ValueError(f"Expected epoched data with shape (N, C, T), got {full_epochs.shape}.")

    end_index = config.crop_start_index + config.keep_samples
    if end_index > full_epochs.shape[-1]:
        raise ValueError(
            f"Crop [{config.crop_start_index}:{end_index}] exceeds epoch length {full_epochs.shape[-1]}."
        )
    cropped_epochs = np.asarray(full_epochs[:, :, config.crop_start_index:end_index], dtype=np.float32)
    event_codes = epochs.events[:, 2].astype(int)
    event_samples = epochs.events[:, 0].astype(int)
    return full_epochs, cropped_epochs, event_codes, event_samples


def _load_epoched_split(
    config: ThingsRawDemoConfig,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    raw, events, continuous_eeg, continuous_sfreq = _load_raw_split(config, split)
    full_epochs, cropped_epochs, event_codes, event_samples = _epoch_raw(config, raw, events)
    return full_epochs, cropped_epochs, event_codes, event_samples, continuous_eeg, continuous_sfreq


def _sort_session_epochs_by_condition(
    cropped_epochs: np.ndarray,
    event_codes: np.ndarray,
    max_rep: int,
) -> np.ndarray:
    img_cond = np.unique(event_codes)
    sorted_data = np.zeros(
        (len(img_cond), max_rep, cropped_epochs.shape[1], cropped_epochs.shape[2]),
        dtype=np.float32,
    )
    for i, event_code in enumerate(img_cond):
        idx = np.where(event_codes == event_code)[0]
        if idx.shape[0] < max_rep:
            raise ValueError(
                f"Event code {event_code} has only {idx.shape[0]} repetitions, expected at least {max_rep}."
            )
        selected = shuffle(idx, random_state=OFFLINE_PREPROCESSING_SEED, n_samples=max_rep)
        sorted_data[i] = cropped_epochs[selected]
    return sorted_data


def _whitening_cache_path(config: ThingsRawDemoConfig) -> Path:
    cache_dir = _output_root(config) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{config.subject}_{config.session}_training_mvnn_whitener_{config.target_sfreq}hz.npy"


def _compute_mvnn_whitener(
    cropped_epochs: np.ndarray,
    event_codes: np.ndarray,
    max_rep: int,
) -> np.ndarray:
    sorted_epochs = _sort_session_epochs_by_condition(cropped_epochs, event_codes, max_rep=max_rep)
    sigma_cond = np.empty(
        (sorted_epochs.shape[0], sorted_epochs.shape[2], sorted_epochs.shape[2]),
        dtype=np.float32,
    )
    for i in range(sorted_epochs.shape[0]):
        cond_data = sorted_epochs[i]
        sigma_cond[i] = np.mean(
            [_cov(cond_data[rep].T, shrinkage="auto") for rep in range(cond_data.shape[0])],
            axis=0,
        )
    sigma_tot = sigma_cond.mean(axis=0)
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)
    return np.asarray(np.real_if_close(sigma_inv), dtype=np.float32)


def load_session_whitener(config: ThingsRawDemoConfig) -> np.ndarray:
    cache_path = _whitening_cache_path(config)
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    train_cfg = replace(config, split="training")
    _, train_data, train_codes, _, _, _ = _load_epoched_split(train_cfg, "training")
    sigma_inv = _compute_mvnn_whitener(train_data, train_codes, max_rep=2)
    np.save(cache_path, sigma_inv)
    return sigma_inv


def apply_whitener(data: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    whitened = (data.transpose(0, 2, 1) @ sigma_inv).transpose(0, 2, 1)
    return np.asarray(whitened, dtype=np.float32)


def _calibration_cache_paths(config: ThingsRawDemoConfig) -> tuple[Path, Path]:
    cache_dir = _output_root(config) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{config.subject}_{config.session}_{config.calibration_split}_"
        f"calibration_{config.calibration_max_conditions}c_{config.calibration_repetitions_per_condition}r_"
        f"{config.target_sfreq}hz"
    )
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _select_calibration_trials(
    config: ThingsRawDemoConfig,
) -> tuple[list[ThingsTrial], np.ndarray, np.ndarray, dict[str, object]]:
    calibration_cfg = replace(
        config,
        split=config.calibration_split,
        apply_whitening=False,
        average_by_event=False,
        compare_to_offline_saved=False,
    )
    full_epochs, cropped_epochs, event_codes, event_samples, _, _ = _load_epoched_split(
        calibration_cfg,
        calibration_cfg.split,
    )
    image_index = build_image_index(config.data_root, calibration_cfg.split)

    eligible_codes: list[int] = []
    for event_code in np.unique(event_codes):
        count = int(np.sum(event_codes == event_code))
        if count >= config.calibration_repetitions_per_condition:
            eligible_codes.append(int(event_code))

    if len(eligible_codes) < config.calibration_max_conditions:
        raise ValueError(
            "Not enough repeated conditions for calibration: "
            f"requested {config.calibration_max_conditions}, found {len(eligible_codes)}."
        )

    selected_codes = eligible_codes[: config.calibration_max_conditions]
    selected_indices: list[int] = []
    trials: list[ThingsTrial] = []
    for condition_offset, event_code in enumerate(selected_codes):
        idx = np.where(event_codes == event_code)[0]
        chosen = shuffle(
            idx,
            random_state=OFFLINE_PREPROCESSING_SEED + condition_offset,
            n_samples=config.calibration_repetitions_per_condition,
        )
        chosen = np.sort(chosen)
        for repetition_offset, trial_index in enumerate(chosen):
            selected_indices.append(int(trial_index))
            trials.append(
                ThingsTrial(
                    trial_index=len(trials),
                    event_code=int(event_code),
                    epoch_full=np.asarray(full_epochs[trial_index], dtype=np.float32),
                    epoch_x250=np.asarray(cropped_epochs[trial_index], dtype=np.float32),
                    image_path=image_index.get(int(event_code)),
                    event_sample=int(event_samples[trial_index]),
                    repetition_count=1,
                )
            )

    selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
    metadata: dict[str, object] = {
        "subject": config.subject,
        "session": config.session,
        "split": calibration_cfg.split,
        "selected_conditions": list(selected_codes),
        "max_conditions": int(config.calibration_max_conditions),
        "repetitions_per_condition": int(config.calibration_repetitions_per_condition),
        "trial_count": int(selected_indices_np.shape[0]),
        "epoch_window_seconds": [float(config.tmin), float(config.tmax)],
    }
    return trials, cropped_epochs[selected_indices_np], event_codes[selected_indices_np], metadata


def load_or_build_calibration_block(config: ThingsRawDemoConfig) -> CalibrationBlock:
    cache_path, metadata_path = _calibration_cache_paths(config)
    trials, selected_epochs, selected_codes, metadata = _select_calibration_trials(config)

    if cache_path.exists():
        sigma_inv = np.load(cache_path).astype(np.float32)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        metadata["cache_path"] = str(cache_path)
        metadata["from_cache"] = True
        return CalibrationBlock(
            trials=trials,
            sigma_inv=sigma_inv,
            cache_path=cache_path,
            metadata_path=metadata_path,
            metadata=metadata,
            from_cache=True,
        )

    sigma_inv = _compute_mvnn_whitener(
        selected_epochs,
        selected_codes,
        max_rep=config.calibration_repetitions_per_condition,
    )
    metadata["cache_path"] = str(cache_path)
    metadata["from_cache"] = False
    np.save(cache_path, sigma_inv)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return CalibrationBlock(
        trials=trials,
        sigma_inv=sigma_inv,
        cache_path=cache_path,
        metadata_path=metadata_path,
        metadata=metadata,
        from_cache=False,
    )


def _load_saved_offline_preprocessed(config: ThingsRawDemoConfig) -> np.ndarray | None:
    saved_path = (
        config.data_root
        / "Preprocessed_data_250Hz"
        / config.subject
        / f"preprocessed_eeg_{config.split}.npy"
    )
    if not saved_path.exists():
        return None
    with open(saved_path, "rb") as handle:
        saved = pickle.load(handle)
    return np.asarray(saved["preprocessed_eeg_data"], dtype=np.float32)


def compare_trial_to_saved_offline(
    trial_epoch_x250: np.ndarray,
    event_code: int,
    saved_data: np.ndarray,
    repetition_count: int = 1,
) -> dict[str, object] | None:
    if saved_data.ndim != 4:
        return None
    condition_index = int(event_code) - 1
    if condition_index < 0 or condition_index >= saved_data.shape[0]:
        return None

    candidates = np.asarray(saved_data[condition_index], dtype=np.float32)
    diff = candidates - trial_epoch_x250[None, :, :]
    mae = np.mean(np.abs(diff), axis=(1, 2))
    rmse = np.sqrt(np.mean(diff**2, axis=(1, 2)))
    max_abs = np.max(np.abs(diff), axis=(1, 2))

    flat_trial = trial_epoch_x250.reshape(-1).astype(np.float64)
    flat_candidates = candidates.reshape(candidates.shape[0], -1).astype(np.float64)
    trial_norm = np.linalg.norm(flat_trial)
    candidate_norm = np.linalg.norm(flat_candidates, axis=1)
    denom = np.maximum(candidate_norm * max(trial_norm, 1e-12), 1e-12)
    cosine = np.sum(flat_candidates * flat_trial[None, :], axis=1) / denom

    best_idx = int(np.argmin(rmse))
    comparison: dict[str, object] = {
        "condition_index": condition_index,
        "available_repetitions": int(candidates.shape[0]),
        "closest_repetition_index": best_idx,
        "closest_mae": float(mae[best_idx]),
        "closest_rmse": float(rmse[best_idx]),
        "closest_max_abs": float(max_abs[best_idx]),
        "closest_cosine_similarity": float(cosine[best_idx]),
    }
    if repetition_count > 1:
        mean_candidate = np.mean(candidates, axis=0)
        mean_diff = mean_candidate - trial_epoch_x250
        mean_norm = np.linalg.norm(mean_candidate.reshape(-1).astype(np.float64))
        mean_cosine = float(
            np.sum(mean_candidate.reshape(-1).astype(np.float64) * flat_trial)
            / max(mean_norm * max(trial_norm, 1e-12), 1e-12)
        )
        comparison["mean_reference"] = {
            "mae": float(np.mean(np.abs(mean_diff))),
            "rmse": float(np.sqrt(np.mean(mean_diff**2))),
            "max_abs": float(np.max(np.abs(mean_diff))),
            "cosine_similarity": mean_cosine,
        }
    return comparison


def load_raw_things_session(
    config: ThingsRawDemoConfig,
    whitener: np.ndarray | None = None,
) -> ThingsReplaySession:
    """Load continuous EEG and true stimulus-locked THINGS trials from raw data."""
    full_epochs, data, event_codes, event_samples, continuous_eeg, continuous_sfreq = _load_epoched_split(
        config,
        config.split,
    )
    if config.apply_whitening:
        sigma_inv = whitener
        if sigma_inv is None:
            if config.calibration_enabled:
                calibration_block = load_or_build_calibration_block(config)
                sigma_inv = calibration_block.sigma_inv
            else:
                sigma_inv = load_session_whitener(config)
        data = apply_whitener(data, sigma_inv)
    image_index = build_image_index(config.data_root, config.split)
    offline_saved = _load_saved_offline_preprocessed(config) if config.compare_to_offline_saved else None

    if config.average_by_event:
        grouped: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
        unique_codes = np.unique(event_codes)
        for event_code in unique_codes:
            idx = np.where(event_codes == event_code)[0]
            grouped.append(
                (
                    int(event_code),
                    np.asarray(full_epochs[idx].mean(axis=0), dtype=np.float32),
                    np.asarray(data[idx].mean(axis=0), dtype=np.float32),
                    int(len(idx)),
                    int(event_samples[idx[0]]),
                )
            )

        trials = [
            ThingsTrial(
                trial_index=trial_index,
                event_code=event_code,
                epoch_full=epoch_full,
                epoch_x250=epoch_x250,
                image_path=image_index.get(event_code),
                event_sample=event_sample,
                repetition_count=repetition_count,
            )
            for trial_index, (event_code, epoch_full, epoch_x250, repetition_count, event_sample) in enumerate(grouped)
        ]
        return ThingsReplaySession(
            continuous_eeg=continuous_eeg,
            continuous_sfreq=continuous_sfreq,
            trials=trials,
            offline_saved_data=offline_saved,
        )

    trials: list[ThingsTrial] = []
    for trial_index, (event_code, epoch_full, epoch, event_sample) in enumerate(
        zip(event_codes, full_epochs, data, event_samples)
    ):
        trials.append(
            ThingsTrial(
                trial_index=trial_index,
                event_code=int(event_code),
                epoch_full=np.asarray(epoch_full, dtype=np.float32),
                epoch_x250=np.asarray(epoch, dtype=np.float32),
                image_path=image_index.get(int(event_code)),
                event_sample=int(event_sample),
                repetition_count=1,
            )
        )
    return ThingsReplaySession(
        continuous_eeg=continuous_eeg,
        continuous_sfreq=continuous_sfreq,
        trials=trials,
        offline_saved_data=offline_saved,
    )


def load_raw_things_trials(config: ThingsRawDemoConfig) -> list[ThingsTrial]:
    return load_raw_things_session(config).trials


class ThingsRawDemoRunner:
    """Replay raw THINGS EEG with true stimulus onsets for demo reconstruction."""

    def __init__(
        self,
        config: ThingsRawDemoConfig,
        encoder: LowLevelEpochEncoder,
        decoder: LowLevelVAEDecoder,
        worker: Optional[SemanticRefinementWorker],
        monitor: RuntimeMonitorState | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.worker = worker
        self.monitor = monitor
        self._stop_event = threading.Event()
        self._plot_lock = threading.Lock()
        self._plot_data: dict[str, object] = {
            "sampling_rate": None,
            "window_seconds": None,
            "traces": [],
            "markers": [],
        }
        self._offline_saved_data: np.ndarray | None = None
        self._calibration_block: CalibrationBlock | None = None

        self.output_root = config.output_root or Path.cwd() / "outputs" / "things_raw_demo"
        self.low_dir = self.output_root / "low_level"
        self.high_dir = self.output_root / "high_level"
        self.target_dir = self.output_root / "targets"
        self.meta_dir = self.output_root / "metadata"
        for directory in [self.low_dir, self.high_dir, self.target_dir, self.meta_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        if self.config.calibration_enabled:
            self._run_calibration_block()
        session = load_raw_things_session(
            self.config,
            whitener=None if self._calibration_block is None else self._calibration_block.sigma_inv,
        )
        self._offline_saved_data = session.offline_saved_data
        trials = session.trials
        selected = trials[self.config.start_trial :]
        if self.config.max_trials is not None:
            selected = selected[: self.config.max_trials]
        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=True,
                eeg_connected=True,
                marker_connected=True,
                mode="things-replay",
                message="Starting THINGS replay.",
            )

        print(
            f"Loaded {len(trials)} trials from {self.config.subject}/{self.config.session}/{self.config.split}; "
            f"running {len(selected)} trial(s)."
        )

        for trial in selected:
            if self._stop_event.is_set():
                break
            self._replay_trial_signal(session, trial)
            self._run_trial(trial)
            if self.config.sleep_seconds > 0:
                time.sleep(self.config.sleep_seconds)
        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=False,
                eeg_connected=False,
                marker_connected=False,
                message="Replay complete.",
            )
        return 0

    def _run_calibration_block(self) -> None:
        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=True,
                eeg_connected=True,
                marker_connected=True,
                mode="things-replay",
                message="Preparing calibration block.",
            )
        calibration_block = load_or_build_calibration_block(self.config)
        self._calibration_block = calibration_block
        calibration_session = ThingsReplaySession(
            continuous_eeg=np.empty((0, 0), dtype=np.float32),
            continuous_sfreq=float(self.config.target_sfreq),
            trials=calibration_block.trials,
            offline_saved_data=None,
        )

        print(
            "Calibration block: "
            f"{len(calibration_block.trials)} trial(s), "
            f"{self.config.calibration_max_conditions} condition(s), "
            f"{self.config.calibration_repetitions_per_condition} repetition(s) each, "
            f"cache={'hit' if calibration_block.from_cache else 'miss'}."
        )

        for trial in calibration_block.trials:
            if self._stop_event.is_set():
                return
            self._run_calibration_trial(calibration_session, trial)
            if self.config.calibration_sleep_seconds > 0:
                time.sleep(self.config.calibration_sleep_seconds)

        if self.monitor is not None:
            self.monitor.set_status(
                engine_running=True,
                eeg_connected=True,
                marker_connected=True,
                mode="things-replay",
                message="Calibration complete. Starting replay.",
            )

    def _run_calibration_trial(self, session: ThingsReplaySession, trial: ThingsTrial) -> None:
        self._set_epoch_plot_data(trial.epoch_full, sfreq=float(self.config.target_sfreq))
        if self.monitor is not None:
            self.monitor.add_marker(
                {
                    "timestamp": float(trial.trial_index),
                    "label": f"calibration {trial.event_code}",
                    "event_name": str(trial.event_code),
                    "raw_value": str(trial.event_code),
                    "status": "calibration",
                }
            )
            self.monitor.set_latest_epoch(
                trial.epoch_full,
                sampling_rate=self.config.target_sfreq,
                start_seconds=self.config.tmin,
            )
            if trial.image_path is not None:
                self.monitor.set_latest_target(
                    trial.image_path,
                    {
                        "image_id": str(trial.event_code),
                        "label": trial.image_path.parent.name,
                        "phase": "calibration",
                    },
                )
        print(
            f"calibration trial {trial.trial_index:05d} event={trial.event_code} target={trial.image_path}"
        )

    def _set_epoch_plot_data(self, epoch: np.ndarray, sfreq: float) -> None:
        if epoch.size == 0:
            return
        data = np.asarray(epoch[:8], dtype=np.float32)
        data = data - data.mean(axis=1, keepdims=True)
        channel_std = np.std(data, axis=1, keepdims=True)
        channel_std[channel_std < 1e-8] = 1.0
        data = data / channel_std
        n_samples = int(data.shape[1])
        times = (
            np.arange(n_samples, dtype=np.float32) / float(sfreq)
        ) + float(self.config.tmin)
        scale = 6.0

        traces = []
        for idx, channel in enumerate(data):
            traces.append(
                {
                    "name": f"Ch {idx + 1}",
                    "x": times.tolist(),
                    "y": (channel + idx * scale).astype(np.float32).tolist(),
                }
            )

        with self._plot_lock:
            self._plot_data = {
                "sampling_rate": float(sfreq),
                "window_seconds": n_samples / float(sfreq),
                "traces": traces,
                "markers": [{"x": 0.0, "label": "stim onset", "status": "calibration"}],
            }

    def stop(self) -> None:
        self._stop_event.set()

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, object]:
        with self._plot_lock:
            sampling_rate = self._plot_data["sampling_rate"]
            window_seconds = self._plot_data["window_seconds"]
            traces = [
                {
                    "name": trace["name"],
                    "x": list(trace["x"]),
                    "y": list(trace["y"]),
                }
                for trace in self._plot_data["traces"][:max_channels]
            ]
            markers = [dict(marker) for marker in self._plot_data["markers"]]
        return {
            "sampling_rate": sampling_rate,
            "window_seconds": window_seconds,
            "traces": traces,
            "markers": markers,
        }

    def _replay_trial_signal(self, session: ThingsReplaySession, trial: ThingsTrial) -> None:
        if self._stop_event.is_set():
            return
        sfreq = float(session.continuous_sfreq)
        if sfreq <= 0 or session.continuous_eeg.size == 0:
            return

        pre_roll_seconds = 3.0
        post_roll_seconds = 1.5
        window_seconds = 6.0
        step_seconds = 0.05
        playback_speed = 4.0

        total_samples = int(session.continuous_eeg.shape[1])
        window_samples = max(int(round(window_seconds * sfreq)), 1)
        min_window_samples = max(int(round(0.75 * sfreq)), 1)
        step_samples = max(int(round(step_seconds * sfreq)), 1)

        segment_start = max(0, trial.event_sample - int(round(pre_roll_seconds * sfreq)))
        segment_end = min(total_samples, trial.event_sample + int(round(post_roll_seconds * sfreq)))
        cursor_start = max(segment_start + min_window_samples, min_window_samples)
        cursor_end = max(cursor_start, segment_end)

        for cursor in range(cursor_start, cursor_end + 1, step_samples):
            if self._stop_event.is_set():
                return
            start = max(0, cursor - window_samples)
            window = session.continuous_eeg[:, start:cursor]
            self._set_plot_data(window, sfreq, trial.event_sample, cursor)
            time.sleep(step_seconds / playback_speed)

    def _set_plot_data(self, window: np.ndarray, sfreq: float, event_sample: int, cursor: int) -> None:
        if window.size == 0:
            return
        data = np.asarray(window[:8], dtype=np.float32)
        data = data - data.mean(axis=1, keepdims=True)
        channel_std = np.std(data, axis=1, keepdims=True)
        channel_std[channel_std < 1e-8] = 1.0
        data = data / channel_std
        n_samples = int(data.shape[1])
        times = np.linspace(-n_samples / sfreq, 0.0, n_samples, endpoint=False, dtype=np.float32)
        scale = 6.0

        traces = []
        for idx, channel in enumerate(data):
            traces.append(
                {
                    "name": f"Ch {idx + 1}",
                    "x": times.tolist(),
                    "y": (channel + idx * scale).astype(np.float32).tolist(),
                }
            )

        marker_x = (float(event_sample) - float(cursor)) / float(sfreq)
        markers: list[dict[str, object]] = []
        if float(times[0]) <= marker_x <= 0.0:
            markers.append({"x": marker_x, "label": "stim onset", "status": "accepted"})

        with self._plot_lock:
            self._plot_data = {
                "sampling_rate": float(sfreq),
                "window_seconds": n_samples / float(sfreq),
                "traces": traces,
                "markers": markers,
            }

    def _run_trial(self, trial: ThingsTrial) -> None:
        tensor = torch.from_numpy(trial.epoch_x250).unsqueeze(0).to(self.encoder.device_name)
        with torch.inference_mode():
            latent = self.encoder(tensor)
            image = self.decoder(latent).astype(np.uint8)

        stem = self._stem_for_trial(trial)
        low_path = self.low_dir / f"{stem}_low.png"
        Image.fromarray(image).save(low_path)

        metadata = {
            "trial_index": int(trial.trial_index),
            "event_code": int(trial.event_code),
            "repetition_count": int(trial.repetition_count),
            "subject": self.config.subject,
            "session": self.config.session,
            "split": self.config.split,
            "epoch_window_seconds": [float(self.config.tmin), float(self.config.tmax)],
            "full_epoch_shape": list(trial.epoch_full.shape),
            "model_epoch_shape": list(trial.epoch_x250.shape),
            "image_path": str(trial.image_path) if trial.image_path else None,
        }
        offline_match = compare_trial_to_saved_offline(
            trial.epoch_x250,
            trial.event_code,
            self._offline_saved_data,
            repetition_count=trial.repetition_count,
        ) if self._offline_saved_data is not None else None
        if offline_match is not None:
            metadata["offline_saved_match"] = offline_match
        (self.meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))

        target_copy: Path | None = None
        if trial.image_path is not None and self.config.copy_targets:
            target_copy = self.target_dir / f"{stem}_target{trial.image_path.suffix}"
            shutil.copy2(trial.image_path, target_copy)

        if self.monitor is not None:
            self.monitor.add_marker(
                {
                    "timestamp": float(trial.trial_index),
                    "label": f"event {trial.event_code}",
                    "event_name": str(trial.event_code),
                    "raw_value": str(trial.event_code),
                    "status": "accepted",
                }
            )
            self.monitor.set_latest_epoch(
                trial.epoch_full,
                sampling_rate=self.config.target_sfreq,
                start_seconds=self.config.tmin,
            )
            self.monitor.set_latest_low_level(
                low_path,
                {
                    "event_code": int(trial.event_code),
                    "marker_label": f"event {trial.event_code}",
                    "event_time": f"trial {trial.trial_index:05d}",
                    "stem": stem,
                },
            )
            if trial.image_path is not None:
                self.monitor.set_latest_target(
                    trial.image_path,
                    {
                        "image_id": str(trial.event_code),
                        "label": trial.image_path.parent.name,
                        "copy_path": None if target_copy is None else str(target_copy),
                    },
                )

        x250 = self.encoder.snapshot_last_x250()
        if self.worker is not None and x250 is not None:
            self.worker.submit(
                stem=stem,
                x250=x250,
                low_level_image=image,
                text_prompt=self._text_prompt_for_trial(trial),
            )

        print(
            f"trial {trial.trial_index:05d} event={trial.event_code} "
            f"repetitions={trial.repetition_count} target={trial.image_path}"
            + (
                ""
                if offline_match is None
                else f" closest_offline_rmse={offline_match['closest_rmse']:.6f}"
            )
        )

    def _stem_for_trial(self, trial: ThingsTrial) -> str:
        return (
            f"{self.config.subject}_{self.config.session}_{self.config.split}_"
            f"trial{trial.trial_index:05d}_event{trial.event_code:05d}"
        )

    def _text_prompt_for_trial(self, trial: ThingsTrial) -> str | None:
        if trial.image_path is None:
            return None
        parent_name = trial.image_path.parent.name
        if "_" in parent_name:
            return parent_name.split("_", 1)[1]
        return parent_name
