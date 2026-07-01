from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image

from maryam_rt.gui.monitor import RuntimeMonitorState

if TYPE_CHECKING:
    from maryam_rt.integration.low_level import LowLevelEpochEncoder, LowLevelVAEDecoder
    from maryam_rt.integration.worker import SemanticRefinementWorker


THINGS_63_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]


STARSTIM_31_CHANNELS = [
    "P8", "T8", "CP6", "FC6", "F8", "F4", "C4", "P4",
    "AF4", "Fp2", "Fp1", "AF3", "FC2", "Cz", "CP2",
    "PO3", "O1", "Oz", "O2", "PO4", "Pz", "CP1", "FC1",
    "P3", "C3", "F3", "F7", "FC5", "CP5", "T7", "P7",
]


@dataclass(frozen=True)
class LabReplayConfig:
    data_root: Path
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


@dataclass(frozen=True)
class LabTrial:
    trial_index: int
    epoch_index: int
    target_index: int
    trigger: int
    subject: str
    category: str
    split: str
    epoch_lab31: np.ndarray
    epoch_model: np.ndarray
    image_path: Optional[Path]
    usable_for_finetune: bool = True


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _default_epoch_path(data_root: Path) -> Path:
    return data_root / "derived" / "lab_starstim31_epochs_250hz.npz"


def _default_target_manifest(data_root: Path) -> Path:
    return data_root / "derived" / "finetune" / "lab_target_manifest.tsv"


def _default_split_manifest(data_root: Path) -> Path:
    return data_root / "derived" / "finetune" / "lab_finetune_split_manifest.tsv"


def _output_root(config: LabReplayConfig) -> Path:
    return config.output_root or Path.cwd() / "outputs" / "lab_replay"


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


def load_lab_trials(config: LabReplayConfig) -> list[LabTrial]:
    epoch_path = config.epoch_npz or _default_epoch_path(config.data_root)
    target_path = config.target_manifest or _default_target_manifest(config.data_root)
    split_path = config.split_manifest or _default_split_manifest(config.data_root)

    if not epoch_path.exists():
        raise FileNotFoundError(f"Lab epoch file not found: {epoch_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Lab target manifest not found: {target_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Lab split manifest not found: {split_path}")

    npz = np.load(epoch_path, allow_pickle=True)
    eeg = np.asarray(npz["eeg"], dtype=np.float32)
    ch_names = npz["ch_names"].astype(str).tolist()
    if eeg.ndim != 3:
        raise ValueError(f"Expected lab EEG shaped (N, C, T), got {eeg.shape}.")
    if eeg.shape[1:] != (31, 250):
        raise ValueError(
            f"Expected model-ready lab epochs shaped (N, 31, 250), got {eeg.shape}."
        )

    target_rows = _read_tsv(target_path)
    target_by_index = {int(row["target_index"]): row for row in target_rows}
    split_rows = _read_tsv(split_path)
    if len(split_rows) != eeg.shape[0]:
        raise ValueError(f"Split manifest rows ({len(split_rows)}) do not match EEG epochs ({eeg.shape[0]}).")

    trials: list[LabTrial] = []
    for row in split_rows:
        epoch_index = int(row["epoch_index"])
        if epoch_index < 0 or epoch_index >= eeg.shape[0]:
            raise ValueError(f"Epoch index {epoch_index} is outside EEG array with {eeg.shape[0]} rows.")

        split = row.get("split", "")
        subject = row.get("subject_folder", "")
        if config.split != "all" and split != config.split:
            continue
        if config.subject is not None and subject != config.subject:
            continue

        target_index = int(row["target_index"])
        target_row = target_by_index.get(target_index, {})
        epoch_lab31 = eeg[epoch_index]
        if config.adapter == "starstim31-to-things63":
            epoch_model = adapt_starstim31_to_things63(epoch_lab31, ch_names)
        elif config.adapter == "none":
            epoch_model = epoch_lab31
        else:
            raise ValueError(f"Unsupported lab adapter: {config.adapter}")

        trials.append(
            LabTrial(
                trial_index=len(trials),
                epoch_index=epoch_index,
                target_index=target_index,
                trigger=int(row["trigger"]),
                subject=subject,
                category=row.get("category") or target_row.get("category", ""),
                split=split,
                epoch_lab31=epoch_lab31,
                epoch_model=epoch_model,
                image_path=_resolve_image_path({**target_row, **row}),
                usable_for_finetune=str(row.get("usable_for_finetune", "true")).lower() == "true",
            )
        )
    return trials


class LabReplayRunner:
    """Replay manifest-aligned native Starstim lab epochs through the realtime GUI pipeline."""

    def __init__(
        self,
        config: LabReplayConfig,
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

        self.output_root = _output_root(config)
        self.low_dir = self.output_root / "low_level"
        self.high_dir = self.output_root / "high_level"
        self.target_dir = self.output_root / "targets"
        self.meta_dir = self.output_root / "metadata"
        for directory in [self.low_dir, self.high_dir, self.target_dir, self.meta_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> int:
        trials = load_lab_trials(self.config)
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
            f"with adapter={self.config.adapter}."
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
                    "status": "accepted" if trial.usable_for_finetune else "artifact_review",
                    "image_id": str(trial.target_index),
                    "subject": trial.subject,
                    "split": trial.split,
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
        with torch.inference_mode():
            latent = self.encoder(input_tensor)
            image = self.decoder(latent).astype(np.uint8)

        low_path = self.low_dir / f"{stem}_low.png"
        Image.fromarray(image).save(low_path)
        metadata = {
            "mode": "lab-replay",
            "adapter": self.config.adapter,
            "trial_index": trial.trial_index,
            "epoch_index": trial.epoch_index,
            "target_index": trial.target_index,
            "trigger": trial.trigger,
            "subject": trial.subject,
            "category": trial.category,
            "split": trial.split,
            "usable_for_finetune": trial.usable_for_finetune,
            "epoch_lab31_shape": list(trial.epoch_lab31.shape),
            "epoch_model_shape": list(trial.epoch_model.shape),
            "image_path": None if trial.image_path is None else str(trial.image_path),
            "low_level_path": str(low_path),
        }
        if trial.image_path is not None and self.config.copy_targets:
            target_copy = self.target_dir / f"{stem}_target{trial.image_path.suffix}"
            shutil.copy2(trial.image_path, target_copy)
            metadata["target_copy_path"] = str(target_copy)
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
                },
            )

        x250 = self.encoder.snapshot_last_x250()
        if self.worker is not None and x250 is not None:
            prompt = trial.category.replace("_", " ") if trial.category else None
            self.worker.submit(stem=stem, x250=x250, low_level_image=image, text_prompt=prompt)
