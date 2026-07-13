from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image

from maryam_rt.hierarchical.atms_pipeline.atms_utils import encoder_low_level, encoder_low_level_transformed
from maryam_rt.integration.resample import (
    prepare_realtime_window_torch,
    prepare_starstim32_native_window_torch,
    prepare_starstim32_things63_window_torch,
)


_LEGACY_LOW_LEVEL_MISSING_KEYS = {
    "upsampler.25.weight",
    "upsampler.25.bias",
    "upsampler.25.running_mean",
    "upsampler.25.running_var",
}


def _load_low_level_state(model: nn.Module, checkpoint_path: str | Path, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        result = model.load_state_dict(state, strict=False)
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if unexpected or not missing.issubset(_LEGACY_LOW_LEVEL_MISSING_KEYS):
            raise


class LowLevelRealtimeEncoder(nn.Module):
    """Fast path encoder for realtime low-level image reconstruction."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_num_channels: int = 63,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.model_num_channels = model_num_channels
        self.model = encoder_low_level(num_channels=model_num_channels).to(device)
        _load_low_level_state(self.model, checkpoint_path, device)
        self.model.eval()
        self._lock = threading.Lock()
        self._last_x250: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x250 = prepare_realtime_window_torch(x)
        latent = self.model(x250)
        with self._lock:
            self._last_x250 = x250.detach().cpu()
        return latent

    def snapshot_last_x250(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._last_x250 is None:
                return None
            return self._last_x250.clone()


class LowLevelStarstimNativeRealtimeEncoder(nn.Module):
    """Native encoder for Starstim32 lab streams and a matching lab checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_num_channels: int = 32,
    ) -> None:
        super().__init__()
        if model_num_channels not in {31, 32}:
            raise ValueError(f"Native Starstim model must use 31 or 32 channels, got {model_num_channels}.")
        self.device_name = device
        self.model_num_channels = model_num_channels
        self.drop_fz = model_num_channels == 31
        self.model = encoder_low_level(num_channels=model_num_channels).to(device)
        _load_low_level_state(self.model, checkpoint_path, device)
        self.model.eval()
        self._lock = threading.Lock()
        self._last_x250: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x250 = prepare_starstim32_native_window_torch(x, drop_fz=self.drop_fz)
        latent = self.model(x250)
        with self._lock:
            self._last_x250 = x250.detach().cpu()
        return latent

    def snapshot_last_x250(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._last_x250 is None:
                return None
            return self._last_x250.clone()


class LowLevelStarstimThingsAdapterRealtimeEncoder(nn.Module):
    """Compatibility encoder for Starstim32 lab streams adapted into THINGS63 model input."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_num_channels: int = 63,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.model_num_channels = model_num_channels
        self.model = encoder_low_level(num_channels=model_num_channels).to(device)
        _load_low_level_state(self.model, checkpoint_path, device)
        self.model.eval()
        self._lock = threading.Lock()
        self._last_x250: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x250 = prepare_starstim32_things63_window_torch(x)
        latent = self.model(x250)
        with self._lock:
            self._last_x250 = x250.detach().cpu()
        return latent

    def snapshot_last_x250(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._last_x250 is None:
                return None
            return self._last_x250.clone()


class LowLevelEpochEncoder(nn.Module):
    """Low-level encoder for already-preprocessed epochs shaped (B, C, 250)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_num_channels: int = 63,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.model_num_channels = model_num_channels
        self.model = encoder_low_level(num_channels=model_num_channels).to(device)
        _load_low_level_state(self.model, checkpoint_path, device)
        self.model.eval()
        self._lock = threading.Lock()
        self._last_x250: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, x250: torch.Tensor) -> torch.Tensor:
        if x250.ndim != 3:
            raise ValueError(f"Expected epoch tensor shaped (B, C, T), got {tuple(x250.shape)}.")
        if x250.shape[1] != self.model_num_channels:
            raise ValueError(
                f"Encoder was initialized for {self.model_num_channels} channels, "
                f"but received {x250.shape[1]} channels."
            )
        latent = self.model(x250)
        with self._lock:
            self._last_x250 = x250.detach().cpu()
        return latent

    def snapshot_last_x250(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._last_x250 is None:
                return None
            return self._last_x250.clone()


class LowLevelTransformedEpochEncoder(nn.Module):
    """Low-level encoder for transformed Starstim31 VAE checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        model_num_channels: int = 31,
        subject_id: int | None = None,
    ) -> None:
        super().__init__()
        if model_num_channels != 31:
            raise ValueError(f"Transformed lab low-level checkpoint expects 31 channels, got {model_num_channels}.")
        self.device_name = device
        self.model_num_channels = model_num_channels
        self.fixed_subject_id = subject_id
        self.current_subject_id = subject_id
        self.model = encoder_low_level_transformed(num_channels=model_num_channels).to(device)
        _load_low_level_state(self.model, checkpoint_path, device)
        self.model.eval()
        self._lock = threading.Lock()
        self._last_x250: Optional[torch.Tensor] = None

    def set_subject_id(self, subject_id: int | None) -> None:
        if self.fixed_subject_id is None:
            self.current_subject_id = subject_id

    @torch.inference_mode()
    def forward(self, x250: torch.Tensor) -> torch.Tensor:
        if x250.ndim != 3:
            raise ValueError(f"Expected epoch tensor shaped (B, C, T), got {tuple(x250.shape)}.")
        if x250.shape[1] != self.model_num_channels:
            raise ValueError(
                f"Encoder was initialized for {self.model_num_channels} channels, "
                f"but received {x250.shape[1]} channels."
            )
        subject_id = 0 if self.current_subject_id is None else self.current_subject_id
        subject_ids = torch.full(
            (x250.shape[0],),
            int(subject_id),
            dtype=torch.long,
            device=x250.device,
        )
        latent = self.model(x250, subject_ids=subject_ids)
        with self._lock:
            self._last_x250 = x250.detach().cpu()
        return latent

    def snapshot_last_x250(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._last_x250 is None:
                return None
            return self._last_x250.clone()


class LowLevelVAEDecoder:
    """Decode low-level VAE latents directly into an image for the fast path."""

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
        scaled_latents: bool = False,
    ) -> None:
        self.device = device
        self.dtype = torch_dtype
        self.scaled_latents = scaled_latents
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae",
            torch_dtype=torch_dtype,
        ).to(device)
        self.vae.eval()

    @torch.inference_mode()
    def __call__(self, latent: torch.Tensor) -> np.ndarray:
        latent = latent.to(self.device, dtype=self.dtype)
        if self.scaled_latents:
            latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)[0].cpu()
        return np.array(to_pil_image(image))
