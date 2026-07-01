from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image

from maryam_rt.hierarchical.atms_pipeline.atms_utils import encoder_low_level
from maryam_rt.integration.resample import (
    prepare_realtime_window_torch,
    prepare_starstim32_native_window_torch,
    prepare_starstim32_things63_window_torch,
)


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
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
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
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
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
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
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
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
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


class LowLevelVAEDecoder:
    """Decode low-level VAE latents directly into an image for the fast path."""

    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = torch_dtype
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae",
            torch_dtype=torch_dtype,
        ).to(device)
        self.vae.eval()

    @torch.inference_mode()
    def __call__(self, latent: torch.Tensor) -> np.ndarray:
        latent = latent.to(self.device, dtype=self.dtype)
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)[0].cpu()
        return np.array(to_pil_image(image))
