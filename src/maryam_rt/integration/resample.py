from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def drop_last_channel_np(data: np.ndarray, keep_channels: int = 63) -> np.ndarray:
    """Keep the first `keep_channels` channels from a (C, T) EEG array."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape}.")
    return np.ascontiguousarray(data[:keep_channels], dtype=np.float32)


def mean_downsample_np(data: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample a (C, T) EEG array by averaging groups of `factor` samples."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape}.")
    n_channels, n_samples = data.shape
    trimmed = n_samples - (n_samples % factor)
    if trimmed == 0:
        raise ValueError(f"Not enough samples ({n_samples}) for factor {factor}.")
    return data[:, :trimmed].reshape(n_channels, -1, factor).mean(axis=2).astype(np.float32)


def prepare_realtime_window_np(
    data: np.ndarray,
    keep_channels: int = 63,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Convert 64x1000 realtime input into 63x250 model input."""
    return mean_downsample_np(drop_last_channel_np(data, keep_channels), downsample_factor)


def prepare_realtime_window_torch(
    x: torch.Tensor,
    keep_channels: int = 63,
    downsample_factor: int = 4,
) -> torch.Tensor:
    """Convert batched realtime tensors from (B, 64, 1000) to (B, 63, 250)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}.")
    x = x[:, :keep_channels, :]
    return F.avg_pool1d(x, kernel_size=downsample_factor, stride=downsample_factor)
