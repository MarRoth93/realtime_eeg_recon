from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


THINGS_63_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

STARSTIM_32_CHANNELS = [
    "P8", "T8", "CP6", "FC6", "F8", "F4", "C4", "P4",
    "AF4", "Fp2", "Fp1", "AF3", "Fz", "FC2", "Cz", "CP2",
    "PO3", "O1", "Oz", "O2", "PO4", "Pz", "CP1", "FC1",
    "P3", "C3", "F3", "F7", "FC5", "CP5", "T7", "P7",
]

STARSTIM_31_CHANNELS = [channel for channel in STARSTIM_32_CHANNELS if channel != "Fz"]


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


def starstim31_to_things63_torch(x31: torch.Tensor) -> torch.Tensor:
    """Map batched Starstim31 tensors from (B, 31, T) into THINGS63 channel slots."""
    if x31.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x31.shape)}.")
    if x31.shape[1] != len(STARSTIM_31_CHANNELS):
        raise ValueError(f"Expected {len(STARSTIM_31_CHANNELS)} Starstim31 channels, got {x31.shape[1]}.")

    things_index = {name: idx for idx, name in enumerate(THINGS_63_CHANNELS)}
    out = x31.new_zeros((x31.shape[0], len(THINGS_63_CHANNELS), x31.shape[2]))
    for lab_idx, channel in enumerate(STARSTIM_31_CHANNELS):
        out[:, things_index[channel], :] = x31[:, lab_idx, :]
    return out


def downsample_starstim_window_torch(
    x: torch.Tensor,
    downsample_factor: int = 4,
) -> torch.Tensor:
    """Downsample batched native Starstim tensors without changing channel layout."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}.")
    if x.shape[1] not in {len(STARSTIM_31_CHANNELS), len(STARSTIM_32_CHANNELS)}:
        raise ValueError(
            f"Expected {len(STARSTIM_31_CHANNELS)} or {len(STARSTIM_32_CHANNELS)} "
            f"Starstim channels, got {x.shape[1]}."
        )
    return F.avg_pool1d(x, kernel_size=downsample_factor, stride=downsample_factor)


def drop_fz_starstim32_torch(x: torch.Tensor) -> torch.Tensor:
    """Drop Fz from batched Starstim32 tensors to match the current lab epoch artifact."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}.")
    if x.shape[1] != len(STARSTIM_32_CHANNELS):
        raise ValueError(f"Expected {len(STARSTIM_32_CHANNELS)} Starstim32 channels, got {x.shape[1]}.")
    keep_indices = [idx for idx, channel in enumerate(STARSTIM_32_CHANNELS) if channel != "Fz"]
    return x[:, keep_indices, :]


def prepare_starstim32_native_window_torch(
    x: torch.Tensor,
    downsample_factor: int = 4,
    drop_fz: bool = False,
) -> torch.Tensor:
    """Convert Starstim32 realtime tensors from (B, 32, 1000) to native 250 Hz lab input."""
    if drop_fz:
        x = drop_fz_starstim32_torch(x)
    elif x.ndim != 3 or x.shape[1] != len(STARSTIM_32_CHANNELS):
        shape = tuple(x.shape) if x.ndim == 3 else tuple(x.shape)
        raise ValueError(f"Expected batched Starstim32 tensor shaped (B, 32, T), got {shape}.")
    return downsample_starstim_window_torch(x, downsample_factor=downsample_factor)


def prepare_starstim32_things63_window_torch(
    x: torch.Tensor,
    downsample_factor: int = 4,
) -> torch.Tensor:
    """Compatibility path: convert Starstim32 live tensors into THINGS63 model slots."""
    x31 = drop_fz_starstim32_torch(x)
    x31 = downsample_starstim_window_torch(x31, downsample_factor=downsample_factor)
    return starstim31_to_things63_torch(x31)
