from __future__ import annotations

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
