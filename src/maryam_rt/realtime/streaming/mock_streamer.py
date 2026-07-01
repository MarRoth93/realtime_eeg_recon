"""Mock LSL streamer for the THINGS EEG dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
from pylsl import StreamInfo, StreamOutlet

DEFAULT_CHANNEL_LABELS: List[str] = [
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]

PREFERRED_DATA_KEYS: Sequence[str] = ("data", "eeg", "EEG", "signals", "signal", "x", "X")


def _select_data_array(container: Mapping[str, np.ndarray], data_key: Optional[str]) -> np.ndarray:
    if data_key:
        if data_key not in container:
            raise KeyError(f"Data key '{data_key}' not found. Keys: {sorted(container.keys())}")
        return container[data_key]

    for key in PREFERRED_DATA_KEYS:
        if key in container:
            return container[key]

    keys = [key for key in container.keys() if not key.startswith("__")]
    if not keys:
        raise KeyError("No data arrays found in file.")
    return container[sorted(keys)[0]]


def _load_eeg_file(file_path: Path, data_key: Optional[str]) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        # Try loading as plain array first, then as pickled dict (THINGS-EEG2 format)
        try:
            return np.load(file_path)
        except ValueError:
            data = np.load(file_path, allow_pickle=True).item()
            if isinstance(data, dict):
                # THINGS-EEG2 format: dict with 'raw_eeg_data' key
                if data_key and data_key in data:
                    return data[data_key]
                if "raw_eeg_data" in data:
                    return data["raw_eeg_data"]
                return _select_data_array(data, data_key)
            return data
    if suffix == ".npz":
        with np.load(file_path) as npz_file:
            return _select_data_array({key: npz_file[key] for key in npz_file.files}, data_key)
    if suffix == ".mat":
        from scipy.io import loadmat

        mat_data = loadmat(file_path)
        return _select_data_array(mat_data, data_key)
    if suffix == ".vhdr":
        # BrainVision format - load with MNE
        import mne
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        # Return data in (samples, channels) format
        return raw.get_data().T
    if suffix == ".edf":
        # EDF format - load with MNE
        import mne
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw.get_data().T
    if suffix == ".fif":
        # MNE FIF format
        import mne
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        return raw.get_data().T
    if suffix == ".set":
        # EEGLAB format
        import mne
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        return raw.get_data().T
    raise ValueError(f"Unsupported file type: {file_path}")


def _reshape_to_samples_channels(data: np.ndarray, channel_count: int) -> np.ndarray:
    data = np.squeeze(np.asarray(data))
    if data.ndim < 2:
        raise ValueError(f"EEG data must have at least 2 dimensions, got shape {data.shape}.")

    if data.ndim == 2:
        if data.shape[1] == channel_count:
            return data
        if data.shape[0] == channel_count:
            return data.T
        raise ValueError(f"Expected {channel_count} channels, got shape {data.shape}.")

    if data.ndim == 3:
        if data.shape[1] == channel_count:
            data = np.transpose(data, (0, 2, 1))
            return data.reshape(-1, channel_count)
        if data.shape[2] == channel_count:
            return data.reshape(-1, channel_count)
        if data.shape[0] == channel_count:
            data = np.transpose(data, (1, 2, 0))
            return data.reshape(-1, channel_count)
        raise ValueError(f"Expected {channel_count} channels, got shape {data.shape}.")

    channel_axes = [axis for axis, size in enumerate(data.shape) if size == channel_count]
    if len(channel_axes) != 1:
        raise ValueError(f"Expected a single channel axis of size {channel_count}, got {data.shape}.")
    channel_axis = channel_axes[0]
    if channel_axis != data.ndim - 1:
        data = np.moveaxis(data, channel_axis, -1)
    return data.reshape(-1, channel_count)


def load_things_eeg(data_path: Path, data_key: Optional[str], channel_count: int = 64) -> np.ndarray:
    """Load THINGS EEG data and return shape (n_samples, channel_count) float32."""
    if data_path.is_dir():
        files = sorted(
            list(data_path.glob("*.npy"))
            + list(data_path.glob("*.npz"))
            + list(data_path.glob("*.mat"))
        )
        if not files:
            raise FileNotFoundError(f"No EEG files found in directory: {data_path}")
        arrays = [_reshape_to_samples_channels(_load_eeg_file(path, data_key), channel_count) for path in files]
        data = np.concatenate(arrays, axis=0)
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"EEG file not found: {data_path}")
        data = _reshape_to_samples_channels(_load_eeg_file(data_path, data_key), channel_count)

    return np.ascontiguousarray(data.astype(np.float32, copy=False))


def load_channel_labels(labels_path: Optional[Path], channel_count: int = 64) -> List[str]:
    """Load channel labels from a text file or return defaults."""
    if labels_path is None:
        labels = DEFAULT_CHANNEL_LABELS
    else:
        labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    if len(labels) != channel_count:
        raise ValueError(f"Expected {channel_count} channel labels, got {len(labels)}.")
    return list(labels)


def create_lsl_outlet(
    name: str,
    stream_type: str,
    channel_labels: Sequence[str],
    sampling_rate: float,
    source_id: str = "things_mock_streamer",
) -> StreamOutlet:
    """Create an LSL outlet with channel metadata."""
    channel_count = len(channel_labels)
    info = StreamInfo(name, stream_type, channel_count, sampling_rate, "float32", source_id)
    desc = info.desc()
    channels = desc.append_child("channels")
    for label in channel_labels:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info, chunk_size=1)


def stream_eeg(
    data: np.ndarray,
    outlet: StreamOutlet,
    sampling_rate: float,
    loop: bool = False,
) -> None:
    """Stream EEG samples one frame at a time at the requested sampling rate."""
    interval = 1.0 / sampling_rate
    while True:
        start_time = time.perf_counter()
        for index, sample in enumerate(data, start=1):
            outlet.push_sample(sample)
            target_time = start_time + (index * interval)
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
        if not loop:
            break


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the mock streamer."""
    parser = argparse.ArgumentParser(description="Stream THINGS EEG data over LSL.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to EEG data file (.vhdr, .edf, .fif, .set, .npy, .npz, .mat) or directory.",
    )
    parser.add_argument(
        "--data-key",
        type=str,
        default=None,
        help="Optional key for selecting arrays from .npz/.mat files.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Sampling rate in Hz. Auto-detected from file if not specified.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor applied before streaming (default: 1).",
    )
    parser.add_argument(
        "--channel-labels",
        type=Path,
        default=None,
        help="Optional path to channel labels file (one label per line).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Number of channels to stream. Auto-detected from file if not specified.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the dataset indefinitely.",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default="MockEEG",
        help="Name of the LSL stream (default: MockEEG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the mock LSL streamer as a standalone script."""
    args = parse_args(argv)
    if args.downsample < 1:
        raise ValueError("downsample must be >= 1.")

    # Load data - auto-detect channel count
    data_path = args.data_path
    suffix = data_path.suffix.lower()

    # For MNE-supported formats, we can extract sampling rate from the file
    sampling_rate = args.sampling_rate
    if suffix in (".vhdr", ".edf", ".fif", ".set"):
        import mne
        if suffix == ".vhdr":
            raw = mne.io.read_raw_brainvision(data_path, preload=True, verbose=False)
        elif suffix == ".edf":
            raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        elif suffix == ".fif":
            raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        elif suffix == ".set":
            raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose=False)

        data = raw.get_data().T  # (samples, channels)
        if sampling_rate is None:
            sampling_rate = raw.info["sfreq"]
        channel_count = data.shape[1]

        # Use channel names from the file if no custom labels provided
        if args.channel_labels is None:
            file_channel_labels = raw.ch_names
        else:
            file_channel_labels = None
    else:
        # Legacy formats
        if sampling_rate is None:
            sampling_rate = 1000.0  # Default
        channel_count = args.channels or 64
        data = load_things_eeg(data_path, args.data_key, channel_count=channel_count)
        file_channel_labels = None

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    # Optionally limit channels
    if args.channels is not None and args.channels < data.shape[1]:
        data = data[:, :args.channels]
        channel_count = args.channels

    # Downsample if requested
    if args.downsample > 1:
        data = data[:: args.downsample]
    effective_rate = sampling_rate / args.downsample

    # Get channel labels
    if args.channel_labels is not None:
        channel_labels = load_channel_labels(args.channel_labels, channel_count=channel_count)
    elif file_channel_labels is not None:
        channel_labels = list(file_channel_labels[:channel_count])
    else:
        channel_labels = DEFAULT_CHANNEL_LABELS[:channel_count]
        if len(channel_labels) < channel_count:
            channel_labels.extend([f"CH{i}" for i in range(len(channel_labels), channel_count)])

    # Create outlet
    outlet = create_lsl_outlet(
        name=args.stream_name,
        stream_type="EEG",
        channel_labels=channel_labels,
        sampling_rate=effective_rate,
    )

    duration_sec = data.shape[0] / effective_rate
    print(
        f"Streaming {data.shape[0]} samples ({duration_sec:.1f}s) at {effective_rate:.2f} Hz "
        f"({channel_count} channels) as '{args.stream_name}'. Press Ctrl+C to stop."
    )
    if args.loop:
        print("Looping enabled.")

    try:
        stream_eeg(data, outlet, sampling_rate=effective_rate, loop=args.loop)
    except KeyboardInterrupt:
        print("\nStreaming stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
