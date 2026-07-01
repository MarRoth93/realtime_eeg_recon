"""Main real-time inference pipeline for EEG-to-Image reconstruction.

This module implements the executive loop that orchestrates:
- LSL data streaming
- Preprocessing
- Neural inference
- Visualization
- Latency monitoring
"""

from __future__ import annotations

import csv
import gc
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class LatencyStats:
    """Statistics for a single pipeline component."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def update(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.samples.append(duration_ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def recent_mean_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        mean = self.recent_mean_ms
        variance = sum((x - mean) ** 2 for x in self.samples) / (len(self.samples) - 1)
        return variance ** 0.5

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


class LatencyMonitor:
    """Real-time latency monitoring for pipeline components.

    Tracks execution time of each pipeline stage and generates reports.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        warn_threshold_ms: float = 30.0,
        critical_threshold_ms: float = 50.0,
    ) -> None:
        self._stats: Dict[str, LatencyStats] = {}
        self._log_dir = Path(log_dir) if log_dir else None
        self._warn_threshold = warn_threshold_ms
        self._critical_threshold = critical_threshold_ms
        self._start_time = time.time()
        self._lock = threading.Lock()

        self._total_iterations = 0
        self._warnings = 0
        self._criticals = 0

        self._loop_times: Deque[float] = deque(maxlen=1000)
        self._last_loop_start: Optional[float] = None

        # CSV file opened lazily via reopen() on first start()
        self._csv_file = None
        self._csv_writer = None

    def __enter__(self) -> "LatencyMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def reopen(self) -> None:
        """Reset stats and reopen CSV logging for a new pipeline run."""
        self.close()

        # Reset per-run counters and stats
        with self._lock:
            self._stats.clear()
        self._total_iterations = 0
        self._warnings = 0
        self._criticals = 0
        self._loop_times.clear()
        self._last_loop_start = None
        self._start_time = time.time()

        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(
                self._log_dir / f"latency_{datetime.now():%Y%m%d_%H%M%S_%f}.csv",
                "w",
                newline="",
            )
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["timestamp", "component", "latency_ms"])

    def start_iteration(self) -> None:
        """Mark the start of a new loop iteration."""
        now = time.perf_counter()
        if self._last_loop_start is not None:
            loop_time = (now - self._last_loop_start) * 1000
            self._loop_times.append(loop_time)
        self._last_loop_start = now
        self._total_iterations += 1

    def time_component(self, name: str):
        """Context manager for timing a pipeline component."""
        return _ComponentTimer(self, name)

    def record(self, name: str, duration_ms: float) -> None:
        """Record a latency measurement.

        Args:
            name: Component name.
            duration_ms: Duration in milliseconds.
        """
        with self._lock:
            if name not in self._stats:
                self._stats[name] = LatencyStats(name=name)
            self._stats[name].update(duration_ms)

        if duration_ms > self._critical_threshold:
            self._criticals += 1
        elif duration_ms > self._warn_threshold:
            self._warnings += 1

        if self._csv_writer and self._csv_file and not self._csv_file.closed:
            self._csv_writer.writerow([time.time(), name, f"{duration_ms:.3f}"])

    def get_stats(self, name: str) -> Optional[LatencyStats]:
        """Get statistics for a component."""
        with self._lock:
            return self._stats.get(name)

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all components."""
        with self._lock:
            return dict(self._stats)

    @property
    def loop_rate_hz(self) -> float:
        """Current loop rate in Hz."""
        if not self._loop_times:
            return 0.0
        mean_ms = sum(self._loop_times) / len(self._loop_times)
        return 1000.0 / mean_ms if mean_ms > 0 else 0.0

    @property
    def total_loop_time_ms(self) -> float:
        """Total average loop time in milliseconds."""
        if not self._loop_times:
            return 0.0
        return sum(self._loop_times) / len(self._loop_times)

    def generate_report(self) -> str:
        """Generate a human-readable latency report."""
        lines = [
            "=" * 60,
            "LATENCY REPORT",
            "=" * 60,
            f"Total iterations: {self._total_iterations}",
            f"Runtime: {time.time() - self._start_time:.1f}s",
            f"Loop rate: {self.loop_rate_hz:.1f} Hz",
            f"Avg loop time: {self.total_loop_time_ms:.2f} ms",
            f"Warnings (>{self._warn_threshold}ms): {self._warnings}",
            f"Critical (>{self._critical_threshold}ms): {self._criticals}",
            "",
            "Component Breakdown:",
            "-" * 60,
        ]

        with self._lock:
            for name, stats in sorted(self._stats.items()):
                lines.append(
                    f"  {name:20s}: mean={stats.mean_ms:6.2f}ms, "
                    f"min={stats.min_ms:6.2f}ms, max={stats.max_ms:6.2f}ms, "
                    f"p95={stats.p95_ms:6.2f}ms"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, path: Optional[Path] = None) -> Path:
        """Save latency report to file."""
        if path is None:
            if self._log_dir:
                path = self._log_dir / f"report_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
            else:
                path = Path(f"latency_report_{datetime.now():%Y%m%d_%H%M%S_%f}.txt")

        report = self.generate_report()
        path.write_text(report)
        return path


class _ComponentTimer:
    """Context manager for timing pipeline components."""

    def __init__(self, monitor: LatencyMonitor, name: str) -> None:
        self._monitor = monitor
        self._name = name
        self._start: Optional[float] = None

    def __enter__(self) -> "_ComponentTimer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            duration_ns = time.perf_counter_ns() - self._start
            duration_ms = duration_ns / 1e6
            self._monitor.record(self._name, duration_ms)


class GPUMemoryManager:
    """GPU memory management for stable long-running inference.

    Provides utilities for:
    - Pre-allocation of tensors
    - Memory monitoring
    - Garbage collection scheduling
    - Memory leak detection
    """

    def __init__(
        self,
        device: str = "cuda",
        gc_interval: int = 100,
        memory_threshold_gb: float = 0.9,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._gc_interval = gc_interval
        self._memory_threshold = memory_threshold_gb
        self._iteration_count = 0
        self._preallocated: Dict[str, torch.Tensor] = {}
        self._initial_memory: Optional[int] = None
        self._peak_memory: int = 0

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
            self._initial_memory = torch.cuda.memory_allocated(self._device)

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Whether using CUDA."""
        return self._device.type == "cuda"

    def preallocate(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Pre-allocate a tensor for reuse.

        Args:
            name: Identifier for the tensor.
            shape: Tensor shape.
            dtype: Data type.

        Returns:
            Pre-allocated tensor.
        """
        tensor = torch.empty(shape, dtype=dtype, device=self._device)
        self._preallocated[name] = tensor
        return tensor

    def get_preallocated(self, name: str) -> Optional[torch.Tensor]:
        """Get a pre-allocated tensor by name."""
        return self._preallocated.get(name)

    def clear_preallocated(self) -> None:
        """Clear all pre-allocated tensors."""
        self._preallocated.clear()
        self._collect_garbage()

    def step(self) -> None:
        """Called each iteration for memory management."""
        self._iteration_count += 1

        if self._iteration_count % self._gc_interval == 0:
            self._collect_garbage()

        if self.is_cuda:
            current = torch.cuda.memory_allocated(self._device)
            if current > self._peak_memory:
                self._peak_memory = current

    def _collect_garbage(self) -> None:
        """Run garbage collection."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics.

        Returns:
            Dictionary with memory stats in GB.
        """
        if not self.is_cuda:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_gb": 0.0,
                "total_gb": 0.0,
            }

        allocated = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self._device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(self._device).total_memory / (1024 ** 3)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "peak_gb": peak,
            "total_gb": total,
        }

    def check_memory_leak(self, tolerance_mb: float = 100.0) -> bool:
        """Check for potential memory leaks.

        Args:
            tolerance_mb: Acceptable memory growth in MB.

        Returns:
            True if a potential leak is detected.
        """
        if not self.is_cuda or self._initial_memory is None:
            return False

        current = torch.cuda.memory_allocated(self._device)
        growth = (current - self._initial_memory) / (1024 ** 2)
        return growth > tolerance_mb

    def memory_warning(self) -> bool:
        """Check if memory usage is above threshold.

        Returns:
            True if memory usage exceeds threshold.
        """
        if not self.is_cuda:
            return False

        stats = self.get_memory_stats()
        usage_ratio = stats["allocated_gb"] / stats["total_gb"]
        return usage_ratio > self._memory_threshold


@dataclass
class PipelineConfig:
    """Configuration for the real-time pipeline."""

    window_samples: int = 1000
    n_channels: int = 64
    sampling_rate: float = 1000.0
    inference_interval_ms: float = 50.0
    max_latency_ms: float = 30.0
    device: str = "cuda"
    log_dir: Optional[Path] = None
    enable_visualization: bool = False
    enable_watchdog: bool = True
    watchdog_timeout_ms: float = 100.0

    # Error recovery settings
    max_consecutive_errors: int = 10
    error_recovery_delay_ms: float = 100.0
    lsl_reconnect_attempts: int = 3
    lsl_reconnect_delay_s: float = 1.0
    reset_on_error_threshold: int = 5

    def __post_init__(self) -> None:
        if self.log_dir is not None:
            self.log_dir = Path(self.log_dir)


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    pass


class RecoverableError(PipelineError):
    """Error that can be recovered from."""

    pass


class FatalError(PipelineError):
    """Error that requires pipeline shutdown."""

    pass


@dataclass
class ErrorState:
    """Tracks error state for recovery decisions."""

    consecutive_errors: int = 0
    total_errors: int = 0
    last_error_time: float = 0.0
    last_error_message: str = ""
    recovery_attempts: int = 0

    def record_error(self, message: str) -> None:
        """Record an error occurrence."""
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error_time = time.monotonic()
        self.last_error_message = message

    def record_success(self) -> None:
        """Record a successful iteration."""
        self.consecutive_errors = 0

    def needs_recovery(self, threshold: int) -> bool:
        """Check if error count exceeds threshold."""
        return self.consecutive_errors >= threshold

    def needs_reset(self, threshold: int) -> bool:
        """Check if a state reset is needed."""
        return self.consecutive_errors >= threshold

    def reset(self) -> None:
        """Reset error counters after recovery."""
        self.consecutive_errors = 0
        self.recovery_attempts += 1


class RealTimePipeline:
    """Main executive loop for real-time EEG-to-Image inference.

    Orchestrates data streaming, preprocessing, inference, and visualization
    with latency monitoring and safety features.
    """

    def __init__(
        self,
        encoder,
        config: Optional[PipelineConfig] = None,
        decoder=None,
        preprocessor=None,
        **kwargs,
    ) -> None:
        if config is None:
            config = PipelineConfig(**kwargs)
        elif kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")

        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.preprocessor = preprocessor

        self.memory_manager = GPUMemoryManager(device=config.device)
        self.latency_monitor = LatencyMonitor(
            log_dir=config.log_dir,
            warn_threshold_ms=config.max_latency_ms,
        )

        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inlet = None
        self._last_inference_time: float = 0.0
        self._latest_latent: Optional[np.ndarray] = None
        self._latest_image: Optional[np.ndarray] = None
        self._callbacks: List[Callable] = []

        self._input_buffer = self.memory_manager.preallocate(
            "input_buffer",
            (1, config.n_channels, config.window_samples),
        )
        self._output_buffer = None

        self._watchdog_time = time.monotonic()
        self._watchdog_triggered = False

        # Error recovery state
        self._error_state = ErrorState()
        self._lsl_stream_name: Optional[str] = None
        self._signals_registered = False

        # Cleanup guard for thread-safe once-only cleanup
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        self._artifact_reject_count = 0

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, stopping pipeline...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._running

    @property
    def latest_latent(self) -> Optional[np.ndarray]:
        """Most recent latent vector output."""
        return self._latest_latent

    @property
    def latest_image(self) -> Optional[np.ndarray]:
        """Most recent reconstructed image."""
        return self._latest_image

    def add_callback(self, callback: Callable[[np.ndarray, Optional[np.ndarray]], None]) -> None:
        """Add a callback to be invoked after each inference.

        Args:
            callback: Function taking (latent, image) arrays.
        """
        self._callbacks.append(callback)

    def connect_lsl(
        self,
        stream_name: str = "MockEEG",
        timeout: float = 10.0,
        auto_detect: bool = True,
    ) -> bool:
        """Connect to an LSL stream.

        Args:
            stream_name: Name of the LSL stream.
            timeout: Connection timeout in seconds.
            auto_detect: If True, auto-detect channel count and sampling rate from stream.

        Returns:
            True if connection successful.
        """
        from .streaming import LSLInletWrapper, LSLInletConfig

        # If auto_detect is True, don't enforce channel count - let the stream define it
        inlet_config = LSLInletConfig(
            stream_name=stream_name,
            stream_type=None,  # Don't filter by type to be more permissive
            channel_count=None if auto_detect else self.config.n_channels,
            sampling_rate=None if auto_detect else self.config.sampling_rate,
            resolve_timeout=min(timeout, 5.0),  # Use shorter resolve timeout for faster retries
        )

        print(f"Searching for LSL stream '{stream_name}'...")
        self._inlet = LSLInletWrapper(inlet_config)
        self._inlet.start()

        if self._inlet.wait_for_stream(timeout):
            # Update config with detected values
            if auto_detect and self._inlet.channel_count is not None:
                detected_channels = self._inlet.channel_count
                detected_rate = self._inlet.sampling_rate or self.config.sampling_rate
                print(f"  Detected: {detected_channels} channels at {detected_rate} Hz")

                # Update internal config and reallocate buffer if needed
                if detected_channels != self.config.n_channels:
                    self.config.n_channels = detected_channels
                    self._input_buffer = self.memory_manager.preallocate(
                        "input_buffer",
                        (1, detected_channels, self.config.window_samples),
                    )
                if detected_rate != self.config.sampling_rate:
                    self.config.sampling_rate = detected_rate

            print(f"Connected to LSL stream: {stream_name}")
            self._lsl_stream_name = stream_name

            # Create encoder now that we know channel count (if not already provided)
            if self.encoder is None and hasattr(self, "_encoder_type"):
                from .models import create_encoder
                self.encoder = create_encoder(
                    self._encoder_type,
                    n_channels=self.config.n_channels,
                    n_samples=self.config.window_samples,
                )
                self.encoder = self.encoder.to(self.memory_manager.device).eval()
                print(f"  Created {self._encoder_type} encoder for {self.config.n_channels} channels")

            return True

        print(f"Failed to connect to LSL stream: {stream_name}")
        print("  Make sure the mock streamer is running in another terminal.")
        return False

    def start(self, blocking: bool = True) -> None:
        """Start the pipeline.

        Args:
            blocking: If True, blocks until stop() is called.
        """
        if self._running:
            return

        self._stop_event.clear()
        self._cleaned_up = False
        self._artifact_reject_count = 0

        # Initialize per-run resources before committing to running state
        try:
            self.latency_monitor.reopen()
        except Exception:
            # Don't leave pipeline in stuck "running" state
            raise

        self._running = True

        # Register signal handlers only from main thread, only once
        if not self._signals_registered and threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                self._signals_registered = True
            except ValueError:
                pass  # Not in main thread despite check (e.g., embedded interpreter)

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)

        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Release pipeline resources. Thread-safe and idempotent."""
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True

        if self._inlet:
            self._inlet.stop()

        if self._artifact_reject_count > 0:
            print(f"  Artifact rejections: {self._artifact_reject_count}")

        print("\n" + self.latency_monitor.generate_report())

        if self.config.log_dir:
            self.latency_monitor.save_report()
        self.latency_monitor.close()

    def _run_loop(self) -> None:
        """Main execution loop with error recovery."""
        print("Pipeline started. Press Ctrl+C to stop.")

        try:
            while self._running and not self._stop_event.is_set():
                self.latency_monitor.start_iteration()
                self._watchdog_time = time.monotonic()

                try:
                    now = time.monotonic() * 1000
                    if now - self._last_inference_time < self.config.inference_interval_ms:
                        time.sleep(0.001)
                        continue

                    with self.latency_monitor.time_component("read_lsl"):
                        eeg_data = self._read_data()

                    if eeg_data is None:
                        continue

                    with self.latency_monitor.time_component("preprocess"):
                        processed = self._preprocess(eeg_data)

                    # Skip frame if artifact rejected
                    if processed is None:
                        continue

                    with self.latency_monitor.time_component("inference"):
                        latent = self._infer(processed)

                    self._latest_latent = latent

                    if self.decoder is not None:
                        with self.latency_monitor.time_component("decode"):
                            image = self._decode(latent)
                        self._latest_image = image
                    else:
                        image = None

                    for callback in self._callbacks:
                        try:
                            callback(latent, image)
                        except Exception as e:
                            print(f"Callback error: {e}")

                    self._last_inference_time = now

                    self.memory_manager.step()

                    self._check_watchdog()

                    # Record successful iteration
                    self._error_state.record_success()

                except FatalError as e:
                    print(f"Fatal pipeline error: {e}")
                    self._running = False
                    break

                except (RecoverableError, Exception) as e:
                    self._handle_error(e)

                    if not self._running:
                        break

            print("Pipeline stopped.")
            self._print_error_summary()
        finally:
            self._cleanup_resources()

    def _handle_error(self, error: Exception) -> None:
        """Handle errors with recovery logic.

        Args:
            error: The exception that occurred.
        """
        error_msg = str(error)
        self._error_state.record_error(error_msg)

        # Log the error
        if self._error_state.consecutive_errors == 1:
            print(f"Pipeline error: {error_msg}")
        elif self._error_state.consecutive_errors % 5 == 0:
            print(f"Pipeline error (x{self._error_state.consecutive_errors}): {error_msg}")

        # Check if we need to attempt recovery
        if self._error_state.needs_reset(self.config.reset_on_error_threshold):
            print(f"Attempting recovery after {self._error_state.consecutive_errors} consecutive errors...")
            if self._attempt_recovery():
                self._error_state.reset()
                print("Recovery successful.")
            else:
                print("Recovery failed.")

        # Check if we've exceeded max errors
        if self._error_state.consecutive_errors >= self.config.max_consecutive_errors:
            print(f"Max consecutive errors ({self.config.max_consecutive_errors}) exceeded. Stopping pipeline.")
            self._running = False
            return

        # Brief delay before retrying
        time.sleep(self.config.error_recovery_delay_ms / 1000.0)

    def _attempt_recovery(self) -> bool:
        """Attempt to recover from errors.

        Returns:
            True if recovery was successful.
        """
        recovery_success = True

        # Step 1: Clear GPU memory
        try:
            self.memory_manager._collect_garbage()
        except Exception as e:
            print(f"  Memory cleanup failed: {e}")
            recovery_success = False

        # Step 2: Reset input buffer
        try:
            self._input_buffer = self.memory_manager.preallocate(
                "input_buffer",
                (1, self.config.n_channels, self.config.window_samples),
            )
        except Exception as e:
            print(f"  Buffer reset failed: {e}")
            recovery_success = False

        # Step 3: Attempt LSL reconnection if we have a stream name
        if self._lsl_stream_name and self._inlet is not None:
            reconnected = self._attempt_lsl_reconnect()
            if not reconnected:
                recovery_success = False

        # Step 4: Reset preprocessor state if available
        if self.preprocessor is not None and hasattr(self.preprocessor, "reset"):
            try:
                self.preprocessor.reset()
            except Exception as e:
                print(f"  Preprocessor reset failed: {e}")

        return recovery_success

    def _attempt_lsl_reconnect(self) -> bool:
        """Attempt to reconnect to the LSL stream.

        Returns:
            True if reconnection was successful.
        """
        if self._lsl_stream_name is None:
            return True

        for attempt in range(1, self.config.lsl_reconnect_attempts + 1):
            print(f"  LSL reconnect attempt {attempt}/{self.config.lsl_reconnect_attempts}...")

            try:
                # Stop existing inlet
                if self._inlet is not None:
                    try:
                        self._inlet.stop()
                    except Exception:
                        pass

                # Wait before reconnecting
                time.sleep(self.config.lsl_reconnect_delay_s)

                # Attempt reconnection
                if self.connect_lsl(self._lsl_stream_name, timeout=5.0):
                    return True

            except Exception as e:
                print(f"    Reconnect failed: {e}")

        return False

    def _print_error_summary(self) -> None:
        """Print a summary of errors that occurred during the run."""
        if self._error_state.total_errors > 0:
            print(f"\nError Summary:")
            print(f"  Total errors: {self._error_state.total_errors}")
            print(f"  Recovery attempts: {self._error_state.recovery_attempts}")
            if self._error_state.last_error_message:
                print(f"  Last error: {self._error_state.last_error_message}")

    def _read_data(self) -> Optional[np.ndarray]:
        """Read data from LSL inlet."""
        if self._inlet is None:
            return np.random.randn(self.config.n_channels, self.config.window_samples).astype(np.float32)

        try:
            data = self._inlet.get_window(self.config.window_samples, timeout=0.1)
            return data
        except Exception:
            return None

    def _preprocess(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Apply preprocessing pipeline.

        Returns:
            Processed data, or None if artifact rejection triggered.
        """
        if self.preprocessor is not None:
            processed, is_valid = self.preprocessor.process(data)
            if not is_valid:
                self._artifact_reject_count += 1
                return None
            return processed
        return data

    @torch.inference_mode()
    def _infer(self, data: np.ndarray) -> np.ndarray:
        """Run encoder inference."""
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized. Connect to LSL stream first.")

        self._input_buffer.copy_(torch.from_numpy(data).unsqueeze(0))

        output = self.encoder(self._input_buffer)
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy().squeeze()
        return np.asarray(output).squeeze()

    @torch.inference_mode()
    def _decode(self, latent: np.ndarray) -> np.ndarray:
        """Run decoder to generate image."""
        latent_tensor = torch.from_numpy(latent).unsqueeze(0).to(self.memory_manager.device)
        image = self.decoder(latent_tensor)
        if isinstance(image, torch.Tensor):
            return image.cpu().numpy().squeeze()
        return np.asarray(image).squeeze()

    def _check_watchdog(self) -> None:
        """Check for pipeline hangs."""
        if not self.config.enable_watchdog:
            return

        elapsed = (time.monotonic() - self._watchdog_time) * 1000
        if elapsed > self.config.watchdog_timeout_ms:
            if not self._watchdog_triggered:
                print(f"WARNING: Watchdog timeout ({elapsed:.1f}ms > {self.config.watchdog_timeout_ms}ms)")
                self._watchdog_triggered = True
        else:
            self._watchdog_triggered = False


def create_pipeline(
    encoder_path: Optional[Union[str, Path]] = None,
    decoder_path: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    encoder_type: str = "cnn1d",
    **kwargs,
) -> RealTimePipeline:
    """Create a real-time pipeline with loaded models.

    Args:
        encoder_path: Path to encoder model (.pt, .onnx, or .trt).
        decoder_path: Path to decoder model.
        config: PipelineConfig instance.
        encoder_type: Type of encoder to use if no path provided.
        **kwargs: Additional config arguments.

    Returns:
        Configured RealTimePipeline instance.
    """
    if config is None:
        config = PipelineConfig(**kwargs)

    encoder = None
    decoder = None

    if encoder_path is not None:
        from .models import InferenceEngine
        encoder = InferenceEngine(encoder_path, device=config.device)

    if decoder_path is not None:
        from .models import InferenceEngine
        decoder = InferenceEngine(decoder_path, device=config.device)

    # If no encoder path, we'll create one lazily when we know the channel count
    # Store the encoder type for lazy creation
    pipeline = RealTimePipeline(encoder, config, decoder=decoder)
    pipeline._encoder_type = encoder_type  # Store for lazy creation
    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run real-time EEG inference pipeline")
    parser.add_argument("--encoder", type=str, help="Path to encoder model")
    parser.add_argument("--decoder", type=str, help="Path to decoder model")
    parser.add_argument("--stream", type=str, default="MockEEG", help="LSL stream name")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--log-dir", type=str, help="Directory for latency logs")
    args = parser.parse_args()

    config = PipelineConfig(
        device=args.device,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )

    pipeline = create_pipeline(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        config=config,
    )

    if args.stream:
        pipeline.connect_lsl(args.stream)

    pipeline.start(blocking=True)
