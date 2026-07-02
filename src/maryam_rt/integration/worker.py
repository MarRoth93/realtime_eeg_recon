from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from maryam_rt.integration.high_level import HighLevelRefiner
from maryam_rt.integration.assessor_ratings import AssessorRatingWriter


@dataclass
class RefinementJob:
    stem: str
    x250: torch.Tensor
    low_level_image: np.ndarray
    text_prompt: str | None = None
    low_level_path: Path | None = None
    target_image_path: Path | None = None


class SemanticRefinementWorker:
    """Background worker that processes refinement requests in FIFO order."""

    def __init__(
        self,
        refiner: HighLevelRefiner,
        output_dir: str | Path,
        maxsize: int = 0,
        on_result: Callable[[Path], None] | None = None,
        assessor_writer: AssessorRatingWriter | None = None,
    ) -> None:
        self.refiner = refiner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[Optional[RefinementJob]] = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.on_result = on_result
        self.assessor_writer = assessor_writer

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)

    def submit(
        self,
        stem: str,
        x250: torch.Tensor,
        low_level_image: np.ndarray,
        text_prompt: str | None = None,
        low_level_path: str | Path | None = None,
        target_image_path: str | Path | None = None,
    ) -> None:
        job = RefinementJob(
            stem=stem,
            x250=x250.cpu(),
            low_level_image=low_level_image.copy(),
            text_prompt=text_prompt,
            low_level_path=None if low_level_path is None else Path(low_level_path),
            target_image_path=None if target_image_path is None else Path(target_image_path),
        )
        self.queue.put(job)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                continue
            try:
                image = self.refiner.refine(job.x250, job.low_level_image, text_prompt=job.text_prompt)
                output_path = self.output_dir / f"{job.stem}_refined.png"
                image.save(output_path)
                if self.assessor_writer is not None:
                    try:
                        if job.low_level_path is not None and job.target_image_path is not None:
                            self.assessor_writer.write_triplet_summary(
                                stem=job.stem,
                                target_path=job.target_image_path,
                                low_level_path=job.low_level_path,
                                high_level_path=output_path,
                                title=job.text_prompt or job.stem,
                            )
                        else:
                            self.assessor_writer.assess_path(output_path, job.stem, "high_level")
                    except Exception as exc:
                        error_path = self.output_dir / f"{job.stem}_assessor_error.txt"
                        error_path.write_text(str(exc))
                if self.on_result is not None:
                    self.on_result(output_path)
            except Exception as exc:
                error_path = self.output_dir / f"{job.stem}_error.txt"
                error_path.write_text(str(exc))
