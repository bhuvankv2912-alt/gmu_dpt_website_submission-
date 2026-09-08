"""Processing metrics for a single video run (M1)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.detection.yolo_detector import Detection


@dataclass
class ProcessingMetrics:
    """Counts and timings collected while processing one video."""

    camera_id: str = ""
    source: str = ""
    frames_read: int = 0
    frames_processed: int = 0
    detections: int = 0
    detections_by_class: Counter = field(default_factory=Counter)
    inference_seconds: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float = 0.0

    def record_frame(self, detections: List[Detection], inference_seconds: float = 0.0) -> None:
        """Record one processed frame and its detections."""
        self.frames_processed += 1
        self.detections += len(detections)
        self.inference_seconds += inference_seconds
        self.detections_by_class.update(d.class_name for d in detections)

    def finish(self) -> "ProcessingMetrics":
        self.ended_at = time.perf_counter()
        return self

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.perf_counter()
        return max(0.0, end - self.started_at)

    @property
    def fps(self) -> float:
        """Processing throughput in frames per second (0 when nothing ran)."""
        elapsed = self.elapsed_seconds
        return self.frames_processed / elapsed if elapsed > 0 else 0.0

    @property
    def average_detections_per_frame(self) -> float:
        return self.detections / self.frames_processed if self.frames_processed else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "detections_by_class": dict(sorted(self.detections_by_class.items())),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "inference_seconds": round(self.inference_seconds, 3),
            "fps": round(self.fps, 2),
            "average_detections_per_frame": round(self.average_detections_per_frame, 3),
        }

    def summary(self) -> str:
        return (
            f"[{self.camera_id}] frames_read={self.frames_read} "
            f"frames_processed={self.frames_processed} detections={self.detections} "
            f"fps={self.fps:.2f} elapsed={self.elapsed_seconds:.2f}s"
        )
