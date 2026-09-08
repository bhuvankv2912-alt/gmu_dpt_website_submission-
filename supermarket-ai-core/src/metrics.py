"""Processing metrics for a single video run (M1 detection, M2 tracking)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from src.detection.yolo_detector import Detection

if TYPE_CHECKING:  # avoids a src.tracking -> src.metrics import cycle at runtime
    from src.tracking.tracker import TrackedDetection


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


@dataclass
class TrackingMetrics(ProcessingMetrics):
    """M1 metrics plus per-track counts for one tracked video."""

    track_frames: Counter = field(default_factory=Counter)
    tracks_by_class: Dict[str, set] = field(default_factory=dict)
    max_concurrent_tracks: int = 0

    def record_frame(
        self, detections: List["TrackedDetection"], inference_seconds: float = 0.0
    ) -> None:
        """Record one processed frame and the tracks visible in it."""
        super().record_frame(detections, inference_seconds)
        self.max_concurrent_tracks = max(self.max_concurrent_tracks, len(detections))
        for track in detections:
            self.track_frames[track.track_id] += 1
            self.tracks_by_class.setdefault(track.class_name, set()).add(track.track_id)

    @property
    def unique_tracks(self) -> int:
        return len(self.track_frames)

    @property
    def average_track_length(self) -> float:
        """Mean number of processed frames a track stayed visible for."""
        return sum(self.track_frames.values()) / self.unique_tracks if self.track_frames else 0.0

    @property
    def average_tracks_per_frame(self) -> float:
        return self.detections / self.frames_processed if self.frames_processed else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "unique_tracks": self.unique_tracks,
                "tracks_by_class": {
                    name: len(ids) for name, ids in sorted(self.tracks_by_class.items())
                },
                "max_concurrent_tracks": self.max_concurrent_tracks,
                "average_tracks_per_frame": round(self.average_tracks_per_frame, 3),
                "average_track_length_frames": round(self.average_track_length, 3),
            }
        )
        return data

    def summary(self) -> str:
        return (
            f"[{self.camera_id}] frames_read={self.frames_read} "
            f"frames_processed={self.frames_processed} tracks={self.unique_tracks} "
            f"observations={self.detections} fps={self.fps:.2f} "
            f"elapsed={self.elapsed_seconds:.2f}s"
        )
