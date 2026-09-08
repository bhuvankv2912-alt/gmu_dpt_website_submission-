"""Writers for the run artefacts: an annotated video and a detection/tracking JSON file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2

from src.detection.yolo_detector import Detection
from src.logging_setup import get_logger

if TYPE_CHECKING:  # avoids a src.tracking -> src.detection import cycle at runtime
    from src.tracking.tracker import TrackedDetection

logger = get_logger(__name__)

FALLBACK_CODECS = ("mp4v", "avc1", "MJPG")


class OutputError(RuntimeError):
    """Raised when an output artefact cannot be written."""


class AnnotatedVideoWriter:
    """Writes annotated frames to a video file, trying codecs in order."""

    def __init__(
        self,
        path: str | Path,
        size: Tuple[int, int],
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.codecs = (codec, *(c for c in FALLBACK_CODECS if c != codec))
        self._writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

    def open(self) -> "AnnotatedVideoWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for codec in self.codecs:
            writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, self.size
            )
            if writer.isOpened():
                self._writer = writer
                logger.info(
                    "Writing annotated video %s (%dx%d @ %.2f fps, codec %s)",
                    self.path,
                    self.size[0],
                    self.size[1],
                    self.fps,
                    codec,
                )
                return self
            writer.release()
        raise OutputError(
            f"No usable codec for {self.path}; tried {self.codecs}. "
            "Install a codec-enabled OpenCV build or choose another OUTPUT.VIDEO_CODEC."
        )

    def write(self, frame: Any) -> None:
        if self._writer is None:
            raise OutputError("write() called before open()")
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def detections_to_frames(detections: List[Detection]) -> List[Dict[str, Any]]:
    """Group flat detections into per-frame records, ordered by frame number."""
    grouped: Dict[int, Dict[str, Any]] = {}
    for detection in detections:
        frame = grouped.setdefault(
            detection.frame_number,
            {
                "frame_number": detection.frame_number,
                "timestamp": detection.timestamp.isoformat(),
                "detections": [],
            },
        )
        frame["detections"].append(
            {
                "class_name": detection.class_name,
                "confidence": round(detection.confidence, 4),
                "bbox": [round(v, 2) for v in detection.bbox],
            }
        )
    return [grouped[key] for key in sorted(grouped)]


def write_detections_json(
    path: str | Path,
    camera_id: str,
    source: str,
    detections: List[Detection],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Path:
    """Write the structured detection results for one processed video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "milestone": "M1",
        "camera_id": camera_id,
        "source": str(source),
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": config,
        "metrics": metrics,
        "frames": detections_to_frames(detections),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote detection JSON %s", path)
    return path


def tracks_to_frames(tracks: List["TrackedDetection"]) -> List[Dict[str, Any]]:
    """Group tracked detections into per-frame records, ordered by frame number."""
    grouped: Dict[int, Dict[str, Any]] = {}
    for track in tracks:
        frame = grouped.setdefault(
            track.frame_number,
            {
                "frame_number": track.frame_number,
                "timestamp": track.timestamp.isoformat(),
                "tracks": [],
            },
        )
        frame["tracks"].append(
            {
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 4),
                "bbox": [round(v, 2) for v in track.bbox],
                "frames_tracked": track.frames_tracked,
            }
        )
    return [grouped[key] for key in sorted(grouped)]


def tracks_to_summary(tracks: List["TrackedDetection"]) -> List[Dict[str, Any]]:
    """One record per track id: class, lifespan and confidence range."""
    summary: Dict[str, Dict[str, Any]] = {}
    for track in tracks:
        record = summary.setdefault(
            track.track_id,
            {
                "track_id": track.track_id,
                "class_name": track.class_name,
                "first_frame": track.frame_number,
                "last_frame": track.frame_number,
                "frames_tracked": 0,
                "max_confidence": 0.0,
            },
        )
        record["first_frame"] = min(record["first_frame"], track.frame_number)
        record["last_frame"] = max(record["last_frame"], track.frame_number)
        record["frames_tracked"] += 1
        record["max_confidence"] = round(max(record["max_confidence"], track.confidence), 4)
    return [summary[key] for key in sorted(summary)]


def write_tracks_json(
    path: str | Path,
    camera_id: str,
    source: str,
    tracks: List["TrackedDetection"],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Path:
    """Write the structured tracking results for one processed video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "milestone": "M2",
        "camera_id": camera_id,
        "source": str(source),
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": config,
        "metrics": metrics,
        "tracks": tracks_to_summary(tracks),
        "frames": tracks_to_frames(tracks),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote tracking JSON %s", path)
    return path
