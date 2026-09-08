"""Writers for M1 artefacts: an annotated video and a detection JSON file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from src.detection.yolo_detector import Detection
from src.logging_setup import get_logger

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
