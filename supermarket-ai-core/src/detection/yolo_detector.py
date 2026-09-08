"""YOLO detection interface and detection schema (M2 placeholder)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single object detected in one frame of one camera."""

    camera_id: str
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    frame_number: int
    timestamp: datetime


class YOLODetector:
    """Wraps an Ultralytics YOLO model.

    Responsibilities (M2):
      * lazy-load weights from `models/`
      * run inference on a frame and filter by `DETECTION_CONFIDENCE`
      * return `Detection` objects, never raw model tensors
    """

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    def load(self) -> None:
        """Load model weights into memory."""
        raise NotImplementedError

    def detect(self, frame: Any, camera_id: str, frame_number: int) -> List[Detection]:
        """Run detection on one frame."""
        raise NotImplementedError
