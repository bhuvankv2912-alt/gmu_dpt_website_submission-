"""Ultralytics YOLO detection (M1).

The detector only turns frames into `Detection` records. It performs no
association across frames, so M2 can add tracking on top without changing this
module or its callers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.logging_setup import get_logger

logger = get_logger(__name__)


class Detection(BaseModel):
    """A single object detected in one frame of one camera."""

    camera_id: str
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    frame_number: int
    timestamp: datetime

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class DetectorError(RuntimeError):
    """Raised when the detection model cannot be loaded or run."""


class BaseDetector:
    """Minimal detector contract the pipeline depends on.

    Anything implementing `detect()` can be plugged into `DetectionPipeline`,
    which keeps the pipeline testable without model weights and lets M2 wrap a
    detector with tracking.
    """

    def detect(
        self,
        frame: Any,
        camera_id: str,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> List[Detection]:
        raise NotImplementedError


class YOLODetector(BaseDetector):
    """Wraps an Ultralytics YOLO model."""

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float,
        device: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = None if device in (None, "", "auto") else device
        self.classes = [c.lower() for c in classes] if classes else None
        self._model: Any = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "YOLODetector":
        """Load weights. `ultralytics` is imported lazily so that importing this
        module (and running non-detection tests) does not require torch."""
        if self._model is not None:
            return self

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DetectorError(
                "ultralytics is not installed; run `pip install -r requirements.txt`"
            ) from exc

        path = Path(self.weights_path)
        if not path.is_file() and path.parent != Path("."):
            raise DetectorError(
                f"Model weights not found: {path}. Download a YOLO checkpoint into "
                f"models/ (see models/README.md)."
            )

        try:
            self._model = YOLO(self.weights_path)
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise DetectorError(f"Failed to load YOLO weights '{self.weights_path}': {exc}") from exc

        if self.device:
            try:
                self._model.to(self.device)
            except Exception as exc:  # noqa: BLE001 - fall back to default device
                logger.warning("Could not move model to device '%s': %s", self.device, exc)

        logger.info(
            "Loaded YOLO weights %s (%d classes, conf>=%.2f)",
            self.weights_path,
            len(self.class_names),
            self.confidence_threshold,
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        """Class names supported by the loaded model."""
        if self._model is None:
            return []
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        return [str(name) for name in names]

    # -- inference ---------------------------------------------------------
    def detect(
        self,
        frame: Any,
        camera_id: str,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> List[Detection]:
        """Run detection on one frame and return `Detection` records."""
        if self._model is None:
            self.load()

        stamp = timestamp or datetime.now(timezone.utc)
        try:
            results = self._model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill a run
            logger.error("Inference failed on frame %d: %s", frame_number, exc)
            return []

        return list(self._to_detections(results, camera_id, frame_number, stamp))

    def _to_detections(
        self,
        results: Iterable[Any],
        camera_id: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Iterable[Detection]:
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                class_name = str(names.get(int(box.cls[0]), int(box.cls[0])))
                if self.classes and class_name.lower() not in self.classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                yield Detection(
                    camera_id=camera_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
