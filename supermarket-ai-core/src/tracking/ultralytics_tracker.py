"""ByteTrack / BoT-SORT tracking through Ultralytics (M2).

Ultralytics runs detection and association in a single `model.track(persist=True)`
call, so this class implements the whole `BaseTracker` contract rather than
wrapping a separate detector. Raw numeric ids from the tracker are mapped onto
stable ids such as `person_001` by `TrackLabeler`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.detection.yolo_detector import DetectorError
from src.logging_setup import get_logger
from src.tracking.tracker import BaseTracker, TrackedDetection, TrackerError, TrackLabeler

logger = get_logger(__name__)

TRACKER_CONFIGS: Dict[str, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


def resolve_tracker_config(name: str) -> str:
    """Map a friendly tracker name onto an Ultralytics tracker config file."""
    key = str(name).strip().lower()
    if key in TRACKER_CONFIGS:
        return TRACKER_CONFIGS[key]
    if key.endswith(".yaml"):
        return key
    raise TrackerError(
        f"Unknown tracker '{name}'. Use one of {sorted(TRACKER_CONFIGS)} "
        "or a path to an Ultralytics tracker YAML."
    )


class UltralyticsTracker(BaseTracker):
    """Multi-object tracking with the trackers bundled in Ultralytics."""

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float,
        tracker: str = "bytetrack",
        device: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.weights_path = str(weights_path)
        self.confidence_threshold = float(confidence_threshold)
        self.tracker_config = resolve_tracker_config(tracker)
        self.name = Path(self.tracker_config).stem
        self.device = None if device in (None, "", "auto") else device
        self.classes = [c.lower() for c in classes] if classes else None
        self.labeler = TrackLabeler()
        self._frames_tracked: Dict[Any, int] = {}
        self._model: Any = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "UltralyticsTracker":
        """Load weights. `ultralytics` is imported lazily so importing this
        module (and running the non-model tests) does not require torch."""
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
            "Loaded YOLO weights %s with tracker %s (conf>=%.2f)",
            self.weights_path,
            self.tracker_config,
            self.confidence_threshold,
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- tracking ----------------------------------------------------------
    def track(
        self,
        frame: Any,
        camera_id: str,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> List[TrackedDetection]:
        """Advance the tracker by one frame; `persist=True` keeps ids alive."""
        if self._model is None:
            self.load()

        stamp = timestamp or datetime.now(timezone.utc)
        try:
            results = self._model.track(
                frame,
                conf=self.confidence_threshold,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill a run
            logger.error("Tracking failed on frame %d: %s", frame_number, exc)
            return []

        tracked = list(self._to_tracked(results, camera_id, frame_number, stamp))
        tracked.sort(key=lambda t: t.track_number)
        return tracked

    def _to_tracked(
        self,
        results: Iterable[Any],
        camera_id: str,
        frame_number: int,
        timestamp: datetime,
    ) -> Iterable[TrackedDetection]:
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "id", None) is None:
                continue  # no confirmed tracks in this frame
            for box in boxes:
                if box.id is None:
                    continue  # detection not yet confirmed as a track
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                class_name = str(names.get(int(box.cls[0]), int(box.cls[0])))
                if self.classes and class_name.lower() not in self.classes:
                    continue

                raw_id = int(box.id[0])
                track_id, track_number = self.labeler.label(class_name, raw_id)
                key = (class_name, raw_id)
                self._frames_tracked[key] = self._frames_tracked.get(key, 0) + 1

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                yield TrackedDetection(
                    camera_id=camera_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    frame_number=frame_number,
                    timestamp=timestamp,
                    track_id=track_id,
                    track_number=track_number,
                    frames_tracked=self._frames_tracked[key],
                )

    def reset(self) -> None:
        """Forget all tracks; the next `track()` call starts a new sequence."""
        self.labeler.reset()
        self._frames_tracked.clear()
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
