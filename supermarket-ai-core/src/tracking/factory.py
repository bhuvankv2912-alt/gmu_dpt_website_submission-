"""Tracker construction from configuration (M2)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.detection.yolo_detector import YOLODetector
from src.tracking.tracker import BaseTracker, IoUTracker, TrackerError
from src.tracking.ultralytics_tracker import TRACKER_CONFIGS, UltralyticsTracker

TRACKER_CHOICES = (*sorted(TRACKER_CONFIGS), "iou")


def create_tracker(
    tracker: str,
    weights_path: str,
    confidence_threshold: float,
    tracking_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
) -> BaseTracker:
    """Build the configured tracker.

    `bytetrack` / `botsort` run inside Ultralytics; `iou` is the dependency-free
    fallback that wraps a plain `YOLODetector`.
    """
    name = str(tracker).strip().lower()
    settings = tracking_config or {}

    if name == "iou":
        detector = YOLODetector(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device,
            classes=classes,
        )
        return IoUTracker(
            detector,
            min_iou=float(settings.get("MIN_IOU", 0.3)),
            max_age=int(settings.get("MAX_AGE", 30)),
            min_hits=int(settings.get("MIN_HITS", 1)),
        )

    if name in TRACKER_CONFIGS or name.endswith(".yaml"):
        return UltralyticsTracker(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            tracker=name,
            device=device,
            classes=classes,
        )

    raise TrackerError(f"Unknown tracker '{tracker}'. Choose one of {list(TRACKER_CHOICES)}.")
