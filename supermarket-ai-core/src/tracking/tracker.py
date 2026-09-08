"""Single-camera multi-object tracking (M2).

Tracking is expressed as one small contract, `BaseTracker.track()`, which turns a
frame into `TrackedDetection` records carrying a persistent temporary id such as
`person_001`. Two implementations ship with M2:

* `UltralyticsTracker` - ByteTrack / BoT-SORT running inside Ultralytics.
* `IoUTracker` - a dependency-free greedy IoU tracker that wraps any
  `BaseDetector`, used as a fallback and in tests.

Ids are camera-local and temporary. Cross-camera identity (M4) and behaviour
analysis (M5+) consume `TrackedDetection` and are not implemented here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import Field

from src.detection.yolo_detector import BaseDetector, Detection
from src.logging_setup import get_logger

logger = get_logger(__name__)


class TrackerError(RuntimeError):
    """Raised when a tracker cannot be built or advanced."""


class TrackedDetection(Detection):
    """A detection associated with a camera-local track identity."""

    track_id: str = Field(description="Persistent temporary id, e.g. 'person_001'")
    track_number: int = Field(ge=1, description="Numeric part of the track id")
    frames_tracked: int = Field(ge=1, description="Frames this track has been matched in")

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TrackLabeler:
    """Turns raw tracker ids into stable, readable ids like `person_001`.

    Counters are per class name, so the first tracked person is `person_001`
    regardless of how many other objects the model saw first.
    """

    def __init__(self, width: int = 3) -> None:
        self.width = width
        self._labels: Dict[Tuple[str, Any], Tuple[str, int]] = {}
        self._counters: Dict[str, int] = {}

    def label(self, class_name: str, raw_id: Any) -> Tuple[str, int]:
        """Return `(track_id, track_number)` for a raw tracker id."""
        key = (class_name, raw_id)
        if key not in self._labels:
            number = self._counters.get(class_name, 0) + 1
            self._counters[class_name] = number
            self._labels[key] = (f"{class_name}_{number:0{self.width}d}", number)
        return self._labels[key]

    @property
    def total(self) -> int:
        return len(self._labels)

    def reset(self) -> None:
        self._labels.clear()
        self._counters.clear()


class BaseTracker:
    """Contract the tracking pipeline depends on.

    Implementations own detection as well as association, because native
    trackers (ByteTrack, BoT-SORT) run both inside one model call.
    """

    name = "base"

    def load(self) -> "BaseTracker":
        """Prepare the tracker (load weights, etc.). Default: nothing to do."""
        return self

    def track(
        self,
        frame: Any,
        camera_id: str,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> List[TrackedDetection]:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop all track state (e.g. between videos)."""
        raise NotImplementedError


class _Track:
    """Internal state of one `IoUTracker` track."""

    __slots__ = ("raw_id", "class_name", "bbox", "hits", "misses")

    def __init__(self, raw_id: int, class_name: str, bbox: List[float]) -> None:
        self.raw_id = raw_id
        self.class_name = class_name
        self.bbox = list(bbox)
        self.hits = 1
        self.misses = 0


class IoUTracker(BaseTracker):
    """Greedy IoU tracker wrapping any `BaseDetector`.

    Detections are matched to existing tracks of the same class by highest IoU.
    Unmatched detections open new tracks (objects entering the frame); tracks
    unmatched for more than `max_age` processed frames are dropped (objects
    leaving the frame), which keeps ids stable across brief misses.
    """

    name = "iou"

    def __init__(
        self,
        detector: BaseDetector,
        min_iou: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.detector = detector
        self.min_iou = float(min_iou)
        self.max_age = int(max_age)
        self.min_hits = max(1, int(min_hits))
        self.labeler = TrackLabeler()
        self._tracks: List[_Track] = []
        self._next_raw_id = 1

    def load(self) -> "IoUTracker":
        loader = getattr(self.detector, "load", None)
        if callable(loader):
            loader()
        return self

    def track(
        self,
        frame: Any,
        camera_id: str,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> List[TrackedDetection]:
        stamp = timestamp or datetime.now(timezone.utc)
        detections = self.detector.detect(frame, camera_id, frame_number, stamp)
        return self.update(detections)

    def update(self, detections: List[Detection]) -> List[TrackedDetection]:
        """Associate one frame of detections with the open tracks."""
        pairs = sorted(
            (
                (iou(track.bbox, detection.bbox), track_index, det_index)
                for track_index, track in enumerate(self._tracks)
                for det_index, detection in enumerate(detections)
                if track.class_name == detection.class_name
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        matched_tracks: Dict[int, int] = {}
        matched_dets: Dict[int, int] = {}
        for score, track_index, det_index in pairs:
            if score < self.min_iou:
                break
            if track_index in matched_tracks or det_index in matched_dets:
                continue
            matched_tracks[track_index] = det_index
            matched_dets[det_index] = track_index

        results: List[TrackedDetection] = []
        for track_index, det_index in matched_tracks.items():
            track = self._tracks[track_index]
            detection = detections[det_index]
            track.bbox = list(detection.bbox)
            track.hits += 1
            track.misses = 0
            results.append(self._as_tracked(detection, track))

        for det_index, detection in enumerate(detections):
            if det_index in matched_dets:
                continue
            track = _Track(self._next_raw_id, detection.class_name, list(detection.bbox))
            self._next_raw_id += 1
            self._tracks.append(track)
            results.append(self._as_tracked(detection, track))

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]

        results = [r for r in results if r.frames_tracked >= self.min_hits]
        results.sort(key=lambda r: r.track_number)
        return results

    def _as_tracked(self, detection: Detection, track: _Track) -> TrackedDetection:
        track_id, track_number = self.labeler.label(track.class_name, track.raw_id)
        return TrackedDetection(
            **detection.model_dump(),
            track_id=track_id,
            track_number=track_number,
            frames_tracked=track.hits,
        )

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_raw_id = 1
        self.labeler.reset()
