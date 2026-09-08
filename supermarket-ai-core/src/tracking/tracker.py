"""Single-camera tracking interface (M2 placeholder)."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel

from src.detection.yolo_detector import Detection


class Track(BaseModel):
    """A detection associated with a camera-local track identity."""

    local_track_id: str  # e.g. "C1_07"
    camera_id: str
    detection: Detection


class Tracker:
    """Associates detections across frames within one camera.

    Backed by BoT-SORT or ByteTrack (M2). Track IDs are camera-scoped and
    formatted as `C<camera index>_<track number>`, e.g. `C1_07`; global identity
    resolution is the `GlobalIDManager`'s job, not the tracker's.
    """

    def __init__(self, camera_id: str, tracker_type: str = "botsort") -> None:
        self.camera_id = camera_id
        self.tracker_type = tracker_type

    def update(self, frame: Any, detections: List[Detection]) -> List[Track]:
        """Advance the tracker by one frame and return current tracks."""
        raise NotImplementedError

    def format_local_id(self, track_number: int) -> str:
        """Build the camera-scoped local track id, e.g. `C1_07`."""
        raise NotImplementedError

    def reset(self) -> None:
        """Drop all track state."""
        raise NotImplementedError
