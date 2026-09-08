"""Tests for single-camera multi-object tracking (M2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from src.detection.annotator import draw_tracks, track_color
from src.detection.output import tracks_to_frames, tracks_to_summary, write_tracks_json
from src.detection.yolo_detector import BaseDetector, Detection
from src.metrics import TrackingMetrics
from src.tracking.factory import create_tracker
from src.tracking.tracker import IoUTracker, TrackedDetection, TrackerError, TrackLabeler, iou
from src.tracking.ultralytics_tracker import UltralyticsTracker, resolve_tracker_config


def detection(bbox, frame_number=1, class_name="person", camera_id="CAM1") -> Detection:
    return Detection(
        camera_id=camera_id,
        class_name=class_name,
        confidence=0.9,
        bbox=list(bbox),
        frame_number=frame_number,
        timestamp=datetime.now(timezone.utc),
    )


class ScriptedDetector(BaseDetector):
    """Replays a fixed list of per-frame boxes."""

    def __init__(self, script: List[List[List[float]]]) -> None:
        self.script = script

    def detect(self, frame, camera_id, frame_number, timestamp=None) -> List[Detection]:
        boxes = self.script[frame_number - 1] if frame_number <= len(self.script) else []
        return [detection(box, frame_number, camera_id=camera_id) for box in boxes]


# -- geometry and id formatting -------------------------------------------
def test_iou_of_identical_and_disjoint_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_labeler_formats_ids_and_is_stable():
    labeler = TrackLabeler()
    assert labeler.label("person", 7) == ("person_001", 1)
    assert labeler.label("person", 7) == ("person_001", 1)
    assert labeler.label("person", 9) == ("person_002", 2)
    assert labeler.label("backpack", 9) == ("backpack_001", 1)
    assert labeler.total == 3


# -- IoU tracker -----------------------------------------------------------
def test_tracker_keeps_one_id_while_object_stays_visible():
    tracker = IoUTracker(ScriptedDetector([[[10, 10, 50, 90]] for _ in range(5)]))
    ids = set()
    for frame_number in range(1, 6):
        tracks = tracker.track(None, "CAM1", frame_number)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert ids == {"person_001"}
    assert tracks[0].frames_tracked == 5


def test_tracker_follows_a_moving_object():
    script = [[[10 + step * 5, 10, 50 + step * 5, 90]] for step in range(6)]
    tracker = IoUTracker(ScriptedDetector(script))
    ids = {tracker.track(None, "CAM1", n)[0].track_id for n in range(1, 7)}
    assert ids == {"person_001"}


def test_tracker_assigns_new_ids_to_objects_entering_the_frame():
    script = [
        [[10, 10, 50, 90]],
        [[10, 10, 50, 90], [200, 10, 240, 90]],
    ]
    tracker = IoUTracker(ScriptedDetector(script))
    tracker.track(None, "CAM1", 1)
    tracks = tracker.track(None, "CAM1", 2)
    assert [t.track_id for t in tracks] == ["person_001", "person_002"]


def test_tracker_drops_tracks_after_an_object_leaves():
    script = [[[10, 10, 50, 90]], [], [], [[10, 10, 50, 90]]]
    tracker = IoUTracker(ScriptedDetector(script), max_age=1)
    tracker.track(None, "CAM1", 1)
    tracker.track(None, "CAM1", 2)
    tracker.track(None, "CAM1", 3)
    assert tracker.active_tracks == 0
    reappeared = tracker.track(None, "CAM1", 4)
    assert reappeared[0].track_id == "person_002"


def test_tracker_survives_a_short_miss_within_max_age():
    script = [[[10, 10, 50, 90]], [], [[10, 10, 50, 90]]]
    tracker = IoUTracker(ScriptedDetector(script), max_age=5)
    tracker.track(None, "CAM1", 1)
    tracker.track(None, "CAM1", 2)
    assert tracker.track(None, "CAM1", 3)[0].track_id == "person_001"


def test_tracker_does_not_match_across_classes():
    tracker = IoUTracker(ScriptedDetector([[]]))
    tracker.update([detection([10, 10, 50, 90], class_name="person")])
    tracks = tracker.update([detection([10, 10, 50, 90], 2, class_name="backpack")])
    assert [t.track_id for t in tracks] == ["backpack_001"]


def test_tracker_reset_clears_ids():
    tracker = IoUTracker(ScriptedDetector([[[10, 10, 50, 90]], [[10, 10, 50, 90]]]))
    tracker.track(None, "CAM1", 1)
    tracker.reset()
    assert tracker.active_tracks == 0
    assert tracker.track(None, "CAM1", 2)[0].track_id == "person_001"


def test_min_hits_delays_reporting_a_new_track():
    tracker = IoUTracker(ScriptedDetector([[[10, 10, 50, 90]] for _ in range(3)]), min_hits=2)
    assert tracker.track(None, "CAM1", 1) == []
    assert tracker.track(None, "CAM1", 2)[0].track_id == "person_001"


# -- factory and Ultralytics wiring ---------------------------------------
def test_create_tracker_returns_the_requested_implementation():
    assert isinstance(create_tracker("iou", "yolov8n.pt", 0.5), IoUTracker)
    bytetrack = create_tracker("bytetrack", "yolov8n.pt", 0.5)
    assert isinstance(bytetrack, UltralyticsTracker)
    assert bytetrack.tracker_config == "bytetrack.yaml"
    assert create_tracker("botsort", "yolov8n.pt", 0.5).tracker_config == "botsort.yaml"


def test_create_tracker_rejects_unknown_names():
    with pytest.raises(TrackerError):
        create_tracker("sortish", "yolov8n.pt", 0.5)
    with pytest.raises(TrackerError):
        resolve_tracker_config("sortish")


def test_ultralytics_tracker_is_lazy_about_weights():
    tracker = UltralyticsTracker("yolov8n.pt", 0.4, tracker="botsort", classes=["Person"])
    assert not tracker.is_loaded
    assert tracker.classes == ["person"]
    assert tracker.name == "botsort"


# -- metrics ---------------------------------------------------------------
def tracked(track_id: str, number: int, frame_number: int, frames_tracked: int = 1):
    return TrackedDetection(
        **detection([0, 0, 10, 10], frame_number).model_dump(),
        track_id=track_id,
        track_number=number,
        frames_tracked=frames_tracked,
    )


def test_tracking_metrics_count_unique_ids_and_lifespans():
    metrics = TrackingMetrics(camera_id="CAM1")
    metrics.record_frame([tracked("person_001", 1, 1), tracked("person_002", 2, 1)], 0.01)
    metrics.record_frame([tracked("person_001", 1, 2, 2)], 0.01)
    metrics.finish()

    data = metrics.to_dict()
    assert data["unique_tracks"] == 2
    assert data["tracks_by_class"] == {"person": 2}
    assert data["max_concurrent_tracks"] == 2
    assert data["average_track_length_frames"] == pytest.approx(1.5)
    assert "tracks=2" in metrics.summary()


# -- annotation and JSON ---------------------------------------------------
def test_draw_tracks_labels_boxes_without_mutating_the_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    annotated = draw_tracks(frame, [tracked("person_001", 1, 1)], overlay={"cam": "CAM1"})
    assert annotated.shape == frame.shape
    assert annotated.any()
    assert not frame.any()
    assert track_color(1) == track_color(9)


def test_write_tracks_json_groups_frames_and_summarises_tracks(tmp_path):
    tracks = [
        tracked("person_001", 1, 1),
        tracked("person_002", 2, 1),
        tracked("person_001", 1, 2, 2),
    ]
    assert [f["frame_number"] for f in tracks_to_frames(tracks)] == [1, 2]

    summary = tracks_to_summary(tracks)
    assert [s["track_id"] for s in summary] == ["person_001", "person_002"]
    assert summary[0] == {
        "track_id": "person_001",
        "class_name": "person",
        "first_frame": 1,
        "last_frame": 2,
        "frames_tracked": 2,
        "max_confidence": 0.9,
    }

    path = write_tracks_json(
        tmp_path / "tracks.json",
        camera_id="CAM1",
        source="cam1.mp4",
        tracks=tracks,
        metrics={"unique_tracks": 2},
        config={"tracker": "bytetrack"},
    )
    import json

    payload = json.loads(path.read_text())
    assert payload["milestone"] == "M2"
    assert payload["config"]["tracker"] == "bytetrack"
    assert payload["frames"][0]["tracks"][0]["track_id"] == "person_001"
