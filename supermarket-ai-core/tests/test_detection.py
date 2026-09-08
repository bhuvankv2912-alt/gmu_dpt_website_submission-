"""M1 tests for detection models, annotation and JSON output."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from src.detection.annotator import class_color, draw_detections
from src.detection.output import (
    AnnotatedVideoWriter,
    detections_to_frames,
    write_detections_json,
)
from src.detection.yolo_detector import Detection, DetectorError, YOLODetector


def make_detection(frame_number=1, class_name="person", confidence=0.9) -> Detection:
    return Detection(
        camera_id="CAM1",
        class_name=class_name,
        confidence=confidence,
        bbox=[10.0, 20.0, 50.0, 80.0],
        frame_number=frame_number,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_detection_geometry():
    detection = make_detection()
    assert (detection.width, detection.height) == (40.0, 60.0)
    assert detection.area == 2400.0


def test_detection_rejects_bad_payloads():
    with pytest.raises(ValidationError):
        Detection(
            camera_id="CAM1",
            class_name="person",
            confidence=1.4,
            bbox=[0, 0, 1, 1],
            frame_number=1,
            timestamp=datetime.now(timezone.utc),
        )
    with pytest.raises(ValidationError):
        Detection(
            camera_id="CAM1",
            class_name="person",
            confidence=0.5,
            bbox=[0, 0, 1],
            frame_number=1,
            timestamp=datetime.now(timezone.utc),
        )


def test_annotator_draws_and_does_not_mutate_input():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    annotated = draw_detections(frame, [make_detection()], overlay={"frame": 1})
    assert annotated.shape == frame.shape
    assert annotated.any()
    assert not frame.any()


def test_person_class_colour_is_stable():
    assert class_color("person") == class_color("person")
    assert class_color("bottle") == class_color("bottle")


def test_detections_are_grouped_by_frame():
    frames = detections_to_frames(
        [make_detection(1), make_detection(3), make_detection(1, "bottle")]
    )
    assert [f["frame_number"] for f in frames] == [1, 3]
    assert len(frames[0]["detections"]) == 2


def test_json_artifact_structure(tmp_path):
    path = write_detections_json(
        tmp_path / "out.json",
        camera_id="CAM1",
        source="videos/cam1.mp4",
        detections=[make_detection(1), make_detection(2, "bottle", 0.6)],
        metrics={"frames_processed": 2, "detections": 2, "fps": 12.5},
        config={"detection_confidence": 0.45},
    )
    payload = json.loads(path.read_text())
    assert payload["milestone"] == "M1"
    assert payload["camera_id"] == "CAM1"
    assert payload["metrics"]["detections"] == 2
    assert payload["config"]["detection_confidence"] == 0.45
    assert len(payload["frames"]) == 2
    assert payload["frames"][1]["detections"][0]["class_name"] == "bottle"


def test_annotated_video_writer_roundtrip(tmp_path):
    path = tmp_path / "annotated.mp4"
    with AnnotatedVideoWriter(path, size=(160, 120), fps=10.0) as writer:
        for _ in range(5):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        assert writer.frames_written == 5
    assert path.is_file() and path.stat().st_size > 0


def test_writer_resizes_mismatched_frames(tmp_path):
    with AnnotatedVideoWriter(tmp_path / "a.mp4", size=(80, 60), fps=10.0) as writer:
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        assert writer.frames_written == 1


def test_detector_reports_missing_weights_clearly():
    pytest.importorskip("ultralytics")
    detector = YOLODetector("models/definitely_missing.pt", confidence_threshold=0.5)
    with pytest.raises(DetectorError, match="not found"):
        detector.load()


def test_detector_defaults():
    detector = YOLODetector("yolov8n.pt", 0.4, device="auto", classes=["Person"])
    assert detector.device is None
    assert detector.classes == ["person"]
    assert not detector.is_loaded
    assert detector.class_names == []
