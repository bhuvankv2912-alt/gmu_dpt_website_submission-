"""Tests for the end-to-end detection (M1) and tracking (M2) pipelines."""

from __future__ import annotations

import json

from src.metrics import ProcessingMetrics
from src.pipeline.detection_pipeline import DetectionPipeline
from src.pipeline.tracking_pipeline import TrackingPipeline
from src.tracking.tracker import IoUTracker
from src.video.source import VideoSource
from tests.conftest import FakeDetector


def test_pipeline_produces_video_json_and_metrics(sample_video, tmp_path, fake_detector):
    pipeline = DetectionPipeline(fake_detector, output_dir=tmp_path)
    with VideoSource("CAM1", sample_video) as source:
        result = pipeline.run(source, run_config={"detection_confidence": 0.45})

    assert result.metrics.frames_processed == 12
    assert result.metrics.detections == 12
    assert result.metrics.detections_by_class["person"] == 12
    assert result.metrics.fps > 0
    assert fake_detector.calls == 12

    assert result.video_path.is_file() and result.video_path.stat().st_size > 0
    payload = json.loads(result.json_path.read_text())
    assert len(payload["frames"]) == 12
    assert payload["metrics"]["frames_processed"] == 12
    assert payload["config"]["detection_confidence"] == 0.45


def test_pipeline_respects_max_frames_and_frame_skip(sample_video, tmp_path):
    pipeline = DetectionPipeline(FakeDetector(per_frame=2), output_dir=tmp_path, write_video=False)
    with VideoSource("CAM1", sample_video, frame_skip=1) as source:
        result = pipeline.run(source, max_frames=3)

    assert result.metrics.frames_processed == 3
    assert result.metrics.detections == 6
    assert result.video_path is None
    assert [d.frame_number for d in result.detections][:3] == [1, 1, 3]


def test_pipeline_can_skip_all_artifacts(sample_video, tmp_path):
    output_dir = tmp_path / "out"
    pipeline = DetectionPipeline(
        FakeDetector(), output_dir=output_dir, write_video=False, write_json=False
    )
    with VideoSource("CAM1", sample_video) as source:
        result = pipeline.run(source)
    assert result.json_path is None
    assert result.video_path is None
    assert not output_dir.exists()


def test_pipeline_handles_video_with_no_detections(sample_video, tmp_path):
    detector = FakeDetector(per_frame=0)
    pipeline = DetectionPipeline(detector, output_dir=tmp_path)
    with VideoSource("CAM1", sample_video) as source:
        result = pipeline.run(source)
    assert result.metrics.detections == 0
    assert json.loads(result.json_path.read_text())["frames"] == []


def test_metrics_summary_and_dict():
    metrics = ProcessingMetrics(camera_id="CAM1", source="videos/cam1.mp4")
    metrics.frames_read = 4
    metrics.record_frame([], 0.01)
    metrics.finish()
    data = metrics.to_dict()
    assert data["frames_processed"] == 1
    assert data["detections"] == 0
    assert data["fps"] >= 0
    assert "CAM1" in metrics.summary()


def test_metrics_are_zero_safe():
    metrics = ProcessingMetrics().finish()
    assert metrics.fps == 0.0
    assert metrics.average_detections_per_frame == 0.0


def test_tracking_pipeline_produces_video_json_and_stable_ids(sample_video, tmp_path):
    pipeline = TrackingPipeline(IoUTracker(FakeDetector(per_frame=2)), output_dir=tmp_path)
    with VideoSource("CAM1", sample_video) as source:
        result = pipeline.run(source, run_config={"tracker": "iou"})

    assert result.metrics.frames_processed == 12
    assert result.metrics.unique_tracks == 2
    assert result.track_ids == ["person_001", "person_002"]
    assert result.metrics.max_concurrent_tracks == 2

    assert result.video_path.name.endswith("_tracked.mp4")
    assert result.video_path.is_file() and result.video_path.stat().st_size > 0

    payload = json.loads(result.json_path.read_text())
    assert payload["milestone"] == "M2"
    assert payload["config"]["tracker"] == "iou"
    assert [t["track_id"] for t in payload["tracks"]] == ["person_001", "person_002"]
    assert payload["tracks"][0]["frames_tracked"] == 12
    assert payload["metrics"]["unique_tracks"] == 2


def test_tracking_pipeline_respects_max_frames(sample_video, tmp_path):
    pipeline = TrackingPipeline(
        IoUTracker(FakeDetector()), output_dir=tmp_path, write_video=False, write_json=False
    )
    with VideoSource("CAM1", sample_video) as source:
        result = pipeline.run(source, max_frames=4)

    assert result.metrics.frames_processed == 4
    assert result.metrics.frames_read == 4
    assert result.video_path is None and result.json_path is None
