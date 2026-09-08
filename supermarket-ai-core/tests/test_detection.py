"""Placeholder tests for YOLO detection (M2)."""

import pytest


@pytest.mark.skip(reason="M2: YOLODetector not implemented yet")
def test_detector_returns_detection_models():
    """detect() should return Detection objects with bbox [x1,y1,x2,y2]."""


@pytest.mark.skip(reason="M2: YOLODetector not implemented yet")
def test_detector_filters_by_confidence_threshold():
    """Detections below DETECTION_CONFIDENCE should be discarded."""


@pytest.mark.skip(reason="M2: YOLODetector not implemented yet")
def test_detector_detects_person_class():
    """A frame containing a person should yield at least one 'person' detection."""
