"""Shared fixtures for the M1 test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from src.detection.yolo_detector import BaseDetector, Detection


def make_video(path: Path, frames: int = 12, size=(160, 120), fps: float = 10.0) -> Path:
    """Write a small synthetic video so tests never need real footage."""
    width, height = size
    for codec in ("mp4v", "MJPG"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            break
        writer.release()
    else:  # pragma: no cover - depends on the OpenCV build
        pytest.skip("No usable OpenCV video codec available")

    for index in range(frames):
        frame = np.full((height, width, 3), index * 5 % 255, dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (60, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "cam.mp4")


class FakeDetector(BaseDetector):
    """Deterministic stand-in for YOLO so pipeline tests need no model weights."""

    def __init__(self, per_frame: int = 1, class_name: str = "person") -> None:
        self.per_frame = per_frame
        self.class_name = class_name
        self.calls = 0

    def detect(self, frame, camera_id, frame_number, timestamp=None) -> List[Detection]:
        self.calls += 1
        stamp = timestamp or datetime.now(timezone.utc)
        return [
            Detection(
                camera_id=camera_id,
                class_name=self.class_name,
                confidence=0.9,
                bbox=[10.0 + index, 10.0, 60.0 + index, 90.0],
                frame_number=frame_number,
                timestamp=stamp,
            )
            for index in range(self.per_frame)
        ]


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
