"""M1 tests for the recorded-video input layer."""

from __future__ import annotations

import pytest

from src.video.manager import VideoManager
from src.video.source import VideoSource, VideoSourceError
from tests.conftest import make_video


def test_reads_every_frame_sequentially(sample_video):
    with VideoSource("CAM1", sample_video) as source:
        numbers = [number for number, _ in source.frames()]
    assert numbers == list(range(1, 13))


def test_metadata_exposed(sample_video):
    with VideoSource("CAM1", sample_video) as source:
        assert (source.width, source.height) == (160, 120)
        assert source.source_fps == pytest.approx(10.0, abs=0.5)
        assert source.frame_count == 12


def test_frame_skip_keeps_every_other_frame(sample_video):
    with VideoSource("CAM1", sample_video, frame_skip=1) as source:
        numbers = [number for number, _ in source.frames()]
        assert numbers == [1, 3, 5, 7, 9, 11]
        assert source.output_fps == pytest.approx(5.0, abs=0.5)


def test_resolution_is_applied(sample_video):
    with VideoSource("CAM1", sample_video, resolution=(80, 60)) as source:
        _, frame = next(source.frames())
        assert frame.shape[:2] == (60, 80)
        assert source.output_size == (80, 60)


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoSourceError, match="not found"):
        VideoSource("CAM1", tmp_path / "nope.mp4").open()


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"not a video")
    with pytest.raises(VideoSourceError, match="Unsupported video extension"):
        VideoSource("CAM1", path).open()


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"still not a video")
    with pytest.raises(VideoSourceError, match="could not open"):
        VideoSource("CAM1", path).open()


def test_read_before_open_raises(tmp_path):
    with pytest.raises(VideoSourceError, match="before open"):
        VideoSource("CAM1", tmp_path / "x.mp4").read()


def test_release_is_idempotent(sample_video):
    source = VideoSource("CAM1", sample_video).open()
    source.release()
    source.release()
    assert not source.is_open()


def test_manager_resolves_camera_source(sample_video):
    manager = VideoManager({"CAM1": {"source": str(sample_video), "next_cameras": []}})
    assert manager.source_path("CAM1") == str(sample_video)
    with pytest.raises(VideoSourceError, match="not defined"):
        manager.source_path("CAM9")


def test_manager_skips_unavailable_sources(tmp_path):
    good = make_video(tmp_path / "good.mp4")
    manager = VideoManager(
        {
            "CAM1": {"source": str(tmp_path / "missing.mp4")},
            "CAM2": {"source": str(good)},
        }
    )
    opened = [source.camera_id for source in manager.iter_sources()]
    assert opened == ["CAM2"]
