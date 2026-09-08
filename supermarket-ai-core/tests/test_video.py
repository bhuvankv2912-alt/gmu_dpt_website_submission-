"""Placeholder tests for the video input layer (M1)."""

import pytest


@pytest.mark.skip(reason="M1: VideoSource not implemented yet")
def test_video_source_reads_local_file():
    """Should open a local mp4 and yield frames with increasing frame numbers."""


@pytest.mark.skip(reason="M1: VideoSource not implemented yet")
def test_video_source_fails_gracefully_on_bad_source():
    """Should report an unavailable source without raising or hanging."""


@pytest.mark.skip(reason="M1: VideoSource not implemented yet")
def test_video_source_respects_resolution_and_frame_skip():
    """Frames should match VIDEO_RESOLUTION and honour FRAME_SKIP."""


@pytest.mark.skip(reason="M1: VideoManager not implemented yet")
def test_video_manager_multiplexes_cameras():
    """Should yield frames from all configured cameras and stop them cleanly."""
