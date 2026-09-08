"""Single video source interface (M1 placeholder)."""

from __future__ import annotations

from typing import Any, Optional, Tuple


class VideoSource:
    """A single video stream: local file, webcam index, or RTSP URL.

    Responsibilities (M1):
      * open a source and fail gracefully when it is unavailable
      * honour the configured FPS, frame skip and target resolution
      * expose frames one at a time with a monotonically increasing frame number
      * release all handles on shutdown
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        fps: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
        frame_skip: int = 0,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.fps = fps
        self.resolution = resolution
        self.frame_skip = frame_skip

    def open(self) -> None:
        """Open the underlying stream. Must not raise on transient failure."""
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Return `(ok, frame)`; `ok` is False when the stream is exhausted."""
        raise NotImplementedError

    def is_open(self) -> bool:
        """Whether the stream is currently readable."""
        raise NotImplementedError

    def release(self) -> None:
        """Release capture handles cleanly."""
        raise NotImplementedError

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
