"""Recorded-video input (M1).

M1 scope is *recorded video only* (`.mp4`, `.avi`, `.mov`). Webcam and RTSP
inputs are deliberately rejected here; they arrive in a later milestone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import cv2

from src.logging_setup import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """A single recorded video file, read sequentially frame by frame.

    Honours the configured target resolution and frame skip, exposes source
    metadata (fps, size, frame count) and releases its capture handle cleanly.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | Path,
        resolution: Optional[Tuple[int, int]] = None,
        frame_skip: int = 0,
    ) -> None:
        self.camera_id = camera_id
        self.source = Path(source)
        self.resolution = resolution
        self.frame_skip = max(0, int(frame_skip))
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_number = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "VideoSource":
        """Open the file, validating existence, extension and decodability."""
        if not self.source.is_file():
            raise VideoSourceError(f"Video file not found: {self.source}")
        suffix = self.source.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise VideoSourceError(
                f"Unsupported video extension '{suffix}' for {self.source}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        capture = cv2.VideoCapture(str(self.source))
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"OpenCV could not open video: {self.source}")

        self._capture = capture
        self._frame_number = 0
        logger.info(
            "[%s] opened %s (%dx%d @ %.2f fps, %d frames)",
            self.camera_id,
            self.source,
            self.width,
            self.height,
            self.source_fps,
            self.frame_count,
        )
        return self

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        """Release the capture handle. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("[%s] released %s", self.camera_id, self.source)

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # -- metadata ----------------------------------------------------------
    def _prop(self, prop: int, default: float = 0.0) -> float:
        if self._capture is None:
            return default
        value = self._capture.get(prop)
        return value if value and value > 0 else default

    @property
    def source_fps(self) -> float:
        return self._prop(cv2.CAP_PROP_FPS, 0.0)

    @property
    def frame_count(self) -> int:
        return int(self._prop(cv2.CAP_PROP_FRAME_COUNT, 0.0))

    @property
    def width(self) -> int:
        return int(self._prop(cv2.CAP_PROP_FRAME_WIDTH, 0.0))

    @property
    def height(self) -> int:
        return int(self._prop(cv2.CAP_PROP_FRAME_HEIGHT, 0.0))

    @property
    def frames_read(self) -> int:
        """Number of frames decoded from the file so far (before frame skipping)."""
        return self._frame_number

    @property
    def output_size(self) -> Tuple[int, int]:
        """`(width, height)` of the frames this source yields."""
        if self.resolution:
            return int(self.resolution[0]), int(self.resolution[1])
        return self.width, self.height

    @property
    def output_fps(self) -> float:
        """Effective fps after frame skipping, used for the annotated video."""
        fps = self.source_fps or 25.0
        return fps / (self.frame_skip + 1)

    # -- reading -----------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[Any]]:
        """Read the next raw frame. Returns `(False, None)` at end of stream."""
        if self._capture is None:
            raise VideoSourceError(f"[{self.camera_id}] read() before open()")
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        self._frame_number += 1
        return True, frame

    def frames(self) -> Iterator[Tuple[int, Any]]:
        """Yield `(frame_number, frame)` sequentially.

        Applies `frame_skip` (1 frame kept out of every `frame_skip + 1`) and
        resizes to the configured resolution. Frame numbers refer to the source
        video, so they stay meaningful when frames are skipped. Decode failures
        mid-file end iteration rather than raising, so a truncated file still
        produces usable output.
        """
        if self._capture is None:
            raise VideoSourceError(f"[{self.camera_id}] frames() before open()")

        step = self.frame_skip + 1
        while True:
            try:
                ok, frame = self.read()
            except cv2.error as exc:  # pragma: no cover - decoder specific
                logger.warning("[%s] decode error, stopping: %s", self.camera_id, exc)
                break
            if not ok:
                break
            if (self._frame_number - 1) % step != 0:
                continue
            yield self._frame_number, self.prepare(frame)

    def prepare(self, frame: Any) -> Any:
        """Resize a frame to the configured resolution, if one is set."""
        if self.resolution is None:
            return frame
        target = (int(self.resolution[0]), int(self.resolution[1]))
        if (frame.shape[1], frame.shape[0]) == target:
            return frame
        return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
