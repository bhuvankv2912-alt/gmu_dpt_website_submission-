"""Multi-video management (M1).

M1 processes recorded files sequentially: the manager resolves camera ids to
file paths from `config/cameras.yaml` and hands out one `VideoSource` at a time.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.logging_setup import get_logger
from src.video.source import VideoSource, VideoSourceError

logger = get_logger(__name__)


class VideoManager:
    """Owns the `VideoSource` objects described by the cameras configuration."""

    def __init__(
        self,
        cameras_config: Dict[str, Any],
        resolution: Optional[Tuple[int, int]] = None,
        frame_skip: int = 0,
    ) -> None:
        self.cameras_config = cameras_config
        self.resolution = resolution
        self.frame_skip = frame_skip
        self.sources: Dict[str, VideoSource] = {}

    def camera_ids(self) -> List[str]:
        return list(self.cameras_config)

    def source_path(self, camera_id: str) -> str:
        """Resolve a camera id to its configured source path."""
        camera = self.cameras_config.get(camera_id)
        if camera is None:
            raise VideoSourceError(
                f"Camera '{camera_id}' is not defined in cameras configuration "
                f"(known: {sorted(self.cameras_config)})"
            )
        source = camera.get("source")
        if not source:
            raise VideoSourceError(f"Camera '{camera_id}' has no 'source' configured")
        return str(source)

    def create_source(self, camera_id: str, source: Optional[str] = None) -> VideoSource:
        """Build (but do not open) a `VideoSource` for one camera."""
        video_source = VideoSource(
            camera_id=camera_id,
            source=source or self.source_path(camera_id),
            resolution=self.resolution,
            frame_skip=self.frame_skip,
        )
        self.sources[camera_id] = video_source
        return video_source

    def iter_sources(self, camera_ids: Optional[List[str]] = None) -> Iterator[VideoSource]:
        """Yield an opened source per camera, skipping unavailable ones.

        One missing or corrupt file must not abort a multi-video run.
        """
        for camera_id in camera_ids or self.camera_ids():
            try:
                source = self.create_source(camera_id).open()
            except VideoSourceError as exc:
                logger.error("Skipping camera %s: %s", camera_id, exc)
                continue
            try:
                yield source
            finally:
                source.release()

    def stop(self) -> None:
        """Release every source this manager created."""
        for source in self.sources.values():
            source.release()
        self.sources.clear()
