"""Multi-camera video management interface (M1 placeholder)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

from src.video.source import VideoSource


class VideoManager:
    """Owns one `VideoSource` per configured camera.

    Responsibilities (M1):
      * build sources from `config/cameras.yaml`
      * iterate frames across cameras without one dead stream stalling the rest
      * reconnect or skip failed sources
      * shut every source down cleanly
    """

    def __init__(self, cameras_config: Dict[str, Any]) -> None:
        self.cameras_config = cameras_config
        self.sources: Dict[str, VideoSource] = {}

    def start(self) -> None:
        """Instantiate and open a `VideoSource` for each configured camera."""
        raise NotImplementedError

    def frames(self) -> Iterator[Tuple[str, int, Any]]:
        """Yield `(camera_id, frame_number, frame)` across all active sources."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release every managed source."""
        raise NotImplementedError
