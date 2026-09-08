"""M2 tracking pipeline: recorded video in, id-annotated video + tracking JSON out.

Mirrors `DetectionPipeline` but depends on the `BaseTracker.track()` contract, so
ByteTrack, BoT-SORT or the IoU fallback plug in interchangeably. Re-ID (M3) and
behaviour analysis (M5+) can consume the `TrackedDetection` stream this produces
without changing this module.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.detection.annotator import draw_tracks
from src.detection.output import AnnotatedVideoWriter, write_tracks_json
from src.logging_setup import get_logger
from src.metrics import TrackingMetrics
from src.tracking.tracker import BaseTracker, TrackedDetection
from src.video.source import VideoSource

logger = get_logger(__name__)


class TrackingResult:
    """Artefacts and metrics produced by one tracked video run."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        tracks: List[TrackedDetection],
        metrics: TrackingMetrics,
        video_path: Optional[Path],
        json_path: Optional[Path],
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.tracks = tracks
        self.metrics = metrics
        self.video_path = video_path
        self.json_path = json_path

    @property
    def track_ids(self) -> List[str]:
        """Every temporary track id seen in this run, in first-seen order."""
        seen: Dict[str, None] = {}
        for track in self.tracks:
            seen.setdefault(track.track_id, None)
        return list(seen)


class TrackingPipeline:
    """Runs a tracker over a recorded video and writes the M2 artefacts."""

    def __init__(
        self,
        tracker: BaseTracker,
        output_dir: str | Path = "output",
        write_video: bool = True,
        write_json: bool = True,
        video_codec: str = "mp4v",
        log_every: int = 50,
    ) -> None:
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.write_video = write_video
        self.write_json = write_json
        self.video_codec = video_codec
        self.log_every = max(1, log_every)

    def run(
        self,
        source: VideoSource,
        max_frames: Optional[int] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> TrackingResult:
        """Process every (kept) frame of `source` sequentially, keeping track state."""
        camera_id = source.camera_id
        metrics = TrackingMetrics(camera_id=camera_id, source=str(source.source))
        tracks: List[TrackedDetection] = []

        stem = f"{camera_id}_{Path(source.source).stem}"
        video_path = self.output_dir / f"{stem}_tracked.mp4" if self.write_video else None
        json_path = self.output_dir / f"{stem}_tracks.json" if self.write_json else None

        writer: Optional[AnnotatedVideoWriter] = None
        if video_path is not None:
            writer = AnnotatedVideoWriter(
                video_path, source.output_size, source.output_fps, self.video_codec
            ).open()

        fps = source.source_fps or 25.0
        start_wall = datetime.now(timezone.utc)

        try:
            for frame_number, frame in source.frames():
                timestamp = start_wall + timedelta(seconds=(frame_number - 1) / fps)

                inference_start = time.perf_counter()
                frame_tracks = self.tracker.track(frame, camera_id, frame_number, timestamp)
                inference_seconds = time.perf_counter() - inference_start

                tracks.extend(frame_tracks)
                metrics.record_frame(frame_tracks, inference_seconds)

                if writer is not None:
                    writer.write(
                        draw_tracks(
                            frame,
                            frame_tracks,
                            overlay={
                                "cam": camera_id,
                                "frame": frame_number,
                                "tracks": len(frame_tracks),
                                "ids": metrics.unique_tracks,
                            },
                        )
                    )

                if metrics.frames_processed % self.log_every == 0:
                    logger.info(
                        "[%s] %d frames processed, %d ids so far (%.2f fps)",
                        camera_id,
                        metrics.frames_processed,
                        metrics.unique_tracks,
                        metrics.fps,
                    )

                if max_frames and metrics.frames_processed >= max_frames:
                    logger.info("[%s] stopping at --max-frames=%d", camera_id, max_frames)
                    break
        finally:
            metrics.frames_read = source.frames_read
            if writer is not None:
                writer.release()

        metrics.finish()
        logger.info(metrics.summary())

        if json_path is not None:
            write_tracks_json(
                json_path,
                camera_id=camera_id,
                source=str(source.source),
                tracks=tracks,
                metrics=metrics.to_dict(),
                config=run_config or {},
            )

        return TrackingResult(
            camera_id=camera_id,
            source=str(source.source),
            tracks=tracks,
            metrics=metrics,
            video_path=video_path,
            json_path=json_path,
        )
