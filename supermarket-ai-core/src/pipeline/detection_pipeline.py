"""M1 detection pipeline: recorded video in, annotated video + JSON + metrics out.

The pipeline depends only on the `BaseDetector.detect()` contract, so M2 can wrap
or replace the detector (e.g. detection + tracking) without rewriting this module.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.detection.annotator import draw_detections
from src.detection.output import AnnotatedVideoWriter, write_detections_json
from src.detection.yolo_detector import BaseDetector, Detection
from src.logging_setup import get_logger
from src.metrics import ProcessingMetrics
from src.video.source import VideoSource

logger = get_logger(__name__)


class PipelineResult:
    """Artefacts and metrics produced by one video run."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        detections: List[Detection],
        metrics: ProcessingMetrics,
        video_path: Optional[Path],
        json_path: Optional[Path],
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.detections = detections
        self.metrics = metrics
        self.video_path = video_path
        self.json_path = json_path


class DetectionPipeline:
    """Runs detection over a recorded video and writes the M1 artefacts."""

    def __init__(
        self,
        detector: BaseDetector,
        output_dir: str | Path = "output",
        write_video: bool = True,
        write_json: bool = True,
        video_codec: str = "mp4v",
        log_every: int = 50,
    ) -> None:
        self.detector = detector
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
    ) -> PipelineResult:
        """Process every (kept) frame of `source` sequentially."""
        camera_id = source.camera_id
        metrics = ProcessingMetrics(camera_id=camera_id, source=str(source.source))
        detections: List[Detection] = []

        stem = f"{camera_id}_{Path(source.source).stem}"
        video_path = self.output_dir / f"{stem}_annotated.mp4" if self.write_video else None
        json_path = self.output_dir / f"{stem}_detections.json" if self.write_json else None

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
                frame_detections = self.detector.detect(
                    frame, camera_id, frame_number, timestamp
                )
                inference_seconds = time.perf_counter() - inference_start

                detections.extend(frame_detections)
                metrics.record_frame(frame_detections, inference_seconds)

                if writer is not None:
                    writer.write(
                        draw_detections(
                            frame,
                            frame_detections,
                            overlay={
                                "cam": camera_id,
                                "frame": frame_number,
                                "det": len(frame_detections),
                            },
                        )
                    )

                if metrics.frames_processed % self.log_every == 0:
                    logger.info(
                        "[%s] %d frames processed, %d detections so far (%.2f fps)",
                        camera_id,
                        metrics.frames_processed,
                        metrics.detections,
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
            write_detections_json(
                json_path,
                camera_id=camera_id,
                source=str(source.source),
                detections=detections,
                metrics=metrics.to_dict(),
                config=run_config or {},
            )

        return PipelineResult(
            camera_id=camera_id,
            source=str(source.source),
            detections=detections,
            metrics=metrics,
            video_path=video_path,
            json_path=json_path,
        )
