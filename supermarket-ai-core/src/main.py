"""CLI entrypoint.

M1 mode: run YOLO detection over a recorded video and write an annotated video,
a detection JSON file and processing metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config_loader import ConfigError, load_cameras, load_config
from src.detection.yolo_detector import DetectorError, YOLODetector
from src.logging_setup import get_logger, setup_logging
from src.metrics import ProcessingMetrics
from src.pipeline.detection_pipeline import DetectionPipeline
from src.video.manager import VideoManager
from src.video.source import VideoSourceError

logger = get_logger("src.main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supermarket-ai-core",
        description="M1: recorded-video object detection (no tracking, no live streams).",
    )
    parser.add_argument("--source", help="Path to a recorded .mp4/.avi/.mov file. "
                                         "Defaults to the source configured for --camera.")
    parser.add_argument("--camera", default="CAM1",
                        help="Camera id from config/cameras.yaml (default: CAM1).")
    parser.add_argument("--weights", help="Override MODEL.WEIGHTS.")
    parser.add_argument("--conf", type=float, help="Override DETECTION_CONFIDENCE.")
    parser.add_argument("--classes", nargs="*",
                        help="Restrict output to these class names (default: all model classes).")
    parser.add_argument("--frame-skip", type=int, help="Override FRAME_SKIP.")
    parser.add_argument("--max-frames", type=int,
                        help="Stop after N processed frames (useful for smoke runs).")
    parser.add_argument("--output-dir", help="Override OUTPUT.DIR.")
    parser.add_argument("--no-video", action="store_true", help="Skip the annotated video.")
    parser.add_argument("--no-json", action="store_true", help="Skip the detection JSON.")
    parser.add_argument("--device", help="Override MODEL.DEVICE (cpu, cuda, cuda:0, mps).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def resolve_resolution(config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Read VIDEO_RESOLUTION, honouring the ENABLED flag."""
    resolution = config.get("VIDEO_RESOLUTION") or {}
    if not resolution.get("ENABLED", True):
        return None
    width, height = resolution.get("width"), resolution.get("height")
    if not width or not height:
        return None
    return int(width), int(height)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        config = load_config()
        cameras = load_cameras()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    model_config = config.get("MODEL") or {}
    output_config = config.get("OUTPUT") or {}

    confidence = args.conf if args.conf is not None else config["DETECTION_CONFIDENCE"]
    frame_skip = args.frame_skip if args.frame_skip is not None else config.get("FRAME_SKIP", 0)
    classes = args.classes if args.classes else (model_config.get("CLASSES") or None)
    resolution = resolve_resolution(config)

    manager = VideoManager(cameras, resolution=resolution, frame_skip=frame_skip)
    try:
        source = manager.create_source(args.camera, args.source).open()
    except VideoSourceError as exc:
        logger.error("Cannot open video source: %s", exc)
        return 2

    detector = YOLODetector(
        weights_path=args.weights or model_config.get("WEIGHTS", "yolov8n.pt"),
        confidence_threshold=confidence,
        device=args.device or model_config.get("DEVICE", "auto"),
        classes=classes,
    )
    try:
        detector.load()
    except DetectorError as exc:
        logger.error("Cannot load detection model: %s", exc)
        source.release()
        return 3

    pipeline = DetectionPipeline(
        detector=detector,
        output_dir=args.output_dir or output_config.get("DIR", "output"),
        write_video=not args.no_video and output_config.get("ANNOTATED_VIDEO", True),
        write_json=not args.no_json and output_config.get("JSON", True),
        video_codec=output_config.get("VIDEO_CODEC", "mp4v"),
        log_every=int(output_config.get("LOG_EVERY", 50)),
    )

    run_config = {
        "detection_confidence": confidence,
        "frame_skip": frame_skip,
        "resolution": list(resolution) if resolution else "native",
        "weights": detector.weights_path,
        "classes": classes or "all",
    }

    try:
        result = pipeline.run(source, max_frames=args.max_frames, run_config=run_config)
    finally:
        source.release()

    report(result.metrics, result.video_path, result.json_path)
    return 0


def report(metrics: ProcessingMetrics, video_path: Optional[Path], json_path: Optional[Path]) -> None:
    """Print the run summary to stdout."""
    data = metrics.to_dict()
    print("\nM1 detection run complete")
    print(f"  camera            : {data['camera_id']}")
    print(f"  source            : {data['source']}")
    print(f"  frames read       : {data['frames_read']}")
    print(f"  frames processed  : {data['frames_processed']}")
    print(f"  detections        : {data['detections']}")
    print(f"  by class          : {data['detections_by_class'] or '{}'}")
    print(f"  elapsed / fps     : {data['elapsed_seconds']}s / {data['fps']}")
    if video_path:
        print(f"  annotated video   : {video_path}")
    if json_path:
        print(f"  detections json   : {json_path}")


if __name__ == "__main__":
    sys.exit(main())
