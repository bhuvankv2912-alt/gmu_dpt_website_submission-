"""CLI demo entrypoint (placeholder for the M12 demo mode)."""

from __future__ import annotations

import argparse
from typing import List, Optional

from src.config_loader import load_cameras, load_config
from src.logging_setup import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supermarket-ai-core",
        description="AI core engine demo runner (pipeline not yet implemented).",
    )
    parser.add_argument(
        "--source",
        help="Video source: file path, webcam index, or RTSP URL. "
             "Defaults to the configured source for --camera.",
    )
    parser.add_argument(
        "--camera",
        default="CAM1",
        help="Camera id from config/cameras.yaml (default: CAM1).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    logger = get_logger("src.main")

    config = load_config()
    cameras = load_cameras()

    camera = cameras.get(args.camera)
    source = args.source or (camera or {}).get("source")

    logger.info("Loaded %d configuration keys and %d cameras", len(config), len(cameras))
    logger.info("Camera: %s | source: %s", args.camera, source)
    if camera is None:
        logger.warning("Camera %s is not defined in config/cameras.yaml", args.camera)

    logger.info(
        "Milestone M0 only: architecture and setup are in place, "
        "but the processing pipeline is not yet implemented."
    )
    print("Pipeline not yet implemented (M0: architecture & setup only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
