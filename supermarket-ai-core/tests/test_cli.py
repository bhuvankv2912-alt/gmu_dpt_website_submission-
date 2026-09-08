"""M1 tests for the CLI wiring."""

from __future__ import annotations

import pytest

from src.main import build_parser, main, resolve_resolution


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.camera == "CAM1"
    assert args.source is None
    assert args.log_level == "INFO"


def test_parser_overrides():
    args = build_parser().parse_args(
        ["--source", "videos/a.mp4", "--camera", "CAM2", "--conf", "0.6",
         "--classes", "person", "bottle", "--max-frames", "5", "--no-video"]
    )
    assert args.conf == 0.6
    assert args.classes == ["person", "bottle"]
    assert args.max_frames == 5
    assert args.no_video is True


def test_resolve_resolution_respects_enabled_flag():
    assert resolve_resolution({"VIDEO_RESOLUTION": {"ENABLED": True, "width": 640, "height": 480}}) == (640, 480)
    assert resolve_resolution({"VIDEO_RESOLUTION": {"ENABLED": False, "width": 640, "height": 480}}) is None
    assert resolve_resolution({}) is None


def test_cli_exits_cleanly_when_source_missing(tmp_path, capsys):
    exit_code = main(["--source", str(tmp_path / "missing.mp4"), "--camera", "CAM1"])
    assert exit_code == 2
    assert "Cannot open video source" in capsys.readouterr().out
