"""Real tests for logging setup (M0)."""

import logging

from src.logging_setup import get_logger, setup_logging


def test_setup_logging_installs_single_handler():
    root = setup_logging()
    setup_logging()
    assert len(root.handlers) == 1
    assert root.level == logging.INFO


def test_logger_emits(capsys):
    setup_logging()
    get_logger("test.module").info("hello")
    assert "hello" in capsys.readouterr().out


def test_get_logger_name():
    assert get_logger("src.video.source").name == "src.video.source"
