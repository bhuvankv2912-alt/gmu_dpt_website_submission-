"""Real tests for the configuration loader (M0/M1)."""

import pytest

from src.config_loader import ConfigError, get_setting, load_cameras, load_config


def test_load_config_contains_required_thresholds():
    config = load_config()
    for key in (
        "DETECTION_CONFIDENCE",
        "REID_SIMILARITY_THRESHOLD",
        "GLOBAL_ID_TIMEOUT",
        "GLOBAL_ID_SEARCH_TIMEOUT",
        "EVENT_CONFIDENCE_THRESHOLD",
        "TRACKING_FPS",
        "FRAME_SKIP",
        "VIDEO_RESOLUTION",
        "MODEL",
        "OUTPUT",
    ):
        assert key in config


def test_reid_threshold_default():
    assert load_config()["REID_SIMILARITY_THRESHOLD"] == 0.70


def test_model_and_output_sections():
    config = load_config()
    assert config["MODEL"]["WEIGHTS"]
    assert config["MODEL"]["DEVICE"]
    assert isinstance(config["MODEL"]["CLASSES"], list)
    assert config["OUTPUT"]["DIR"]
    assert config["OUTPUT"]["VIDEO_CODEC"]


def test_camera_topology():
    cameras = load_cameras()
    assert set(cameras) == {"CAM1", "CAM2", "CAM3", "CAM4"}
    assert cameras["CAM1"]["next_cameras"] == ["CAM2", "CAM3"]
    assert cameras["CAM2"]["next_cameras"] == ["CAM4"]
    assert cameras["CAM3"]["next_cameras"] == ["CAM4"]
    assert cameras["CAM4"]["next_cameras"] == []
    for camera in cameras.values():
        assert camera["source"]


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("config/does_not_exist.yaml")


def test_get_setting_missing_key():
    with pytest.raises(ConfigError):
        get_setting({}, "DETECTION_CONFIDENCE")
