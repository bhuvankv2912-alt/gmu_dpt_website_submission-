"""Loading of YAML configuration files.

This module is functional (part of M0 setup): every other module must read
thresholds and camera topology from here rather than hard-coding values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_CAMERAS_PATH = CONFIG_DIR / "cameras.yaml"


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or malformed."""


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return data


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the main tuning configuration (`config/config.yaml`)."""
    return load_yaml(path)


def load_cameras(path: str | Path = DEFAULT_CAMERAS_PATH) -> Dict[str, Any]:
    """Load the camera topology configuration (`config/cameras.yaml`).

    Returns the mapping of camera id -> camera settings.
    """
    data = load_yaml(path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ConfigError(f"'cameras' mapping missing in {path}")
    return cameras


def get_setting(config: Dict[str, Any], key: str) -> Any:
    """Fetch a required setting, raising a clear error when it is absent."""
    if key not in config:
        raise ConfigError(f"Missing required configuration key: {key}")
    return config[key]
