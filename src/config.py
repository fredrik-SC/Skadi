"""Configuration loading and access for Skaði."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file. Defaults to config/default.yaml.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    logger.info("Loading configuration from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        logger.warning("Configuration file %s is empty, returning empty dict", path)
        return {}

    return config
