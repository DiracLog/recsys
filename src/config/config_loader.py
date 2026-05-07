from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from dotenv import load_dotenv

from src.paths.get_project_root import get_project_root

from src.logging.setup_logging import setup_logging

if TYPE_CHECKING:
    from src.config.config_schema import Config

logger = logging.getLogger(__name__)
setup_logging(log_file="config_loading.log")


def load_config(override_file: str | None = None) -> Config:
    """
    Load pipeline configuration from YAML.

    Resolves sensitive values from .env file. Environment variables
    override configs values when present.

    Parameters
    ----------
    override_file : str | None
        Overrride file name (.yaml)

    Returns
    -------
    Config
        Parsed configuration dictionary.
    """
    load_dotenv()

    project_root: Path = get_project_root()

    config_base_path: Path = project_root / "configs" / "common_config.yaml"

    if override_file is not None:
        config_override_path: Path = project_root / "configs" / override_file
        with config_override_path.open("r", encoding="utf-8") as f:
            override: Config = yaml.safe_load(f)
    else:
        override = {}

    with config_base_path.open("r", encoding="utf-8") as f:
        base: Config = yaml.safe_load(f)

    cfg = deep_update(base, override)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be mapping")

    logger.info(f"current mode: {cfg['data']['mode']}")

    if cfg["data"].get("injection"):
        logger.info("Injecting dataset paths from environment variables")

        mode = cfg["data"].get("mode")

        env_map = {
            "ml1": {
                "ratings_path": "RATINGS_PATH",
                "movies_path": "MOVIES_PATH",
                "users_path": "USERS_PATH",
            },
            "ml32": {
                "ratings_path": "RATINGS_PATH_ML32",
                "movies_path": "MOVIES_PATH_ML32",
            },
        }

        for key, env_var in env_map.get(mode, {}).items():
            value = os.getenv(env_var)
            if value:
                cfg["data"][key] = value

    return cfg


def deep_update(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = deep_update(result[k], v)
            else:
                result[k] = v
        return result
    else:
        return base
