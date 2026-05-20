"""Config loader — reads config/config.yaml and validates required keys."""

from pathlib import Path

import yaml

from src.exceptions import ConfigError

_REQUIRED_TOP_KEYS = {"seed", "data", "model", "evaluation", "monitoring", "api"}


def load_config(path: str | Path) -> dict:
    """Load config/config.yaml. Raise ConfigError on invalid YAML or missing keys."""
    try:
        with open(str(path)) as fh:
            cfg = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(cfg)}")
    missing = _REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ConfigError(f"Config missing required keys: {missing}")
    return cfg
