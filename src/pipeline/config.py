"""Configuration helpers for the anomaly detection pipeline."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path("config/pipeline.yaml")


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load pipeline configuration from YAML and return as a dictionary."""
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative_path: str) -> Path:
    """Return absolute Path for a project-relative location."""
    return Path(relative_path).expanduser().resolve()

