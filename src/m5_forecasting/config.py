"""Configuration loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve relative paths against project root."""
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str(ROOT / p)

    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create output / data directories referenced in config."""
    for key, value in cfg.get("paths", {}).items():
        if key.endswith("_dir"):
            Path(value).mkdir(parents=True, exist_ok=True)
