"""Configuration loading and path resolution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML and resolve dataset/project paths without machine defaults."""
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("Configuration must contain a dataset mapping.")
    project_root = next(
        (candidate for candidate in (config_path.parent, *config_path.parents) if (candidate / "pyproject.toml").is_file()),
        config_path.parent,
    )
    config["_config_path"] = config_path
    config["_project_root"] = project_root
    config["dataset"]["rhb_root"] = Path(config["dataset"]["rhb_root"]).expanduser().resolve()
    if not config["dataset"]["rhb_root"].is_dir():
        raise FileNotFoundError(config["dataset"]["rhb_root"])
    return config


def config_hash(config: dict) -> str:
    """Hash only user-visible configuration values."""
    plain = {k: v for k, v in config.items() if not k.startswith("_")}
    payload = json.dumps(plain, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def project_path(config: dict, key: str) -> Path:
    """Resolve one entry below config.paths against the project root."""
    return config["_project_root"] / config["paths"][key]
