from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("data", "windowing", "preprocessing", "outputs")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config section(s): {', '.join(missing)}")

    lookback = int(config["windowing"]["lookback"])
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    if lookback <= 0:
        raise ValueError("windowing.lookback must be positive")
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("windowing.horizons must contain positive integers")

    chunks = config["data"].get("chunks", [])
    if not chunks:
        raise ValueError("data.chunks must contain at least one chunk")
    for chunk in chunks:
        for key in ("id", "start_mhz", "end_mhz"):
            if key not in chunk:
                raise ValueError(f"Chunk is missing {key!r}: {chunk}")
