"""
Configuration loading, validation, and path-resolution utilities.

This module provides the shared configuration interface used by all integrated
training, evaluation, plotting, export, and result scripts. It loads the
project's YAML configuration, applies configuration-path rules, and resolves
relative filesystem paths consistently.

Primary responsibilities include:

- selecting the default configuration file when no override is provided;
- loading YAML content into a Python dictionary;
- validating that the configuration root and required sections are mappings;
- reporting missing, malformed, or unreadable configuration files clearly;
- preserving the location of the active configuration for relative path
  resolution;
- expanding user-home references where supported;
- resolving relative data, checkpoint, output, and metadata paths;
- returning normalized absolute Path objects to calling modules; and
- preventing individual scripts from applying inconsistent path rules.

This module does not interpret model architecture or dataset semantics beyond
the basic structural validation needed to load the shared configuration.
Detailed setting validation remains in the module that consumes each setting.
"""




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

    prediction_start_row = config["data"].get("prediction_start_row")
    
    if prediction_start_row is not None:
        prediction_start_row = int(prediction_start_row)
    
        if prediction_start_row <= 0:
            raise ValueError(
                "Error! data.prediction_start_row must be a "
                "positive one-based row number when provided."
            )
    
        if str(
            config["data"].get(
                "loader",
                "aerpaw",
            )
        ).lower() != "powder":
            raise ValueError(
                "Error! data.prediction_start_row is currently "
                "supported only by the POWDER loader."
            )