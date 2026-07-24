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

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def resolve_path(value: str | Path) -> Path:
    """Add root to path"""
    
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def unique_run_dir(path: Path) -> Path:
    """Create and return the first available path with a numeric suffix."""
    suffix = 0
    while True:
        candidate = path if suffix == 0 else path.with_name(f"{path.name}_{suffix}")
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            suffix += 1


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """ Loads configuration file"""
    
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """" Checks to ensure all expected config settings are present"""
    
    # List of main config sections
    required = ("data", "windowing", "preprocessing", "training")
    
    # Checks for ecag section in config
    missing = [key for key in required if key not in config]
    
    #Throw error for missing section
    if missing:
        raise ValueError(f"Missing config section(s): {', '.join(missing)}")

    model_names(config)

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
        if float(chunk["start_mhz"]) > float(chunk["end_mhz"]):
            raise ValueError(f"Chunk start_mhz exceeds end_mhz: {chunk}")

    data = config["data"]
    if "representation" in data:
        representation = str(data["representation"]).lower()
        if representation not in {"1d", "2d", "4d"}:
            raise ValueError("data.representation must be one of: 1d, 2d, 4d")
        files = data.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("data.files must be a non-empty list")
        partitions = set()
        for entry in files:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ValueError("Each data.files entry requires path and partition")
            partition = entry.get("partition")
            if partition not in {"train", "test"}:
                raise ValueError("Each data.files partition must be train or test")
            partitions.add(partition)
        if partitions != {"train", "test"}:
            raise ValueError("data.files must include both train and test partitions")
        if "train_files" in data or "test_files" in data:
            raise ValueError("Use data.files; train_files and test_files are no longer supported")
        if data.get("split", {}).get("test_rows") is not None:
            raise ValueError("Test data is selected by file partition; remove data.split.test_rows")
        ranges = (data.get("split") or {}).get("ranges")
        if ranges is not None:
            if not isinstance(ranges, dict) or set(ranges) != {"train", "validation", "test"}:
                raise ValueError("data.split.ranges must define train, validation, and test")
            for name, bounds in ranges.items():
                bounds_list = bounds if isinstance(bounds, list) else [bounds]
                if not bounds_list:
                    raise ValueError(f"data.split.ranges.{name} must not be empty")
                for bounds_item in bounds_list:
                    if not isinstance(bounds_item, dict) or set(bounds_item) != {"start", "end"}:
                        raise ValueError(f"data.split.ranges.{name} requires start and end")
                    start = pd.Timestamp(bounds_item["start"])
                    end = pd.Timestamp(bounds_item["end"])
                    if start.tzinfo is None or end.tzinfo is None:
                        raise ValueError(f"data.split.ranges.{name} timestamps must include a UTC timezone")
                    start = start.tz_convert("UTC")
                    end = end.tz_convert("UTC")
                    if start > end:
                        raise ValueError(f"data.split.ranges.{name}.start must not exceed end")
        if representation == "4d":
            map_config = data.get("map") or {}
            if not map_config.get("name"):
                raise ValueError("4d configuration requires data.map.name")
            if data["files"] and not map_config.get("locations"):
                raise ValueError("4d map generation requires data.map.locations")
        mask_config = data.get("mask") or {}
        if mask_config:
            if representation == "1d":
                raise ValueError("data.mask is supported only for 2d and 4d")
            if data.get("frequency_bins") or data.get("frequency_ranges"):
                raise ValueError("data.mask cannot be combined with frequency selection")
            if not mask_config.get("frequency_ranges"):
                raise ValueError("data.mask.frequency_ranges is required")
            if "noise_floor" not in mask_config:
                raise ValueError("data.mask.noise_floor is required")

    max_missing_gap = int(config["preprocessing"].get("max_missing_gap", 0))
    if max_missing_gap < 0:
        raise ValueError("preprocessing.max_missing_gap must be non-negative")

    prediction_start_row = config["data"].get("prediction_start_row")
    
    if prediction_start_row is not None:
        prediction_start_row = int(prediction_start_row)
    
        if prediction_start_row <= 0:
            raise ValueError(
                "Error! data.prediction_start_row must be a "
                "positive one-based row number when provided."
            )
    
        representation = str(config["data"].get("representation", "")).lower()
        loader = str(config["data"].get("loader", "aerpaw")).lower()
        if representation not in {"1d", "2d", "4d"} and loader != "powder":
            raise ValueError(
                "Error! data.prediction_start_row is currently "
                "supported only by the unified representation loaders or POWDER."
            )


def model_names(config: dict[str, Any]) -> list[str]:
    """Return the ordered model list from a single- or multi-model config."""
    training = config.get("training") or {}
    raw_names = training.get("models")
    if raw_names is None:
        raw_name = training.get("model_name")
        raw_names = [] if raw_name is None else [raw_name]
    if not isinstance(raw_names, list) or not raw_names:
        raise ValueError("training requires model_name or a non-empty models list")

    names = [str(name).lower() for name in raw_names]
    if len(names) != len(set(names)):
        raise ValueError("training.models must not contain duplicates")
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"Missing model configuration section(s): {', '.join(missing)}")
    return names
