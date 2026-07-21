"""
Shared data-selection interface for frequency-chunk training and evaluation.

This module translates the common project configuration into calls to the
appropriate dataset loader. It provides a small model-agnostic interface that
allows training and evaluation scripts to request one configured frequency
chunk without implementing loader-specific branching.

Primary responsibilities include:

- defining the immutable frequency-chunk specification used across the pipeline;
- converting configured chunk dictionaries into validated ChunkSpec objects;
- resolving project-relative data paths;
- selecting the configured 1D, 2D, or 4D loading path;
- validating the unified source-file and map configuration;
- forwarding chunk frequency bounds and preprocessing settings;
- forwarding validation-fraction information used for leakage-safe fitting;
- forwarding max-row limits used for reduced test runs;
- forwarding chronological split settings, including data.prediction_start_row; and
- returning a LoadedSpectrumData object through one consistent interface.

This module does not parse source files or perform normalization directly.
Those operations are delegated to shared source, map-building, and preprocessing
utilities.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.common.config import resolve_path
from training.common.preprocessing import (
    LoadedSpectrumData,
    SequenceSegment,
    SplitArrays,
    apply_per_frequency_normalization,
    fit_per_frequency_normalization,
)
from training.common.data_sources import load_csv_sources
from training.common.map_builder import load_4d


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    start_mhz: float
    end_mhz: float


def chunk_specs(config: dict[str, Any]) -> list[ChunkSpec]:
    return [
        ChunkSpec(str(chunk["id"]), float(chunk["start_mhz"]), float(chunk["end_mhz"]))
        for chunk in config["data"]["chunks"]
    ]


def _split_segments(
    segments: tuple[SequenceSegment, ...],
    start: int,
    end: int,
) -> tuple[SequenceSegment, ...]:
    result = []
    for segment in segments:
        clipped_start = max(segment.start, start)
        clipped_end = min(segment.end, end)
        if clipped_start < clipped_end:
            result.append(
                SequenceSegment(
                    clipped_start - start,
                    clipped_end - start,
                    segment.label,
                )
            )
    return tuple(result)


def _flatten_segments(
    data: np.ndarray,
    segments: tuple[SequenceSegment, ...],
) -> tuple[np.ndarray, tuple[SequenceSegment, ...]]:
    values = []
    flattened_segments = []
    offset = 0
    for segment in segments:
        block = data[segment.start:segment.end]
        for frequency_index in range(block.shape[1]):
            series = block[:, frequency_index]
            values.append(series)
            flattened_segments.append(
                SequenceSegment(
                    offset,
                    offset + len(series),
                    f"{segment.label}:frequency_{frequency_index}",
                )
            )
            offset += len(series)
    if not values:
        return np.empty(0, dtype=np.float32), tuple()
    return np.concatenate(values).astype(np.float32), tuple(flattened_segments)


def load_chunk( config: dict[str, Any], chunk: ChunkSpec, val_fraction: float,) -> LoadedSpectrumData:
    """ Load and process data chunk automatically routing by file extension. """
    
    # Load data config
    normalize = bool(config["preprocessing"].get("normalize", True))
    impute = bool(config["preprocessing"].get("impute", False))
    data_cfg = config["data"]

    if "representation" not in data_cfg or "files" not in data_cfg:
        raise ValueError(
            "The unified loader requires data.representation and data.files"
        )
    return _load_configured_representation(
        config=config,
        chunk=chunk,
        val_fraction=val_fraction,
        normalize=normalize,
        impute=impute,
    )


def _load_configured_representation(
    config: dict[str, Any],
    chunk: ChunkSpec,
    val_fraction: float,
    normalize: bool,
    impute: bool,
) -> LoadedSpectrumData:
    data_cfg = config["data"]
    representation = str(data_cfg["representation"]).lower()
    if representation not in {"1d", "2d", "4d"}:
        raise ValueError("data.representation must be one of: 1d, 2d, 4d")

    raw_files = data_cfg.get("files", [])
    if not isinstance(raw_files, list) or (representation != "4d" and not raw_files):
        raise ValueError("data.files must be a non-empty list unless loading a cached 4d map")
    files = [resolve_path(value) for value in raw_files]
    frequency_bins = data_cfg.get("frequency_bins")
    frequency_ranges = data_cfg.get("frequency_ranges")
    mask_cfg = data_cfg.get("mask") or {}
    mask_ranges = mask_cfg.get("frequency_ranges")
    if mask_cfg and representation == "1d":
        raise ValueError("Frequency masking is supported only for 2d and 4d data")
    if mask_cfg and (frequency_bins or frequency_ranges):
        raise ValueError("data.mask cannot be combined with frequency selection")
    concat = str(data_cfg.get("concat", "rows")).lower()
    if representation == "1d" and concat != "rows":
        raise ValueError("1d loading only supports row-wise concatenation")
    max_missing_gap = int(config.get("preprocessing", {}).get("max_missing_gap", 0))

    if representation in {"1d", "2d"}:
        source = load_csv_sources(
            files,
            frequency_bins=frequency_bins,
            frequency_ranges=frequency_ranges,
            concat=concat,
            impute=impute,
            max_missing_gap=max_missing_gap,
            mask_ranges=mask_ranges,
            noise_floor=float(mask_cfg["noise_floor"]) if mask_cfg else None,
        )
        raw_matrix = source.data
        feature_labels = source.feature_labels
    else:
        map_cfg = data_cfg.get("map") or {}
        map_name = map_cfg.get("name")
        locations = map_cfg.get("locations")
        if not map_name:
            raise ValueError("4d loading requires data.map.name")
        source = load_4d(
            files=files,
            map_name=str(map_name),
            map_dir=resolve_path(map_cfg.get("output_dir", "data/maps")),
            locations_path=resolve_path(locations) if locations else None,
            collection_key=str(map_cfg.get("collection_key", "endpoints")),
            grid=dict(map_cfg.get("grid") or {}),
            map_key=str(data_cfg.get("map_key", "map_db")),
            frequency_bins=frequency_bins,
            frequency_ranges=frequency_ranges,
            impute=impute,
            max_missing_gap=max_missing_gap,
            force_rebuild=bool(map_cfg.get("force_rebuild", False)),
            permute=bool(map_cfg.get("permute", False)),
            permute_seed=(
                int(map_cfg["permute_seed"])
                if map_cfg.get("permute_seed") is not None
                else None
            ),
            mask_ranges=mask_ranges,
            noise_floor=float(mask_cfg["noise_floor"]) if mask_cfg else None,
        )
        raw_matrix = source.data
        feature_labels = source.feature_labels

    max_rows = data_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("data.max_rows must be positive when provided")
        raw_matrix = raw_matrix[:max_rows]

    prediction_start_row = data_cfg.get("prediction_start_row")
    if prediction_start_row is not None:
        train_end = int(prediction_start_row) - 1
        if train_end <= 0 or train_end >= len(raw_matrix):
            raise ValueError(
                "data.prediction_start_row must leave non-empty train and test data"
            )
    else:
        test_rows = data_cfg.get("split", {}).get("test_rows")
        if test_rows is None:
            test_fraction = float(data_cfg.get("split", {}).get("test_fraction", 0.2))
            if not 0.0 < test_fraction < 1.0:
                raise ValueError("data.split.test_fraction must be between 0 and 1")
            test_rows = max(1, int(len(raw_matrix) * test_fraction))
        test_rows = int(test_rows)
        if test_rows <= 0 or test_rows >= len(raw_matrix):
            raise ValueError("data.split.test_rows must leave non-empty train and test data")
        train_end = len(raw_matrix) - test_rows
    fit_end = int(train_end * (1.0 - val_fraction))
    if fit_end <= 0 or fit_end >= train_end:
        raise ValueError("validation split leaves an empty normalization-training portion")

    train_raw = raw_matrix[:train_end]
    test_raw = raw_matrix[train_end:]
    train_segments = _split_segments(source.segments, 0, train_end)
    test_segments = _split_segments(source.segments, train_end, len(raw_matrix))
    if not np.isfinite(train_raw).all() or not np.isfinite(test_raw).all():
        raise ValueError("Missing or non-finite values remain after preprocessing")
    train_model = train_raw
    test_model = test_raw
    normalization = None
    if normalize:
        mean, std = fit_per_frequency_normalization(
            train_raw[:fit_end],
            allow_zero_variance=bool(mask_cfg),
        )
        train_model = apply_per_frequency_normalization(train_raw, mean, std).astype(np.float32)
        test_model = apply_per_frequency_normalization(test_raw, mean, std).astype(np.float32)
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": str(data_cfg.get("reference_site", representation)),
            "source_split": "train_before_validation",
            "frequency_axis": -1,
        }

    if representation == "1d":
        train_time_segments = train_segments
        test_time_segments = test_segments
        train_raw, train_segments = _flatten_segments(train_raw, train_time_segments)
        test_raw, test_segments = _flatten_segments(test_raw, test_time_segments)
        train_model, _ = _flatten_segments(
            train_model.reshape(-1, len(source.frequencies)), train_time_segments
        )
        test_model, _ = _flatten_segments(
            test_model.reshape(-1, len(source.frequencies)), test_time_segments
        )

    train_row_start = 0
    train_row_end = len(train_raw) - 1
    test_row_start = len(train_raw) if representation == "1d" else train_end
    test_row_end = test_row_start + len(test_raw) - 1

    reference_site = str(data_cfg.get("reference_site", representation))
    return LoadedSpectrumData(
        files={str(index): path for index, path in enumerate(source.files)},
        raw_frames={},
        filled_frames={},
        reference_site=reference_site,
        frequencies=[float(value) for value in source.frequencies],
        splits={
            f"{reference_site}_train": SplitArrays(
                train_raw, train_model, train_row_start, train_row_end, train_segments
            ),
            f"{reference_site}_test": SplitArrays(
                test_raw, test_model, test_row_start, test_row_end, test_segments
            ),
        },
        normalization=normalization,
        feature_labels=feature_labels,
    )
