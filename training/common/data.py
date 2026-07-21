"""Unified source loader and train/validation/test preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from training.common.config import resolve_path
from training.common.data_sources import LoadedSource, load_csv_sources
from training.common.map_builder import load_4d
from training.common.preprocessing import (
    LoadedSpectrumData,
    SequenceSegment,
    SplitArrays,
    apply_per_frequency_normalization,
    fit_per_frequency_normalization,
)


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
    return tuple(
        SequenceSegment(
            max(segment.start, start) - start,
            min(segment.end, end) - start,
            segment.label,
        )
        for segment in segments
        if max(segment.start, start) < min(segment.end, end)
    )


def _flatten_segments(
    data: np.ndarray,
    segments: tuple[SequenceSegment, ...],
) -> tuple[np.ndarray, tuple[SequenceSegment, ...]]:
    values = []
    output_segments = []
    offset = 0
    for segment in segments:
        block = data[segment.start:segment.end]
        for frequency_index in range(block.shape[1]):
            series = block[:, frequency_index]
            values.append(series)
            output_segments.append(
                SequenceSegment(
                    offset,
                    offset + len(series),
                    f"{segment.label}:frequency_{frequency_index}",
                )
            )
            offset += len(series)
    if not values:
        return np.empty(0, dtype=np.float32), tuple()
    return np.concatenate(values).astype(np.float32), tuple(output_segments)


def _partition_files(raw_files: list[dict[str, str]]) -> dict[str, list[Path]]:
    partitions = {"train": [], "test": []}
    for entry in raw_files:
        if not isinstance(entry, dict) or entry.get("partition") not in partitions:
            raise ValueError("Each data.files entry requires partition: train or test")
        if not entry.get("path"):
            raise ValueError("Each data.files entry requires path")
        partitions[entry["partition"]].append(resolve_path(entry["path"]))
    if not partitions["train"] or not partitions["test"]:
        raise ValueError("data.files must include both train and test partitions")
    return partitions


def _slice_source(source: LoadedSource, start: int, end: int) -> LoadedSource:
    timestamps = None if source.timestamps is None else source.timestamps[start:end]
    return replace(
        source,
        data=source.data[start:end],
        timestamps=timestamps,
        segments=_split_segments(source.segments, start, end),
    )


def _load_partition(
    files: list[Path],
    partition: str,
    representation: str,
    data_cfg: dict[str, Any],
    frequency_bins: list[float] | None,
    frequency_ranges: list[list[float]] | None,
    concat: str,
    impute: bool,
    max_missing_gap: int,
    mask_ranges: list[list[float]] | None,
    noise_floor: float | None,
) -> LoadedSource:
    if representation in {"1d", "2d"}:
        return load_csv_sources(
            files,
            frequency_bins=frequency_bins,
            frequency_ranges=frequency_ranges,
            concat=concat,
            impute=impute,
            max_missing_gap=max_missing_gap,
            mask_ranges=mask_ranges,
            noise_floor=noise_floor,
        )

    map_cfg = data_cfg.get("map") or {}
    map_name = map_cfg.get("name")
    if not map_name:
        raise ValueError("4d loading requires data.map.name")
    locations = map_cfg.get("locations")
    return load_4d(
        files=files,
        map_name=f"{map_name}_{partition}",
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
        noise_floor=noise_floor,
    )


def load_chunk(
    config: dict[str, Any],
    chunk: ChunkSpec,
    val_fraction: float,
) -> LoadedSpectrumData:
    """Load one configured dataset and prepare train/validation/test splits."""
    data_cfg = config["data"]
    representation = str(data_cfg.get("representation", "")).lower()
    if representation not in {"1d", "2d", "4d"}:
        raise ValueError("data.representation must be one of: 1d, 2d, 4d")
    if representation == "1d" and str(data_cfg.get("concat", "rows")) != "rows":
        raise ValueError("1d loading only supports row-wise concatenation")

    partitions = _partition_files(data_cfg.get("files", []))
    frequency_bins = data_cfg.get("frequency_bins")
    frequency_ranges = data_cfg.get("frequency_ranges")
    mask_cfg = data_cfg.get("mask") or {}
    mask_ranges = mask_cfg.get("frequency_ranges")
    if mask_cfg and representation == "1d":
        raise ValueError("Frequency masking is supported only for 2d and 4d data")
    if mask_cfg and (frequency_bins or frequency_ranges):
        raise ValueError("data.mask cannot be combined with frequency selection")

    preprocessing = config.get("preprocessing", {})
    impute = bool(preprocessing.get("impute", False))
    max_missing_gap = int(preprocessing.get("max_missing_gap", 0))
    noise_floor = float(mask_cfg["noise_floor"]) if mask_cfg else None
    concat = str(data_cfg.get("concat", "rows"))

    train_source = _load_partition(
        partitions["train"], "train", representation, data_cfg,
        frequency_bins, frequency_ranges, concat, impute, max_missing_gap,
        mask_ranges, noise_floor,
    )
    test_source = _load_partition(
        partitions["test"], "test", representation, data_cfg,
        frequency_bins, frequency_ranges, concat, impute, max_missing_gap,
        mask_ranges, noise_floor,
    )
    if not np.array_equal(train_source.frequencies, test_source.frequencies):
        raise ValueError("Train and test partitions have different frequency columns")

    max_rows = data_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("data.max_rows must be positive when provided")
        train_source = _slice_source(train_source, 0, max_rows)
        test_source = _slice_source(test_source, 0, max_rows)

    prediction_start_row = data_cfg.get("prediction_start_row")
    if prediction_start_row is not None:
        test_source = _slice_source(test_source, int(prediction_start_row) - 1, len(test_source.data))

    if not len(train_source.data) or not len(test_source.data):
        raise ValueError("Train and test partitions must both contain data")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1")

    fit_end = int(len(train_source.data) * (1.0 - val_fraction))
    if fit_end <= 0 or fit_end >= len(train_source.data):
        raise ValueError("validation split leaves an empty normalization-training portion")

    train_data = train_source.data
    test_data = test_source.data
    normalization = None
    train_model = train_data
    test_model = test_data
    if bool(preprocessing.get("normalize", True)):
        mean, std = fit_per_frequency_normalization(
            train_data[:fit_end], allow_zero_variance=bool(mask_cfg)
        )
        train_model = apply_per_frequency_normalization(train_data, mean, std).astype(np.float32)
        test_model = apply_per_frequency_normalization(test_data, mean, std).astype(np.float32)
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": str(data_cfg.get("reference_site", representation)),
            "source_split": "train_before_validation",
            "frequency_axis": -1,
        }

    train_segments = train_source.segments
    test_segments = test_source.segments
    if representation == "1d":
        train_data, train_segments = _flatten_segments(train_data, train_segments)
        test_data, test_segments = _flatten_segments(test_data, test_segments)
        train_model, _ = _flatten_segments(
            train_model.reshape(-1, len(train_source.frequencies)), train_source.segments
        )
        test_model, _ = _flatten_segments(
            test_model.reshape(-1, len(test_source.frequencies)), test_source.segments
        )

    reference_site = str(data_cfg.get("reference_site", representation))
    train_files = {f"train_{i}": path for i, path in enumerate(train_source.files)}
    test_files = {f"test_{i}": path for i, path in enumerate(test_source.files)}
    train_start = 0
    test_start = len(train_data)
    return LoadedSpectrumData(
        files={**train_files, **test_files},
        raw_frames={},
        filled_frames={},
        reference_site=reference_site,
        frequencies=[float(value) for value in train_source.frequencies],
        splits={
            f"{reference_site}_train": SplitArrays(
                train_data, train_model, train_start, len(train_data) - 1, train_segments
            ),
            f"{reference_site}_test": SplitArrays(
                test_data, test_model, test_start, test_start + len(test_data) - 1, test_segments
            ),
        },
        normalization=normalization,
        feature_labels=train_source.feature_labels,
    )
