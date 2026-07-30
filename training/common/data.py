"""Unified source loader and train/validation/test preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.common.config import resolve_path
from training.common.data_sources import LoadedSource, load_csv_sources
from training.common.map_builder import load_4d, load_map_layout, prepare_4d_partitions
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


def frequency_range_mask(
    frequencies: np.ndarray,
    frequency_ranges: list[list[float]],
) -> np.ndarray:
    """Return bins contained in any inclusive frequency range."""
    frequencies = np.asarray(frequencies, dtype=np.float32)
    keep = np.zeros(len(frequencies), dtype=bool)
    for start, stop in frequency_ranges:
        keep |= (frequencies >= float(start)) & (frequencies <= float(stop))
    if not keep.any():
        raise ValueError("Frequency ranges do not include any available channels")
    return keep


def apply_2d_spectral_mask(
    data: np.ndarray,
    frequencies: np.ndarray,
    mask_config: dict[str, Any],
    *,
    training_data: np.ndarray,
    split_seed_offset: int,
) -> np.ndarray:
    """Replace non-target 2D inputs using training-derived noise."""
    keep = frequency_range_mask(frequencies, mask_config["frequency_ranges"])
    result = np.asarray(data, dtype=np.float32).copy()
    replacement = str(mask_config.get("replacement", "constant")).lower()
    if replacement == "constant":
        result[:, ~keep] = float(mask_config["noise_floor"])
        return result

    if replacement != "low_tail_gaussian":
        raise ValueError(f"Unsupported spectral mask replacement: {replacement!r}")
    calibration_ranges = mask_config.get("calibration_frequency_ranges")
    calibration_mask = (
        frequency_range_mask(frequencies, calibration_ranges)
        if calibration_ranges else np.ones(len(frequencies), dtype=bool)
    )
    samples = np.asarray(training_data[:, calibration_mask], dtype=np.float64).reshape(-1)
    quantile = float(mask_config.get("low_tail_quantile", 0.05))
    center = float(np.quantile(samples, quantile))
    tail_limit = np.quantile(samples, min(0.25, max(quantile * 4, quantile)))
    lower_tail = samples[samples <= tail_limit]
    median = float(np.median(lower_tail))
    noise_std = float(1.4826 * np.median(np.abs(lower_tail - median)))
    noise_std = max(noise_std, float(mask_config.get("minimum_noise_std_db", 0.05)))
    rng = np.random.default_rng(int(mask_config.get("seed", 42)) + split_seed_offset)
    result[:, ~keep] = rng.normal(
        center,
        noise_std,
        size=(len(result), int(np.count_nonzero(~keep))),
    ).astype(np.float32)
    return result


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
        return np.empty((0, 1), dtype=np.float32), tuple()
    return np.concatenate(values).astype(np.float32).reshape(-1, 1), tuple(output_segments)


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


def _select_chunk(source: LoadedSource, chunk: ChunkSpec) -> LoadedSource:
    mask = (source.frequencies >= chunk.start_mhz) & (source.frequencies <= chunk.end_mhz)
    if not np.any(mask):
        raise ValueError(
            f"Chunk {chunk.chunk_id!r} contains no frequencies in "
            f"{chunk.start_mhz:g}-{chunk.end_mhz:g} MHz"
        )
    indices = np.flatnonzero(mask)
    return replace(
        source,
        data=np.take(source.data, indices, axis=-1),
        frequencies=source.frequencies[indices],
        feature_labels=[source.feature_labels[index] for index in indices],
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
    expected_layout: dict[str, object] | None = None,
    selected_sites: list[str] | None = None,
    excluded_sites: list[str] | None = None,
    outage_threshold: int | None = None,
    timestamp_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
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
            timestamp_ranges=timestamp_ranges,
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
        expected_layout=expected_layout,
        selected_sites=selected_sites,
        excluded_sites=excluded_sites,
        outage_threshold=outage_threshold,
        timestamp_ranges=timestamp_ranges,
    )


def load_chunk(
    config: dict[str, Any],
    chunk: ChunkSpec,
    val_fraction: float,
) -> LoadedSpectrumData:
    """Load one configured dataset and prepare train/validation/test splits."""
    print(f"[DEBUG] load_chunk entry: chunk={chunk.chunk_id}, val_fraction={val_fraction}")
    data_cfg = config["data"]
    representation = str(data_cfg.get("representation", "")).lower()
    print(f"[DEBUG] load_chunk: representation={representation}")
    if representation not in {"1d", "2d", "4d"}:
        raise ValueError("data.representation must be one of: 1d, 2d, 4d")
    if representation == "1d" and str(data_cfg.get("concat", "rows")) != "rows":
        raise ValueError("1d loading only supports row-wise concatenation")

    print(f"[DEBUG] load_chunk: parsing file list ...")
    partitions = _partition_files(data_cfg.get("files", []))
    print(f"[DEBUG] load_chunk: train files={partitions.get('train', [])}, test files={partitions.get('test', [])}")
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
    noise_floor = float(mask_cfg["noise_floor"]) if mask_cfg and "noise_floor" in mask_cfg else None
    concat = str(data_cfg.get("concat", "rows"))
    raw_ranges = (data_cfg.get("split") or {}).get("ranges")
    timestamp_ranges = None
    if raw_ranges is not None:
        timestamp_ranges = {
            name: [
                (
                    pd.Timestamp(item["start"]).tz_convert("UTC"),
                    pd.Timestamp(item["end"]).tz_convert("UTC"),
                )
                for item in (bounds if isinstance(bounds, list) else [bounds])
            ]
            for name, bounds in raw_ranges.items()
        }
        partitions = {
            "train": partitions["train"],
            "validation": partitions["train"],
            "test": partitions["test"],
        }

    selected_sites = excluded_sites = None
    outage_threshold = None
    if representation == "4d":
        print(f"[DEBUG] load_chunk: entering 4d branch ...")
        map_cfg = data_cfg.get("map") or {}
        locations = map_cfg.get("locations")
        if not locations:
            raise ValueError("4d map generation requires data.map.locations")
        outage_threshold = max(int(value) for value in config["windowing"]["horizons"])
        print(f"[DEBUG] load_chunk: calling prepare_4d_partitions with locations={locations} ...")
        partitions, selected_sites, excluded_sites = prepare_4d_partitions(
            partitions, resolve_path(locations), str(map_cfg.get("collection_key", "endpoints")),
            frequency_bins, frequency_ranges, outage_threshold, timestamp_ranges,
        )
    print(f"[DEBUG] load_chunk: prepare_4d_partitions done, selected_sites={selected_sites}")

    print(f"[DEBUG] load_chunk: loading train partition ...")
    source_mask_ranges = mask_ranges if representation == "4d" else None
    source_noise_floor = noise_floor if representation == "4d" else None
    train_source = _load_partition(
        partitions["train"], "train", representation, data_cfg,
        frequency_bins, frequency_ranges, concat, impute, max_missing_gap,
        source_mask_ranges, source_noise_floor, None, selected_sites, excluded_sites, outage_threshold,
        None if timestamp_ranges is None else timestamp_ranges["train"],
    )
    print(f"[DEBUG] load_chunk: train data shape={train_source.data.shape}")
    training_layout = None
    if representation == "4d":
        if not train_source.files:
            raise RuntimeError("4d training map did not provide a cache path")
        map_path = train_source.files[0]
        print(f"[DEBUG] load_chunk: loading training layout from {map_path}")
        training_layout = load_map_layout(map_path)
        print(f"[DEBUG] load_chunk: training layout loaded")
    validation_source = None
    if timestamp_ranges is not None:
        validation_source = _load_partition(
            partitions["validation"], "validation", representation, data_cfg,
            frequency_bins, frequency_ranges, concat, impute, max_missing_gap,
            source_mask_ranges, source_noise_floor, training_layout, selected_sites, excluded_sites,
            outage_threshold, timestamp_ranges["validation"],
        )
    print(f"[DEBUG] load_chunk: loading test partition ...")
    test_source = _load_partition(
        partitions["test"], "test", representation, data_cfg,
        frequency_bins, frequency_ranges, concat, impute, max_missing_gap,
        source_mask_ranges, source_noise_floor, training_layout, selected_sites, excluded_sites, outage_threshold,
        None if timestamp_ranges is None else timestamp_ranges["test"],
    )
    print(f"[DEBUG] load_chunk: test data shape={test_source.data.shape}")
    train_source = _select_chunk(train_source, chunk)
    if validation_source is not None:
        validation_source = _select_chunk(validation_source, chunk)
    test_source = _select_chunk(test_source, chunk)
    print(f"[DEBUG] load_chunk: after _select_chunk train.shape={train_source.data.shape} test.shape={test_source.data.shape}")
    if not np.array_equal(train_source.frequencies, test_source.frequencies):
        raise ValueError("Train and test partitions have different frequency columns")
    if validation_source is not None and not np.array_equal(
        train_source.frequencies, validation_source.frequencies
    ):
        raise ValueError("Train and validation partitions have different frequency columns")

    max_rows = data_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("data.max_rows must be positive when provided")
        train_source = _slice_source(train_source, 0, max_rows)
        if validation_source is not None:
            validation_source = _slice_source(validation_source, 0, max_rows)
        test_source = _slice_source(test_source, 0, max_rows)
        print(f"[DEBUG] load_chunk: after max_rows truncation train.shape={train_source.data.shape}")

    prediction_start_row = data_cfg.get("prediction_start_row")
    if prediction_start_row is not None:
        test_source = _slice_source(test_source, int(prediction_start_row) - 1, len(test_source.data))
        print(f"[DEBUG] load_chunk: after prediction_start_row truncation test.shape={test_source.data.shape}")

    if not len(train_source.data) or not len(test_source.data) or (
        validation_source is not None and not len(validation_source.data)
    ):
        raise ValueError("Configured train, validation, and test partitions must contain data")
    if timestamp_ranges is None and not 0.0 < val_fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1")

    fit_end = (
        int(len(train_source.data) * (1.0 - val_fraction))
        if timestamp_ranges is None else len(train_source.data)
    )
    print(f"[DEBUG] load_chunk: fit_end={fit_end} / {len(train_source.data)} (val_fraction={val_fraction})")
    if fit_end <= 0 or (timestamp_ranges is None and fit_end >= len(train_source.data)):
        raise ValueError("validation split leaves an empty normalization-training portion")

    train_data = train_source.data
    validation_data = None if validation_source is None else validation_source.data
    test_data = test_source.data
    train_input = train_data
    validation_input = validation_data
    test_input = test_data
    if mask_cfg and representation == "2d":
        training_fit_data = train_data[:fit_end]
        train_input = apply_2d_spectral_mask(
            train_data, train_source.frequencies, mask_cfg,
            training_data=training_fit_data, split_seed_offset=0,
        )
        if validation_data is not None:
            validation_input = apply_2d_spectral_mask(
                validation_data, validation_source.frequencies, mask_cfg,
                training_data=training_fit_data, split_seed_offset=1,
            )
        test_input = apply_2d_spectral_mask(
            test_data, test_source.frequencies, mask_cfg,
            training_data=training_fit_data, split_seed_offset=2,
        )
    normalization = None
    train_model = train_input
    validation_model = validation_input
    test_model = test_input
    if bool(preprocessing.get("normalize", True)):
        print(f"[DEBUG] load_chunk: fitting per-frequency normalization on first {fit_end} rows ...")
        mean, std = fit_per_frequency_normalization(
            train_input[:fit_end], allow_zero_variance=bool(mask_cfg)
        )
        print(f"[DEBUG] load_chunk: normalization stats: mean.shape={mean.shape}, std.shape={std.shape}")
        train_model = apply_per_frequency_normalization(train_input, mean, std).astype(np.float32)
        if validation_input is not None:
            validation_model = apply_per_frequency_normalization(
                validation_input, mean, std
            ).astype(np.float32)
        test_model = apply_per_frequency_normalization(test_input, mean, std).astype(np.float32)
        print(f"[DEBUG] load_chunk: normalization applied, train_model.shape={train_model.shape}")
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": str(data_cfg.get("reference_site", representation)),
            "source_split": "train" if timestamp_ranges is not None else "train_before_validation",
            "frequency_axis": -1,
        }

    train_segments = train_source.segments
    validation_segments = () if validation_source is None else validation_source.segments
    test_segments = test_source.segments
    if representation == "1d":
        print(f"[DEBUG] load_chunk: flattening 1d segments ...")
        train_data, train_segments = _flatten_segments(train_data, train_segments)
        if validation_source is not None:
            validation_data, validation_segments = _flatten_segments(
                validation_data, validation_segments
            )
        test_data, test_segments = _flatten_segments(test_data, test_segments)
        train_model, _ = _flatten_segments(
            train_model.reshape(-1, len(train_source.frequencies)), train_source.segments
        )
        if validation_source is not None:
            validation_model, _ = _flatten_segments(
                validation_model.reshape(-1, len(validation_source.frequencies)),
                validation_source.segments,
            )
        test_model, _ = _flatten_segments(
            test_model.reshape(-1, len(test_source.frequencies)), test_source.segments
        )
        if normalization is not None:
            n_freq = len(train_source.frequencies)
            per_freq_mean = mean.reshape(1, -1).astype(np.float32)
            per_freq_std = std.reshape(1, -1).astype(np.float32)
            normalization = {
                "mean_dbm": per_freq_mean,
                "std_dbm": per_freq_std,
                "site": normalization["site"],
                "source_split": normalization["source_split"],
                "frequency_axis": normalization["frequency_axis"],
            }

    print(f"[DEBUG] load_chunk: building LoadedSpectrumData ...")
    reference_site = str(data_cfg.get("reference_site", representation))
    train_files = {f"train_{i}": path for i, path in enumerate(train_source.files)}
    test_files = {f"test_{i}": path for i, path in enumerate(test_source.files)}
    train_start = 0
    test_start = len(train_data)
    validation_start = len(train_data)
    if validation_data is not None:
        test_start += len(validation_data)
    print(f"[DEBUG] load_chunk: returning, train_model.shape={train_model.shape if hasattr(train_model, 'shape') else '?'}, test_model.shape={test_model.shape if hasattr(test_model, 'shape') else '?'}")
    splits = {
        f"{reference_site}_train": SplitArrays(
            train_data, train_model, train_start, len(train_data) - 1, train_segments
        ),
        f"{reference_site}_test": SplitArrays(
            test_data, test_model, test_start, test_start + len(test_data) - 1, test_segments
        ),
    }
    if validation_data is not None:
        splits[f"{reference_site}_validation"] = SplitArrays(
            validation_data,
            validation_model,
            validation_start,
            validation_start + len(validation_data) - 1,
            validation_segments,
        )
    return LoadedSpectrumData(
        files={**train_files, **test_files},
        raw_frames={},
        filled_frames={},
        reference_site=reference_site,
        frequencies=[float(value) for value in train_source.frequencies],
        splits=splits,
        normalization=normalization,
        feature_labels=train_source.feature_labels,
    )
