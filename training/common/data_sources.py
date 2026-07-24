"""Shared source loading and bounded missing-value imputation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from training.common.preprocessing import SequenceSegment


@dataclass(frozen=True)
class LoadedSource:
    data: np.ndarray
    frequencies: np.ndarray
    timestamps: pd.DatetimeIndex | None
    files: list[Path]
    feature_labels: list[str]
    segments: tuple[SequenceSegment, ...]


def _normalize_frequency_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    columns = [str(float(column)) for column in frame.columns]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} contains duplicate frequency columns")
    frame.columns = columns
    return frame.loc[:, sorted(frame.columns, key=float)]


def _read_csv(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {path}")

    timestamps = None
    if "timestamp_utc" in frame.columns:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("timestamp_utc"), utc=True, errors="raise")
        )
        if timestamps.hasnans or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError(f"{path} has invalid timestamp ordering or duplicates")

    if frame.shape[1] == 0:
        raise ValueError(f"CSV file has no frequency columns: {path}")
    frame = _normalize_frequency_columns(frame, path)
    return frame.apply(pd.to_numeric, errors="coerce"), timestamps


def impute_frame(frame: pd.DataFrame, max_missing_gap: int) -> pd.DataFrame:
    """Interpolate/fill only missing runs no longer than max_missing_gap."""
    values = impute_array(frame.to_numpy(dtype=np.float32), max_missing_gap)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)


def impute_array(data: np.ndarray, max_missing_gap: int) -> np.ndarray:
    """Apply bounded temporal imputation independently to every feature."""
    if data.ndim < 2:
        raise ValueError("imputation requires a time axis and at least one feature axis")
    if max_missing_gap < 0:
        raise ValueError("max_missing_gap must be non-negative")

    original_shape = data.shape
    values = np.asarray(data, dtype=np.float32).reshape(data.shape[0], -1).copy()
    valid = np.isfinite(values)
    time = np.arange(values.shape[0], dtype=np.int64)[:, None]
    previous = np.maximum.accumulate(np.where(valid, time, -1), axis=0)
    following = np.minimum.accumulate(
        np.where(valid, time, values.shape[0])[::-1], axis=0
    )[::-1]
    gap = following - previous - 1
    eligible = (~valid) & (gap <= max_missing_gap) & ((previous >= 0) | (following < values.shape[0]))
    if not np.any(eligible):
        return values.reshape(original_shape)

    previous_safe = np.clip(previous, 0, values.shape[0] - 1)
    following_safe = np.clip(following, 0, values.shape[0] - 1)
    previous_values = np.take_along_axis(values, previous_safe, axis=0)
    following_values = np.take_along_axis(values, following_safe, axis=0)
    internal = (previous >= 0) & (following < values.shape[0])
    fraction = (time - previous) / np.maximum(following - previous, 1)
    interpolated = previous_values + fraction * (following_values - previous_values)
    filled = np.where(internal, interpolated, np.where(previous >= 0, previous_values, following_values))
    values[eligible] = filled[eligible]
    return values.reshape(original_shape)


def select_frequencies(
    frame: pd.DataFrame,
    frequency_bins: list[float] | None = None,
    frequency_ranges: list[list[float]] | None = None,
) -> pd.DataFrame:
    if frequency_bins and frequency_ranges:
        raise ValueError("Specify either frequency_bins or frequency_ranges, not both")
    available = np.asarray([float(column) for column in frame.columns])
    if frequency_bins:
        selected = []
        for requested in frequency_bins:
            matches = np.flatnonzero(np.isclose(available, float(requested), atol=1e-6))
            if len(matches) != 1:
                raise ValueError(f"Frequency bin {requested} MHz was not found")
            selected.append(frame.columns[matches[0]])
    elif frequency_ranges:
        selected = [
            column
            for column in frame.columns
            if any(float(start) <= float(column) <= float(stop) for start, stop in frequency_ranges)
        ]
    else:
        selected = list(frame.columns)
    if not selected:
        raise ValueError("Frequency selection produced no columns")
    return frame.loc[:, selected]


def mask_outside_frequency_ranges(
    data: np.ndarray,
    frequencies: np.ndarray,
    frequency_ranges: list[list[float]],
    noise_floor: float,
) -> np.ndarray:
    """Replace every frequency channel outside the allowed ranges."""
    if not frequency_ranges:
        raise ValueError("mask.frequency_ranges must contain at least one range")
    frequencies = np.asarray(frequencies, dtype=np.float32)
    keep = np.zeros(len(frequencies), dtype=bool)
    for start, stop in frequency_ranges:
        if float(start) > float(stop):
            raise ValueError(f"Invalid frequency mask range: {start}, {stop}")
        keep |= (frequencies >= float(start)) & (frequencies <= float(stop))
    if not keep.any():
        raise ValueError("Frequency mask does not include any available channels")
    result = np.array(data, dtype=np.float32, copy=True)
    result[..., ~keep] = float(noise_floor)
    return result


def load_csv_sources(
    files: list[Path],
    concat: str = "rows",
    frequency_bins: list[float] | None = None,
    frequency_ranges: list[list[float]] | None = None,
    impute: bool = False,
    max_missing_gap: int = 0,
    mask_ranges: list[list[float]] | None = None,
    noise_floor: float | None = None,
) -> LoadedSource:
    if not files:
        raise ValueError("At least one CSV file is required")
    if concat not in {"rows", "columns"}:
        raise ValueError("concat must be 'rows' or 'columns'")

    frames = []
    timestamps = []
    for path in files:
        frame, stamp = _read_csv(path)
        frame = select_frequencies(frame, frequency_bins, frequency_ranges)
        if impute:
            frame = impute_frame(frame, max_missing_gap)
        if mask_ranges is not None:
            if noise_floor is None:
                raise ValueError("noise_floor is required when frequency masking is enabled")
            frame = pd.DataFrame(
                mask_outside_frequency_ranges(
                    frame.to_numpy(),
                    np.asarray([float(column) for column in frame.columns]),
                    mask_ranges,
                    noise_floor,
                ),
                columns=frame.columns,
                index=frame.index,
            )
        frames.append(frame)
        timestamps.append(stamp)

    if concat == "rows":
        first_columns = list(frames[0].columns)
        for frame in frames[1:]:
            if list(frame.columns) != first_columns:
                raise ValueError("Row-wise CSV concatenation requires matching frequency columns")
        data_frame = pd.concat(frames, axis=0, ignore_index=True)
        if all(stamp is not None for stamp in timestamps):
            combined_timestamps = pd.DatetimeIndex(np.concatenate([stamp.to_numpy() for stamp in timestamps]))
        else:
            combined_timestamps = None
        labels = first_columns
        segments = []
        offset = 0
        for path, frame in zip(files, frames):
            segments.append(SequenceSegment(offset, offset + len(frame), str(path)))
            offset += len(frame)
    else:
        if all(stamp is not None for stamp in timestamps):
            common = timestamps[0]
            for stamp in timestamps[1:]:
                common = common.intersection(stamp)
            if common.empty:
                raise ValueError("Column-wise CSV concatenation has no common timestamps")
            aligned = [frame.loc[stamp.get_indexer(common)] for frame, stamp in zip(frames, timestamps)]
            combined_timestamps = common.sort_values()
        else:
            if any(stamp is not None for stamp in timestamps) or len({len(frame) for frame in frames}) != 1:
                raise ValueError("Column-wise concatenation requires timestamps or equal row counts")
            aligned = frames
            combined_timestamps = None
        labels = []
        for path, frame in zip(files, aligned):
            labels.extend(f"{path.parent.name}:{column}" for column in frame.columns)
        data_frame = pd.concat(aligned, axis=1)
        segments = [SequenceSegment(0, len(data_frame), "columns")]

    data = data_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Missing or non-finite values remain after preprocessing")
    return LoadedSource(
        data=data,
        frequencies=np.asarray([float(column) for column in data_frame.columns], dtype=np.float32),
        timestamps=combined_timestamps,
        files=files,
        feature_labels=labels,
        segments=tuple(segments),
    )


def load_locations(path: Path, collection_key: str) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return {str(item["name"]).lower(): item for item in document[collection_key]}


def clean_name(value: str) -> str:
    value = re.sub(r"(?:-nuc1|_nuc1)$", "", value.lower())
    return re.sub(r"[^a-z0-9]", "", value)


def find_location(csv_path: Path, locations: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    lookup = {clean_name(name): (name, location) for name, location in locations.items()}
    for parent in csv_path.parents:
        key = clean_name(parent.name)
        if key in lookup:
            return lookup[key]
    for segment in csv_path.stem.split("_"):
        key = clean_name(segment)
        if key in lookup:
            return lookup[key]
    raise ValueError(f"Could not match a location to CSV path: {csv_path}")
