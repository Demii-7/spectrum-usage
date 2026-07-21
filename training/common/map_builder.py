"""Loader and cache builder for spatial spectrum maps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from training.common.data_sources import (
    LoadedSource,
    find_location,
    impute_array,
    impute_frame,
    load_locations,
    mask_outside_frequency_ranges,
    select_frequencies,
    _read_csv,
)
from training.common.preprocessing import SequenceSegment


def _local_xy(longitudes, latitudes, origin_lon, origin_lat):
    radius = 6_371_000.0
    x = radius * np.radians(longitudes - origin_lon) * np.cos(np.radians(origin_lat))
    y = radius * np.radians(latitudes - origin_lat)
    return x, y


def _map_from_sites(site_data, site_x, site_y, grid):
    height = int(grid.get("height", grid.get("grid_height", 50)))
    width = int(grid.get("width", grid.get("grid_width", 50)))
    if height <= 0 or width <= 0:
        raise ValueError("map grid dimensions must be positive")
    origin_lon = float(np.mean(site_x))
    origin_lat = float(np.mean(site_y))
    x, y = _local_xy(site_x, site_y, origin_lon, origin_lat)
    width_m = float(grid.get("width_meters", max(np.ptp(x), 1.0)))
    height_m = float(grid.get("height_meters", max(np.ptp(y), 1.0)))
    grid_x = np.linspace(-width_m / 2.0, width_m / 2.0, width)
    grid_y = np.linspace(-height_m / 2.0, height_m / 2.0, height)
    gx, gy = np.meshgrid(grid_x, grid_y)
    distances = np.hypot(gx[..., None] - x, gy[..., None] - y)
    power = float(grid.get("power", 2.0))
    weights = 1.0 / np.maximum(distances, 1e-6) ** power
    exact = distances == 0
    for site_index in range(len(site_x)):
        weights[exact[..., site_index], :] = 0.0
        weights[exact[..., site_index], site_index] = 1.0

    # Site data arrives as (site, time, frequency); map output is time-first.
    working = np.moveaxis(site_data.astype(np.float64), 0, 1)
    if grid.get("power_domain", "db") == "linear":
        working = np.power(10.0, working / 10.0)
    valid = np.isfinite(working)
    working = np.moveaxis(working, -1, 1)
    valid = np.moveaxis(valid, -1, 1)
    weighted = np.where(
        valid[:, :, None, None, :],
        working[:, :, None, None, :] * weights[None, None, :, :, :],
        0.0,
    )
    denominator = np.sum(
        np.where(valid[:, :, None, None, :], weights[None, None, :, :, :], 0.0),
        axis=-1,
    )
    mapped = np.divide(
        np.sum(weighted, axis=-1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    if grid.get("power_domain", "db") == "linear":
        mapped = 10.0 * np.log10(np.maximum(mapped, 1e-30))
    return np.moveaxis(mapped, 1, -1).astype(np.float32), gx, gy


def _load_cached(path: Path, map_key: str, frequency_bins, frequency_ranges) -> LoadedSource:
    with np.load(path, allow_pickle=False) as archive:
        if map_key not in archive or "freqs_mhz" not in archive:
            raise KeyError(f"{path} must contain {map_key!r} and 'freqs_mhz'")
        data = np.asarray(archive[map_key], dtype=np.float32)
        frequencies = np.asarray(archive["freqs_mhz"], dtype=np.float32)
        timestamps = None
        if "timestamps" in archive:
            import pandas as pd
            timestamps = pd.DatetimeIndex(pd.to_datetime(archive["timestamps"], utc=True))
    if data.ndim != 4:
        raise ValueError(f"cached map must have shape (T, H, W, F), got {data.shape}")
    valid_timesteps = np.isfinite(data).any(axis=(1, 2, 3))
    data = data[valid_timesteps]
    if timestamps is not None:
        timestamps = timestamps[valid_timesteps]
    selected = np.ones(len(frequencies), dtype=bool)
    if frequency_bins:
        selected = np.zeros(len(frequencies), dtype=bool)
        for value in frequency_bins:
            selected |= np.isclose(frequencies, float(value), atol=1e-6)
    elif frequency_ranges:
        selected = np.zeros(len(frequencies), dtype=bool)
        for start, stop in frequency_ranges:
            selected |= (frequencies >= start) & (frequencies <= stop)
    if not selected.any():
        raise ValueError("Frequency selection produced no map channels")
    return LoadedSource(
        data[:, :, :, selected],
        frequencies[selected],
        timestamps,
        [path],
        [str(value) for value in frequencies[selected]],
        (SequenceSegment(0, len(data), str(path)),),
    )


def load_4d(
    files: list[Path],
    map_name: str,
    map_dir: Path,
    locations_path: Path | None,
    collection_key: str,
    grid: dict,
    map_key: str = "map_db",
    frequency_bins: list[float] | None = None,
    frequency_ranges: list[list[float]] | None = None,
    impute: bool = False,
    max_missing_gap: int = 0,
    force_rebuild: bool = False,
    permute: bool = False,
    permute_seed: int | None = 42,
    mask_ranges: list[list[float]] | None = None,
    noise_floor: float | None = None,
) -> LoadedSource:
    if not map_name:
        raise ValueError("data.map.name is required for 4d loading")
    cache_path = map_dir / f"{map_name}.npz"
    if cache_path.exists() and not force_rebuild:
        source = _load_cached(cache_path, map_key, frequency_bins, frequency_ranges)
        if impute:
            source = replace(source, data=impute_array(source.data, max_missing_gap))
        if mask_ranges is not None:
            if noise_floor is None:
                raise ValueError("noise_floor is required when frequency masking is enabled")
            source = replace(
                source,
                data=mask_outside_frequency_ranges(
                    source.data, source.frequencies, mask_ranges, noise_floor
                ),
            )
        if not np.isfinite(source.data).all():
            raise ValueError("Missing or non-finite values remain after 4d imputation")
        return source
    if not files:
        raise ValueError(
            "4d map is unavailable and data.files contains no CSV sources"
        )
    if locations_path is None:
        raise ValueError("4d map generation requires data.map.locations")

    locations = load_locations(locations_path, collection_key)
    frames = []
    stamps = []
    site_names = []
    site_lons = []
    site_lats = []
    for path in files:
        frame, stamp = _read_csv(path)
        frame = select_frequencies(frame, frequency_bins, frequency_ranges)
        if impute:
            frame = impute_frame(frame, max_missing_gap)
        if mask_ranges is not None:
            if noise_floor is None:
                raise ValueError("noise_floor is required when frequency masking is enabled")
            frame = frame.copy()
            frame.iloc[:, :] = mask_outside_frequency_ranges(
                frame.to_numpy(),
                np.asarray([float(column) for column in frame.columns]),
                mask_ranges,
                noise_floor,
            )
        name, location = find_location(path, locations)
        frames.append(frame)
        stamps.append(stamp.floor("min") if stamp is not None else stamp)
        site_names.append(name)
        site_lons.append(float(location["longitude"]))
        site_lats.append(float(location["latitude"]))

    columns = list(frames[0].columns)
    if any(list(frame.columns) != columns for frame in frames[1:]):
        raise ValueError("Map source CSVs must contain matching frequency columns")
    if all(stamp is not None for stamp in stamps):
        common = stamps[0]
        for stamp in stamps[1:]:
            common = common.intersection(stamp)
        if common.empty:
            raise ValueError("Map source CSVs have no common timestamps")
        common = common.sort_values()
        site_data = np.stack([frame.loc[stamp.get_indexer(common)].to_numpy(np.float32) for frame, stamp in zip(frames, stamps)])
        timestamps = common
    else:
        if any(stamp is not None for stamp in stamps) or len({len(frame) for frame in frames}) != 1:
            raise ValueError("Map source CSVs require common timestamps or equal row counts")
        site_data = np.stack([frame.to_numpy(np.float32) for frame in frames])
        timestamps = None

    site_lons_array = np.asarray(site_lons, dtype=np.float64)
    site_lats_array = np.asarray(site_lats, dtype=np.float64)
    position_permutation = np.arange(len(site_lons_array))
    if permute:
        rng = np.random.default_rng(permute_seed)
        position_permutation = rng.permutation(len(site_lons_array))
        site_lons_array = site_lons_array[position_permutation]
        site_lats_array = site_lats_array[position_permutation]

    mapped, grid_x, grid_y = _map_from_sites(
        site_data,
        site_lons_array,
        site_lats_array,
        grid,
    )
    valid_timesteps = np.isfinite(mapped).any(axis=(1, 2, 3))
    mapped = mapped[valid_timesteps]
    if timestamps is not None:
        timestamps = timestamps[valid_timesteps]
    map_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        map_key: mapped,
        "freqs_mhz": np.asarray([float(column) for column in columns], dtype=np.float32),
        "site_names": np.asarray(site_names),
        "site_lons": site_lons_array,
        "site_lats": site_lats_array,
        "position_permutation": position_permutation,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "metadata": np.asarray(json.dumps({
            "files": [str(path) for path in files],
            "grid": grid,
            "permute": permute,
            "permute_seed": permute_seed,
        })),
    }
    if timestamps is not None:
        payload["timestamps"] = np.asarray(timestamps.astype(str))
    np.savez_compressed(cache_path, **payload)
    return LoadedSource(
        mapped,
        payload["freqs_mhz"],
        timestamps,
        [cache_path],
        [str(value) for value in payload["freqs_mhz"]],
        (SequenceSegment(0, len(mapped), str(cache_path)),),
    )
