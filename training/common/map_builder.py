"""Loader and cache builder for spatial spectrum maps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from training.common.data_sources import (
    LoadedSource,
    clean_name,
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


def find_site_grid_indices(
    cache_path: Path,
    map_key: str = "map_db",
) -> dict[str, tuple[int, int]]:
    """Return the nearest grid (h, w) index for each collection site."""
    with np.load(cache_path, allow_pickle=True) as archive:
        grid_x = np.asarray(archive["grid_x"], dtype=np.float64)
        grid_y = np.asarray(archive["grid_y"], dtype=np.float64)
        site_names = list(archive["site_names"])
        site_lons = np.asarray(archive["site_lons"], dtype=np.float64)
        site_lats = np.asarray(archive["site_lats"], dtype=np.float64)
        perm = np.asarray(archive["position_permutation"], dtype=np.intp)

    site_lons = site_lons[perm]
    site_lats = site_lats[perm]
    origin_lon = float(np.mean(site_lons))
    origin_lat = float(np.mean(site_lats))
    sx, sy = _local_xy(site_lons, site_lats, origin_lon, origin_lat)

    result: dict[str, tuple[int, int]] = {}
    for i, name in enumerate(site_names):
        dist = np.hypot(grid_x - sx[i], grid_y - sy[i])
        idx = np.unravel_index(int(dist.argmin()), dist.shape)
        result[str(name)] = (int(idx[0]), int(idx[1]))
    return result


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
    with np.load(path, allow_pickle=True) as archive:
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


def _request_metadata(
    files, grid, permute, permute_seed, frequency_bins, frequency_ranges,
    selected_sites, excluded_sites, outage_threshold,
):
    return {
        "files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
        "grid": grid,
        "permute": permute,
        "permute_seed": permute_seed,
        "frequency_bins": frequency_bins,
        "frequency_ranges": frequency_ranges,
        "selected_sites": selected_sites,
        "excluded_sites": excluded_sites,
        "outage_threshold": outage_threshold,
    }


def _has_long_nonfinite_run(frame, threshold):
    invalid = ~np.isfinite(frame.to_numpy(dtype=np.float64))
    for values in invalid.T:
        padded = np.concatenate(([False], values, [False])).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        if np.any(edges[1::2] - edges[::2] > threshold):
            return True
    return False


def prepare_4d_partitions(
    partitions: dict[str, list[Path]], locations_path: Path, collection_key: str,
    frequency_bins, frequency_ranges, outage_threshold: int,
) -> tuple[dict[str, list[Path]], list[str], list[str]]:
    """Validate and deterministically filter paired 4D site partitions."""
    locations = load_locations(locations_path, collection_key)
    indexed = {}
    outages = set()
    for partition, files in partitions.items():
        sites = {}
        for path in files:
            name, location = find_location(path, locations)
            identity = clean_name(name)
            if identity in sites:
                raise ValueError(f"Duplicate canonical site {identity!r} in {partition} partition")
            frame, _ = _read_csv(path)
            frame = select_frequencies(frame, frequency_bins, frequency_ranges)
            sites[identity] = (path, float(location["longitude"]), float(location["latitude"]))
            if _has_long_nonfinite_run(frame, outage_threshold):
                outages.add(identity)
        indexed[partition] = sites

    train_sites = set(indexed["train"])
    test_sites = set(indexed["test"])
    if train_sites != test_sites:
        missing = sorted(train_sites - test_sites)
        extra = sorted(test_sites - train_sites)
        raise ValueError(f"Train/test canonical site sets differ; missing from test: {missing}; extra in test: {extra}")
    for identity in sorted(train_sites):
        if indexed["train"][identity][1:] != indexed["test"][identity][1:]:
            raise ValueError(f"Train/test coordinates differ for canonical site {identity!r}")

    selected = sorted(train_sites - outages)
    excluded = sorted(outages)
    if not selected:
        raise ValueError("All 4D sites were excluded by the non-finite outage threshold")
    ordered = {
        partition: [indexed[partition][identity][0] for identity in selected]
        for partition in ("train", "test")
    }
    return ordered, selected, excluded


def load_map_layout(path: Path) -> dict[str, object]:
    """Load the source-site layout and grid coordinate system from a map cache."""
    with np.load(path, allow_pickle=True) as archive:
        required = {"site_names", "site_lons", "site_lats", "position_permutation", "grid_x", "grid_y"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Map cache {path} is missing layout fields: {', '.join(sorted(missing))}")
        permutation = np.asarray(archive["position_permutation"], dtype=np.intp)
        inverse = np.argsort(permutation)
        return {
            "site_names": [str(value) for value in archive["site_names"]],
            "site_lons": np.asarray(archive["site_lons"], dtype=np.float64)[inverse],
            "site_lats": np.asarray(archive["site_lats"], dtype=np.float64)[inverse],
            "grid_x": np.asarray(archive["grid_x"], dtype=np.float64),
            "grid_y": np.asarray(archive["grid_y"], dtype=np.float64),
        }


def _layout_error(actual_names, actual_lons, actual_lats, expected):
    expected_names = list(expected["site_names"])
    if actual_names != expected_names:
        return f"site names/order differ: expected {expected_names}, got {actual_names}"
    if not np.array_equal(np.asarray(actual_lons), np.asarray(expected["site_lons"])) or not np.array_equal(
        np.asarray(actual_lats), np.asarray(expected["site_lats"])
    ):
        return "site coordinates differ from the training layout"
    return None


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
    expected_layout: dict[str, object] | None = None,
    selected_sites: list[str] | None = None,
    excluded_sites: list[str] | None = None,
    outage_threshold: int | None = None,
) -> LoadedSource:
    if not map_name:
        raise ValueError("data.map.name is required for 4d loading")
    cache_path = map_dir / f"{map_name}.npz"
    locations = None
    if files and locations_path is not None:
        locations = load_locations(locations_path, collection_key)
        files = sorted(files, key=lambda path: clean_name(find_location(path, locations)[0]))
    request_metadata = _request_metadata(
        files, grid, permute, permute_seed, frequency_bins, frequency_ranges,
        selected_sites, excluded_sites, outage_threshold,
    )
    requested_layout = None
    if files and locations_path is not None:
        requested_names = []
        requested_lons = []
        requested_lats = []
        for path in files:
            name, location = find_location(path, locations)
            requested_names.append(name)
            requested_lons.append(float(location["longitude"]))
            requested_lats.append(float(location["latitude"]))
        requested_layout = (requested_names, requested_lons, requested_lats)
    cache_error = None
    if cache_path.exists() and not force_rebuild:
        try:
            with np.load(cache_path, allow_pickle=True) as archive:
                if "metadata" not in archive:
                    raise ValueError("cache has no request metadata")
                cached_metadata = json.loads(str(archive["metadata"].item()))
            if files and cached_metadata != request_metadata:
                raise ValueError("cached request metadata does not match the requested sources or map settings")
            if requested_layout is not None:
                cached_layout = load_map_layout(cache_path)
                error = _layout_error(*requested_layout, cached_layout)
                if error:
                    raise ValueError(error)
            if expected_layout is not None:
                cached_layout = load_map_layout(cache_path)
                error = _layout_error(
                    cached_layout["site_names"], cached_layout["site_lons"],
                    cached_layout["site_lats"], expected_layout,
                )
                if error:
                    raise ValueError(error)
                if not np.array_equal(cached_layout["grid_x"], expected_layout["grid_x"]) or not np.array_equal(
                    cached_layout["grid_y"], expected_layout["grid_y"]
                ):
                    raise ValueError("grid coordinates differ from the training map")
            source = _load_cached(cache_path, map_key, frequency_bins, frequency_ranges)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            cache_error = str(exc)
            source = None
        if source is not None:
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
            f"4d map is unavailable or incompatible ({cache_error}) and data.files contains no CSV sources"
        )
    if locations_path is None:
        raise ValueError("4d map generation requires data.map.locations")

    if locations is None:
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
        if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Missing or non-finite values remain after 4d imputation: {path}")
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

    if expected_layout is not None:
        error = _layout_error(site_names, site_lons, site_lats, expected_layout)
        if error:
            raise ValueError(f"Test map must use the training site layout: {error}")

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
    if expected_layout is not None and (
        not np.array_equal(grid_x, expected_layout["grid_x"])
        or not np.array_equal(grid_y, expected_layout["grid_y"])
    ):
        raise ValueError("Test map grid coordinates differ from the training map")
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
        "metadata": np.asarray(json.dumps(request_metadata)),
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
