"""Pure primitives for evaluating spatial checkpoints at receiver locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from training.common.data_sources import _read_csv, impute_array
from training.common.map_builder import _local_xy


@dataclass(frozen=True)
class FrozenGrid:
    x: np.ndarray
    y: np.ndarray
    origin_longitude: float
    origin_latitude: float
    power: float = 2.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape


@dataclass(frozen=True)
class ReceiverSeries:
    values_dbm: np.ndarray
    timestamps: pd.DatetimeIndex
    frequencies: np.ndarray
    names: tuple[str, ...]
    longitudes: np.ndarray
    latitudes: np.ndarray


def _permutation(permutation: Sequence[int] | None, size: int) -> np.ndarray:
    result = np.arange(size, dtype=np.intp) if permutation is None else np.asarray(permutation, dtype=np.intp)
    if result.shape != (size,) or not np.array_equal(np.sort(result), np.arange(size)):
        raise ValueError("coordinate_permutation must be a bijection over receiver indices")
    return result


def derive_frozen_grid(
    longitudes: Sequence[float],
    latitudes: Sequence[float],
    *,
    coordinate_permutation: Sequence[int] | None = None,
    height: int = 10,
    width: int = 10,
    width_meters: float | None = None,
    height_meters: float | None = None,
    power: float = 2.0,
) -> FrozenGrid:
    """Derive the fixed grid using the exact geometry of map_builder._map_from_sites."""
    lons = np.asarray(longitudes, dtype=np.float64)
    lats = np.asarray(latitudes, dtype=np.float64)
    if lons.ndim != 1 or lons.shape != lats.shape or not len(lons):
        raise ValueError("receiver coordinates must be non-empty matching vectors")
    if height <= 0 or width <= 0 or power <= 0:
        raise ValueError("grid dimensions and IDW power must be positive")
    order = _permutation(coordinate_permutation, len(lons))
    lons, lats = lons[order], lats[order]
    origin_lon, origin_lat = float(np.mean(lons)), float(np.mean(lats))
    site_x, site_y = _local_xy(lons, lats, origin_lon, origin_lat)
    width_m = float(max(np.ptp(site_x), 1.0) if width_meters is None else width_meters)
    height_m = float(max(np.ptp(site_y), 1.0) if height_meters is None else height_meters)
    gx, gy = np.meshgrid(
        np.linspace(-width_m / 2.0, width_m / 2.0, width),
        np.linspace(-height_m / 2.0, height_m / 2.0, height),
    )
    return FrozenGrid(gx, gy, origin_lon, origin_lat, float(power))


def _idw_weights(target_x: np.ndarray, target_y: np.ndarray, source_x: np.ndarray,
                 source_y: np.ndarray, power: float) -> np.ndarray:
    distances = np.hypot(target_x[..., None] - source_x, target_y[..., None] - source_y)
    weights = 1.0 / np.maximum(distances, 1e-6) ** power
    exact = distances == 0
    for index in range(source_x.size):
        weights[exact[..., index], :] = 0.0
        weights[exact[..., index], index] = 1.0
    return weights


def build_idw_maps(
    values_dbm: np.ndarray,
    longitudes: Sequence[float],
    latitudes: Sequence[float],
    grid: FrozenGrid,
    *,
    coordinate_permutation: Sequence[int] | None = None,
) -> np.ndarray:
    """Map frequency-last receiver values ``(..., R, F)`` to ``(..., H, W, F)``."""
    values = np.asarray(values_dbm)
    lons, lats = np.asarray(longitudes, dtype=np.float64), np.asarray(latitudes, dtype=np.float64)
    if lons.shape != lats.shape or values.ndim < 2 or values.shape[-2] != len(lons):
        raise ValueError("values and receiver coordinates do not agree")
    order = _permutation(coordinate_permutation, len(lons))
    site_x, site_y = _local_xy(lons[order], lats[order], grid.origin_longitude, grid.origin_latitude)
    weights = _idw_weights(grid.x, grid.y, site_x, site_y, grid.power)
    valid = np.isfinite(values)
    weighted = np.einsum("...rf,hwr->...hwf", np.where(valid, values, 0.0), weights)
    denominator = np.einsum("...rf,hwr->...hwf", valid, weights)
    return np.divide(weighted, denominator, out=np.full_like(weighted, np.nan, dtype=np.float64),
                     where=denominator > 0).astype(np.float32)


def sample_forecast_grid(
    maps: np.ndarray,
    grid: FrozenGrid,
    longitudes: Sequence[float],
    latitudes: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IDW-sample frequency-last maps without clamping, including extrapolation metadata."""
    values = np.asarray(maps)
    if values.ndim < 3 or values.shape[-3:-1] != grid.shape:
        raise ValueError("map spatial dimensions do not match the frozen grid")
    x, y = _local_xy(np.asarray(longitudes), np.asarray(latitudes), grid.origin_longitude, grid.origin_latitude)
    weights = _idw_weights(x, y, grid.x.ravel(), grid.y.ravel(), grid.power)
    flat = values.reshape(*values.shape[:-3], -1, values.shape[-1])
    sampled = np.einsum("...gf,rg->...rf", flat, weights) / weights.sum(axis=-1)[:, None]
    inside = (x >= grid.x.min()) & (x <= grid.x.max()) & (y >= grid.y.min()) & (y <= grid.y.max())
    dx = np.maximum.reduce((grid.x.min() - x, np.zeros_like(x), x - grid.x.max()))
    dy = np.maximum.reduce((grid.y.min() - y, np.zeros_like(y), y - grid.y.max()))
    return sampled.astype(np.float32), inside, np.hypot(dx, dy)


def load_aligned_receiver_csvs(
    paths: Sequence[Path],
    checkpoint_frequencies: Sequence[float],
    coordinates: Mapping[str, tuple[float, float]],
    *,
    receiver_names: Sequence[str] | None = None,
    max_missing_gap: int = 0,
) -> ReceiverSeries:
    """Read, frequency-order, inner-align, and bounded-impute receiver CSVs."""
    names = tuple(receiver_names or [Path(path).stem for path in paths])
    if len(names) != len(paths) or len(set(names)) != len(names):
        raise ValueError("receiver_names must uniquely identify every CSV")
    frequencies = np.asarray(checkpoint_frequencies, dtype=np.float64)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame, timestamps = _read_csv(Path(path))
        if timestamps is None:
            raise ValueError(f"{path} has no timestamp_utc column")
        available = np.asarray([float(column) for column in frame.columns])
        indices = []
        for frequency in frequencies:
            match = np.flatnonzero(np.isclose(available, frequency, rtol=0, atol=1e-6))
            if len(match) != 1:
                raise ValueError(f"{path} does not contain checkpoint frequency {frequency}")
            indices.append(int(match[0]))
        ordered = frame.iloc[:, indices].copy()
        ordered.columns = frequencies
        ordered.index = timestamps.floor("min")
        frames.append(ordered)
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index, sort=False)
    if common.empty:
        raise ValueError("receiver CSVs have no common timestamps")
    values = np.stack([frame.loc[common].to_numpy(np.float32) for frame in frames], axis=1)
    values = impute_array(values, max_missing_gap)
    missing = [name for name in names if name not in coordinates]
    if missing:
        raise ValueError(f"missing coordinates for: {', '.join(missing)}")
    return ReceiverSeries(values, pd.DatetimeIndex(common), frequencies.astype(np.float32), names,
                          np.asarray([coordinates[name][0] for name in names]),
                          np.asarray([coordinates[name][1] for name in names]))


def normalize_with_checkpoint(values: np.ndarray, normalization: Mapping[str, Any] | None) -> np.ndarray:
    """Apply checkpoint statistics without fitting evaluation data."""
    values = np.asarray(values, dtype=np.float32)
    if normalization is None:
        return values.copy()
    mean = np.asarray(normalization["mean_dbm"], dtype=np.float32).squeeze()
    std = np.asarray(normalization["std_dbm"], dtype=np.float32).squeeze()
    if mean.ndim != 1 or mean.shape != std.shape or values.shape[-1] != len(mean) or np.any(std <= 0):
        raise ValueError("checkpoint normalization is incompatible with frequency axis")
    return ((values - mean) / std).astype(np.float32)


def select_window_origins(
    series: ReceiverSeries,
    input_length: int,
    horizons: Sequence[int],
    *,
    model_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid origins, model inputs, and untouched physical targets."""
    horizon = np.asarray(horizons, dtype=np.int64)
    if input_length <= 0 or horizon.ndim != 1 or not len(horizon) or np.any(horizon <= 0):
        raise ValueError("input_length and horizons must be positive")
    inputs = series.values_dbm if model_values is None else np.asarray(model_values)
    if inputs.shape != series.values_dbm.shape:
        raise ValueError("model_values must match receiver series shape")
    stamps = series.timestamps.view("i8")
    minute = pd.Timedelta(minutes=1).value
    origins = []
    for origin in range(input_length - 1, len(stamps) - int(horizon.max())):
        indices = np.concatenate((np.arange(origin - input_length + 1, origin + 1), origin + horizon))
        if np.all(np.diff(stamps[indices]) == np.diff(indices) * minute):
            origins.append(origin)
    origin_array = np.asarray(origins, dtype=np.int64)
    x = np.stack([inputs[o - input_length + 1:o + 1] for o in origins]) if origins else np.empty((0, input_length, *inputs.shape[1:]))
    y = np.stack([series.values_dbm[o + horizon] for o in origins]) if origins else np.empty((0, len(horizon), *inputs.shape[1:]))
    return origin_array, x, y


def infer_frozen_model(
    checkpoint_path: Path,
    model_name: str,
    config: dict[str, Any],
    input_maps: np.ndarray,
    frequencies: Sequence[float],
    horizons: Sequence[int],
    *,
    normalization: dict[str, Any] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Load a frozen map model and return frequency-last forecasts at requested horizons."""
    import torch
    from training.common.forecasting import forecast
    from training.common.model_factory import build_model, load_checkpoint_into_model

    maps = np.asarray(input_maps, dtype=np.float32)
    if maps.ndim != 5 or batch_size <= 0:
        raise ValueError("input_maps must have shape (B, T, H, W, F)")
    if model_name.lower() == "lookbackmean4d":
        return np.stack([maps.mean(axis=1) for _ in horizons], axis=1)
    horizon = np.asarray(horizons, dtype=np.int64)
    rollout = int(horizon.max())
    prediction_horizon = int(config[model_name.lower()]["model"]["prediction_horizon"])
    model = build_model(model_name, config, maps[0])
    torch_device = torch.device(device)
    model, _ = load_checkpoint_into_model(checkpoint_path=Path(checkpoint_path), model=model,
        model_name=model_name.lower(), data_normalization=normalization,
        data_frequencies=list(frequencies), device=torch_device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(maps), batch_size):
            batch = (
                torch.from_numpy(maps[start:start + batch_size])
                .permute(0, 1, 4, 2, 3)
                .contiguous()
                .to(torch_device)
            )
            predicted = forecast(model, batch, prediction_horizon, rollout)
            outputs.append(predicted[:, horizon - 1].permute(0, 1, 3, 4, 2).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def aggregate_region_metrics(
    predictions: np.ndarray,
    physical_targets: np.ndarray,
    frequencies: Sequence[float],
    regions: Sequence[Mapping[str, Any]],
    *,
    normalization: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate physical-unit errors by frequency region."""
    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(physical_targets, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if normalization is not None:
        mean = np.asarray(normalization["mean_dbm"], dtype=np.float32).squeeze()
        std = np.asarray(normalization["std_dbm"], dtype=np.float32).squeeze()
        pred = pred * std + mean
    frequencies = np.asarray(frequencies)
    result = []
    for region in regions:
        keep = (frequencies >= float(region["start_mhz"])) & (frequencies <= float(region["end_mhz"]))
        if not np.any(keep):
            raise ValueError(f"region {region['region_id']!r} contains no frequencies")
        error = pred[..., keep] - target[..., keep]
        result.append({"region_id": str(region["region_id"]),
                       "is_noise_floor": bool(region.get("is_noise_floor", False)),
                       "count": int(np.isfinite(error).sum()),
                       "mae_db": float(np.nanmean(np.abs(error))),
                       "rmse_db": float(np.sqrt(np.nanmean(error ** 2)))})
    return result
