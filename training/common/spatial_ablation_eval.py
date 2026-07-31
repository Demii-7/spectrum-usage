"""Evaluation-only spatial ablations for frozen 4D checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from training.common.data_sources import clean_name, load_locations
from training.common.spatial_checkpoint_evaluation import (
    ReceiverSeries,
    aggregate_region_metrics,
    build_idw_maps,
    derive_frozen_grid,
    infer_frozen_model,
    load_aligned_receiver_csvs,
    normalize_with_checkpoint,
    sample_forecast_grid,
    select_window_origins,
)


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _coordinates(path: Path, collection_key: str, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    locations = load_locations(path, collection_key)
    by_key = {clean_name(name): value for name, value in locations.items()}
    result = {}
    for name in names:
        key = clean_name(name)
        if key not in by_key:
            raise ValueError(f"No coordinate found for receiver {name!r}")
        location = by_key[key]
        result[name] = (float(location["longitude"]), float(location["latitude"]))
    return result


def _checkpoint_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any] | None]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "frequencies" not in checkpoint:
        raise ValueError(f"{path} is not an integrated checkpoint")
    frequencies = np.asarray(checkpoint["frequencies"], dtype=np.float32)
    normalization = checkpoint.get("normalization")
    return frequencies, normalization


def _frequencies_from_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, nrows=0)
    columns = [column for column in frame.columns if column != "timestamp_utc"]
    return np.asarray([float(column) for column in columns], dtype=np.float32)


def _regions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "region_id": row.get("region_id") or row.get("band_id"),
            "start_mhz": float(row["start_mhz"]),
            "end_mhz": float(row["end_mhz"]),
            "is_noise_floor": str(row.get("is_noise_floor", "false")).lower() == "true",
        }
        for row in rows
    ]


def _permutation(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    result = rng.permutation(size)
    while np.array_equal(result, identity):
        result = rng.permutation(size)
    return result


def _subset_series(series: ReceiverSeries, indices: Sequence[int]) -> ReceiverSeries:
    selected = np.asarray(indices, dtype=np.intp)
    return ReceiverSeries(
        series.values_dbm[:, selected],
        series.timestamps,
        series.frequencies,
        tuple(series.names[index] for index in selected),
        series.longitudes[selected],
        series.latitudes[selected],
    )


def _filter_origins(
    series: ReceiverSeries,
    origins: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    horizons: Sequence[int],
    *,
    score_start: str | None,
    score_end: str | None,
    stride: int,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError("origin stride must be positive")
    keep = np.ones(len(origins), dtype=bool)
    max_target = origins + max(horizons)
    if score_start:
        keep &= series.timestamps[origins] >= pd.Timestamp(score_start)
    if score_end:
        keep &= series.timestamps[max_target] <= pd.Timestamp(score_end)
    selected = np.flatnonzero(keep)[::stride]
    if limit is not None:
        selected = selected[:limit]
    if not len(selected):
        raise ValueError("No valid forecast origins remain after filtering")
    return origins[selected], inputs[selected], targets[selected]


def _finite_origin_mask(
    series: ReceiverSeries,
    origins: np.ndarray,
    input_indices: Sequence[int],
    raw_targets: np.ndarray,
    lookback: int,
) -> np.ndarray:
    target_axes = tuple(range(1, raw_targets.ndim))
    target_valid = np.isfinite(raw_targets).all(axis=target_axes)
    input_valid = np.asarray(
        [
            np.isfinite(
                series.values_dbm[
                    origin - lookback + 1:origin + 1,
                    input_indices,
                ]
            ).all()
            for origin in origins
        ],
        dtype=bool,
    )
    return input_valid & target_valid


def _nearest_grid_indices(
    grid: Any,
    longitudes: Sequence[float],
    latitudes: Sequence[float],
    origin_longitude: float,
    origin_latitude: float,
) -> list[tuple[int, int]]:
    from training.common.map_builder import _local_xy

    x, y = _local_xy(
        np.asarray(longitudes, dtype=np.float64),
        np.asarray(latitudes, dtype=np.float64),
        origin_longitude,
        origin_latitude,
    )
    return [
        tuple(int(value) for value in np.unravel_index(
            int(np.hypot(grid.x - site_x, grid.y - site_y).argmin()),
            grid.shape,
        ))
        for site_x, site_y in zip(x, y)
    ]


def _map_metric_rows(
    predicted_maps: np.ndarray,
    target_maps: np.ndarray,
    frequencies: np.ndarray,
    regions: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    normalization: Mapping[str, Any] | None,
    grid: Any,
    receiver_indices: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(horizons):
        predicted = predicted_maps[:, horizon_index]
        target = target_maps[:, horizon_index]
        comparisons = [("full_grid_idw", predicted, target)]
        receiver_prediction = np.stack(
            [predicted[:, row, column] for row, column in receiver_indices],
            axis=1,
        )
        receiver_target = np.stack(
            [target[:, row, column] for row, column in receiver_indices],
            axis=1,
        )
        comparisons.append(("receiver_grid_idw", receiver_prediction, receiver_target))
        for scope, prediction, expected in comparisons:
            for row in aggregate_region_metrics(
                prediction,
                expected,
                frequencies,
                regions,
                normalization=normalization,
            ):
                row.update(
                    metric_scope=scope,
                    horizon=horizon,
                    n_origins=int(prediction.shape[0]),
                )
                rows.append(row)
    return rows


def _condition_data(
    experiment: Mapping[str, Any],
    condition: str,
    frequencies: np.ndarray,
    coordinates: Mapping[str, tuple[float, float]],
    root: Path,
) -> tuple[ReceiverSeries, list[int], list[int]]:
    baseline = list(experiment["baseline_receivers"])
    added = list(experiment["added_receivers"])
    if condition.startswith("geometry_"):
        names = baseline
        file_group = experiment["geometry_files"]
        input_indices = list(range(len(baseline)))
        target_indices = input_indices
    elif condition in {"six_to_new", "eight_to_new"}:
        names = baseline + added
        file_group = experiment["new_receiver_files"]
        input_indices = list(range(len(baseline))) if condition == "six_to_new" else list(range(len(names)))
        target_indices = list(range(len(baseline), len(names)))
    elif condition in {"six_to_guesthouse", "seven_to_guesthouse"}:
        guesthouse = str(experiment["guesthouse_receiver"])
        names = baseline + [guesthouse]
        file_group = experiment["guesthouse_files"]
        input_indices = (
            list(range(len(baseline)))
            if condition == "six_to_guesthouse"
            else list(range(len(names)))
        )
        target_indices = [len(baseline)]
    else:
        raise ValueError(f"Unsupported condition {condition!r}")
    paths = [_resolve(file_group[name], root) for name in names]
    series = load_aligned_receiver_csvs(
        paths,
        frequencies,
        coordinates,
        receiver_names=names,
        max_missing_gap=int(experiment.get("max_missing_gap", 9)),
    )
    return series, input_indices, target_indices


def evaluate_condition(
    *,
    experiment_path: Path,
    model_config_path: Path,
    checkpoint_path: Path | None,
    model_name: str,
    seed: int,
    condition: str,
    output_directory: Path,
    permutation_seed: int | None = None,
    origin_stride: int = 1,
    origin_limit: int | None = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint and write physical-receiver metrics."""
    started = time.perf_counter()
    experiment_path = experiment_path.resolve()
    model_config_path = model_config_path.resolve()
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
    root = experiment_path.parents[2]
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    if model_name == "lookbackmean4d":
        frequencies = _frequencies_from_csv(
            _resolve(experiment["geometry_files"][experiment["baseline_receivers"][0]], root)
        )
        normalization = None
    else:
        frequencies, normalization = _checkpoint_metadata(checkpoint_path)
    all_names = list(experiment["baseline_receivers"]) + list(experiment["added_receivers"])
    if experiment.get("guesthouse_receiver"):
        all_names.append(str(experiment["guesthouse_receiver"]))
    coordinates = _coordinates(
        _resolve(experiment["locations"], root),
        str(experiment.get("collection_key", "endpoints")),
        all_names,
    )
    baseline = list(experiment["baseline_receivers"])
    baseline_lons = [coordinates[name][0] for name in baseline]
    baseline_lats = [coordinates[name][1] for name in baseline]
    grid_cfg = experiment.get("grid", {})
    grid = derive_frozen_grid(
        baseline_lons,
        baseline_lats,
        height=int(grid_cfg.get("height", 10)),
        width=int(grid_cfg.get("width", 10)),
        power=float(grid_cfg.get("power", 2.0)),
    )
    series, input_indices, target_indices = _condition_data(
        experiment, condition, frequencies, coordinates, root
    )
    input_series = _subset_series(series, input_indices)
    coordinate_permutation = None
    if condition == "geometry_permuted":
        if permutation_seed is None:
            raise ValueError("geometry_permuted requires permutation_seed")
        coordinate_permutation = _permutation(len(input_indices), permutation_seed)
    elif permutation_seed is not None:
        raise ValueError("permutation_seed is only valid for geometry_permuted")

    input_maps = build_idw_maps(
        input_series.values_dbm,
        input_series.longitudes,
        input_series.latitudes,
        grid,
        coordinate_permutation=coordinate_permutation,
    )
    normalized_maps = normalize_with_checkpoint(input_maps, normalization)
    map_series = ReceiverSeries(
        normalized_maps,
        series.timestamps,
        frequencies,
        ("map",),
        np.asarray([grid.origin_longitude]),
        np.asarray([grid.origin_latitude]),
    )
    horizons = [int(value) for value in experiment.get("horizons", [1, 15, 60])]
    origins, windows, _ = select_window_origins(
        map_series,
        int(experiment.get("lookback", 60)),
        horizons,
    )
    raw_targets = np.stack(
        [series.values_dbm[origins + horizon][:, target_indices] for horizon in horizons],
        axis=1,
    )
    lookback = int(experiment.get("lookback", 60))
    finite = _finite_origin_mask(
        series,
        origins,
        input_indices,
        raw_targets,
        lookback,
    )
    origins, windows, raw_targets = origins[finite], windows[finite], raw_targets[finite]
    if condition.startswith("geometry_"):
        timing_key = "geometry_timing"
    elif condition.endswith("_guesthouse"):
        timing_key = "guesthouse_timing"
    else:
        timing_key = "new_receiver_timing"
    timing = experiment.get(timing_key, {})
    origins, windows, raw_targets = _filter_origins(
        series,
        origins,
        windows,
        raw_targets,
        horizons,
        score_start=timing.get("start"),
        score_end=timing.get("end"),
        stride=origin_stride,
        limit=origin_limit,
    )
    predictions = infer_frozen_model(
        checkpoint_path,
        model_name,
        model_config,
        windows,
        frequencies,
        horizons,
        normalization=normalization,
        batch_size=batch_size,
        device=device,
    )
    target_names = [series.names[index] for index in target_indices]
    target_lons = series.longitudes[target_indices]
    target_lats = series.latitudes[target_indices]
    sampled, inside, outside_distance = sample_forecast_grid(
        predictions, grid, target_lons, target_lats
    )
    region_definitions = _regions(_resolve(experiment["regions"], root))
    metric_rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(horizons):
        for receiver_index, receiver in enumerate(target_names):
            rows = aggregate_region_metrics(
                sampled[:, horizon_index, receiver_index],
                raw_targets[:, horizon_index, receiver_index],
                frequencies,
                region_definitions,
                normalization=normalization,
            )
            for row in rows:
                row.update(
                    model=model_name,
                    seed=seed,
                    condition=condition,
                    permutation_seed=permutation_seed,
                    horizon=horizon,
                    receiver=receiver,
                    n_origins=len(origins),
                    inside_training_grid=bool(inside[receiver_index]),
                    distance_outside_grid_m=float(outside_distance[receiver_index]),
                )
            metric_rows.extend(rows)

    map_metric_rows: list[dict[str, Any]] = []
    if condition == "geometry_correct":
        target_maps = np.stack(
            [
                build_idw_maps(
                    series.values_dbm[origins + horizon][:, input_indices],
                    input_series.longitudes,
                    input_series.latitudes,
                    grid,
                )
                for horizon in horizons
            ],
            axis=1,
        )
        receiver_grid_indices = _nearest_grid_indices(
            grid,
            input_series.longitudes,
            input_series.latitudes,
            grid.origin_longitude,
            grid.origin_latitude,
        )
        map_metric_rows = _map_metric_rows(
            predictions,
            target_maps,
            frequencies,
            region_definitions,
            horizons,
            normalization=normalization,
            grid=grid,
            receiver_indices=receiver_grid_indices,
        )
        for row in map_metric_rows:
            row.update(model=model_name, seed=seed, condition=condition)

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_directory / "region_metrics.csv", index=False)
    if map_metric_rows:
        pd.DataFrame(map_metric_rows).to_csv(
            output_directory / "map_metrics.csv", index=False
        )
    np.savez_compressed(
        output_directory / "receiver_forecasts.npz",
        predictions_normalized=sampled,
        predicted_maps_normalized=predictions,
        target_maps_dbm=(target_maps if map_metric_rows else np.empty(0)),
        targets_dbm=raw_targets,
        origins=origins,
        origin_timestamps=series.timestamps[origins].astype(str).to_numpy(),
        frequencies_mhz=frequencies,
        receivers=np.asarray(target_names),
    )
    permutation_names = None
    if coordinate_permutation is not None:
        permutation_names = {
            input_series.names[index]: input_series.names[int(coordinate_permutation[index])]
            for index in range(len(input_series.names))
        }
    checkpoint_sha256 = (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_path is not None else None
    )
    metadata = {
        "evaluation_only": True,
        "model": model_name,
        "seed": seed,
        "condition": condition,
        "permutation_seed": permutation_seed,
        "permutation_assignment": permutation_names,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "input_receivers": list(input_series.names),
        "target_receivers": target_names,
        "n_origins": len(origins),
        "origin_stride": origin_stride,
        "horizons": horizons,
        "sampling_policy": "idw_extrapolation_from_frozen_grid",
        "validation_period_overlap": condition in {"six_to_new", "eight_to_new"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    non_noise = [row for row in metric_rows if not row["is_noise_floor"]]
    summary: dict[str, Any] = {
        **metadata,
        "checkpoint_sha256": checkpoint_sha256,
        "completed": True,
    }
    for horizon in horizons:
        values = [row["mae_db"] for row in non_noise if row["horizon"] == horizon]
        summary[f"mae_db_t{horizon}"] = float(np.mean(values))
    summary["mean_horizon_mae_db"] = float(
        np.mean([summary[f"mae_db_t{horizon}"] for horizon in horizons])
    )
    (output_directory / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--condition",
        choices=(
            "geometry_correct",
            "geometry_permuted",
            "six_to_new",
            "eight_to_new",
            "six_to_guesthouse",
            "seven_to_guesthouse",
        ),
        required=True,
    )
    parser.add_argument("--permutation-seed", type=int)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--origin-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_condition(
        experiment_path=args.experiment,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        seed=args.seed,
        condition=args.condition,
        output_directory=args.output_dir,
        permutation_seed=args.permutation_seed,
        origin_stride=args.origin_stride,
        origin_limit=args.origin_limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
