"""Evaluation-only transfer of frozen 4D map forecasters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from training.common.data_sources import clean_name, load_locations
from training.common.spatial_checkpoint_evaluation import (
    FrozenGrid,
    ReceiverSeries,
    build_idw_maps,
    derive_frozen_grid,
    load_aligned_receiver_csvs,
    normalize_with_checkpoint,
    sample_forecast_grid,
    select_window_origins,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "training/configs/transfer_4d_manifest.yaml"
TRANSFER_MODELS = (
    "convlstm",
    "residualconvlstm",
    "linearar4d",
    "residuallinearar4d",
    "dswinlstm_i",
    "lookbackmean4d",
)
SEED = 42
HORIZONS = (1, 15, 60)
LOOKBACK = 60


@dataclass(frozen=True)
class TransferTask:
    task_id: str
    evaluation: str
    platform: str
    frequency_start_mhz: float
    frequency_end_mhz: float
    geometry_receivers: tuple[str, ...]
    input_receivers: tuple[str, ...]
    score_receivers: tuple[str, ...]
    raw_start: pd.Timestamp
    raw_end: pd.Timestamp
    score_start: pd.Timestamp
    score_end: pd.Timestamp
    temporal_overlap: bool
    specification: Mapping[str, Any]


def repository_path(value: str | Path) -> Path:
    """Resolve a repository-relative manifest path without allowing escape."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest paths must be repository-relative: {value}")
    result = (ROOT / path).resolve()
    root = ROOT.resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"Manifest path escapes repository root: {value}")
    return result


def _timestamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Transfer timestamps must include a timezone: {value}")
    return stamp.tz_convert("UTC")


def _task(task_id: str, entry: Mapping[str, Any]) -> TransferTask:
    band = entry.get("band")
    raw = entry.get("raw_interval")
    score = entry.get("score_interval")
    if not isinstance(band, list) or len(band) != 2 or float(band[1]) - float(band[0]) != 200:
        raise ValueError(f"Task {task_id!r} must declare one 200 MHz band")
    if not isinstance(raw, list) or len(raw) != 2 or not isinstance(score, list) or len(score) != 2:
        raise ValueError(f"Task {task_id!r} must declare raw and score intervals")
    raw_start, raw_end = (_timestamp(str(value)) for value in raw)
    score_start, score_end = (_timestamp(str(value)) for value in score)
    if not raw_start < raw_end or not score_start < score_end:
        raise ValueError(f"Task {task_id!r} has an empty interval")
    names = {}
    for field in ("geometry_receivers", "input_receivers", "score_receivers"):
        values = entry.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"Task {task_id!r} requires unique {field}")
        names[field] = tuple(str(value) for value in values)
    if not set(names["geometry_receivers"]) <= set(names["input_receivers"]):
        raise ValueError(f"Task {task_id!r} geometry receivers must be map inputs")
    if not set(names["score_receivers"]) <= set(names["input_receivers"]):
        raise ValueError(f"Task {task_id!r} score receivers must be map inputs")
    files = entry.get("files")
    if not isinstance(files, dict) or set(files) != set(names["input_receivers"]):
        raise ValueError(f"Task {task_id!r} files must exactly match input receivers")
    return TransferTask(
        task_id=task_id,
        evaluation=str(entry.get("evaluation", "")),
        platform=str(entry.get("platform", "")),
        frequency_start_mhz=float(band[0]),
        frequency_end_mhz=float(band[1]),
        geometry_receivers=names["geometry_receivers"],
        input_receivers=names["input_receivers"],
        score_receivers=names["score_receivers"],
        raw_start=raw_start,
        raw_end=raw_end,
        score_start=score_start,
        score_end=score_end,
        temporal_overlap=bool(entry.get("temporal_overlap", False)),
        specification=entry,
    )


def load_transfer_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate the complete 8-task, 48-cell transfer catalog."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or int(document.get("schema_version", 0)) != 1:
        raise ValueError("4D transfer manifest schema_version must be 1")
    if int(document.get("seed", -1)) != SEED:
        raise ValueError("4D transfer uses only seed 42")
    if tuple(document.get("models", ())) != TRANSFER_MODELS:
        raise ValueError(f"Manifest models must be exactly {list(TRANSFER_MODELS)}")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, dict) or tuple(checkpoints) != TRANSFER_MODELS:
        raise ValueError("Manifest checkpoints must follow the model catalog")
    if checkpoints["lookbackmean4d"] is not None:
        raise ValueError("LookbackMean4D must not have a checkpoint")
    for model, entry in checkpoints.items():
        if model == "lookbackmean4d":
            continue
        if not isinstance(entry, dict) or entry.get("source") not in {"archive", "local", "s3"}:
            raise ValueError(f"Invalid checkpoint source for {model}")
        required = "key" if entry["source"] == "s3" else ("member" if entry["source"] == "archive" else "path")
        if not isinstance(entry.get(required), str):
            raise ValueError(f"Checkpoint for {model} requires {required}")
    if document.get("normalization_checkpoint") not in TRANSFER_MODELS[:-1]:
        raise ValueError("normalization_checkpoint must identify a frozen model")
    if not isinstance(document.get("config"), str):
        raise ValueError("Manifest requires a repository-relative model config")
    repository_path(document["config"])
    entries = document.get("tasks")
    if not isinstance(entries, dict) or len(entries) != 8:
        raise ValueError("4D transfer manifest must contain exactly 8 tasks")
    tasks = [_task(str(task_id), entry) for task_id, entry in entries.items()]
    if sum(task.evaluation == "reference" for task in tasks) != 1:
        raise ValueError("Manifest must contain exactly one reference task")
    new_point = [task for task in tasks if task.evaluation == "new_point"]
    if len(new_point) != 1 or new_point[0].specification.get("option") != 2 \
            or not new_point[0].temporal_overlap or new_point[0].score_receivers != ("guesthouse",):
        raise ValueError("New-point task must use overlapping option 2 and score only guesthouse")
    ara = [task for task in tasks if task.platform == "ara"]
    if len(ara) != 2 or any(set(task.score_receivers) != {"ames", "horticulture"} for task in ara):
        raise ValueError("ARA tasks must score Ames and Horticulture")
    for task in tasks:
        for value in task.specification["files"].values():
            repository_path(value)
        repository_path(task.specification["locations"])
        if task.specification.get("regions"):
            repository_path(task.specification["regions"])
    return document


def transfer_tasks(manifest: Mapping[str, Any] | None = None) -> tuple[TransferTask, ...]:
    document = load_transfer_manifest() if manifest is None else manifest
    return tuple(_task(str(task_id), entry) for task_id, entry in document["tasks"].items())


def campaign_cells(tasks: Sequence[str] | None = None, models: Sequence[str] | None = None) -> list[dict[str, Any]]:
    task_ids = list(tasks or [task.task_id for task in transfer_tasks()])
    model_names = list(models or TRANSFER_MODELS)
    known_tasks = {task.task_id for task in transfer_tasks()}
    if set(task_ids) - known_tasks:
        raise ValueError(f"Unknown 4D transfer tasks: {sorted(set(task_ids) - known_tasks)}")
    if set(model_names) - set(TRANSFER_MODELS):
        raise ValueError(f"Unknown 4D transfer models: {sorted(set(model_names) - set(TRANSFER_MODELS))}")
    return [{"task_id": task_id, "model": model, "seed": SEED}
            for task_id in task_ids for model in model_names]


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def target_frequencies(task: TransferTask) -> np.ndarray:
    values = np.arange(task.frequency_start_mhz + 0.5, task.frequency_end_mhz, 1.0, dtype=np.float32)
    validate_target_frequencies(values)
    return values


def validate_target_frequencies(frequencies: Sequence[float]) -> None:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.shape != (200,) or not np.allclose(np.diff(values), 1.0, rtol=0, atol=1e-6):
        raise ValueError("4D transfer requires exactly 200 ordered 1 MHz bins")


def apply_checkpoint_normalization(values: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    """Apply source statistics by channel position, including on shifted bands."""
    return normalize_with_checkpoint(values, normalization)


def inverse_checkpoint_normalization(values: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(normalization["mean_dbm"], dtype=np.float32).squeeze()
    std = np.asarray(normalization["std_dbm"], dtype=np.float32).squeeze()
    validate_target_frequencies(np.arange(len(mean), dtype=np.float32))
    if mean.shape != std.shape or np.any(std <= 0) or values.shape[-1] != len(mean):
        raise ValueError("Checkpoint normalization is incompatible with map channels")
    return (np.asarray(values, dtype=np.float32) * std + mean).astype(np.float32)


def resolve_checkpoint(
    entry: Mapping[str, Any], *, archive: Path | None = None, cache_dir: Path | None = None,
    endpoint: str | None = None, default_bucket: str | None = None,
) -> Path:
    """Materialize a local, archive-member, or S3/MinIO checkpoint atomically."""
    source = str(entry["source"])
    if source == "local":
        path = repository_path(str(entry["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    identity = str(entry["member"] if source == "archive" else entry["key"])
    cache = cache_dir or Path(tempfile.gettempdir()) / "spectrum-usage-transfer-4d"
    destination = cache / hashlib.sha256(identity.encode()).hexdigest()[:16] / Path(identity).name
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        if source == "archive":
            bundle_path = Path(archive or ROOT / "checkpoints.tgz")
            with tarfile.open(bundle_path, "r:*") as bundle:
                try:
                    member = bundle.getmember(identity)
                except KeyError as exc:
                    raise FileNotFoundError(f"Checkpoint archive has no {identity}") from exc
                handle = bundle.extractfile(member)
                if handle is None or not member.isfile():
                    raise FileNotFoundError(f"Checkpoint archive member is not a file: {identity}")
                with temporary.open("wb") as output:
                    while block := handle.read(1024 * 1024):
                        output.write(block)
        elif source == "s3":
            import boto3

            resolved_endpoint = endpoint or os.environ.get("AWS_ENDPOINT_URL") \
                or os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT")
            client = boto3.client("s3", endpoint_url=resolved_endpoint)
            bucket = str(entry.get("bucket") or default_bucket or os.environ.get("S3_BUCKET", ""))
            if not bucket:
                raise ValueError("S3 checkpoint requires a bucket")
            client.download_file(bucket, identity, str(temporary))
        else:
            raise ValueError(f"Unsupported checkpoint source: {source}")
        if destination.exists():
            temporary.unlink(missing_ok=True)
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def checkpoint_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "frequencies" not in checkpoint \
            or checkpoint.get("normalization") is None:
        raise ValueError(f"{path} is not a normalized integrated checkpoint")
    frequencies = np.asarray(checkpoint["frequencies"], dtype=np.float32)
    validate_target_frequencies(frequencies)
    expected = np.arange(600.5, 800.0, 1.0, dtype=np.float32)
    if not np.allclose(frequencies, expected, rtol=0, atol=1e-6):
        raise ValueError("Transfer checkpoint must use source frequencies 600.5..799.5 MHz")
    normalization = dict(checkpoint["normalization"])
    apply_checkpoint_normalization(np.empty((1, 200), dtype=np.float32), normalization)
    return frequencies, normalization


def load_checkpoint_model(
    checkpoint_path: Path | None, model_name: str, config: dict[str, Any],
    *, normalization: Mapping[str, Any] | None = None, device: str = "cpu",
):
    """Construct the resolved 200-channel architecture and strict-load its state."""
    import torch
    from training.common.model_factory import build_model

    if model_name not in TRANSFER_MODELS:
        raise ValueError(f"Unsupported transfer model {model_name!r}")
    model = build_model(model_name, config, np.empty((LOOKBACK, 10, 10, 200), dtype=np.float32))
    metadata_normalization = dict(normalization) if normalization is not None else None
    if model_name == "lookbackmean4d":
        if checkpoint_path is not None:
            raise ValueError("LookbackMean4D must not load a checkpoint")
    else:
        if checkpoint_path is None:
            raise ValueError(f"{model_name} requires a frozen checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_name = str(checkpoint.get("model_name", model_name)).lower()
        if saved_name != model_name:
            raise ValueError(f"Checkpoint identifies itself as {saved_name!r}, expected {model_name!r}")
        checkpoint_frequencies = np.asarray(checkpoint["frequencies"], dtype=np.float32)
        validate_target_frequencies(checkpoint_frequencies)
        if not np.allclose(checkpoint_frequencies, np.arange(600.5, 800.0, 1.0),
                           rtol=0, atol=1e-6):
            raise ValueError("Transfer checkpoint must use source frequencies 600.5..799.5 MHz")
        metadata_normalization = dict(checkpoint.get("normalization") or {})
        if not metadata_normalization:
            raise ValueError("Frozen checkpoint has no normalization")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if metadata_normalization is None:
        raise ValueError("Evaluation requires checkpoint normalization")
    apply_checkpoint_normalization(np.empty((1, 200), dtype=np.float32), metadata_normalization)
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    return model, metadata_normalization


def infer_transfer_model(
    model: Any, model_name: str, config: Mapping[str, Any], windows: np.ndarray,
    horizons: Sequence[int] = HORIZONS, *, batch_size: int = 8, device: str = "cpu",
) -> np.ndarray:
    import torch
    from training.common.forecasting import forecast

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 5 or values.shape[-3:] != (10, 10, 200):
        raise ValueError("Transfer windows must have shape (B, T, 10, 10, 200)")
    requested = np.asarray(horizons, dtype=np.int64)
    prediction_horizon = int(config[model_name]["model"]["prediction_horizon"])
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[offset:offset + batch_size]).permute(0, 1, 4, 2, 3).to(device)
            prediction = forecast(model, batch, prediction_horizon, int(requested.max()))
            outputs.append(prediction[:, requested - 1].permute(0, 1, 3, 4, 2).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _coordinates(task: TransferTask) -> dict[str, tuple[float, float]]:
    locations = load_locations(repository_path(task.specification["locations"]),
                               str(task.specification["collection_key"]))
    indexed = {clean_name(name): value for name, value in locations.items()}
    aliases = task.specification.get("coordinate_names", {})
    result = {}
    for receiver in task.input_receivers:
        key = clean_name(str(aliases.get(receiver, receiver)))
        if key not in indexed:
            raise ValueError(f"No coordinate found for receiver {receiver!r}")
        result[receiver] = (float(indexed[key]["longitude"]), float(indexed[key]["latitude"]))
    return result


def derive_task_grid(task: TransferTask, coordinates: Mapping[str, tuple[float, float]]) -> FrozenGrid:
    """Derive the frozen grid only from the task's declared geometry set."""
    return derive_frozen_grid(
        [coordinates[name][0] for name in task.geometry_receivers],
        [coordinates[name][1] for name in task.geometry_receivers],
        height=10, width=10, power=2.0,
    )


def build_target_maps(
    future_values: np.ndarray, task: TransferTask,
    coordinates: Mapping[str, tuple[float, float]], grid: FrozenGrid,
) -> np.ndarray:
    """Build future physical-unit IDW maps from the same receivers used for inputs."""
    return build_idw_maps(
        future_values,
        [coordinates[name][0] for name in task.input_receivers],
        [coordinates[name][1] for name in task.input_receivers],
        grid,
    )


def _all_band_rows(
    predictions: np.ndarray, targets: np.ndarray, horizons: Sequence[int], scope: str,
    *, receivers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if predictions.shape != targets.shape:
        raise ValueError("Metric predictions and targets must have identical shapes")
    rows = []
    receiver_axis = predictions.ndim == 4
    names: Sequence[str | None] = list(receivers) if receiver_axis else [None]
    for horizon_index, horizon in enumerate(horizons):
        for receiver_index, receiver in enumerate(names):
            prediction = predictions[:, horizon_index]
            target = targets[:, horizon_index]
            if receiver_axis:
                prediction, target = prediction[:, receiver_index], target[:, receiver_index]
            error = prediction - target
            row = {
                "metric_scope": scope,
                "region_id": "all-band",
                "horizon": int(horizon),
                "receiver": receiver,
                "count": int(np.isfinite(error).sum()),
                "n_origins": int(predictions.shape[0]),
                "mae_db": float(np.nanmean(np.abs(error))),
                "rmse_db": float(np.sqrt(np.nanmean(error ** 2))),
            }
            rows.append(row)
    return rows


def compare_transfer_outputs(
    predicted_maps_dbm: np.ndarray, target_maps_dbm: np.ndarray,
    predicted_receivers_dbm: np.ndarray, raw_receivers_dbm: np.ndarray,
    target_map_receivers_dbm: np.ndarray, horizons: Sequence[int], receiver_names: Sequence[str],
    *, predicted_map_receivers_dbm: np.ndarray | None = None,
    all_target_map_receivers_dbm: np.ndarray | None = None,
    map_receiver_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return map and physical-receiver all-band metrics without mixing targets."""
    map_predictions = (predicted_receivers_dbm if predicted_map_receivers_dbm is None
                       else predicted_map_receivers_dbm)
    map_targets = (target_map_receivers_dbm if all_target_map_receivers_dbm is None
                   else all_target_map_receivers_dbm)
    map_names = receiver_names if map_receiver_names is None else map_receiver_names
    map_rows = _all_band_rows(predicted_maps_dbm, target_maps_dbm, horizons, "full_grid_idw")
    map_rows.extend(_all_band_rows(map_predictions, map_targets, horizons,
                                   "receiver_grid_idw", receivers=map_names))
    receiver_rows = _all_band_rows(predicted_receivers_dbm, raw_receivers_dbm,
                                   horizons, "physical_receiver", receivers=receiver_names)
    return map_rows, receiver_rows


def _regions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{"region_id": row.get("region_id") or row.get("band_id"),
             "start_mhz": float(row["start_mhz"]), "end_mhz": float(row["end_mhz"]),
             "is_noise_floor": str(row.get("is_noise_floor", "false")).lower() == "true"}
            for row in rows]


def _region_rows(
    prediction: np.ndarray, target: np.ndarray, frequencies: np.ndarray,
    horizons: Sequence[int], scope: str, regions: Sequence[Mapping[str, Any]],
    *, receivers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    names: Sequence[str | None] = list(receivers) if prediction.ndim == 4 else [None]
    for horizon_index, horizon in enumerate(horizons):
        for receiver_index, receiver in enumerate(names):
            left, right = prediction[:, horizon_index], target[:, horizon_index]
            if prediction.ndim == 4:
                left, right = left[:, receiver_index], right[:, receiver_index]
            for region in regions:
                keep = (frequencies >= float(region["start_mhz"])) \
                    & (frequencies <= float(region["end_mhz"]))
                if not np.any(keep):
                    continue
                error = left[..., keep] - right[..., keep]
                rows.append({"metric_scope": scope, "region_id": str(region["region_id"]),
                             "is_noise_floor": bool(region.get("is_noise_floor", False)),
                             "horizon": int(horizon), "receiver": receiver,
                             "count": int(np.isfinite(error).sum()),
                             "n_origins": int(prediction.shape[0]),
                             "mae_db": float(np.nanmean(np.abs(error))),
                             "rmse_db": float(np.sqrt(np.nanmean(error ** 2)))})
    return rows


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_transfer_task(
    *, task: TransferTask, model_name: str, config: dict[str, Any],
    checkpoint_path: Path | None, normalization_checkpoint_path: Path | None,
    output_directory: Path, seed: int = SEED, device: str = "cpu",
    batch_size: int | None = None, origin_stride: int = 1, origin_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen model/task cell and persist all transfer artifacts."""
    started = time.perf_counter()
    if seed != SEED:
        raise ValueError("4D transfer uses only seed 42")
    if origin_stride <= 0:
        raise ValueError("origin_stride must be positive")
    fallback_normalization = None
    if model_name == "lookbackmean4d":
        if normalization_checkpoint_path is None:
            raise ValueError("LookbackMean4D requires a frozen normalization checkpoint")
        _, fallback_normalization = checkpoint_metadata(normalization_checkpoint_path)
    model, normalization = load_checkpoint_model(
        checkpoint_path, model_name, config, normalization=fallback_normalization, device=device
    )
    frequencies = target_frequencies(task)
    coordinates = _coordinates(task)
    paths = [repository_path(task.specification["files"][name]) for name in task.input_receivers]
    series = load_aligned_receiver_csvs(
        paths, frequencies, coordinates, receiver_names=task.input_receivers,
        max_missing_gap=int(task.specification.get("max_missing_gap", 9)),
    )
    in_raw = (series.timestamps >= task.raw_start) & (series.timestamps <= task.raw_end)
    series = ReceiverSeries(series.values_dbm[in_raw], series.timestamps[in_raw], series.frequencies,
                            series.names, series.longitudes, series.latitudes)
    grid = derive_task_grid(task, coordinates)
    input_maps = build_idw_maps(series.values_dbm, series.longitudes, series.latitudes, grid)
    normalized_maps = apply_checkpoint_normalization(input_maps, normalization)
    map_series = ReceiverSeries(normalized_maps, series.timestamps, series.frequencies,
                                ("map",), np.asarray([grid.origin_longitude]),
                                np.asarray([grid.origin_latitude]))
    origins, windows, _ = select_window_origins(map_series, LOOKBACK, HORIZONS)
    first_targets = origins + min(HORIZONS)
    last_targets = origins + max(HORIZONS)
    keep = (series.timestamps[first_targets] >= task.score_start) \
        & (series.timestamps[last_targets] <= task.score_end)
    future = np.stack([series.values_dbm[origins + horizon] for horizon in HORIZONS], axis=1)
    finite_axes = tuple(range(1, future.ndim))
    keep &= np.isfinite(windows).all(axis=tuple(range(1, windows.ndim))) \
        & np.isfinite(future).all(axis=finite_axes)
    selected = np.flatnonzero(keep)[::origin_stride]
    if origin_limit is not None:
        selected = selected[:origin_limit]
    if not len(selected):
        raise ValueError(f"Task {task.task_id} produced no valid forecast origins")
    origins, windows, future = origins[selected], windows[selected], future[selected]
    target_maps = build_target_maps(future, task, coordinates, grid)
    configured_batch = int(config[model_name].get("evaluation", {}).get("batch_size", 8))
    predicted_normalized = infer_transfer_model(
        model, model_name, config, windows, HORIZONS,
        batch_size=int(batch_size or configured_batch), device=device,
    )
    predicted_maps = inverse_checkpoint_normalization(predicted_normalized, normalization)
    score_indices = [task.input_receivers.index(name) for name in task.score_receivers]
    input_lons = [coordinates[name][0] for name in task.input_receivers]
    input_lats = [coordinates[name][1] for name in task.input_receivers]
    all_predicted_receivers, all_inside, all_outside_distance = sample_forecast_grid(
        predicted_maps, grid, input_lons, input_lats
    )
    all_target_map_receivers, _, _ = sample_forecast_grid(target_maps, grid, input_lons, input_lats)
    predicted_receivers = all_predicted_receivers[:, :, score_indices]
    target_map_receivers = all_target_map_receivers[:, :, score_indices]
    inside = all_inside[score_indices]
    outside_distance = all_outside_distance[score_indices]
    raw_receivers = future[:, :, score_indices]
    map_rows, receiver_rows = compare_transfer_outputs(
        predicted_maps, target_maps, predicted_receivers, raw_receivers,
        target_map_receivers, HORIZONS, task.score_receivers,
        predicted_map_receivers_dbm=all_predicted_receivers,
        all_target_map_receivers_dbm=all_target_map_receivers,
        map_receiver_names=task.input_receivers,
    )
    common = {"task_id": task.task_id, "model": model_name, "seed": seed}
    for row in map_rows + receiver_rows:
        row.update(common)
    for row in receiver_rows:
        index = task.score_receivers.index(str(row["receiver"]))
        row.update(inside_grid=bool(inside[index]), distance_outside_grid_m=float(outside_distance[index]))

    region_rows: list[dict[str, Any]] = []
    if task.specification.get("regions"):
        region_rows = [dict(row) for row in map_rows + receiver_rows]
        definitions = _regions(repository_path(task.specification["regions"]))
        region_rows.extend(_region_rows(predicted_maps, target_maps, frequencies, HORIZONS,
                                        "full_grid_idw", definitions))
        region_rows.extend(_region_rows(all_predicted_receivers, all_target_map_receivers, frequencies,
                                        HORIZONS, "receiver_grid_idw", definitions,
                                        receivers=task.input_receivers))
        region_rows.extend(_region_rows(predicted_receivers, raw_receivers, frequencies,
                                        HORIZONS, "physical_receiver", definitions,
                                        receivers=task.score_receivers))
        for row in region_rows:
            row.update(common)

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(map_rows).to_csv(output_directory / "map_metrics.csv", index=False)
    pd.DataFrame(receiver_rows).to_csv(output_directory / "receiver_metrics.csv", index=False)
    if task.specification.get("regions"):
        pd.DataFrame(region_rows).to_csv(output_directory / "region_metrics.csv", index=False)
    np.savez_compressed(
        output_directory / "forecasts.npz",
        predicted_maps_dbm=predicted_maps,
        target_maps_dbm=target_maps,
        predicted_receivers_dbm=predicted_receivers,
        target_map_receivers_dbm=target_map_receivers,
        all_predicted_map_receivers_dbm=all_predicted_receivers,
        all_target_map_receivers_dbm=all_target_map_receivers,
        raw_receivers_dbm=raw_receivers,
        origins=origins,
        origin_timestamps=series.timestamps[origins].astype(str).to_numpy(),
        frequencies_mhz=frequencies,
        receivers=np.asarray(task.score_receivers),
    )
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {"task_id": task.task_id, "model": model_name, "seed": seed,
                              "completed": True, "elapsed_seconds": elapsed}
    for horizon_index, horizon in enumerate(HORIZONS):
        physical_error = predicted_receivers[:, horizon_index] - raw_receivers[:, horizon_index]
        grid_error = predicted_maps[:, horizon_index] - target_maps[:, horizon_index]
        result[f"mae_db_t{horizon}"] = float(np.nanmean(np.abs(physical_error)))
        result[f"full_grid_mae_db_t{horizon}"] = float(np.nanmean(np.abs(grid_error)))
    metadata = {
        **common,
        "evaluation_only": True,
        "temporal_overlap": task.temporal_overlap,
        "option": task.specification.get("option"),
        "raw_interval": [task.raw_start.isoformat(), task.raw_end.isoformat()],
        "score_interval": [task.score_start.isoformat(), task.score_end.isoformat()],
        "geometry_receivers": list(task.geometry_receivers),
        "input_receivers": list(task.input_receivers),
        "score_receivers": list(task.score_receivers),
        "grid": {"height": 10, "width": 10, "idw_power": 2.0,
                 "origin_longitude": grid.origin_longitude, "origin_latitude": grid.origin_latitude},
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "normalization_source": "frozen_checkpoint_by_bin_position",
        "normalization_fitted_on_target": False,
        "checkpoint_frequency_range_mhz": [600.5, 799.5],
        "target_frequency_range_mhz": [float(frequencies[0]), float(frequencies[-1])],
        "n_origins": int(len(origins)),
    }
    _json_write(output_directory / "metadata.json", metadata)
    _json_write(output_directory / "result.json", result)
    return result
