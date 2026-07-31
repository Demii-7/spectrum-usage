"""Zero-shot evaluation of frozen, flattened 1D transfer checkpoints."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from training.common.data_sources import load_csv_sources
from training.common.evaluation_integrated import (
    calculate_errors_and_export_arrays,
    evaluation_parameters,
)
from training.common.forecast_export import export_map_forecasts
from training.common.forecasting import forecast
from training.common.model_factory import build_model
from training.common.preprocessing import SequenceSegment
from training.common.transfer_2d import TransferTask, transfer_tasks as catalog_transfer_tasks
from training.common.windowing import make_window_batch_array, make_window_starts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "training/configs/transfer_1d_manifest.json"
TRANSFER_MODELS = (
    "vanillalstm1d",
    "residualvanillalstm",
    "linearar1d",
    "residuallinearar1d",
)
SEED = 42
EXPECTED_BINS = 200


def transfer_tasks() -> tuple[TransferTask, ...]:
    """Return the shared, stable 12-task transfer catalog."""
    return catalog_transfer_tasks()


def repository_path(value: str | Path) -> Path:
    """Resolve a repository-relative manifest path without allowing escape."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest paths must be repository-relative: {value}")
    resolved = (ROOT / path).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Manifest path escapes repository root: {value}")
    return resolved


def load_transfer_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the complete 12-task manifest for the four supported 1D models."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("1D transfer manifest schema_version must be 1")
    if not isinstance(document.get("config"), str):
        raise ValueError("1D transfer manifest requires one config path")
    repository_path(document["config"])

    models = document.get("models")
    if not isinstance(models, dict) or tuple(models) != TRANSFER_MODELS:
        raise ValueError(f"Manifest models must contain exactly {list(TRANSFER_MODELS)}")
    for model_name, checkpoint in models.items():
        if not isinstance(checkpoint, str) or "seed-42" not in checkpoint:
            raise ValueError(f"Manifest checkpoint for {model_name!r} must be a seed-42 path")
        repository_path(checkpoint)

    entries = document.get("tasks")
    known = {task.task_id for task in transfer_tasks()}
    if not isinstance(entries, dict) or set(entries) != known:
        raise ValueError("Manifest must contain all stable transfer task IDs")
    for task_id, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list) \
                or not entry["files"]:
            raise ValueError(f"Manifest task {task_id!r} requires a non-empty files list")
        for value in entry["files"]:
            if not isinstance(value, str):
                raise ValueError(f"Manifest task {task_id!r} contains an invalid file path")
            repository_path(value)
    return document


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_checkpoint(value: str) -> Path:
    """Resolve an extracted checkpoint or materialize its archive member."""
    path = repository_path(value)
    if path.exists():
        return path
    archive_path = ROOT / "checkpoints.tgz"
    if not archive_path.exists() or not value.startswith("checkpoints/"):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    cache = Path(tempfile.gettempdir()) / "spectrum-usage-transfer-checkpoints" / value
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        try:
            member = archive.getmember(value)
        except KeyError as exc:
            raise FileNotFoundError(f"Checkpoint archive has no {value}") from exc
        source = archive.extractfile(member)
        if source is None or not member.isfile():
            raise FileNotFoundError(f"Checkpoint archive member is not a file: {value}")
        temporary = cache.with_suffix(cache.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes(source.read())
        if cache.exists():
            temporary.unlink()
        else:
            temporary.replace(cache)
    return cache


def _normalization_stats(
    normalization: Mapping[str, Any],
    *,
    expected_bins: int | None = EXPECTED_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(normalization["mean_dbm"], dtype=np.float32).reshape(-1)
    std = np.asarray(normalization["std_dbm"], dtype=np.float32).reshape(-1)
    if mean.shape != std.shape or (expected_bins is not None and len(mean) != expected_bins):
        expected = f"{expected_bins} " if expected_bins is not None else ""
        raise ValueError(f"1D transfer normalization must contain {expected}bins")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Checkpoint normalization contains invalid statistics")
    return mean, std


def apply_source_normalization(
    data: np.ndarray,
    normalization: Mapping[str, Any],
) -> np.ndarray:
    """Apply frozen source statistics by frequency-column position."""
    values = np.asarray(data, dtype=np.float32)
    mean, std = _normalization_stats(normalization, expected_bins=None)
    if values.ndim != 2 or values.shape[1] != len(mean):
        raise ValueError("Target data shape does not match normalization columns")
    return ((values - mean) / std).astype(np.float32)


def validate_target_frequencies(task: TransferTask, frequencies: np.ndarray) -> str:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.shape != (EXPECTED_BINS,) or not np.all(np.diff(values) > 0) or not np.allclose(
        np.diff(values), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"Task {task.task_id} requires exactly {EXPECTED_BINS} ordered target bins at 1 MHz spacing"
        )
    return (
        "source_checkpoint_stats_by_bin_position"
        if task.frequency_start_mhz != 600.0
        else "source_checkpoint_stats_same_band_by_bin_position"
    )


def _flatten_values(
    data: np.ndarray,
    segments: tuple[SequenceSegment, ...],
) -> tuple[np.ndarray, tuple[SequenceSegment, ...]]:
    values = []
    flattened_segments = []
    offset = 0
    for segment in segments:
        block = np.asarray(data[segment.start:segment.end], dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(f"Expected source data shaped (time, frequency), got {block.shape}")
        for frequency_index in range(block.shape[1]):
            series = block[:, frequency_index]
            values.append(series)
            flattened_segments.append(SequenceSegment(
                offset,
                offset + len(series),
                f"{segment.label}:frequency_{frequency_index}",
            ))
            offset += len(series)
    if not values:
        return np.empty((0, 1), dtype=np.float32), tuple()
    return np.concatenate(values).reshape(-1, 1).astype(np.float32), tuple(flattened_segments)


def flatten_source(source):
    """Convert each source frequency column into an independent scalar series."""
    if source.timestamps is None:
        raise ValueError("1D transfer CSVs require timestamp_utc")
    data, segments = _flatten_values(source.data, source.segments)
    timestamps = []
    for segment in source.segments:
        block_timestamps = source.timestamps[segment.start:segment.end]
        for _ in range(source.data.shape[1]):
            timestamps.extend(block_timestamps)
    return replace(
        source,
        data=data,
        timestamps=pd.DatetimeIndex(timestamps),
        segments=segments,
        feature_labels=["scalar"],
    )


def _flattened_frequency_indices(
    segments: tuple[SequenceSegment, ...],
    rows: np.ndarray,
    *,
    frequency_count: int | None = None,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    indices = np.full(rows.shape, -1, dtype=np.int64)
    for segment in segments:
        token = str(segment.label).rsplit("frequency_", 1)[-1]
        if not token.isdigit():
            raise ValueError(f"Flattened 1D segment is missing a frequency index: {segment.label!r}")
        selected = (rows >= segment.start) & (rows < segment.end)
        indices[selected] = int(token)
    if np.any(indices < 0) or (
        frequency_count is not None and np.any(indices >= frequency_count)
    ):
        raise ValueError("Flattened 1D rows are outside their frequency segments")
    return indices


def row_normalization(
    normalization: Mapping[str, Any],
    segments: tuple[SequenceSegment, ...],
    rows: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Select checkpoint statistics for flattened rows by positional bin."""
    mean, std = _normalization_stats(normalization, expected_bins=None)
    frequency_indices = _flattened_frequency_indices(segments, rows)
    if np.any(frequency_indices >= len(mean)):
        raise ValueError("Flattened 1D segment frequency exceeds normalization statistics")
    return {
        **dict(normalization),
        "mean_dbm": mean[frequency_indices].reshape(-1, 1),
        "std_dbm": std[frequency_indices].reshape(-1, 1),
    }, frequency_indices


def apply_flattened_source_normalization(
    data: np.ndarray,
    normalization: Mapping[str, Any],
    segments: tuple[SequenceSegment, ...],
) -> np.ndarray:
    """Normalize flattened scalar rows using their original bin positions."""
    rows = np.arange(len(data), dtype=np.int64)
    selected, _ = row_normalization(normalization, segments, rows)
    values = np.asarray(data, dtype=np.float32)
    if values.shape != (len(rows), 1):
        raise ValueError("Flattened 1D data must have shape (rows, 1)")
    return ((values - selected["mean_dbm"]) / selected["std_dbm"]).astype(np.float32)


def preflight_checkpoints(config: dict[str, Any], checkpoints: Mapping[str, Path]) -> None:
    """Strict-load every seed-42 checkpoint against the scalar architecture."""
    dummy = np.empty((121, 1), dtype=np.float32)
    for model_name in TRANSFER_MODELS:
        checkpoint = torch.load(checkpoints[model_name], map_location="cpu", weights_only=False)
        saved_name = str(checkpoint.get("model_name", model_name)).lower()
        if saved_name != model_name:
            raise ValueError(f"Checkpoint for {model_name} identifies itself as {saved_name!r}")
        normalization = checkpoint.get("normalization")
        if normalization is None:
            raise ValueError(f"Checkpoint for {model_name} lacks normalization")
        _normalization_stats(normalization)
        model = build_model(model_name, config, dummy)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def target_window_starts(
    timestamps: pd.DatetimeIndex,
    segments: tuple[SequenceSegment, ...],
    *,
    lookback: int,
    max_horizon: int,
    stride: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> np.ndarray:
    """Build segment-safe windows whose complete forecast is in the target interval."""
    starts = make_window_starts(len(timestamps), lookback, max_horizon, stride, segments)
    first = starts + lookback
    last = first + max_horizon - 1
    keep = (timestamps[first] >= target_start) & (timestamps[last] < target_end)
    return starts[np.asarray(keep)]


def load_target(task: TransferTask, files: list[Path], *, impute: bool, max_missing_gap: int):
    source = load_csv_sources(
        files,
        concat="rows",
        frequency_ranges=[[task.frequency_start_mhz, task.frequency_end_mhz]],
        impute=impute,
        max_missing_gap=max_missing_gap,
        timestamp_ranges=[(task.raw_start, task.raw_end)],
    )
    if source.timestamps is None:
        raise ValueError("Transfer CSVs require timestamp_utc")
    validate_target_frequencies(task, source.frequencies)
    return flatten_source(source)


def evaluate_transfer_model(
    *,
    task: TransferTask,
    source,
    model_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    if model_name not in TRANSFER_MODELS:
        raise ValueError(f"Unknown 1D transfer model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("model_name", model_name)).lower() != model_name:
        raise ValueError(f"Checkpoint model does not match configured model {model_name!r}")
    normalization = checkpoint.get("normalization")
    if normalization is None:
        raise ValueError(f"Checkpoint {checkpoint_path} lacks normalization metadata")
    _normalization_stats(normalization)

    parameters = evaluation_parameters(config, model_name)
    normalized = apply_flattened_source_normalization(source.data, normalization, source.segments)
    normalization_policy = validate_target_frequencies(task, source.frequencies)
    model = build_model(model_name, config, normalized)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    starts = target_window_starts(
        source.timestamps,
        source.segments,
        lookback=parameters["lookback"],
        max_horizon=parameters["max_horizon"],
        stride=parameters["test_stride"],
        target_start=task.target_start,
        target_end=task.target_end,
    )
    if not len(starts):
        raise ValueError(f"Task {task.task_id} produced no target windows")
    windows = make_window_batch_array(
        full_x=normalized,
        start_rows=starts,
        lookback=parameters["lookback"],
    )
    parts = {horizon: [] for horizon in parameters["horizons"]}
    with torch.no_grad():
        for offset in range(0, len(windows), parameters["batch_size"]):
            values = torch.from_numpy(
                windows[offset:offset + parameters["batch_size"]]
            ).float().to(device)
            predicted = forecast(
                model=model,
                x=values,
                prediction_horizon=parameters["prediction_horizon"],
                rollout_horizon=parameters["max_horizon"],
                targets=None,
            )
            for horizon in parts:
                parts[horizon].append(predicted[:, horizon - 1].cpu().numpy())

    predictions, targets, rows, frequency_indices_by_horizon, metrics = {}, {}, {}, {}, []
    for horizon, values in parts.items():
        target_rows = starts + parameters["lookback"] + horizon - 1
        row_normalization_payload, frequency_indices = row_normalization(
            normalization, source.segments, target_rows
        )
        prediction, target, absolute, squared = calculate_errors_and_export_arrays(
            prediction_normalized=np.concatenate(values),
            target_raw_model_layout=source.data[target_rows],
            normalization=row_normalization_payload,
        )
        predictions[horizon] = prediction
        targets[horizon] = target
        rows[horizon] = target_rows
        frequency_indices_by_horizon[horizon] = frequency_indices
        metrics.append({
            "task_id": task.task_id,
            "model": model_name,
            "seed": seed,
            "horizon": horizon,
            "n_targets": len(target_rows),
            "mae_db": float(absolute.mean()),
            "rmse_db": float(np.sqrt(squared.mean())),
        })

    export_map_forecasts(
        output_dir,
        task.task_id,
        model_name,
        predictions,
        targets,
        rows,
        {
            "task_id": task.task_id,
            "evaluation": task.evaluation,
            "platform": task.platform,
            "sites": task.sites,
            "raw_interval": [task.raw_start.isoformat(), task.raw_end.isoformat()],
            "target_interval": [task.target_start.isoformat(), task.target_end.isoformat()],
            "target_interval_end_exclusive": True,
            "frequencies_mhz": source.frequencies,
            "flattened_scalar_series": True,
            "frequency_indices_by_horizon": frequency_indices_by_horizon,
            "checkpoint_path": checkpoint_path,
            "normalization_source": "checkpoint",
            "normalization_alignment_policy": normalization_policy,
            "normalization_fitted_on_target": False,
            "seed": seed,
        },
    )
    return metrics
