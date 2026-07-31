"""Zero-shot evaluation of archived 2D checkpoints on target-only data."""

from __future__ import annotations

from dataclasses import dataclass
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
from training.common.preprocessing import SequenceSegment, apply_per_frequency_normalization
from training.common.windowing import make_window_batch_array, make_window_starts

ROOT = Path(__file__).resolve().parents[2]

TRANSFER_MODELS = (
    "autoformer_csa", "linearar2d", "lstmattn", "residuallinearar2d",
    "residualvanillalstm", "temporalconvnet", "vanillalstm", "lookbackmean2d",
)
SEED = 42


@dataclass(frozen=True)
class TransferTask:
    task_id: str
    evaluation: str
    platform: str
    raw_start: pd.Timestamp
    raw_end: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    frequency_start_mhz: float
    frequency_end_mhz: float
    sites: tuple[str, ...]


_ROWS = (
    ("reference", "Reference", "powder", "2026-07-03 18:39", "2026-07-05 20:39", "2026-07-03 19:39", "2026-07-05 19:39", ((600, 800),), ("cpg", "ebc", "humanities", "madsen", "moran", "sagepoint")),
    ("new-point", "New point", "powder", "2026-06-18 00:36", "2026-06-20 02:36", "2026-06-18 01:36", "2026-06-20 01:36", ((600, 800),), ("guesthouse",)),
    ("new-bands", "New bands", "powder", "2026-07-03 18:39", "2026-07-05 20:39", "2026-07-03 19:39", "2026-07-05 19:39", ((800, 1000), (2400, 2600), (3500, 3700), (5725, 5925)), ("cpg", "ebc", "humanities", "madsen", "moran", "sagepoint")),
    ("new-site", "New site", "cosmos", "2026-06-19 18:15", "2026-06-21 20:15", "2026-06-19 19:15", "2026-06-21 19:15", ((600, 800), (2400, 2600)), ("sdr2-md1", "sdr2-s1-lg1")),
    ("new-site", "New site", "ara", "2026-07-02 19:20", "2026-07-04 21:20", "2026-07-02 20:20", "2026-07-04 20:20", ((600, 800), (2400, 2600)), ("ames", "horticulture")),
    ("new-site", "New site", "aerpaw", "2022-02-08 17:57", "2022-02-10 19:57", "2022-02-08 18:57", "2022-02-10 18:57", ((600, 800), (2400, 2600)), ("CC1", "CC2", "LW1")),
)


def transfer_tasks() -> tuple[TransferTask, ...]:
    """Return the stable task catalog specified by TRANSFER.md."""
    tasks = []
    for prefix, label, platform, raw_start, raw_end, target_start, target_end, bands, sites in _ROWS:
        for low, high in bands:
            kind = prefix if low == 600 else ("new-site-new-band" if prefix == "new-site" else prefix)
            task_id = f"{kind}-{platform}-{low}-{high}"
            tasks.append(TransferTask(
                task_id, label if low == 600 else ("New site, new band" if prefix == "new-site" else label),
                platform, *_timestamps(raw_start, raw_end, target_start, target_end),
                float(low), float(high), tuple(sites),
            ))
    return tuple(tasks)


def _timestamps(*values: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(value, tz="UTC") for value in values)


def load_transfer_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("Transfer manifest schema_version must be 1")
    known = {task.task_id for task in transfer_tasks()}
    entries = document.get("tasks")
    if not isinstance(entries, dict) or set(entries) - known:
        raise ValueError("Manifest tasks must be a mapping containing only stable transfer task IDs")
    if not isinstance(document.get("config"), str):
        raise ValueError("Transfer manifest requires one repository-relative config path")
    models = document.get("models")
    if not isinstance(models, dict) or set(models) != set(TRANSFER_MODELS):
        raise ValueError(f"Manifest models must contain exactly {list(TRANSFER_MODELS)}")
    for model_name, entry in models.items():
        if model_name == "lookbackmean2d" and entry is None:
            continue
        if not isinstance(entry, str):
            raise ValueError(f"Manifest checkpoint for {model_name!r} must be a relative path")
    if set(entries) != known:
        raise ValueError("Manifest must contain all stable transfer task IDs")
    for task_id, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list) or not entry["files"]:
            raise ValueError(f"Manifest task {task_id!r} requires a non-empty files list")
    return document


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def repository_path(value: str) -> Path:
    """Resolve a manifest path under the repository root on the current node."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest paths must be repository-relative: {value}")
    resolved = (ROOT / path).resolve()
    if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
        raise ValueError(f"Manifest path escapes repository root: {value}")
    return resolved


def resolve_checkpoint(value: str) -> Path:
    """Resolve an extracted checkpoint or materialize its member from checkpoints.tgz."""
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
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(value)
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


def apply_source_normalization(data: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    """Apply archived source statistics without deriving anything from target data."""
    mean = np.asarray(normalization["mean_dbm"], dtype=np.float32).reshape(-1)
    std = np.asarray(normalization["std_dbm"], dtype=np.float32).reshape(-1)
    if data.ndim != 2 or data.shape[1] != len(mean) or mean.shape != std.shape:
        raise ValueError("Target frequency-bin count does not match checkpoint normalization")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Checkpoint normalization contains invalid statistics")
    return apply_per_frequency_normalization(np.asarray(data, dtype=np.float32), mean, std).astype(np.float32)


def validate_target_frequencies(task: TransferTask, frequencies: np.ndarray) -> str:
    """Validate the positional transfer contract used by the 600--800 MHz checkpoint."""
    values = np.asarray(frequencies, dtype=np.float64)
    if values.shape != (200,) or not np.all(np.diff(values) > 0) or not np.allclose(
        np.diff(values), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"Task {task.task_id} requires exactly 200 ordered target bins at 1 MHz spacing"
        )
    return (
        "source_checkpoint_stats_by_bin_position"
        if task.frequency_start_mhz != 600.0
        else "source_checkpoint_stats_same_band_by_bin_position"
    )


def preflight_checkpoints(config: dict[str, Any], checkpoints: Mapping[str, Path]) -> None:
    """Build all seven 200-bin architectures and strict-load seed-42 states."""
    dummy = np.empty((121, 200), dtype=np.float32)
    for model_name in TRANSFER_MODELS:
        if model_name == "lookbackmean2d":
            continue
        checkpoint = torch.load(checkpoints[model_name], map_location="cpu", weights_only=False)
        saved_name = str(checkpoint.get("model_name", model_name)).lower()
        if saved_name != model_name:
            raise ValueError(f"Checkpoint for {model_name} identifies itself as {saved_name!r}")
        normalization = checkpoint.get("normalization")
        if normalization is None:
            raise ValueError(f"Checkpoint for {model_name} lacks normalization")
        apply_source_normalization(dummy, normalization)
        model = build_model(model_name, config, dummy)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def target_window_starts(
    timestamps: pd.DatetimeIndex, segments: tuple[SequenceSegment, ...], *, lookback: int,
    max_horizon: int, stride: int, target_start: pd.Timestamp, target_end: pd.Timestamp,
) -> np.ndarray:
    """Build segment-safe windows whose full forecast lies in [start, end)."""
    starts = make_window_starts(len(timestamps), lookback, max_horizon, stride, segments)
    first = starts + lookback
    last = first + max_horizon - 1
    keep = (timestamps[first] >= target_start) & (timestamps[last] < target_end)
    return starts[np.asarray(keep)]


def load_target(task: TransferTask, files: list[Path], *, impute: bool, max_missing_gap: int):
    ranges = [(task.raw_start, task.raw_end)]
    source = load_csv_sources(
        files, concat="rows", frequency_ranges=[[task.frequency_start_mhz, task.frequency_end_mhz]],
        impute=impute, max_missing_gap=max_missing_gap, timestamp_ranges=ranges,
    )
    if source.timestamps is None:
        raise ValueError("Transfer CSVs require timestamp_utc")
    validate_target_frequencies(task, source.frequencies)
    return source


def evaluate_transfer_model(
    *, task: TransferTask, source, model_name: str, checkpoint_path: Path,
    config: dict[str, Any], output_dir: Path, seed: int = SEED,
) -> list[dict[str, Any]]:
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization") if checkpoint is not None else None
    if normalization is None:
        raise ValueError(f"Checkpoint {checkpoint_path} lacks normalization metadata")
    parameters = evaluation_parameters(config, model_name)
    normalized = apply_source_normalization(source.data, normalization)
    normalization_policy = validate_target_frequencies(task, source.frequencies)
    model = build_model(model_name, config, normalized)
    if checkpoint is not None and model_name != "lookbackmean2d":
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    starts = target_window_starts(
        source.timestamps, source.segments, lookback=parameters["lookback"],
        max_horizon=parameters["max_horizon"], stride=parameters["test_stride"],
        target_start=task.target_start, target_end=task.target_end,
    )
    if not len(starts):
        raise ValueError(f"Task {task.task_id} produced no target windows")
    windows = make_window_batch_array(
        full_x=normalized,
        start_rows=starts,
        lookback=parameters["lookback"],
    )
    parts = {h: [] for h in parameters["horizons"]}
    with torch.no_grad():
        for offset in range(0, len(windows), parameters["batch_size"]):
            values = torch.from_numpy(windows[offset:offset + parameters["batch_size"]]).float().to(device)
            predicted = forecast(model=model, x=values, prediction_horizon=parameters["prediction_horizon"],
                                 rollout_horizon=parameters["max_horizon"], targets=None)
            for horizon in parts:
                parts[horizon].append(predicted[:, horizon - 1].cpu().numpy())
    predictions, targets, rows, metrics = {}, {}, {}, []
    for horizon, values in parts.items():
        target_rows = starts + parameters["lookback"] + horizon - 1
        prediction, target, absolute, squared = calculate_errors_and_export_arrays(
            prediction_normalized=np.concatenate(values), target_raw_model_layout=source.data[target_rows],
            normalization=dict(normalization),
        )
        predictions[horizon], targets[horizon], rows[horizon] = prediction, target, target_rows
        metrics.append({"task_id": task.task_id, "model": model_name, "seed": seed, "horizon": horizon,
                        "n_targets": len(target_rows), "mae_db": float(absolute.mean()),
                        "rmse_db": float(np.sqrt(squared.mean()))})
    export_map_forecasts(
        output_dir, task.task_id, model_name, predictions, targets, rows,
        {"task_id": task.task_id, "evaluation": task.evaluation, "platform": task.platform,
         "sites": task.sites, "raw_interval": [task.raw_start.isoformat(), task.raw_end.isoformat()],
         "target_interval": [task.target_start.isoformat(), task.target_end.isoformat()],
         "target_interval_end_exclusive": True, "frequencies_mhz": source.frequencies,
          "checkpoint_path": checkpoint_path, "normalization_source": "checkpoint",
         "normalization_alignment_policy": normalization_policy,
         "normalization_fitted_on_target": False, "seed": seed},
    )
    return metrics
