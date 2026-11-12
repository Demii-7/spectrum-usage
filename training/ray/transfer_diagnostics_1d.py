"""Level-vs-dynamics diagnostics for frozen 1D transfer checkpoints, run on Ray.

Re-evaluates the frozen seed-42 1D checkpoints under three input treatments and
splits each error into a per-series level bias and the remaining shape error:

- ``source_norm``: the published protocol (training per-bin mean/std on target).
- ``target_norm``: per-bin mean/std fitted on the target data, isolating
  normalization shift from model behaviour.
- ``warmup_norm``: per-bin mean/std fitted only on the first ``--warmup-hours``
  of target data (causal); windows whose inputs start inside the warm-up are
  not scored, in any mode, so all modes share one window set.
- ``window_centered``: each input window's mean is removed before the frozen
  model runs and added back to the output (post-hoc level-free probe).

A parameter-free dB-space ``lookbackmean`` row is scored on the same windows.
Results are printed between CSV_BEGIN/CSV_END markers so a job driver can be
read from the Ray dashboard logs.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SPECTRUM_REPO_ROOT", "/workspace/spectrum-usage"))
MODES = ("source_norm", "target_norm", "warmup_norm", "window_centered")


def _series_ids(segments, rows):
    starts = np.array([segment.start for segment in segments])
    return np.searchsorted(starts, rows, side="right") - 1


def _error_stats(prediction_db, target_db, series):
    error = np.asarray(prediction_db, dtype=np.float64).reshape(-1) - np.asarray(
        target_db, dtype=np.float64
    ).reshape(-1)
    count = np.bincount(series)
    bias = np.bincount(series, weights=error) / np.maximum(count, 1)
    shape_error = error - bias[series]
    absolute = np.abs(error)
    return {
        "n_targets": int(len(error)),
        "mae_db": float(absolute.mean()),
        "rmse_db": float(np.sqrt((error ** 2).mean())),
        "median_abs_db": float(np.median(absolute)),
        "p95_abs_db": float(np.percentile(absolute, 95)),
        "mean_signed_db": float(error.mean()),
        "series_bias_mae_db": float(np.abs(bias[count > 0]).mean()),
        "shape_mae_db": float(np.abs(shape_error).mean()),
    }


def _target_normalization(source, normalization, mask=None):
    from training.common import transfer_1d as t1

    mean, _ = t1._normalization_stats(normalization, expected_bins=None)
    index = t1._flattened_frequency_indices(
        source.segments, np.arange(len(source.data)), frequency_count=len(mean)
    )
    values = source.data[:, 0].astype(np.float64)
    if mask is not None:
        index, values = index[mask], values[mask]
    count = np.maximum(np.bincount(index, minlength=len(mean)), 1)
    target_mean = np.bincount(index, weights=values, minlength=len(mean)) / count
    squares = np.bincount(index, weights=values ** 2, minlength=len(mean)) / count
    target_std = np.sqrt(np.maximum(squares - target_mean ** 2, 1e-6))
    return {
        **dict(normalization),
        "mean_dbm": target_mean.astype(np.float32),
        "std_dbm": target_std.astype(np.float32),
    }


def evaluate_unit(task, files, model_name, config_path, checkpoints, stride, warmup_hours):
    """Evaluate one (task, model) pair, or the dB-space reference when model_name is None."""
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    import yaml

    from training.common import transfer_1d as t1

    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    source = t1.load_target(
        task, [t1.repository_path(value) for value in files], impute=True, max_missing_gap=10
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    def record(name, mode, horizon, extra, stats):
        rows.append({
            "task_id": task.task_id, "evaluation": task.evaluation, "model": name,
            "mode": mode, "horizon": horizon, "stride": stride, "warmup_hours": warmup_hours,
            **extra, **stats,
        })

    parameters = t1.evaluation_parameters(config, model_name or "linearar1d")
    starts = t1.target_window_starts(
        source.timestamps, source.segments, lookback=parameters["lookback"],
        max_horizon=parameters["max_horizon"], stride=stride,
        target_start=task.target_start, target_end=task.target_end,
    )
    import pandas as pd

    warmup_end = task.raw_start + pd.Timedelta(hours=warmup_hours)
    starts = starts[np.asarray(source.timestamps[starts] >= warmup_end)]
    if not len(starts):
        raise ValueError(f"Task {task.task_id} has no windows after the warm-up")
    lookback = parameters["lookback"]

    if model_name is None:
        # Parameter-free reference on the same windows, computed directly in dB.
        windows_db = t1.make_window_batch_array(
            full_x=source.data, start_rows=starts, lookback=lookback
        )
        level = windows_db.reshape(len(windows_db), lookback, -1).mean(axis=1)
        nan = {"input_mean_shift_db": float("nan"), "input_std_ratio": float("nan")}
        for horizon in parameters["horizons"]:
            target_rows = starts + lookback + horizon - 1
            record("lookbackmean", "db_space", horizon, nan,
                   _error_stats(level, source.data[target_rows],
                                _series_ids(source.segments, target_rows)))
        return rows

    checkpoint = torch.load(t1.resolve_checkpoint(checkpoints[model_name]),
                            map_location="cpu", weights_only=False)
    source_norm = checkpoint["normalization"]
    source_mean, source_std = t1._normalization_stats(source_norm, expected_bins=None)
    target_norm = _target_normalization(source, source_norm)
    warmup_norm = _target_normalization(
        source, source_norm, mask=np.asarray(source.timestamps < warmup_end)
    )
    extra = {"input_mean_shift_db": float(np.abs(target_norm["mean_dbm"] - source_mean).mean()),
             "input_std_ratio": float(np.median(target_norm["std_dbm"] / source_std)),
             "warmup_mean_err_db": float(np.abs(warmup_norm["mean_dbm"] - target_norm["mean_dbm"]).mean())}
    for mode in MODES:
        norm = {"target_norm": target_norm, "warmup_norm": warmup_norm}.get(mode, source_norm)
        normalized = t1.apply_flattened_source_normalization(source.data, norm, source.segments)
        model = t1.build_model(model_name, config, normalized)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        windows = t1.make_window_batch_array(full_x=normalized, start_rows=starts, lookback=lookback)
        parts = {h: [] for h in parameters["horizons"]}
        with torch.no_grad():
            for offset in range(0, len(windows), parameters["batch_size"]):
                values = torch.from_numpy(
                    windows[offset:offset + parameters["batch_size"]]
                ).float().to(device)
                centre = values.mean(dim=1, keepdim=True) if mode == "window_centered" else None
                predicted = t1.forecast(
                    model=model, x=values - centre if centre is not None else values,
                    prediction_horizon=parameters["prediction_horizon"],
                    rollout_horizon=parameters["max_horizon"], targets=None,
                )
                if centre is not None:
                    predicted = predicted + centre
                for horizon in parts:
                    parts[horizon].append(predicted[:, horizon - 1].cpu().numpy())
        for horizon, chunks in parts.items():
            target_rows = starts + lookback + horizon - 1
            payload, _ = t1.row_normalization(norm, source.segments, target_rows)
            prediction, target, _, _ = t1.calculate_errors_and_export_arrays(
                prediction_normalized=np.concatenate(chunks),
                target_raw_model_layout=source.data[target_rows],
                normalization=payload,
            )
            record(model_name, mode, horizon, extra,
                   _error_stats(prediction, target, _series_ids(source.segments, target_rows)))
    return rows


def launch(args):
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import pandas as pd
    import ray

    from training.common import transfer_1d as t1

    manifest = t1.load_transfer_manifest(ROOT / "training/configs/transfer_1d_manifest.json")
    tasks = {task.task_id: task for task in t1.transfer_tasks()}
    selected = args.task_ids or list(tasks)
    unknown = set(selected) - set(tasks)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    models = tuple(args.model_names or t1.TRANSFER_MODELS)
    ray.init(address="auto", ignore_reinit_error=True)
    # 0.25 GPU per (task, model) unit lets four run per GPU; SPREAD keeps all GPUs busy.
    gpu_unit = ray.remote(num_cpus=1, num_gpus=args.gpu_fraction, max_retries=1,
                          scheduling_strategy="SPREAD")(evaluate_unit)
    cpu_unit = ray.remote(num_cpus=1, num_gpus=0, max_retries=1,
                          scheduling_strategy="SPREAD")(evaluate_unit)
    futures = []
    for task_id in selected:
        entry = manifest["tasks"][task_id]["files"]
        futures.append(cpu_unit.remote(tasks[task_id], entry, None, manifest["config"],
                                       manifest["models"], args.stride, args.warmup_hours))
        for model_name in models:
            futures.append(gpu_unit.remote(tasks[task_id], entry, model_name, manifest["config"],
                                           manifest["models"], args.stride, args.warmup_hours))
    frame = pd.DataFrame([row for rows in ray.get(futures) for row in rows])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    print("CSV_BEGIN")
    print(buffer.getvalue())
    print("CSV_END")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="task_ids")
    parser.add_argument("--model", action="append", dest="model_names")
    parser.add_argument("--stride", type=int, default=10, help="window stride (diagnostic subsample)")
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    parser.add_argument("--warmup-hours", type=float, default=6.0,
                        help="initial target hours used to fit warmup_norm; excluded from scoring")
    launch(parser.parse_args(argv))


if __name__ == "__main__":
    main()
