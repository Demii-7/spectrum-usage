"""Level drift or local level variation? Per-origin diagnostics for the 1-minute models with a missing level pathway (task J; evaluation only).

Question: the plain 2D LSTM and LSTMAttn (low offset-response gain: they do not carry a level change through to the forecast) are 13-16% worse than the tuned
lookback median at 1 minute. Is that because the level varies locally (then the gap exists in the training range too), or because the level drifts away from
what the models saw in training (then it appears or grows on the test week)?

Ranges (1-minute raw CSVs of the core receivers; the models were trained on ``train``, ``validation`` selected their configuration, ``test`` is the benchmark week):
    train       2026-10-15 14:34 to 10-22 13:13  (run 20261015T1434Z)
    validation  2026-10-25 13:55 to 10-27 13:55  (run 20261025T1355Z)
    test        2026-10-27 20:00 to 11-03 20:00  (run 20261025T1355Z)
    week2       2026-11-03 20:00 to 11-09 02:00  (same run; unofficial: the collection is not declared final)

Two job kinds (Ray, CPU):
* ``level``: per (band, receiver, range), every origin (stride 1): level distance d = |mean of the window's 60-minute lookback - the bin's training-range mean|
  and the same distance divided by the bin's training standard deviation; quantiles over all (window, bin) values, and per-day quantiles (median, 90th) for the
  test and week2 ranges.
* ``model``: per (band, receiver, checkpoint), every ``stride``-th origin of each range: per-origin MAE over the 200 bins of the checkpoint, of the validation-tuned
  lookback median, tuned mean and lookback mean (identical origins), and the sufficient statistics of the per-(window, bin) regression of the model's excess
  error over the median, ``excess = |model error| - |median error|``, on the level distance ``d`` (n, sum d, sum excess, sum d*excess, sum d^2, sum excess^2).

Outputs under ``level-drift-oct2026/``: ``level/<band>/<receiver>/summary.json`` and ``model/<band>/<receiver>/<source tag>/<model>/seed-N/per_origin.csv`` (+ summary).

    python -m training.ray.level_diagnostics_origins level --skip-done
    python -m training.ray.level_diagnostics_origins model --models vanillalstm lstmattn residualvanillalstm --stride 10 --skip-done
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE = ("humanities", "madsen", "moran", "sagepoint")
BANDS = ("600_800", "2400_2600")
LOOKBACK, HORIZONS = 60, (1, 15, 60)
RUNS = {"train": "20261015T1434Z", "validation": "20261025T1355Z", "test": "20261025T1355Z", "week2": "20261025T1355Z"}
BOUNDS = {"train": ("2026-10-15T14:34:00Z", "2026-10-22T13:14:00Z"), "validation": ("2026-10-25T13:55:00Z", "2026-10-27T13:55:00Z"),
          "test": ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z"), "week2": ("2026-11-03T20:00:00Z", "2026-11-09T02:00:00Z")}
DAY_ORIGIN = {"test": "2026-10-27T20:00:00Z", "week2": "2026-11-03T20:00:00Z"}     # day index 0 starts here (the test week = days 0-6, week2 = days 7-12)
TRAIN_END = "2026-10-22T13:13:00Z"
QUANTILES = (0.1, 0.5, 0.9, 0.99)
PREFIX = "level-drift-oct2026"


# ----------------------------------------------------------------------------------------------------- pure helpers
def window_means(data: np.ndarray, starts: np.ndarray, lookback: int) -> np.ndarray:
    """Mean over each window's ``lookback`` rows, per column: (N, F). ``data`` (T, F), ``starts`` the first row of each window."""
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(data, dtype=np.float64), lookback, axis=0)    # (T - lookback + 1, F, lookback)
    return view.mean(axis=-1)[np.asarray(starts)]


def level_distance(means: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray | None = None) -> np.ndarray:
    """|window mean - the bin's training mean| (dB), divided by the bin's training standard deviation when it is given."""
    d = np.abs(means - train_mean[None, :])
    return d if train_std is None else d / np.maximum(train_std[None, :], 1e-6)


def quantile_dict(values: np.ndarray, qs=QUANTILES) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in qs} | {"mean": float(v.mean()), "n": int(v.size)}


def day_index(times, origin: str) -> np.ndarray:
    """Whole days since ``origin`` (floor), for an array of timestamps."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return np.floor((t - pd.Timestamp(origin)) / pd.Timedelta(days=1)).astype(int).to_numpy()


def regression_stats(d: np.ndarray, excess: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row sufficient statistics (over the bins of each row) of the regression of ``excess`` on ``d``; both (N, F)."""
    return {"n": np.full(d.shape[0], d.shape[1], dtype=np.float64), "sx": d.sum(1), "sy": excess.sum(1), "sxy": (d * excess).sum(1),
            "sxx": (d * d).sum(1), "syy": (excess * excess).sum(1)}


def ols_from_stats(n: float, sx: float, sy: float, sxy: float, sxx: float, syy: float) -> dict[str, float]:
    """Slope, intercept and R^2 of y on x from sums (n, sum x, sum y, sum xy, sum x^2, sum y^2)."""
    vx = n * sxx - sx * sx
    cov = n * sxy - sx * sy
    vy = n * syy - sy * sy
    if n < 3 or vx <= 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": float(n)}
    slope = cov / vx
    return {"slope": float(slope), "intercept": float((sy - slope * sx) / n), "r2": float(cov * cov / (vx * vy)) if vy > 0 else float("nan"), "n": float(n)}


# --------------------------------------------------------------------------------------------------------- data
def load_range(root: str, band: str, receiver: str, name: str):
    """(task, source) of one receiver and band over one range, rows with all bins finite (no imputation), lookback history included."""
    from training.common import transfer_2d
    from training.common.transfer_2d import TransferTask
    from training.ray import transfer15 as t15

    start, end = (pd.Timestamp(t) for t in BOUNDS[name])
    low, high = t15.BAND_LIMITS[band]
    task = TransferTask(task_id=f"level-{name}-{band}-{receiver}", evaluation="level diagnostics", platform="powder",
                        raw_start=start - pd.Timedelta(hours=2), raw_end=end, target_start=start, target_end=end,
                        frequency_start_mhz=low, frequency_end_mhz=high, sites=(receiver,))
    path = Path(root) / "data/powder" / t15.RECEIVERS[receiver] / RUNS[name] / band / "power_1mhz_avg_per_minute.csv"
    return task, transfer_2d.load_target(task, [path], impute=False, max_missing_gap=0)


def origins(task, source, stride: int) -> np.ndarray:
    from training.common import transfer_2d

    return transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=max(HORIZONS), stride=stride,
                                            target_start=task.target_start, target_end=task.target_end)


def training_stats(root: str, band: str, receiver: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean and standard deviation (dB) over the receiver's training range."""
    task, source = load_range(root, band, receiver, "train")
    data = np.asarray(source.data, dtype=np.float64)
    keep = (source.timestamps >= task.target_start) & (source.timestamps < task.target_end)
    return data[np.asarray(keep)].mean(axis=0), data[np.asarray(keep)].std(axis=0)


# ------------------------------------------------------------------------------------------------------ level job
def level_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.storage import spectrum_s3_client

    band, receiver = spec["band"], spec["receiver"]
    client, bucket = spectrum_s3_client(), args["bucket"]
    key = f"{args['prefix']}/level/{band}/{receiver}/summary.json"
    record: dict[str, Any] = {"kind": "level", "band": band, "receiver": receiver}
    started = time.time()
    try:
        mu, sd = training_stats(root, band, receiver)
        out = {}
        for name in BOUNDS:
            task, source = load_range(root, band, receiver, name)
            starts = origins(task, source, 1)
            d = level_distance(window_means(source.data, starts, LOOKBACK), mu)
            dn = d / np.maximum(sd[None, :], 1e-6)
            entry: dict[str, Any] = {"n_origins": int(len(starts)), "d_db": quantile_dict(d), "d_sd": quantile_dict(dn)}
            if name in DAY_ORIGIN:
                times = source.timestamps[starts + LOOKBACK - 1]
                days = day_index(times, DAY_ORIGIN["test"])
                entry["per_day"] = {int(day): {"n_origins": int((days == day).sum()), "d_db": quantile_dict(d[days == day]),
                                               "d_sd": quantile_dict(dn[days == day])} for day in np.unique(days)}
            out[name] = entry
        record.update(ranges=out, ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
    return record


# ------------------------------------------------------------------------------------------------------ model job
def model_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry, band, receiver, stride = spec["entry"], spec["band"], spec["receiver"], int(args["stride"])
    from training.ray.inference_benchmark import build_config
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    tag = entry["prefix"].replace("/", "_")
    base = f"{args['prefix']}/model/{band}/{receiver}/{tag}/{entry['model']}/seed-{entry['seed']}"
    record: dict[str, Any] = {"kind": "level_model", "band": band, "receiver": receiver, "source_tag": tag, "model": entry["model"], "seed": int(entry["seed"]),
                              "stride": stride, "checkpoint_key": entry["pt_key"]}
    started = time.time()
    try:
        import torch

        from training.common import transfer_2d
        from training.common.evaluation_integrated import calculate_errors_and_export_arrays, evaluation_parameters
        from training.common.forecasting import forecast
        from training.common.model_factory import build_model
        from training.common.windowing import make_window_batch_array
        from training.ray.origin_errors import TUNED_K, rolled_forecast

        run_args = {"repo_root": root, "test_start": "2026-10-27T20:00:00Z", "test_end": "2026-11-03T20:00:00Z", "device": None,
                    "strides": {k: 1 for k in ("oct", "band", "p15", "p15tuned", "direct")}}
        mu, _ = training_stats(root, band, receiver)
        frames = []
        with tempfile.TemporaryDirectory(prefix="level-diag-") as scratch:
            config, checkpoint_path = build_config(entry, run_args, client, bucket, Path(scratch))
            name = entry.get("pipeline", entry["model"])
            config[name].setdefault("train", {})["test_stride"] = 1
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            normalization = checkpoint["normalization"]
            parameters = evaluation_parameters(config, name)
            if parameters["lookback"] != LOOKBACK:
                raise ValueError(f"lookback {parameters['lookback']}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = None
            for rng in spec["ranges"]:
                task, source = load_range(root, band, receiver, rng)
                starts = origins(task, source, stride)
                if not len(starts):
                    continue
                normalized = transfer_2d.apply_source_normalization(source.data, normalization)
                if model is None:
                    model = build_model(name, config, normalized)
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    model.to(device).eval()
                    torch.manual_seed(int(entry["seed"]))
                windows = make_window_batch_array(full_x=normalized, start_rows=starts, lookback=LOOKBACK)
                parts = {h: [] for h in parameters["horizons"]}
                with torch.no_grad():
                    for off in range(0, len(windows), parameters["batch_size"]):
                        values = torch.from_numpy(windows[off:off + parameters["batch_size"]]).float().to(device)
                        predicted = forecast(model=model, x=values, prediction_horizon=parameters["prediction_horizon"],
                                             rollout_horizon=parameters["max_horizon"], targets=None)
                        for h in parts:
                            parts[h].append(predicted[:, h - 1].cpu().numpy())
                raw = np.asarray(source.data, dtype=np.float32)
                raw_windows = np.stack([raw[s:s + LOOKBACK] for s in starts])
                x = torch.from_numpy(raw_windows)
                med = {k: rolled_forecast(x, "median", k, max(HORIZONS)) for k in sorted(set(TUNED_K[f"1min|{band}|median"].values()))}
                mea = {k: rolled_forecast(x, "mean", k, max(HORIZONS)) for k in sorted(set(TUNED_K[f"1min|{band}|mean"].values()))}
                d = level_distance(raw_windows.astype(np.float64).mean(axis=1), mu)
                times = pd.DatetimeIndex(source.timestamps[starts + LOOKBACK - 1]).strftime("%Y-%m-%dT%H:%M:%SZ")
                for h in HORIZONS:
                    rows = starts + LOOKBACK + h - 1
                    prediction, target, _, _ = calculate_errors_and_export_arrays(prediction_normalized=np.concatenate(parts[h]),
                                                                                  target_raw_model_layout=source.data[rows], normalization=dict(normalization))
                    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
                    e_model = np.abs(prediction - target)
                    e_med = np.abs(med[TUNED_K[f"1min|{band}|median"][h]][h].astype(np.float64) - target)
                    e_mean = np.abs(mea[TUNED_K[f"1min|{band}|mean"][h]][h].astype(np.float64) - target)
                    e_lb = np.abs(raw_windows.astype(np.float64).mean(axis=1) - target)
                    stats = regression_stats(d, e_model - e_med)
                    frame = pd.DataFrame({"range": rng, "time": np.asarray(times), "horizon": h, "mae_model": e_model.mean(1), "mae_median": e_med.mean(1),
                                          "mae_mean": e_mean.mean(1), "mae_lbmean": e_lb.mean(1), "d_mean": d.mean(1), **stats})
                    frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        client.put_object(Bucket=bucket, Key=f"{base}/per_origin.csv", Body=buffer.getvalue().encode())
        record.update(ok=True, n_rows=int(len(frame)), ranges=list(spec["ranges"]), code_revision=args["code_revision"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


# ------------------------------------------------------------------------------------------------------- driver
def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("level", "model"))
    p.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    p.add_argument("--receivers", nargs="+", default=list(CORE), choices=list(CORE))
    p.add_argument("--models", nargs="+", default=["vanillalstm", "lstmattn", "residualvanillalstm"])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--ranges", nargs="+", default=["train", "validation", "test", "week2"], choices=list(BOUNDS))
    p.add_argument("--stride", type=int, default=10, help="every n-th origin of a range (model jobs); the level job uses every origin")
    p.add_argument("--memory-gb", type=float, default=6.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    try:
        revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        revision = "unknown"

    def done(key: str) -> bool:
        try:
            return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
        except Exception:
            return False

    jobs = []
    if args.kind == "level":
        for band in args.bands:
            for receiver in args.receivers:
                if not (args.skip_done and done(f"{args.prefix}/level/{band}/{receiver}/summary.json")):
                    jobs.append({"band": band, "receiver": receiver})
        fn = level_job
    else:
        for entry in discover_sources(client, bucket, ["oct", "band"], args.models):
            if entry["rep"] != "2d" or entry["band"] not in args.bands or int(entry["seed"]) not in args.seeds:
                continue
            for receiver in args.receivers:
                tag = entry["prefix"].replace("/", "_")
                key = f"{args.prefix}/model/{entry['band']}/{receiver}/{tag}/{entry['model']}/seed-{entry['seed']}/summary.json"
                if not (args.skip_done and done(key)):
                    jobs.append({"band": entry["band"], "receiver": receiver, "entry": entry, "ranges": args.ranges})
        fn = model_job
    print(f"[level-drift] {args.kind}: {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix.strip("/"), "stride": args.stride, "code_revision": revision}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(fn)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.remote(queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[level-drift] {row.get('kind')} {row.get('band')} {row.get('receiver')} {row.get('source_tag', '')} {row.get('model', '')} seed{row.get('seed')} "
              f"ok={row.get('ok')} {row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[level-drift] done: {ok} ok of {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
