"""Per-origin errors on the core receivers' October test week, for paired blocked-bootstrap intervals (evaluation only).

Why: the "ties" claims of the paper (a learned model against the tuned median or the lookback mean) need paired differences per forecast
origin. The tuned direct finals and the 1-minute checkpoints only stored aggregate and per-bin tables. This job re-runs saved checkpoints
(weights and source normalization frozen, the same forecaster as ``training.ray.transfer15.evaluate_frozen``) on each core receiver's
test week and writes, per (horizon, forecast-origin time), the mean absolute error over the 200 bins. The baselines are computed in the
same job family from the same CSV and the same origin set, so every model and baseline row can be matched on the origin timestamp.

Resolutions
* ``15min``: lookback 96 x 15 min, horizons 4/12/24/48, data ``data/powder15`` (``transfer15`` target files).
* ``1min``: lookback 60 min, horizons 1/15/60, the raw 1-minute CSVs of the test run, origins on every minute (stride 1) with a full
  60-step history and 60-step future inside one gap-free segment (``load_target(impute=False)``; the repository's own 1-minute
  evaluator fills gaps up to 9 minutes, so the origin sets can differ slightly: ``summary.json`` records the origin count).

Models: ``direct`` finals of ``tune_direct`` (one checkpoint per trained horizon; ``--models`` such as revinmlp patchtst dlinear nlinear
residuallinearar2d) and, at 1 minute, the rolled-out ``oct`` / ``band`` checkpoints (residuallinearar2d, linearar2d). Baselines
(``--baselines``): persistence, lookback mean (96 / 60 steps), the validation-tuned median and mean (window per horizon from
``evaluation/results/baselines_all_horizons.csv``, selection ``per_horizon``), and the per-bin Kalman filter (EWMA with the fitted
per-bin gains of ``median_study.train_gains``).

Output: ``<prefix>/<res>/<band>/<receiver>/<kind>/<model>/<h or roll>/seed-N/{per_origin.csv, summary.json}`` and
``<prefix>/<res>/<band>/<receiver>/baseline/{per_origin.csv, summary.json}``.

    python -m training.ray.origin_errors --res 15min --baselines --models revinmlp patchtst --skip-done
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
HORIZONS = {"15min": (4, 12, 24, 48), "1min": (1, 15, 60)}
LOOKBACK = {"15min": 96, "1min": 60}
STEP = {"15min": pd.Timedelta(minutes=15), "1min": pd.Timedelta(minutes=1)}
# window K per horizon of the validation-tuned median / mean (evaluation/results/baselines_all_horizons.csv, selection per_horizon)
TUNED_K = {"15min|2400_2600|mean": {1: 3, 4: 6, 12: 8, 24: 96, 48: 96}, "15min|2400_2600|median": {1: 3, 4: 3, 12: 6, 24: 96, 48: 96},
           "15min|600_800|mean": {1: 3, 4: 4, 12: 3, 24: 96, 48: 96}, "15min|600_800|median": {1: 3, 4: 3, 12: 2, 24: 96, 48: 96},
           "1min|2400_2600|mean": {1: 15, 15: 30, 60: 30}, "1min|2400_2600|median": {1: 20, 15: 20, 60: 20},
           "1min|600_800|mean": {1: 15, 15: 30, 60: 30}, "1min|600_800|median": {1: 20, 15: 30, 60: 30}}


def source_file(root: str, res: str, band: str, receiver: str) -> Path:
    from training.ray import transfer15 as t15

    if res == "15min":
        return t15.target_file(root, band, receiver)
    return Path(root) / "data/powder" / t15.RECEIVERS[receiver] / "20261025T1355Z" / band / "power_1mhz_avg_per_minute.csv"


def load_source(root: str, res: str, band: str, receiver: str):
    from training.common import transfer_2d
    from training.ray import transfer15 as t15
    from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

    if res == "15min":
        ensure_powder15(band, root, SPATIAL_NODES)
    task = t15.target_task(band, receiver)
    source = transfer_2d.load_target(task, [source_file(root, res, band, receiver)], impute=False, max_missing_gap=0)
    return task, source


def origin_frame(source, rows: np.ndarray, horizon: int, res: str, mae: np.ndarray, **extra) -> pd.DataFrame:
    times = pd.DatetimeIndex(source.timestamps[rows] - horizon * STEP[res]).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame({**extra, "time": np.asarray(times), "horizon": int(horizon), "mae_db": mae.astype(np.float64)})


# --------------------------------------------------------------------------------------------------------- models
def model_origin_errors(task, source, res: str, name: str, checkpoint_path: Path, config: dict[str, Any], seed: int) -> pd.DataFrame:
    import torch

    from training.common import transfer_2d
    from training.common.evaluation_integrated import calculate_errors_and_export_arrays, evaluation_parameters
    from training.common.forecasting import forecast
    from training.common.model_factory import build_model
    from training.common.windowing import make_window_batch_array

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization")
    if normalization is None:
        raise ValueError("checkpoint lacks normalization")
    parameters = evaluation_parameters(config, name)
    if parameters["lookback"] != LOOKBACK[res]:
        raise ValueError(f"lookback {parameters['lookback']} is not the {res} lookback {LOOKBACK[res]}")
    normalized = transfer_2d.apply_source_normalization(source.data, normalization)
    model = build_model(name, config, normalized)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=parameters["lookback"],
                                              max_horizon=parameters["max_horizon"], stride=1,
                                              target_start=task.target_start, target_end=task.target_end)
    if not len(starts):
        raise ValueError("no origins")
    windows = make_window_batch_array(full_x=normalized, start_rows=starts, lookback=parameters["lookback"])
    parts = {h: [] for h in parameters["horizons"]}
    with torch.no_grad():
        for offset in range(0, len(windows), parameters["batch_size"]):
            values = torch.from_numpy(windows[offset:offset + parameters["batch_size"]]).float().to(device)
            predicted = forecast(model=model, x=values, prediction_horizon=parameters["prediction_horizon"],
                                 rollout_horizon=parameters["max_horizon"], targets=None)
            for h in parts:
                parts[h].append(predicted[:, h - 1].cpu().numpy())
    frames = []
    for h, chunk in parts.items():
        rows = starts + parameters["lookback"] + h - 1
        prediction, target, _, _ = calculate_errors_and_export_arrays(
            prediction_normalized=np.concatenate(chunk), target_raw_model_layout=source.data[rows], normalization=dict(normalization))
        prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError(f"unexpected forecast shape {prediction.shape} vs {target.shape}")
        frames.append(origin_frame(source, rows, h, res, np.abs(prediction - target).mean(axis=1)))
    return pd.concat(frames, ignore_index=True)


def summarize(frame: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    return {**extra, "n_rows": int(len(frame)),
            "by_horizon": {str(h): {"n_origins": int(len(g)), "mae_db": float(g.mae_db.mean())} for h, g in frame.groupby("horizon")}}


def model_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry, receiver, band, res = spec["entry"], spec["receiver"], spec["band"], spec["res"]
    from training.ray.inference_benchmark import build_config
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    horizon = f"h{entry['horizon']}" if entry.get("source") == "direct" else "roll"
    base = f"{args['prefix'].strip('/')}/{res}/{band}/{receiver}/{entry['source']}/{entry['model']}/{horizon}/seed-{entry['seed']}"
    record = {"kind": "origin_errors_model", "res": res, "band": band, "receiver": receiver, "source": entry["source"], "model": entry["model"],
              "trained_horizon": entry.get("horizon"), "seed": int(entry["seed"]), "checkpoint_key": entry["pt_key"]}
    started = time.time()
    try:
        run_args = {"repo_root": root, "test_start": "2026-10-27T20:00:00Z", "test_end": "2026-11-03T20:00:00Z", "device": None,
                    "strides": {k: 1 for k in ("oct", "band", "p15", "p15tuned", "direct")}}
        with tempfile.TemporaryDirectory(prefix="origin-errors-") as scratch_dir:
            config, checkpoint = build_config(entry, run_args, client, bucket, Path(scratch_dir))
            name = entry.get("pipeline", entry["model"])
            config[name].setdefault("train", {})["test_stride"] = 1
            task, source = load_source(root, res, band, receiver)
            frame = model_origin_errors(task, source, res, name, checkpoint, config, int(entry["seed"]))
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        client.put_object(Bucket=bucket, Key=f"{base}/per_origin.csv", Body=buffer.getvalue().encode())
        record.update(summarize(frame, {}), ok=True, code_revision=args["code_revision"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


# ------------------------------------------------------------------------------------------------------- baselines
def rolled_forecast(x, statistic: str, k: int, steps: int, gain=None):
    """Autoregressive roll-out of a one-step window statistic (the repository's ``lookbackmean2d`` and ``forecast``): each step appends
    its own prediction to the window. Returns {step: (N, F) forecast} for steps 1..``steps``. ``statistic`` median / mean over the newest
    ``k`` steps, or 'ewma' (per-bin gain over the whole window of length ``k``). The median is the lower median (torch.median)."""
    import torch

    buffer = x[:, -k:].clone()
    if statistic == "ewma":
        age = torch.arange(k - 1, -1, -1, dtype=x.dtype).view(1, -1, 1)
        g = torch.as_tensor(np.asarray(gain, dtype=np.float32)).view(1, 1, -1)
        weights = g * (1.0 - g) ** age
        weights = weights / weights.sum(dim=1, keepdim=True)
    out = {}
    for step in range(1, steps + 1):
        if statistic == "median":
            pred = buffer.median(dim=1).values
        elif statistic == "mean":
            pred = buffer.mean(dim=1)
        else:
            pred = (buffer * weights).sum(dim=1)
        out[step] = pred.numpy()
        buffer = torch.cat([buffer[:, 1:], pred.unsqueeze(1)], dim=1)
    return out


def baseline_forecasts(data: np.ndarray, starts: np.ndarray, lookback: int, band: str, res: str, horizons, gains) -> dict[str, dict[int, np.ndarray]]:
    """{statistic: {horizon: (N, F) forecast in dB}} from the window ending at each origin (statistics rolled out as in the model tables)."""
    import torch

    windows = np.stack([data[s:s + lookback] for s in starts]).astype(np.float32)           # (N, T, F)
    x = torch.from_numpy(windows)
    top = max(horizons)
    out: dict[str, dict[int, np.ndarray]] = {"persistence": {h: windows[:, -1] for h in horizons},
                                              "lookbackmean": {h: windows.mean(axis=1) for h in horizons}}   # a fixed point of the roll-out
    for stat in ("median", "mean"):
        by_k: dict[int, dict[int, np.ndarray]] = {}
        for h in horizons:
            k = TUNED_K[f"{res}|{band}|{stat}"][h]
            if k not in by_k:
                by_k[k] = rolled_forecast(x, stat, k, top)
        out[f"tuned_{stat}"] = {h: by_k[TUNED_K[f"{res}|{band}|{stat}"][h]][h] for h in horizons}
    out["kalman"] = rolled_forecast(x, "ewma", lookback, top, gain=gains)
    return out


def baseline_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    receiver, band, res = spec["receiver"], spec["band"], spec["res"]
    from training.common import transfer_2d
    from training.ray.median_study import train_gains
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    base = f"{args['prefix'].strip('/')}/{res}/{band}/{receiver}/baseline"
    record = {"kind": "origin_errors_baseline", "res": res, "band": band, "receiver": receiver}
    started = time.time()
    try:
        task, source = load_source(root, res, band, receiver)
        lookback, horizons = LOOKBACK[res], HORIZONS[res]
        starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=lookback, max_horizon=max(horizons), stride=1,
                                                  target_start=task.target_start, target_end=task.target_end)
        if not len(starts):
            raise ValueError("no origins")
        data = np.asarray(source.data, dtype=np.float32)
        gains = train_gains(res, band, root)
        frames = []
        forecasts = baseline_forecasts(data, starts, lookback, band, res, horizons, gains)
        for h in horizons:
            rows = starts + lookback + h - 1
            for stat, by_h in forecasts.items():
                frames.append(origin_frame(source, rows, h, res, np.abs(by_h[h].astype(np.float64) - data[rows]).mean(axis=1), stat=stat))
        frame = pd.concat(frames, ignore_index=True)
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        client.put_object(Bucket=bucket, Key=f"{base}/per_origin.csv", Body=buffer.getvalue().encode())
        record.update(ok=True, n_origins=int(len(starts)), code_revision=args["code_revision"],
                      by_stat={s: {str(h): float(g2.mae_db.mean()) for h, g2 in g.groupby("horizon")} for s, g in frame.groupby("stat")})
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


# ----------------------------------------------------------------------------------------------------------- driver
def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", choices=("15min", "1min"), required=True)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=["600_800", "2400_2600"])
    p.add_argument("--receivers", nargs="+", default=list(CORE), choices=list(CORE))
    p.add_argument("--models", nargs="+", default=[], help="model names (direct finals; at 1 minute also the rolled-out oct/band checkpoints)")
    p.add_argument("--sources", nargs="+", default=["direct"], choices=["direct", "oct", "band"])
    p.add_argument("--horizons", type=int, nargs="+", help="trained horizons of the direct finals (default: all of the resolution)")
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--max-parallel", type=int, default=32)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="origin-errors-oct2026")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = "unknown"
    try:
        revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass

    def done(key: str) -> bool:
        try:
            return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
        except Exception:
            return False

    jobs: list[tuple[str, dict[str, Any]]] = []
    if args.baselines:
        for band in args.bands:
            for receiver in args.receivers:
                spec = {"res": args.res, "band": band, "receiver": receiver}
                if not (args.skip_done and done(f"{args.prefix}/{args.res}/{band}/{receiver}/baseline/summary.json")):
                    jobs.append(("baseline", spec))
    if args.models:
        horizons = set(args.horizons or HORIZONS[args.res])
        for entry in discover_sources(client, bucket, args.sources, args.models):
            if entry["band"] not in args.bands or int(entry["seed"]) not in args.seeds or entry["rep"] != "2d":
                continue
            if entry["source"] == "direct" and (entry["resolution"] != args.res or int(entry["horizon"]) not in horizons):
                continue
            if entry["source"] != "direct" and args.res != "1min":
                continue                                                    # 15-minute rolled-out models: transfer15-oct2026 already has per_origin.csv
            for receiver in args.receivers:
                spec = {"res": args.res, "band": entry["band"], "receiver": receiver, "entry": entry}
                tag = f"h{entry['horizon']}" if entry["source"] == "direct" else "roll"
                key = f"{args.prefix}/{args.res}/{entry['band']}/{receiver}/{entry['source']}/{entry['model']}/{tag}/seed-{entry['seed']}/summary.json"
                if not (args.skip_done and done(key)):
                    jobs.append(("model", spec))
    print(f"[origin-errors] {len(jobs)} jobs ({sum(k == 'baseline' for k, _ in jobs)} baseline), revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision}
    ray.init(address=args.address, ignore_reinit_error=True)
    options = dict(num_cpus=2, num_gpus=0, memory=int(args.memory_gb * 2 ** 30), max_retries=1)
    remote = {"model": ray.remote(**options)(model_job), "baseline": ray.remote(**options)(baseline_job)}
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            kind, spec = queue.pop(0)
            pending.append(remote[kind].remote(spec, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[origin-errors] {row.get('kind')} {row.get('res')} {row.get('band')} {row.get('receiver')} {row.get('model', '')} "
              f"h{row.get('trained_horizon')} seed{row.get('seed')} ok={row.get('ok')} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[origin-errors] done: {ok} ok of {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
