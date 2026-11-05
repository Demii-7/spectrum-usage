"""Train and evaluate our models and baselines on the ElectroSense-Madrid datasets (SpectrumLLM paper).

Data: https://github.com/LIUC-000/SP/tree/main/ElectroSense-Madrid_Spain (spec4/spec25/spec151, one sensor,
500-800 MHz, 15-minute samples). Protocol follows the Time-Series-Library ``Dataset_Custom`` that the
SpectrumLLM baselines use: 70/10/20 chronological train/validation/test, normalization fit on train only, input
192 steps, horizons 12/24/48/96/192, and validation/test windows whose *input* starts 192 rows before the
partition boundary (history from the previous partition; targets stay inside the partition).

Our pipeline needs timestamps, so each step is stamped as one synthetic minute (row index); the real 15-minute
cadence is irrelevant to the models, which only see the sequence. No time features are used.

Metrics are reported in the data's own units ("raw") and in train-standardized units (the Time-Series-Library
default), per horizon and averaged over horizons.

    python -m training.ray.electrosense baselines
    python -m training.ray.electrosense ours --mode rollout --seeds 40 41 42 43 44
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DATASETS = {"spec4": "spec4_data.csv", "spec25": "spec25_data.csv", "spec151": "spec151_data.csv"}
RAW_PREFIX = "datasets/electrosense-madrid"
SEQ_LEN = 192
HORIZONS = (12, 24, 48, 96, 192)
SEASONAL_PERIOD = 96  # one day of 15-minute steps
BASE_TIME = pd.Timestamp("2021-01-01T00:00:00Z")
FREQ_MIN, FREQ_MAX = 500.0, 800.0


# ---------------------------------------------------------------------------------------------- data
def split_borders(n: int, seq_len: int = SEQ_LEN) -> dict[str, tuple[int, int]]:
    """Row ranges [start, end) per partition, exactly as Time-Series-Library Dataset_Custom (70/10/20)."""
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    return {
        "train": (0, num_train),
        "validation": (num_train - seq_len, num_train + num_vali),
        "test": (n - num_test - seq_len, n),
    }


def convert_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Raw ElectroSense frame (date + frequency columns) -> pipeline CSV and split/normalization stats."""
    values = raw.drop(columns=["date"]).astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("ElectroSense data contains missing values")
    n = len(values)
    borders = split_borders(n)
    train = values.iloc[borders["train"][0]:borders["train"][1]]
    stamps = [(BASE_TIME + pd.Timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]
    series = values.copy()
    series.insert(0, "timestamp_utc", stamps)
    series.columns = ["timestamp_utc"] + [str(float(c)) for c in values.columns]
    stats = {
        "n_rows": n,
        "borders": {k: list(v) for k, v in borders.items()},
        "frequencies": [float(c) for c in values.columns],
        "train_mean": [float(x) for x in train.mean(axis=0)],
        "train_std": [float(x) for x in train.std(axis=0, ddof=0)],  # StandardScaler is population std
    }
    return series, stats


def ensure_dataset(name: str, repo_root: str, bucket: str, client) -> tuple[Path, dict[str, Any]]:
    """Build (once per worker) data/electrosense/<name>/series.csv + stats.json from the raw file in the bucket."""
    out = Path(repo_root) / "data" / "electrosense" / name
    series_path, stats_path = out / "series.csv", out / "stats.json"
    if not (series_path.exists() and stats_path.exists()):
        out.mkdir(parents=True, exist_ok=True)
        body = client.get_object(Bucket=bucket, Key=f"{RAW_PREFIX}/{DATASETS[name]}")["Body"].read()
        series, stats = convert_raw(pd.read_csv(io.BytesIO(body)))
        for target, write in ((series_path, lambda p: series.to_csv(p, index=False)),
                              (stats_path, lambda p: Path(p).write_text(json.dumps(stats)))):
            with tempfile.NamedTemporaryFile("w", dir=out, delete=False, suffix=".tmp") as tmp:
                pass
            write(tmp.name)
            os.replace(tmp.name, target)
    return series_path, json.loads(stats_path.read_text())


def window_ranges(stats: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Inclusive synthetic-timestamp ranges per partition (row i is minute i)."""
    def stamp(i: int) -> str:
        return (BASE_TIME + pd.Timedelta(minutes=int(i))).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {name: {"start": stamp(a), "end": stamp(b - 1)} for name, (a, b) in stats["borders"].items()}


# ------------------------------------------------------------------------------------------- baselines
def _test_windows(values: np.ndarray, start: int, seq_len: int, horizon: int):
    """Yield (context, future) for every window of the partition starting at row ``start`` (history included)."""
    for i in range(len(values) - start - seq_len - horizon + 1):
        base = start + i
        yield values[base:base + seq_len], values[base + seq_len:base + seq_len + horizon]


def predict_baseline(kind: str, context: np.ndarray, horizon: int, train_mean: np.ndarray) -> np.ndarray:
    """(horizon, channels) forecast from a (seq_len, channels) context. No training."""
    if kind == "train_mean":
        return np.repeat(train_mean[None, :], horizon, axis=0)
    if kind == "last_value":
        return np.repeat(context[-1:], horizon, axis=0)
    if kind == "window_mean":
        return np.repeat(context.mean(axis=0, keepdims=True), horizon, axis=0)
    steps = np.arange(1, horizon + 1)
    length = len(context)
    if kind == "seasonal_naive":  # same time of day, nearest day inside the window
        lag = SEASONAL_PERIOD * np.ceil(steps / SEASONAL_PERIOD).astype(int)
        return context[length + steps - 1 - lag]
    if kind == "seasonal_mean":  # mean over every earlier day that falls inside the window
        out = np.zeros((horizon, context.shape[1]))
        for j, step in enumerate(steps):
            days = [k for k in range(1, 100) if SEASONAL_PERIOD * k >= step and length + step - 1 - SEASONAL_PERIOD * k >= 0]
            out[j] = np.mean([context[length + step - 1 - SEASONAL_PERIOD * k] for k in days], axis=0)
        return out
    raise ValueError(f"Unknown baseline {kind!r}")


def baseline_metrics(values: np.ndarray, stats: dict[str, Any], kind: str, horizons=HORIZONS,
                     seq_len: int = SEQ_LEN, split: str = "test") -> dict[str, float]:
    """Raw and standardized MSE/MAE per horizon for one trivial predictor on the given partition's windows."""
    start = stats["borders"][split][0]
    end = stats["borders"][split][1]
    part = values[:end]
    mean = np.asarray(stats["train_mean"])
    std = np.asarray(stats["train_std"])
    out: dict[str, float] = {}
    for horizon in horizons:
        raw_se, raw_ae, std_se, std_ae, count = 0.0, 0.0, 0.0, 0.0, 0
        for context, future in _test_windows(part, start, seq_len, horizon):
            error = predict_baseline(kind, context, horizon, mean) - future
            raw_se += float(np.mean(error ** 2)); raw_ae += float(np.mean(np.abs(error)))
            scaled = error / std
            std_se += float(np.mean(scaled ** 2)); std_ae += float(np.mean(np.abs(scaled)))
            count += 1
        out[f"mse_raw_h{horizon}"], out[f"mae_raw_h{horizon}"] = raw_se / count, raw_ae / count
        out[f"mse_std_h{horizon}"], out[f"mae_std_h{horizon}"] = std_se / count, std_ae / count
        out["n_windows_h%d" % horizon] = count
    for metric in ("mse_raw", "mae_raw", "mse_std", "mae_std"):
        out[f"{metric}_mean"] = float(np.mean([out[f"{metric}_h{h}"] for h in horizons]))
    return out


def baseline_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one (dataset, baseline) cell."""
    import sys

    sys.path.insert(0, args["repo_root"])
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    result = {"dataset": job["dataset"], "model": job["kind"], "kind": "baseline"}
    try:
        series_path, stats = ensure_dataset(job["dataset"], args["repo_root"], args["bucket"], client)
        values = pd.read_csv(series_path).drop(columns=["timestamp_utc"]).to_numpy(dtype=np.float64)
        result.update(baseline_metrics(values, stats, job["kind"]))
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
    return result


# ----------------------------------------------------------------------------------------- our models
def standardized_from_per_frequency(per_freq: pd.DataFrame, stats: dict[str, Any], split_suffix: str = "test"):
    """Standardized MSE/MAE per horizon from per-bin rows (data units): mean over bins of mse/std^2 and mae/std."""
    std_by_freq = {round(f, 3): s for f, s in zip(stats["frequencies"], stats["train_std"])}
    rows = per_freq[per_freq["split"].astype(str).str.endswith(split_suffix)].copy()
    rows["std"] = rows["frequency_mhz"].round(3).map(std_by_freq)
    if rows["std"].isna().any():
        raise ValueError("per-bin rows contain frequencies without a training std")
    rows["mse_std"] = rows["mse_db2"] / rows["std"] ** 2
    rows["mae_std"] = rows["mae_db"] / rows["std"]
    out: dict[str, float] = {}
    for horizon, group in rows.groupby("horizon"):
        out[f"mse_std_h{int(horizon)}"] = float(group["mse_std"].mean())
        out[f"mae_std_h{int(horizon)}"] = float(group["mae_std"].mean())
        out[f"mse_raw_h{int(horizon)}"] = float(group["mse_db2"].mean())
        out[f"mae_raw_h{int(horizon)}"] = float(group["mae_db"].mean())
    for metric in ("mse_std", "mae_std", "mse_raw", "mae_raw"):
        values = [v for k, v in out.items() if k.startswith(metric + "_h")]
        out[f"{metric}_mean"] = float(np.mean(values))
    return out


def apply_dataset(config: dict[str, Any], series_path: Path, stats: dict[str, Any], name: str,
                  horizons: list[int]) -> dict[str, Any]:
    """Point a model config at an ElectroSense dataset with the 70/10/20 protocol (lookback 192)."""
    config = copy.deepcopy(config)
    path = str(series_path)
    config["data"]["files"] = [{"path": path, "partition": p} for p in ("train", "validation", "test")]
    config["data"]["chunks"] = [{"id": f"electrosense_{name}", "start_mhz": FREQ_MIN, "end_mhz": FREQ_MAX}]
    config["data"]["split"] = {"ranges": window_ranges(stats)}
    config["data"].pop("map", None)
    config["data"]["concat"] = "rows"
    config["windowing"] = {"lookback": SEQ_LEN, "horizons": sorted(int(h) for h in horizons)}
    config["preprocessing"] = {**config.get("preprocessing", {}), "normalize": True, "impute": False,
                               "max_missing_gap": 0}
    return config


def set_lookback_and_horizon(config: dict[str, Any], model: str, direct_horizon: int | None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config[model]["model"]["input_sequence_length"] = SEQ_LEN
    config[model]["model"]["prediction_horizon"] = int(direct_horizon) if direct_horizon else 1
    for key in ("train_stride", "val_stride", "test_stride"):
        config[model]["train"][key] = 1  # tiny dataset: use every window
    return config


def ours_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one of our models on one dataset/seed(/horizon) and score it."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.final_eval import build_eval_config, config_file_for
    from training.ray.fixed_train import REFERENCE_MODELS
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, rep, seed, name = job["model"], job["rep"], int(job["seed"]), job["dataset"]
    horizon = job.get("horizon")
    result = {"dataset": name, "model": model, "rep": rep, "seed": seed, "kind": "ours", "mode": args["mode"],
              "horizon": horizon or "", "hyperparameters": "inherited", "source_prefix": job.get("source_prefix", "")}
    started = time.time()
    try:
        series_path, stats = ensure_dataset(name, args["repo_root"], bucket, client)
        base = yaml.safe_load(Path(config_file_for(model, rep)).read_text())
        if model in REFERENCE_MODELS:
            config = copy.deepcopy(base)
            config["training"] = {**config.get("training", {}), "models": [model], "device": "cpu"}
        else:
            costs = json.loads(client.get_object(Bucket=bucket, Key=job["costs_key"])["Body"].read())
            config = build_eval_config(base, model, costs, test_start="x", test_end="y", device=args.get("device"))
        horizons = [horizon] if horizon else list(HORIZONS)
        config = apply_dataset(config, series_path, stats, name, horizons)
        config = set_lookback_and_horizon(config, model, horizon)
        config[model]["train"]["seed"] = seed
        if job.get("model_overrides"):  # an architecture variant (for example a seasonal baseline), same hyperparameters
            config[model]["model"].update(job["model_overrides"])
            result["variant"] = job.get("variant", "")
        with tempfile.TemporaryDirectory(prefix="electrosense-") as scratch:
            run_dir, eval_dir = Path(scratch) / "run", Path(scratch) / "eval"
            train_one_model(config, model, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, checkpoint, skip_plots=True)
            per_freq = pd.read_csv(eval_dir / "per_frequency_metrics.csv")
            result.update(standardized_from_per_frequency(per_freq, stats))
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-800:])
    result["wall_seconds"] = round(time.time() - started, 1)
    try:
        suffix = f"/h{horizon}" if horizon else ""
        suffix += f"/variant-{job['variant']}" if job.get("model_overrides") else ""
        key = f"{args['output_prefix'].strip('/')}/{args['mode']}/{name}/{rep}/{model}{suffix}/seed-{seed}.json"
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


# ------------------------------------------------------------------------------------------------ CLI
def ours_jobs(entries, datasets, seeds, mode, horizons):
    from training.ray.fixed_train import ONE_STEP_ONLY, REFERENCE_MODELS

    jobs = []
    for dataset in datasets:
        for entry in entries:
            model = entry["model"]
            is_ref = model in REFERENCE_MODELS
            use_seeds = [42] if is_ref else seeds
            if mode == "rollout":
                jobs += [{**entry, "dataset": dataset, "seed": s} for s in use_seeds]
            elif model not in ONE_STEP_ONLY:  # direct: one model per horizon
                jobs += [{**entry, "dataset": dataset, "seed": s, "horizon": h} for h in horizons for s in use_seeds]
    return jobs


def expand_variants(jobs: list[dict[str, Any]], variants: list[str] | None) -> list[dict[str, Any]]:
    """``NAME=JSON`` variants -> one job per (job, variant) carrying model-config overrides; no variants: unchanged."""
    if not variants:
        return jobs
    parsed = []
    for token in variants:
        name, _, text = token.partition("=")
        if not name or not text:
            raise ValueError(f"variant {token!r} must look like NAME={{json overrides}}")
        parsed.append((name, json.loads(text)))
    return [{**job, "variant": name, "model_overrides": overrides} for job in jobs for name, overrides in parsed]


def launch(args: argparse.Namespace) -> None:
    import ray

    from .fixed_train import discover_models
    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
                   "mode": getattr(args, "mode", "baselines"), "device": args.device}
    futures = []
    if args.command == "baselines":
        kinds = ("train_mean", "last_value", "window_mean", "seasonal_naive", "seasonal_mean")
        remote = ray.remote(num_cpus=2)(baseline_job)
        futures = [remote.remote({"dataset": d, "kind": k}, worker_args) for d in args.datasets for k in kinds]
        prefix = "baselines"
    else:
        entries = discover_models(client, bucket, args.prefixes, args.models)
        if args.include_reference:
            entries += [{"rep": rep, "model": m, "source_prefix": "", "costs_key": ""}
                        for m, rep in (("lookbackmean1d", "1d"), ("lookbackmean2d", "2d"))
                        if not args.models or m in args.models]
        entries = [e for e in entries if e["rep"] in ("1d", "2d")]
        jobs = expand_variants(ours_jobs(entries, args.datasets, args.seeds, args.mode, args.horizons), args.variant)
        print(f"[electrosense] {len(jobs)} runs ({args.mode})", flush=True)
        for job in jobs:
            spec = get_model_spec(job["model"])
            resources = dict(spec.resources.as_ray())
            gpu = args.gpu_fraction if resources.get("gpu", 0) > 0 else 0
            futures.append(ray.remote(num_cpus=resources["cpu"], num_gpus=gpu, max_retries=1)(ours_job)
                           .remote(job, worker_args))
        prefix = args.mode
    frame = pd.DataFrame(ray.get(futures))
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    key = f"{args.output_prefix.strip('/')}/summary_{prefix}_{stamp}.csv"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    failed = frame[~frame["ok"].astype(bool)] if "ok" in frame else frame
    print(f"[electrosense] wrote s3://{bucket}/{key}; {len(frame)} rows, {len(failed)} failed", flush=True)
    for _, row in failed.head(8).iterrows():
        print(f"[electrosense] FAILED {row.get('dataset')}/{row.get('model')}: {row.get('error')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baselines", "ours"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
        p.add_argument("--bucket", default=None)
        p.add_argument("--output-prefix", default="electrosense-results")
        p.add_argument("--address", default="auto")
        p.add_argument("--repo-root", default="/workspace/spectrum-usage")
        p.add_argument("--device", default=None)
    ours = sub.choices["ours"]
    ours.add_argument("--mode", choices=("rollout", "direct"), default="rollout")
    ours.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS), help="for --mode direct")
    ours.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    ours.add_argument("--models", nargs="+")
    ours.add_argument("--prefixes", nargs="+", default=["checkpoints-oct2026", "checkpoints-oct2026-r2"])
    ours.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    ours.add_argument("--variant", action="append", metavar="NAME=JSON",
                      help='Model-config overrides, e.g. seasonal=\'{"baseline":"seasonal","seasonal_period":96}\' '
                           "(repeatable; each variant is run in addition to nothing else, so include the plain run separately)")
    ours.add_argument("--gpu-fraction", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
