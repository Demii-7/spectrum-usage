"""POWDER at 15-minute resolution: train and evaluate our models and trivial baselines on block-averaged data.

Same protocol as the one-minute October 2026 runs (train run 20261015T1434Z, validation = first two days of run
20261025T1355Z, test = Oct 27 20:00Z to Nov 3 20:00Z, four core receivers, same timestamp ranges), with each
per-minute trace replaced by aligned 15-minute block means computed in LINEAR power (mW) and converted back to dB.
Blocks with fewer than 80% valid minutes are filled by linear interpolation in dB (gaps up to 16 blocks; the
imputed fraction is reported). Block averaging happens per file, i.e. within a partition, so nothing leaks across
the split. Lookback is 96 steps (24 h); horizons are 4, 12, 24, 48 steps (1, 3, 6, 12 h), limited by the 2-day
validation partition (192 blocks).

Metrics are MAE/RMSE/MSE in dB (the pipeline's own), per horizon and averaged. Hyperparameters are inherited from
the one-minute POWDER winners (no search); this is stated in every result row.

    python -m training.ray.powder15 baselines
    python -m training.ray.powder15 ours --mode rollout --seeds 40 41 42 43 44
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

RAW_ROOT = "data/powder"
OUT_ROOT = "data/powder15"
RAW_NAME = "power_1mhz_avg_per_minute.csv"
OUT_NAME = "power_15min.csv"
BLOCK_MINUTES = 15
MIN_VALID_FRACTION = 0.8
MAX_GAP_BLOCKS = 16
LOOKBACK = 96
HORIZONS = (4, 12, 24, 48)
BANDS = ("600_800", "2400_2600")
CORE_NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
TRAIN_RUN, TEST_RUN = "20261015T1434Z", "20261025T1355Z"


# ----------------------------------------------------------------------------------------------- data
def site_name(node: str) -> str:
    """Receiver name as the file-restriction helpers expect it: 'humanities-nuc1' -> 'humanities'."""
    return node.removesuffix("-nuc1")


def block_average_db(frame: pd.DataFrame, minutes: int = BLOCK_MINUTES) -> pd.DataFrame:
    """Per-minute dBm (UTC minute index) -> aligned block means in dB, averaged in linear power."""
    power = np.power(10.0, frame / 10.0)
    block = power.index.floor(f"{minutes}min")
    grouped = power.groupby(block)
    mean = grouped.mean().where(grouped.count() >= int(np.ceil(MIN_VALID_FRACTION * minutes)))
    full = pd.date_range(mean.index.min(), mean.index.max(), freq=f"{minutes}min")
    return 10.0 * np.log10(mean.reindex(full))


def fill_gaps(blocks: pd.DataFrame, max_gap: int = MAX_GAP_BLOCKS) -> tuple[pd.DataFrame, float]:
    """Linear interpolation in dB over short gaps; raises if anything non-finite is left. Returns (frame, imputed share)."""
    missing = float(blocks.isna().to_numpy().mean())
    filled = blocks.interpolate(method="linear", limit=max_gap, limit_area="inside")
    if not np.isfinite(filled.to_numpy()).all():
        raise ValueError(f"gaps longer than {max_gap} blocks or leading/trailing NaN remain")
    return filled, missing


def convert_trace(raw_csv: Path, out_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(raw_csv)
    stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
    frame.index = stamps
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="min"))
    blocks = block_average_db(frame)
    first, last = blocks.first_valid_index(), blocks.last_valid_index()
    blocks, missing = fill_gaps(blocks.loc[first:last])
    out = blocks.copy()
    out.insert(0, "timestamp_utc", out.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out_csv.parent, delete=False, suffix=".tmp") as tmp:
        pass
    out.to_csv(tmp.name, index=False)
    os.replace(tmp.name, out_csv)
    return {"blocks": len(out), "imputed_share": missing}


def ensure_powder15(band: str, repo_root: str) -> dict[str, Any]:
    """Convert every needed trace once per worker; returns the imputed share per file."""
    report = {}
    for node in CORE_NODES:
        for run in (TRAIN_RUN, TEST_RUN):
            raw = Path(repo_root) / RAW_ROOT / node / run / band / RAW_NAME
            out = Path(repo_root) / OUT_ROOT / node / run / band / OUT_NAME
            if not out.exists():
                report[f"{node}/{run}"] = convert_trace(raw, out)
    return report


def apply_powder15(config: dict[str, Any], band: str, repo_root: str, horizons: list[int]) -> dict[str, Any]:
    """Point a one-minute POWDER config at the 15-minute files (same timestamp ranges), lookback 96."""
    config = copy.deepcopy(config)
    for entry in config["data"]["files"]:
        path = entry["path"].replace(RAW_ROOT + "/", OUT_ROOT + "/", 1).replace(RAW_NAME, OUT_NAME)
        entry["path"] = path.replace("/600_800/", f"/{band}/")
    config["data"]["chunks"] = [{"id": f"powder15_{band}", "start_mhz": float(band.split("_")[0]),
                                 "end_mhz": float(band.split("_")[1])}]
    config["data"].pop("map", None)
    config["windowing"] = {"lookback": LOOKBACK, "horizons": sorted(int(h) for h in horizons)}
    config["preprocessing"] = {**config.get("preprocessing", {}), "normalize": True, "impute": False,
                               "max_missing_gap": 0}
    return config


def set_lookback_and_horizon(config: dict[str, Any], model: str, direct_horizon: int | None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config[model]["model"]["input_sequence_length"] = LOOKBACK
    config[model]["model"]["prediction_horizon"] = int(direct_horizon) if direct_horizon else 1
    for key in ("train_stride", "val_stride", "test_stride"):
        config[model]["train"][key] = 1
    return config


# ------------------------------------------------------------------------------------------ baselines
def baseline_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one (band, baseline) cell, averaged over the four core receivers' test partitions."""
    import sys

    sys.path.insert(0, args["repo_root"])
    from training.ray.electrosense import baseline_metrics

    band, kind = job["band"], job["kind"]
    result = {"band": band, "model": kind, "kind": "baseline", "hyperparameters": "none"}
    try:
        result["conversion"] = json.dumps(ensure_powder15(band, args["repo_root"]))
        start, end = pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z")
        per_node = []
        for node in CORE_NODES:
            train = pd.read_csv(Path(args["repo_root"]) / OUT_ROOT / node / TRAIN_RUN / band / OUT_NAME)
            test = pd.read_csv(Path(args["repo_root"]) / OUT_ROOT / node / TEST_RUN / band / OUT_NAME)
            stamps = pd.to_datetime(test.pop("timestamp_utc"), utc=True)
            test = test[(stamps >= start) & (stamps < end)].to_numpy(dtype=np.float64)
            train = train.drop(columns=["timestamp_utc"]).to_numpy(dtype=np.float64)
            stats = {"borders": {"test": [0, len(test)]}, "train_mean": train.mean(axis=0).tolist(),
                     "train_std": train.std(axis=0, ddof=0).tolist()}
            per_node.append(baseline_metrics(test, stats, kind, horizons=HORIZONS, seq_len=LOOKBACK))
        for key in per_node[0]:
            result[key] = float(np.mean([m[key] for m in per_node]))
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
    return result


# --------------------------------------------------------------------------------------- our models
def ours_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one of our models on 15-minute POWDER and score it on the test range."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.final_eval import build_eval_config, config_file_for, summarize_metrics
    from training.ray.fixed_train import REFERENCE_MODELS
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, rep, seed, band = job["model"], job["rep"], int(job["seed"]), job["band"]
    horizon = job.get("horizon")
    result = {"band": band, "model": model, "rep": rep, "seed": seed, "kind": "ours", "mode": args["mode"],
              "horizon": horizon or "", "hyperparameters": "inherited", "source_prefix": job.get("source_prefix", "")}
    started = time.time()
    try:
        result["conversion"] = json.dumps(ensure_powder15(band, args["repo_root"]))
        base = yaml.safe_load(Path(config_file_for(model, rep)).read_text())
        if model in REFERENCE_MODELS:
            config = copy.deepcopy(base)
            config["training"] = {**config.get("training", {}), "models": [model], "device": "cpu"}
        else:
            costs = json.loads(client.get_object(Bucket=bucket, Key=job["costs_key"])["Body"].read())
            config = build_eval_config(base, model, costs, test_start="2026-10-27T20:00:00Z",
                                       test_end="2026-11-03T20:00:00Z", device=args.get("device"))
        horizons = [horizon] if horizon else list(HORIZONS)
        config = apply_powder15(config, band, args["repo_root"], horizons)
        config = set_lookback_and_horizon(config, model, horizon)
        config[model]["train"]["seed"] = seed
        site = job.get("site")
        if site:  # a receiver-specific model: only that receiver's files, so its own normalization statistics
            from training.ray.final_eval import restrict_files_to_site

            config = restrict_files_to_site(config, site_name(site))
            result["site"] = site
        with tempfile.TemporaryDirectory(prefix="powder15-") as scratch:
            run_dir, eval_dir = Path(scratch) / "run", Path(scratch) / "eval"
            train_one_model(config, model, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, checkpoint, skip_plots=True)
            rows = pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")
            result.update(summarize_metrics(rows))
            from training.ray.artifacts import save_run

            result.update(save_run(client, bucket, f"{args['output_prefix'].strip('/')}/weights/{band}/{rep}/{model}"
                                   f"{'/h' + str(horizon) if horizon else ''}{'/site-' + str(site) if site else ''}/seed-{seed}",
                                   run_dir, config, {"result_kind": "powder15_ours", "seed": seed, "horizon": horizon,
                                                     "site": site}))
            if args.get("per_site_eval") and not site:
                # score the pooled model one receiver at a time (training/validation data, so normalization, stay pooled)
                from training.ray.final_eval import restrict_files_to_site

                for node in CORE_NODES:
                    out = Path(scratch) / f"eval-{node}"
                    out.mkdir()
                    evaluate_one_model(restrict_files_to_site(config, site_name(node), ("test",)), model, out,
                                       checkpoint, skip_plots=True)
                    per_site = summarize_metrics(pd.read_csv(out / "aggregate_metrics.csv").to_dict("records"))
                    result.update({f"{site_name(node)}__{k}": v for k, v in per_site.items()})
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-800:])
    result["wall_seconds"] = round(time.time() - started, 1)
    try:
        suffix = f"/h{horizon}" if horizon else ""
        suffix += f"/site-{job['site']}" if job.get("site") else ""
        key = f"{args['output_prefix'].strip('/')}/{args['mode']}/{band}/{rep}/{model}{suffix}/seed-{seed}.json"
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


# ------------------------------------------------------------------------------------------------ CLI
def expand_sites(jobs: list[dict[str, Any]], per_site: bool) -> list[dict[str, Any]]:
    """Per-receiver mode: one job per (job, core receiver); otherwise unchanged (pooled over the four receivers)."""
    if not per_site:
        return jobs
    return [{**job, "site": node} for job in jobs for node in CORE_NODES]


def launch(args: argparse.Namespace) -> None:
    import ray

    from .electrosense import ours_jobs
    from .fixed_train import discover_models
    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
                   "mode": getattr(args, "mode", "baselines"), "device": args.device,
                   "per_site_eval": getattr(args, "per_site_eval", False)}
    futures = []
    if args.command == "baselines":
        kinds = ("train_mean", "last_value", "window_mean", "seasonal_naive", "seasonal_mean")
        remote = ray.remote(num_cpus=2)(baseline_job)
        futures = [remote.remote({"band": b, "kind": k}, worker_args) for b in args.bands for k in kinds]
        prefix = "baselines"
    else:
        entries = discover_models(client, bucket, args.prefixes, args.models)
        if args.include_reference:
            entries += [{"rep": rep, "model": m, "source_prefix": "", "costs_key": ""}
                        for m, rep in (("lookbackmean1d", "1d"), ("lookbackmean2d", "2d"))
                        if not args.models or m in args.models]
        entries = [e for e in entries if e["rep"] in ("1d", "2d")]
        jobs = ours_jobs(entries, args.bands, args.seeds, args.mode, args.horizons)
        for job in jobs:
            job["band"] = job.pop("dataset")
        jobs = expand_sites(jobs, args.per_site)
        print(f"[powder15] {len(jobs)} runs ({args.mode})", flush=True)
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
    print(f"[powder15] wrote s3://{bucket}/{key}; {len(frame)} rows, {len(failed)} failed", flush=True)
    for _, row in failed.head(8).iterrows():
        print(f"[powder15] FAILED {row.get('band')}/{row.get('model')}: {row.get('error')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baselines", "ours"):
        p = sub.add_parser(name)
        p.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
        p.add_argument("--bucket", default=None)
        p.add_argument("--output-prefix", default="powder15-results")
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
    ours.add_argument("--per-site", action="store_true",
                      help="Train one model per core receiver (own data and normalization); default is pooled over four")
    ours.add_argument("--per-site-eval", action="store_true",
                      help="Also score each pooled model on each receiver's test windows separately")
    ours.add_argument("--gpu-fraction", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
