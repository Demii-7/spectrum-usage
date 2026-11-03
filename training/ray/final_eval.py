"""Evaluate exported checkpoints on a chosen test range, on any Ray cluster.

Standalone by design: everything comes from the durable Chameleon bucket (checkpoints plus
the ``*_run_costs.json`` next to each one, which holds the exact model/train config) and the
data files, so it needs no MinIO campaign state. Results go back to the bucket as CSV.

    python -m training.ray.final_eval --prefixes checkpoints-oct2026 checkpoints-oct2026-r2 \
        --test-end 2026-11-03T20:00:00Z --output-prefix oct2026-results/final-eval
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = "training/configs/oct2026"
DEFAULT_TEST_START = "2026-10-27T20:00:00Z"
DEFAULT_TEST_END = "2026-11-03T20:00:00Z"


def parse_checkpoint_key(key: str) -> dict[str, Any] | None:
    """``<prefix>/<rep>/<model>/seed-NN/<file>.pt`` -> entry fields, else None."""
    parts = key.split("/")
    if len(parts) < 5 or not parts[-1].endswith(".pt") or not parts[-2].startswith("seed-"):
        return None
    return {"prefix": "/".join(parts[:-4]), "rep": parts[-4], "model": parts[-3],
            "seed": int(parts[-2].removeprefix("seed-")), "pt_key": key}


def config_file_for(model: str, rep: str) -> str:
    from .campaign_all import PLAN

    for plan_model, plan_rep, config in PLAN:
        if plan_model == model and plan_rep == rep:
            return f"{CONFIG_DIR}/{config}.yaml"
    raise KeyError(f"No tuning config for {model} ({rep})")


def build_eval_config(base: dict[str, Any], model: str, costs: dict[str, Any], *,
                      test_start: str, test_end: str, test_stride: int | None = None,
                      data_dir_swap: tuple[str, str] | None = None,
                      device: str | None = None) -> dict[str, Any]:
    """Config that rebuilds the checkpoint's exact architecture and scores the given test range."""
    config = copy.deepcopy(base)
    config[model] = {"model": json.loads(costs["cost_model_config_json"]),
                     "train": json.loads(costs["cost_train_config_json"])}
    if test_stride is not None:
        config[model]["train"]["test_stride"] = int(test_stride)
    config["training"] = {**config.get("training", {}), "models": [model]}
    config["training"].pop("model_name", None)
    if device:
        config["training"]["device"] = device
    ranges = config["data"]["split"]["ranges"]
    ranges["test"] = {"start": test_start, "end": test_end}
    if data_dir_swap:
        old, new = data_dir_swap
        for entry in config["data"]["files"]:
            entry["path"] = entry["path"].replace(old, new, 1)
    return config


CORE_SITES = ("humanities", "madsen", "moran", "sagepoint")


def restrict_files_to_site(config: dict[str, Any], site: str, partitions: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Keep only one receiver's files, in all partitions or just the given ones."""
    config = copy.deepcopy(config)
    kept = []
    for entry in config["data"]["files"]:
        in_scope = partitions is None or entry["partition"] in partitions
        if not in_scope or f"/{site}-nuc1/" in entry["path"]:
            kept.append(entry)
    for partition in (partitions or ("train", "validation", "test")):
        if not any(e["partition"] == partition and f"/{site}-nuc1/" in e["path"] for e in kept):
            raise ValueError(f"No {partition} files for site {site!r}")
    config["data"]["files"] = kept
    return config


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse aggregate_metrics rows to per-split, per-horizon means plus an overall mean."""
    out: dict[str, Any] = {}
    for split in ("validation", "test"):
        selected = [r for r in rows if str(r["split"]).endswith(split)]
        if not selected:
            continue
        by_h: dict[int, list[float]] = {}
        for r in selected:
            by_h.setdefault(int(r["horizon"]), []).append(float(r["mae_db"]))
        for horizon, values in sorted(by_h.items()):
            out[f"{split}_mae_db_t{horizon}"] = sum(values) / len(values)
        out[f"{split}_mean_horizon_mae_db"] = sum(sum(v) / len(v) for v in by_h.values()) / len(by_h)
    return out


def evaluate_entry(entry: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: download one checkpoint, evaluate it, return a flat result row."""
    import os
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    started = time.time()
    result = {k: entry[k] for k in ("prefix", "rep", "model", "seed")}
    try:
        with tempfile.TemporaryDirectory(prefix="final-eval-") as scratch:
            checkpoint = Path(scratch) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(checkpoint))
            costs = json.loads(client.get_object(Bucket=bucket, Key=entry["costs_key"])["Body"].read())
            base = yaml.safe_load(Path(config_file_for(entry["model"], entry["rep"])).read_text())
            config = build_eval_config(
                base, entry["model"], costs, test_start=args["test_start"], test_end=args["test_end"],
                test_stride=args["test_stride"],
                data_dir_swap=tuple(args["data_dir_swap"]) if args["data_dir_swap"] else None,
                device=args["device"])
            sites = args.get("test_sites") or [None]
            per_site = []
            for site in sites:
                site_config = config if site is None else restrict_files_to_site(config, site, ("test",))
                out_dir = Path(scratch) / f"eval-{site or 'all'}"
                out_dir.mkdir()
                evaluate_one_model(site_config, entry["model"], out_dir, checkpoint, skip_plots=True)
                rows = pd.read_csv(out_dir / "aggregate_metrics.csv").to_dict("records")
                summary = summarize_metrics(rows)
                summary["n_test_windows"] = int(max((r.get("n_targets") or 0) for r in rows
                                                    if str(r["split"]).endswith("test")) or 0)
                per_site.append((site, summary))
        if len(per_site) == 1 and per_site[0][0] is None:
            result.update(per_site[0][1])
        else:
            # one flat row per site: <site>__<metric>
            for site, summary in per_site:
                result.update({f"{site}__{k}": v for k, v in summary.items()})
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300],
                      trace=traceback.format_exc()[-800:])
    result["eval_seconds"] = round(time.time() - started, 1)
    return result


def discover(client, bucket: str, prefixes: list[str], models: list[str] | None,
             seeds: list[int] | None) -> list[dict[str, Any]]:
    entries = []
    for prefix in prefixes:
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix.rstrip("/") + "/") for o in page.get("Contents", [])]
        costs = {k.rsplit("/", 1)[0]: k for k in keys if k.endswith("_run_costs.json")}
        for key in keys:
            entry = parse_checkpoint_key(key)
            if not entry or (models and entry["model"] not in models) or (seeds and entry["seed"] not in seeds):
                continue
            costs_key = costs.get(key.rsplit("/", 1)[0])
            if costs_key:
                entries.append({**entry, "costs_key": costs_key})
    return sorted(entries, key=lambda e: (e["prefix"], e["rep"], e["model"], e["seed"]))


def launch(args: argparse.Namespace) -> None:
    import pandas as pd
    import ray

    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or __import__("os").environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    entries = discover(client, bucket, args.prefixes, args.models, args.seeds)
    print(f"[final_eval] {len(entries)} checkpoints", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    options = {"num_cpus": 2, "num_gpus": args.gpu_fraction, "max_retries": 1}
    remote = ray.remote(**options)(evaluate_entry)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "test_start": args.test_start,
                   "test_end": args.test_end, "test_stride": args.test_stride,
                   "data_dir_swap": args.data_dir_swap, "device": args.device,
                   "test_sites": args.test_sites}
    frame = pd.DataFrame(ray.get([remote.remote(e, worker_args) for e in entries]))
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    key = f"{args.output_prefix.strip('/')}/final_eval_{stamp}.csv"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    failed = frame[~frame["ok"].astype(bool)] if "ok" in frame else frame
    print(f"[final_eval] wrote s3://{bucket}/{key} ({len(frame)} rows, {len(failed)} failed)", flush=True)
    for _, row in failed.head(5).iterrows():
        print(f"[final_eval] FAILED {row['rep']}/{row['model']}/seed-{row['seed']}: {row.get('error')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", nargs="+", default=["checkpoints-oct2026", "checkpoints-oct2026-r2"])
    parser.add_argument("--models", nargs="+", help="Only these models")
    parser.add_argument("--seeds", nargs="+", type=int, help="Only these seeds")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--test-stride", type=int, default=1,
                        help="Window stride for the test pass (default 1 = every minute)")
    parser.add_argument("--data-dir-swap", nargs=2, metavar=("OLD", "NEW"),
                        help="Replace this path prefix in data.files, e.g. data/powder/ data/powder_final/")
    parser.add_argument("--device", default=None, help="Override training.device (cuda/cpu)")
    parser.add_argument("--test-sites", nargs="+", help="Also score the test range one receiver at a time "
                        "(training/validation data, and so the normalization, stay pooled)")
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default="oct2026-results/final-eval")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
