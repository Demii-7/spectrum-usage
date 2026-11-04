"""Train each model once with its already-selected hyperparameters on another band.

No search: every model reuses the configuration chosen for 600-800 (rebuilt from the
``*_run_costs.json`` stored with its exported checkpoint), with the data files and frequency
chunk swapped to the target band. Standalone, like ``final_eval``: it reads configs from the
durable Chameleon bucket and writes checkpoints, costs and metrics back to it, so it runs on
any Ray cluster with the repo and the band's data.

    python -m training.ray.fixed_train --band 2400_2600 --seeds 42
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

import yaml

from .final_eval import (
    CORE_SITES, DEFAULT_TEST_END, DEFAULT_TEST_START, build_eval_config, config_file_for, ensure_site_files,
    parse_checkpoint_key, pool_sites, restrict_files_to_site, summarize_metrics,
)

REFERENCE_MODELS = {"lookbackmean1d": "1d", "lookbackmean2d": "2d", "lookbackmean4d": "4d"}
# Rough relative cost, used only to start the slowest runs first.
COST_ORDER = {"dswinlstm_i": 100, "residualconvlstm": 40, "convlstm": 40, "vanillalstm1d": 30,
              "residualvanillalstm1d": 30, "autoformer_csa": 20, "temporalconvnet": 20}

DIRECT_AT_60 = {"autoformer_csa", "dswinlstm_i"}  # already trained directly for 60 steps
ONE_STEP_ONLY = {"linearar1d", "linearar2d", "linearar4d", "residuallinearar1d", "residuallinearar2d",
                 "residuallinearar4d", *REFERENCE_MODELS}  # no direct multi-step form; rolled out instead
STANDARD_HORIZONS = (1, 15, 60)


def horizon_jobs(entries: list[dict[str, Any]], horizons: list[int]) -> list[dict[str, Any]]:
    """One job per (model, horizon) for models that can be trained directly for that many steps."""
    jobs = []
    for entry in entries:
        if entry["model"] in ONE_STEP_ONLY:
            continue
        for horizon in horizons:
            if horizon == 60 and entry["model"] in DIRECT_AT_60:
                continue  # the main run is already a direct 60-step model
            jobs.append({**entry, "horizon": int(horizon)})
    return jobs


def set_direct_horizon(config: dict[str, Any], model: str, horizon: int) -> dict[str, Any]:
    """Train the model directly for ``horizon`` steps and score at the standard horizons up to it."""
    config = copy.deepcopy(config)
    config["windowing"]["horizons"] = sorted({h for h in STANDARD_HORIZONS if h < horizon} | {horizon})
    config[model]["model"]["prediction_horizon"] = int(horizon)
    return config


def band_limits(band: str) -> tuple[float, float]:
    low, high = band.split("_")
    return float(low), float(high)


def swap_band(config: dict[str, Any], band: str, *, source_band: str = "600_800") -> dict[str, Any]:
    """Point a config at another band: data files, frequency chunk, and the 4D map cache name."""
    config = copy.deepcopy(config)
    low, high = band_limits(band)
    for entry in config["data"]["files"]:
        entry["path"] = entry["path"].replace(f"/{source_band}/", f"/{band}/")
    for chunk in config["data"]["chunks"]:
        chunk["id"] = f"powder_{band}"
        chunk["start_mhz"], chunk["end_mhz"] = low, high
    if config["data"].get("map"):
        config["data"]["map"]["name"] = f"oct2026_core4_{band}"
    return config


def pick_best_prefix(candidates: dict[str, float]) -> str:
    """Of several exported rounds for one model, the prefix with the lowest validation error."""
    return min(candidates, key=lambda prefix: (candidates[prefix], prefix))


def discover_models(client, bucket: str, prefixes: list[str], models: list[str] | None) -> list[dict[str, Any]]:
    """One entry per (rep, model): the better round by mean validation, plus a costs key to rebuild it."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for prefix in prefixes:
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix.rstrip("/") + "/") for o in page.get("Contents", [])]
        costs = {k.rsplit("/", 1)[0]: k for k in keys if k.endswith("_run_costs.json")}
        manifests = {"/".join(k.split("/")[:-1]): k for k in keys if k.endswith("MANIFEST.json")}
        for key in keys:
            entry = parse_checkpoint_key(key)
            if not entry or (models and entry["model"] not in models):
                continue
            group = (entry["rep"], entry["model"])
            manifest_key = manifests.get(f"{entry['prefix']}/{entry['rep']}/{entry['model']}")
            objective = float("inf")
            if manifest_key:
                data = json.loads(client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read())
                values = [e["metrics"]["objective"] for e in data["entries"]
                          if "metrics" in e and e["metrics"].get("objective") is not None]
                objective = sum(values) / len(values) if values else float("inf")
            costs_key = costs.get(key.rsplit("/", 1)[0])
            if costs_key and (group not in best or objective < best[group]["source_objective"]):
                best[group] = {"rep": group[0], "model": group[1], "source_prefix": entry["prefix"],
                               "costs_key": costs_key, "source_objective": objective}
    return sorted(best.values(), key=lambda e: (-COST_ORDER.get(e["model"], 1), e["rep"], e["model"]))


def train_entry(entry: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one model for one seed on the target band, evaluate, upload."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, rep, seed = entry["model"], entry["rep"], int(entry["seed"])
    result = {"rep": rep, "model": model, "seed": seed, "band": args["band"],
              "source_prefix": entry.get("source_prefix", "")}
    started = time.time()
    try:
        base = yaml.safe_load(Path(config_file_for(model, rep)).read_text())
        if model in REFERENCE_MODELS:
            config = copy.deepcopy(base)
            config["training"] = {**config.get("training", {}), "models": [model], "device": "cpu"}
            config["data"]["split"]["ranges"]["test"] = {"start": args["test_start"], "end": args["test_end"]}
        else:
            costs = json.loads(client.get_object(Bucket=bucket, Key=entry["costs_key"])["Body"].read())
            config = build_eval_config(base, model, costs, test_start=args["test_start"],
                                       test_end=args["test_end"], test_stride=args["test_stride"],
                                       device=args["device"])
        config = swap_band(config, args["band"])
        site = entry.get("site")
        if site:  # a receiver-specific model: its own data, its own normalization statistics
            config = restrict_files_to_site(ensure_site_files(config, site), site)
            result["site"] = site
        horizon = entry.get("horizon")
        if horizon:  # a model trained directly for this many steps (no new search)
            config = set_direct_horizon(config, model, int(horizon))
            result["horizon"] = int(horizon)
        pooled = entry.get("pooled_sites")
        if pooled:  # one model over exactly these receivers, normalization pooled across them
            config = pool_sites(config, pooled)
            result["pooled_sites"] = ",".join(pooled)
        config[model]["train"]["seed"] = seed
        with tempfile.TemporaryDirectory(prefix="fixed-train-") as scratch:
            run_dir = Path(scratch) / "run"
            train_one_model(config, model, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir = Path(scratch) / "eval"
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, checkpoint, skip_plots=True)
            rows = pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")
            result.update(summarize_metrics(rows))
            costs_file = next(run_dir.rglob("*_run_costs.json"), None)
            if costs_file:
                result.update({k: v for k, v in json.loads(costs_file.read_text()).items()
                               if k.startswith("cost_") and not k.endswith("_json")})
            site_part = f"/site-{site}" if site else (f"/pooled-{len(pooled)}" if pooled else "")
            horizon_part = f"/h{int(horizon)}" if horizon else ""
            destination = (f"{args['output_prefix'].strip('/')}/{args['band']}/{rep}/{model}{horizon_part}"
                           f"{site_part}/seed-{seed}")
            result["destination"] = destination
            for path in [checkpoint] + ([costs_file] if costs_file else []):
                client.upload_file(str(path), bucket, f"{destination}/{path.name}")
            result["checkpoint_key"] = f"{destination}/{checkpoint.name}"
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-800:])
    result["wall_seconds"] = round(time.time() - started, 1)
    # Per-run record next to the checkpoint (or where it would be), so failures are visible as they happen.
    try:
        site_part = f"/site-{entry['site']}" if entry.get("site") else ""
        horizon_part = f"/h{int(entry['horizon'])}" if entry.get("horizon") else ""
        record_key = (f"{args['output_prefix'].strip('/')}/{args['band']}/_results/{rep}-{model}{horizon_part}"
                      f"{site_part.replace('/', '-')}-seed-{seed}.json")
        client.put_object(Bucket=bucket, Key=record_key, Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


def prewarm_4d(args: dict[str, Any], band: str) -> None:
    """Build the 4D map caches once, so concurrent 4D runs do not each rebuild them."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.common.data import chunk_specs, load_chunk

    config = swap_band(yaml.safe_load(Path(f"training/configs/oct2026/ray_tuning_4d.yaml").read_text()), band)
    config["data"]["split"]["ranges"]["test"] = {"start": args["test_start"], "end": args["test_end"]}
    load_chunk(config, chunk_specs(config)[0], 0.1)


def launch(args: argparse.Namespace) -> None:
    import pandas as pd
    import ray

    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    entries = discover_models(client, bucket, args.prefixes, args.models)
    if args.include_reference:
        entries += [{"rep": rep, "model": m, "source_prefix": "", "costs_key": ""}
                    for m, rep in REFERENCE_MODELS.items() if not args.models or m in args.models]
    if args.per_site:
        entries = [e for e in entries if e["rep"] != "4d"]
        sites = args.sites or list(CORE_SITES)
        jobs = [{**e, "seed": seed, "site": site} for e in entries for site in sites
                for seed in (args.seeds if e["model"] not in REFERENCE_MODELS else [42])]
    elif args.horizon:
        jobs = [{**j, "seed": seed} for j in horizon_jobs([e for e in entries if e["model"] not in REFERENCE_MODELS],
                                                          args.horizon) for seed in args.seeds]
    elif args.pooled_sites:
        jobs = [{**e, "seed": seed, "pooled_sites": args.pooled_sites} for e in entries if e["rep"] == "2d"
                for seed in (args.seeds if e["model"] not in REFERENCE_MODELS else [42])]
    else:
        jobs = [{**e, "seed": seed} for e in entries
                for seed in (args.seeds if e["model"] not in REFERENCE_MODELS else [42])]
    print(f"[fixed_train] {len(entries)} models x seeds -> {len(jobs)} runs on band {args.band}", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "band": args.band, "test_start": args.test_start,
                   "test_end": args.test_end, "test_stride": args.test_stride, "device": args.device,
                   "output_prefix": args.output_prefix}
    if any(j["rep"] == "4d" for j in jobs):
        ray.get(ray.remote(num_cpus=4)(prewarm_4d).remote(worker_args, args.band))
        print("[fixed_train] 4D map caches built", flush=True)
    futures = []
    for job in jobs:
        spec = get_model_spec(job["model"])
        resources = dict(spec.resources.as_ray())
        if args.gpu_fraction is not None and resources.get("gpu", 0) > 0:
            resources["gpu"] = args.gpu_fraction
        futures.append(ray.remote(num_cpus=resources["cpu"], num_gpus=resources["gpu"],
                                  max_retries=1)(train_entry).remote(job, worker_args))
    frame = pd.DataFrame(ray.get(futures))
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    tag = "per_site_" if args.per_site else ("pooled_" if args.pooled_sites else ("horizon_" if args.horizon else ""))
    key = f"{args.output_prefix.strip('/')}/{args.band}/{tag}summary_{stamp}.csv"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    failed = frame[~frame["ok"].astype(bool)]
    print(f"[fixed_train] wrote s3://{bucket}/{key}; {len(frame)} runs, {len(failed)} failed", flush=True)
    for _, row in failed.head(8).iterrows():
        print(f"[fixed_train] FAILED {row['rep']}/{row['model']}: {row.get('error')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", default="2400_2600")
    parser.add_argument("--prefixes", nargs="+", default=["checkpoints-oct2026", "checkpoints-oct2026-r2"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-site", action="store_true",
                        help="Train a separate model per receiver (own data and normalization); not for 4D")
    parser.add_argument("--horizon", nargs="+", type=int,
                        help="Train each model directly for these horizons (e.g. 15 60) with its selected "
                             "hyperparameters; linear and reference models are skipped (one-step by design)")
    parser.add_argument("--pooled-sites", nargs="+",
                        help="Train one pooled 2D model over exactly these receivers (may include held-out ones)")
    parser.add_argument("--sites", nargs="+", help="Receivers for --per-site (default: the four core nodes; held-out ones such as "
                        "law73 and guesthouse work too, trained on their own data)")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default="2026-10-30T18:00:00Z",
                        help="Preliminary cutoff by default; pass the final end once the data is synced")
    parser.add_argument("--test-stride", type=int, default=None, help="Override test stride (default: as selected)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-fraction", type=float, default=None, help="Override the per-run GPU fraction")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default="oct2026-band-runs")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
