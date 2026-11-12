"""Spectral-context ablation (full context vs region-only) with inherited hyperparameters, on any Ray cluster.

Standalone like ``fixed_train``: each model's selected configuration is rebuilt from the
``*_run_costs.json`` next to its exported checkpoint (no search, no MinIO campaign state). For every
(model, non-noise region, condition, seed) one model is trained and its test error over the region's bins
is recorded. ``region_only`` replaces all other bins with noise-floor values; both conditions compute the
loss over the target region only, so the comparison isolates whether out-of-region bins help.

    python -m training.ray.spectral_context_fixed --models temporalconvnet vanillalstm
    python -m training.ray.spectral_context_fixed --resolution 15min --output-prefix oct2026-spectral-context-15min

``--resolution 15min`` runs the same ablation on the block-averaged 15-minute POWDER data (lookback 96, horizons 4/12/24/48 steps, the
one-step models rolled out, as in ``powder15``), with each model's inherited 1-minute 600-800 configuration, over the same regions.
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

from .final_eval import DEFAULT_TEST_START, build_eval_config, config_file_for
from .fixed_train import discover_models

DEFAULT_MODELS = ("vanillalstm", "residualvanillalstm", "lstmattn", "temporalconvnet", "autoformer_csa")
DEFAULT_SEEDS = (40, 41, 42, 43, 44)
CONDITIONS = ("full_context", "region_only")


def make_jobs(entries: list[dict[str, Any]], regions: list[dict[str, Any]], seeds, conditions=CONDITIONS):
    """One job per (model, region, condition, seed); entries are discover_models() rows for 2D models."""
    return [{**entry, "region": region, "condition": condition, "seed": int(seed)}
            for entry in entries if entry["rep"] == "2d"
            for region in regions for condition in conditions for seed in seeds]


def record_key(prefix: str, job: dict[str, Any]) -> str:
    return (f"{prefix.strip('/')}/{job['model']}/{job['region']['band_id']}-{job['condition']}"
            f"-seed-{job['seed']}.json")


def tuned15_entries(client, bucket: str, band: str, models: list[str]) -> list[dict[str, Any]]:
    """One entry per 2D model carrying the 15-minute-tuned winner's parameters (``tuning/powder15/<band>/2d/<model>/final/seed-N.json``; the winner is the same
    for every seed, so the first finished final is read)."""
    entries = []
    for model in models:
        for seed in range(40, 45):
            try:
                record = json.loads(client.get_object(Bucket=bucket, Key=f"tuning/powder15/{band}/2d/{model}/final/seed-{seed}.json")["Body"].read())
            except Exception:
                continue
            if record.get("ok") and record.get("parameters") is not None:
                entries.append({"rep": "2d", "model": model, "source_prefix": "tuning/powder15", "costs_key": None, "source_objective": record.get("search_objective"),
                                "parameters": record["parameters"], "candidate_id": record.get("candidate_id"),
                                "config_source": f"tuned15:{record.get('candidate_id')}"})
                break
        else:
            raise FileNotFoundError(f"no finished tuned 15-minute final for {model} in {band}")
    return entries


def ablation_entry(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one (model, region, condition, seed) cell and record its region test metrics."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.spectral_context import regional_config, region_metrics_from_forecasts
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, region = job["model"], job["region"]
    result = {"model": model, "seed": job["seed"], "condition": job["condition"],
              "band_id": str(region["band_id"]), "start_mhz": float(region["start_mhz"]),
              "end_mhz": float(region["end_mhz"]), "source_prefix": job.get("source_prefix", ""),
              "resolution": args.get("resolution", "1min"), "config_source": job.get("config_source", "inherited_1min_run_costs"),
              "band": args.get("band")}
    started = time.time()
    try:
        if job.get("parameters") is not None:
            # the 15-minute-tuned winner: the tuned finals' own path (tune_fixed: base config of the model, the dataset applied, the winner's parameters injected),
            # on the full test week
            from training.ray.tune_fixed import base_config, prepared

            config, _ = base_config("powder15", args["band"], model, job["rep"], args["repo_root"], client, bucket)
            config = prepared(config, "powder15", model, job["parameters"], job["seed"])
            config["data"]["split"]["ranges"]["test"] = {"start": args["test_start"], "end": args["test_end"]}
            if args["test_stride"] is not None:
                config[model]["train"]["test_stride"] = int(args["test_stride"])
            config["training"] = {**config.get("training", {}), "models": [model]}
            config["training"].pop("model_name", None)
            if args["device"]:
                config["training"]["device"] = args["device"]
            result["candidate_id"] = job.get("candidate_id")
        else:
            base = yaml.safe_load(Path(config_file_for(model, job["rep"])).read_text())
            costs = json.loads(client.get_object(Bucket=bucket, Key=job["costs_key"])["Body"].read())
            config = build_eval_config(base, model, costs, test_start=args["test_start"], test_end=args["test_end"],
                                       test_stride=args["test_stride"], device=args["device"])
        if args.get("resolution") == "15min" and job.get("parameters") is None:
            from training.ray.powder15 import HORIZONS, apply_powder15, ensure_powder15, set_lookback_and_horizon

            ensure_powder15(args["band"], args["repo_root"])
            config = apply_powder15(config, args["band"], args["repo_root"], list(HORIZONS))
            config = set_lookback_and_horizon(config, model, None)
        config = regional_config(config, region, job["condition"], job["seed"], noise_ranges=args.get("noise_ranges"))
        config[model]["train"]["seed"] = job["seed"]
        with tempfile.TemporaryDirectory(prefix="spectral-ctx-") as scratch:
            run_dir = Path(scratch) / "run"
            train_one_model(config, model, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir = Path(scratch) / "eval"
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, checkpoint, skip_plots=True)
            result.update(region_metrics_from_forecasts(config, eval_dir, region))
            if job["condition"] == "region_only_fullloss":
                from training.ray.spectral_context import out_of_region_stats

                result.update(out_of_region_stats(config, eval_dir, region))
        try:
            import torch

            if torch.cuda.is_available():
                result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        except Exception:
            pass
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-800:])
    result["wall_seconds"] = round(time.time() - started, 1)
    try:  # per-cell record, so progress and failures are visible while the job runs and reruns can skip done cells
        client.put_object(Bucket=bucket, Key=record_key(args["output_prefix"], job),
                          Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


def launch(args: argparse.Namespace) -> None:
    import pandas as pd
    import ray

    from .registry import get_model_spec
    from .spectral_context import noise_ranges_from_plan, non_noise_regions
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    if args.hyperparameters == "tuned15":
        if args.resolution != "15min":
            raise SystemExit("--hyperparameters tuned15 needs --resolution 15min")
        entries = tuned15_entries(client, bucket, args.band, list(args.models))
    else:
        entries = discover_models(client, bucket, args.prefixes, list(args.models))
    regions = non_noise_regions(args.regions)
    if args.regions_only:
        regions = [r for r in regions if str(r["band_id"]) in set(args.regions_only)]
    jobs = make_jobs(entries, regions, args.seeds, tuple(args.conditions))
    done = set()
    if args.skip_done:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=args.output_prefix.strip("/") + "/"):
            for obj in page.get("Contents", []):
                body = json.loads(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()) \
                    if obj["Key"].endswith(".json") else {}
                if body.get("ok"):
                    done.add(obj["Key"])
    todo = [j for j in jobs if record_key(args.output_prefix, j) not in done]
    if args.reverse:
        todo.reverse()                 # a second driver on another cluster works from the other end, so two drivers rarely meet on a cell
    print(f"[spectral_context] {len(entries)} models, {len(regions)} regions, {len(args.seeds)} seeds: "
          f"{len(jobs)} cells, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
                   "test_start": args.test_start, "test_end": args.test_end, "test_stride": args.test_stride,
                   "device": args.device, "resolution": args.resolution, "band": args.band,
                   "noise_ranges": noise_ranges_from_plan(args.regions) if args.noise_from_plan else None}
    def submit(job):
        spec = get_model_spec(job["model"])
        resources = dict(spec.resources.as_ray())
        gpu = args.gpu_fraction if resources.get("gpu", 0) > 0 else 0
        return ray.remote(num_cpus=resources["cpu"], num_gpus=gpu, max_retries=1, memory=int(args.memory_gb * 2 ** 30))(ablation_entry).remote(job, worker_args)

    # At most --max-parallel cells in flight: submitting every cell up front queued over a thousand tasks that sat ahead of every
    # other job's tasks for hours (Ray schedules same-shape tasks first come, first served).
    queue = list(todo)
    pending = [submit(queue.pop(0)) for _ in range(min(args.max_parallel, len(queue)))]
    rows = []
    while pending:
        done_refs, pending = ray.wait(pending, num_returns=1)
        rows.append(ray.get(done_refs[0]))
        if queue:
            pending.append(submit(queue.pop(0)))
        if len(rows) % 10 == 0:
            print(f"[spectral_context] {len(rows)} of {len(todo)} cells done", flush=True)
    frame = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    key = f"{args.output_prefix.strip('/')}/summary_{stamp}.csv"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    failed = frame[~frame["ok"].astype(bool)] if len(frame) else frame
    print(f"[spectral_context] wrote s3://{bucket}/{key}; {len(frame)} cells, {len(failed)} failed", flush=True)
    for _, row in failed.head(8).iterrows():
        print(f"[spectral_context] FAILED {row['model']} {row['band_id']} {row['condition']} seed {row['seed']}: "
              f"{row.get('error')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--prefixes", nargs="+", default=["checkpoints-oct2026", "checkpoints-oct2026-r2"])
    parser.add_argument("--regions", type=Path, default=Path("evaluation/analysis/plan_regions_600_800.csv"))
    parser.add_argument("--regions-only", nargs="+", help="Only these band ids (e.g. R1 R3)")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=("full_context", "region_only", "region_only_fullloss"),
                        help="conditions to train (default: the original pair); region_only_fullloss = v2 of the rolled-out ablation: noise-floor bins in inputs and targets, loss over all bins")
    parser.add_argument("--noise-from-plan", action="store_true",
                        help="calibrate the replacement noise on the plan's noise-floor regions instead of the hardcoded 600-800 ranges (needed for 2400-2600)")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default="2026-11-03T20:00:00Z")
    parser.add_argument("--test-stride", type=int, default=None, help="Override test stride (default: as selected)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resolution", choices=("1min", "15min"), default="1min")
    parser.add_argument("--hyperparameters", choices=("inherited", "tuned15"), default="inherited",
                        help="inherited: each model's 1-minute selected configuration (the original ablation); tuned15: the 15-minute-tuned winner of tuning/powder15 (needs --resolution 15min)")
    parser.add_argument("--band", default="600_800", help="band for --resolution 15min (the region plan exists for 600_800)")
    parser.add_argument("--memory-gb", type=float, default=8.0, help="memory reserved per cell so Ray never oversubscribes a node")
    parser.add_argument("--max-parallel", type=int, default=8, help="most cells in flight at once (keeps the queue short for other jobs)")
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    parser.add_argument("--skip-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse", action="store_true", help="process the remaining cells in reverse order (for a second driver on another cluster)")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default="oct2026-spectral-context")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
