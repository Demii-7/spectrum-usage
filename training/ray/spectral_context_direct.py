"""Spectral-context ablation for the direct (one model per horizon) cross-frequency models at 15-minute POWDER resolution.

For every (model, band, non-noise region, condition, seed) a model is trained per horizon (4/12/24/48 steps) with its tuned 15-minute
configuration (``direct-tuning/15min/<band>/<model>/winners.json``, the same candidate for every horizon, as in the tuned finals), and its
test error over the region's bins on the full test week (stride 1) is recorded. Conditions:

  full_context  the real inputs, loss over the region's bins (identical to the 2D ablation in ``spectral_context_fixed``)
  region_only   the other bins' INPUTS replaced by low-tail Gaussian noise-floor values, loss over the region's bins (identical to the 2D ablation)
  input_only    the same masked inputs but the loss over ALL bins with the REAL targets (``data.mask.real_targets``): separates losing
                the out-of-region inputs from losing the out-of-region training signal. Only for the matched pair (revinmlp, revinmlp_mix)

Both full_context and region_only therefore use the region loss, so the tuned finals (full-band loss) are not the same condition; the smoke check
compares ``full_context`` with the final's per-bin error over the region.

Records: ``<prefix>/<band>/<model>/<region>-<condition>-h<horizon>-seed-<seed>.json`` (failures: ``....error.json``). Resumable (--skip-done).

    python -m training.ray.spectral_context_direct --models revinmlp patchtst_joint --seeds 40 41 42 --max-parallel 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CONDITIONS = ("full_context", "region_only", "input_only")
INPUT_ONLY_MODELS = ("revinmlp", "revinmlp_mix")
HORIZONS = (4, 12, 24, 48)
BANDS = ("600_800", "2400_2600")
DEFAULT_SEEDS = {"itransformer": (42,)}
ALL_SEEDS = (40, 41, 42)
MODELS = ("revinmlp", "revinmlp_mix", "patchtst_joint", "itransformer")
PREFIX = "oct2026-spectral-context-15min-strong"
TUNING_PREFIX = "direct-tuning"
# CPU / GPU share of one task; the shares are tune_direct's (a task trains its horizons one after the other)
SHARES = {"revinmlp": (2, 0.125), "revinmlp_mix": (2, 0.125), "patchtst_joint": (4, 0.25), "itransformer": (4, 0.25)}


def plan_path(band: str) -> Path:
    return Path("evaluation/analysis") / f"plan_regions_{band}.csv"


def record_key(prefix: str, band: str, model: str, region: str, condition: str, horizon: int, seed: int, failed: bool = False) -> str:
    return f"{prefix.strip('/')}/{band}/{model}/{region}-{condition}-h{horizon}-seed-{seed}{'.error' if failed else ''}.json"


def conditions_for(model: str, requested=CONDITIONS) -> list[str]:
    return [c for c in requested if c != "input_only" or model in INPUT_ONLY_MODELS]


def make_tasks(models, bands, regions_by_band, seeds_by_model, conditions=CONDITIONS, horizons=HORIZONS) -> list[dict[str, Any]]:
    """One task per (model, band, region, condition, seed); a task trains the horizons one after the other. Ordered by seed first, so
    an interrupted run still has every cell for the first seeds."""
    tasks = [{"model": model, "band": band, "region": region, "condition": condition, "seed": int(seed), "horizons": list(horizons)}
             for model in models for band in bands for region in regions_by_band[band]
             for condition in conditions_for(model, conditions) for seed in seeds_by_model[model]]
    return sorted(tasks, key=lambda t: (t["seed"], t["band"], str(t["region"]["band_id"]), t["model"], t["condition"]))


def run_task(task: dict[str, Any], args: dict[str, Any]) -> list[dict[str, Any]]:
    """Remote worker: train and score the task's horizons; one record per horizon."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.llm_powder import use_isolated_libs
    from training.ray.powder15 import ensure_powder15
    from training.ray.spectral_context import region_metrics_from_forecasts, regional_config
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_direct import build_config, default_train_stride
    from training.ray.tune_fixed import train_prepared

    use_isolated_libs()
    import torch

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, band, region, condition, seed = task["model"], task["band"], task["region"], task["condition"], task["seed"]
    rows = []
    for horizon in task["horizons"]:
        record = {"model": model, "band": band, "region": region["band_id"], "start_mhz": float(region["start_mhz"]), "end_mhz": float(region["end_mhz"]),
                  "condition": condition, "horizon": horizon, "seed": seed, "params": task["params"], "config_source": task["config_source"],
                  "loss": "all_bins" if condition == "input_only" else "region_bins", "train_stride": default_train_stride(model, "15min"), "test_stride": 1}
        started = time.time()
        try:
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(min(0.95, SHARES[model][1]))
                torch.cuda.reset_peak_memory_stats()
            ensure_powder15(band, root)
            config, name = build_config(model, "15min", band, horizon, seed, task["params"], root,
                                        train_stride=record["train_stride"], test_stride=1)
            config = regional_config(config, region, condition, seed, noise_ranges=task.get("noise_ranges"))
            with tempfile.TemporaryDirectory(prefix="spectral-direct-") as scratch:
                result = train_prepared(config, name, Path(scratch) / "run")
                record.update(objective=result["objective"], objective_name=result["objective_name"])
                eval_dir = Path(scratch) / "eval"
                eval_dir.mkdir()
                evaluate_one_model(config, name, eval_dir, result["checkpoint"], skip_plots=True)
                record.update(region_metrics_from_forecasts(config, eval_dir, region))
            if torch.cuda.is_available():
                record["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            record["ok"] = math.isfinite(float(record[f"test_mae_db_t{horizon}"]))
        except Exception as error:
            import traceback

            record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
        record["wall_seconds"] = round(time.time() - started, 1)
        key = record_key(args["output_prefix"], band, model, str(region["band_id"]), condition, horizon, seed, failed=not record["ok"])
        try:
            client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
        except Exception:
            pass
        rows.append(record)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def tuned_params(client, bucket: str, band: str, model: str) -> tuple[dict[str, Any], str] | None:
    key = f"{TUNING_PREFIX}/15min/{band}/{model}/winners.json"
    try:
        winner = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None
    return winner["params"], f"{key}:candidate={winner['index']}"


def done_keys(client, bucket: str, prefix: str) -> set[str]:
    done = set()
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.strip("/") + "/"):
        done.update(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".json") and not o["Key"].endswith(".error.json"))
    return done


def launch(args: argparse.Namespace) -> None:
    import ray

    from .spectral_context import noise_ranges_from_plan, non_noise_regions
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    regions = {band: non_noise_regions(plan_path(band)) for band in args.bands}
    if args.regions_only:
        regions = {band: [r for r in rs if str(r["band_id"]) in set(args.regions_only)] for band, rs in regions.items()}
    seeds = {model: tuple(args.seeds) if args.seeds else DEFAULT_SEEDS.get(model, ALL_SEEDS) for model in args.models}
    tasks = make_tasks(args.models, args.bands, regions, seeds, args.conditions, args.horizons)
    tuned, missing = {}, set()
    for model in args.models:
        for band in args.bands:
            found = tuned_params(client, bucket, band, model)
            if found is None:
                missing.add((model, band))
            else:
                tuned[(model, band)] = found
    if missing:
        print(f"[spectral_direct] no tuned winner yet for {sorted(missing)}: skipped", flush=True)
    done = done_keys(client, bucket, args.output_prefix) if args.skip_done else set()
    todo = []
    for task in tasks:
        if (task["model"], task["band"]) not in tuned:
            continue
        task["params"], task["config_source"] = tuned[(task["model"], task["band"])]
        task["noise_ranges"] = noise_ranges_from_plan(plan_path(task["band"])) if task["band"] == "2400_2600" else None
        task["horizons"] = [h for h in task["horizons"]
                            if record_key(args.output_prefix, task["band"], task["model"], str(task["region"]["band_id"]),
                                          task["condition"], h, task["seed"]) not in done]
        if task["horizons"]:
            todo.append(task)
    if args.reverse:
        todo.reverse()
    cells = sum(len(t["horizons"]) for t in todo)
    print(f"[spectral_direct] {len(tasks)} tasks, {len(todo)} with work left ({cells} cells)", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix}

    def submit(task):
        cpus, gpu = SHARES[task["model"]]
        return ray.remote(num_cpus=cpus, num_gpus=gpu, max_retries=2, memory=int(args.memory_gb * 2 ** 30))(run_task).remote(task, worker_args)

    queue = list(todo)
    pending = [submit(queue.pop(0)) for _ in range(min(args.max_parallel, len(queue)))]
    finished = failed = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for row in ray.get(ready[0]):
            finished += 1
            if not row.get("ok"):
                failed += 1
                print(f"[spectral_direct] FAILED {row['model']} {row['band']} {row['region']} {row['condition']} h{row['horizon']} seed {row['seed']}: "
                      f"{row.get('error')}", flush=True)
        if queue:
            pending.append(submit(queue.pop(0)))
        print(f"[spectral_direct] {finished} of {cells} cells done, {failed} failed", flush=True)
    print(f"[spectral_direct] finished: {finished} cells, {failed} failed", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    parser.add_argument("--seeds", nargs="+", type=int, help="default: 40 41 42 (itransformer: 42 only)")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--regions-only", nargs="+", help="only these region ids (e.g. R3)")
    parser.add_argument("--max-parallel", type=int, default=4, help="most tasks in flight at once")
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--skip-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default=PREFIX)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
