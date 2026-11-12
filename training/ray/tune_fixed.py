"""Equal-budget hyperparameter search for our models on ElectroSense and 15-minute POWDER, then five seeds.

For every (dataset, model) we evaluate the same number of candidates (default 16: the tiny/small/reference anchors
plus random draws from the model's registry search space, as in the October campaign), each trained to completion
with seed 42 and scored by validation MAE (best epoch, no early stopping). The lowest-objective candidate is then
retrained with seeds 40-44 and scored on the test partition. Test data is never used for selection. Unlike the
October campaign this search does not use ASHA early stopping (trials are short on these datasets).

Per-trial and per-seed records go to the durable bucket as they finish, so a restart skips completed work:

    python -m training.ray.tune_fixed --source electrosense --datasets spec25 --models vanillalstm:2d linearar1d:1d
    python -m training.ray.tune_fixed --source powder15 --datasets 600_800
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CAPACITIES = ("tiny", "small", "reference")
SEARCH_SEED = 42
FINAL_SEEDS = (40, 41, 42, 43, 44)
DEFAULT_MODELS = (
    "vanillalstm:2d", "residualvanillalstm:2d", "lstmattn:2d", "temporalconvnet:2d", "autoformer_csa:2d",
    "linearar2d:2d", "residuallinearar2d:2d", "linearar1d:1d", "residuallinearar1d:1d",
)
SOURCE_DATASETS = {"electrosense": ("spec4", "spec25", "spec151"), "powder15": ("600_800", "2400_2600"),
                   "powder15_4d": ("600_800", "2400_2600")}    # 15-minute maps of the four core receivers (default IDW map, six-receiver grid frame)


# STS-PredNet keeps its own flat config (training/STS-PredNet/train_integrated.py reads batch_size, epochs, learning_rate ... directly
# from the model's section). Only the recent-history (closeness) branch is enabled, with ``lc`` set to the dataset's lookback so it sees
# the same input window as every other model; the daily and weekly branches stay off (``daily history`` is a separate experiment).
STS_SECTION = {
    "use_closeness": True, "use_period": False, "use_trend": False, "allow_unavailable_branches": False,
    "lc": 96, "lp": 1, "lq": 1, "period_interval": 1, "trend_interval": 1, "prediction_offset": 1, "share_branch_weights": False,
    "batch_size": 2, "epochs": 20, "learning_rate": 2e-4, "weight_decay": 0.0, "gradient_clip_norm": 5.0,
    "early_stopping_patience": 20, "seed": 42,
    "model": {"input_channels": 1, "map_height": 1, "hidden_dim": 32, "num_layers": 1, "kernel_size": [3, 3],
              "output_activation": "none", "fusion_weight_shape": "per_location", "output_channels": 1},
}


def tuning_base(model: str, rep: str) -> dict[str, Any]:
    """The October tuning config for ``model``; STS-PredNet has no entry in the campaign plan, so its section is added here."""
    from .final_eval import config_file_for

    if model == "stsprednet":
        base = yaml.safe_load(Path(config_file_for("residualconvlstm", "4d")).read_text())
        base["stsprednet"] = json.loads(json.dumps(STS_SECTION))
        return base
    return yaml.safe_load(Path(config_file_for(model, rep)).read_text())


# ------------------------------------------------------------------------------------- search space
def draw(domain, rng: random.Random):
    """One sample from a registry Domain (choice / uniform / loguniform / randint)."""
    if domain.kind == "choice":
        return rng.choice(list(domain.choices))
    if domain.kind == "uniform":
        return rng.uniform(domain.lower, domain.upper)
    if domain.kind == "loguniform":
        return math.exp(rng.uniform(math.log(domain.lower), math.log(domain.upper)))
    if domain.kind == "randint":
        return rng.randrange(int(domain.lower), int(domain.upper))
    raise ValueError(f"Unsupported domain kind {domain.kind!r}")


def candidate_list(spec, budget: int, rng: random.Random) -> list[dict[str, Any]]:
    """Anchors first, then random draws, de-duplicated by content, up to ``budget`` candidates."""
    from .candidate import canonical_json

    candidates = [dict(spec.anchor(capacity)) for capacity in CAPACITIES]
    seen = {canonical_json(c) for c in candidates}
    attempts = 0
    while len(candidates) < budget and attempts < budget * 50:
        attempts += 1
        sample = {name: draw(domain, rng) for name, domain in spec.space.items()}
        key = canonical_json(sample)
        if key not in seen:
            seen.add(key)
            candidates.append(sample)
    return candidates


def pick_winner(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest validation objective among completed trials (ties broken by candidate id)."""
    done = [r for r in records if r.get("ok") and r.get("objective") is not None and r["objective"] == r["objective"]]
    if not done:
        raise ValueError("no completed trials to choose a winner from")
    return min(done, key=lambda r: (r["objective"], r["candidate_id"]))


def group_prefix(root: str, source: str, dataset: str, rep: str, model: str) -> str:
    return f"{root.strip('/')}/{source}/{dataset}/{rep}/{model}"


# --------------------------------------------------------------------------------------- data sources
def base_config(source: str, dataset: str, model: str, rep: str, repo_root: str, client, bucket: str):
    """(config, metric_fn): the model's tuning config pointed at the dataset, plus a test-metrics extractor."""
    from .final_eval import config_file_for, summarize_metrics

    base = tuning_base(model, rep)
    if source == "powder15_4d":
        from .powder15 import CORE_NODES, HORIZONS, SPATIAL_NODES, apply_powder15_4d, ensure_powder15
        from .spatial15 import GRID_DEFAULT, six_receiver_frame

        ensure_powder15(dataset, repo_root, SPATIAL_NODES)
        grid = {**GRID_DEFAULT, **six_receiver_frame(repo_root)}
        # the map name matches spatial15's default variant, so both runners share one map cache per band
        config = apply_powder15_4d(base, dataset, list(HORIZONS), CORE_NODES, grid, f"p15_{dataset}_core4-p2")

        def metrics(eval_dir: Path) -> dict[str, Any]:
            return summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records"))
        return config, metrics
    if source == "electrosense":
        from .electrosense import HORIZONS, apply_dataset, ensure_dataset, standardized_from_per_frequency

        series_path, stats = ensure_dataset(dataset, repo_root, bucket, client)
        config = apply_dataset(base, series_path, stats, dataset, list(HORIZONS))

        def metrics(eval_dir: Path) -> dict[str, Any]:
            return standardized_from_per_frequency(pd.read_csv(eval_dir / "per_frequency_metrics.csv"), stats)
    elif source == "powder15":
        from .powder15 import HORIZONS, apply_powder15, ensure_powder15

        ensure_powder15(dataset, repo_root)
        config = apply_powder15(base, dataset, repo_root, list(HORIZONS))

        def metrics(eval_dir: Path) -> dict[str, Any]:
            return summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records"))
    else:
        raise ValueError(f"unknown source {source!r}")
    return config, metrics


def prepared(config: dict[str, Any], source: str, model: str, parameters: dict[str, Any], seed: int):
    """Inject parameters, then re-apply the dataset's lookback/stride (bundles may carry the old lookback)."""
    from .config import inject_parameters

    injected = inject_parameters(config, model, parameters, seed=seed)
    if source == "electrosense":
        from .electrosense import set_lookback_and_horizon
    else:
        from .powder15 import set_lookback_and_horizon
    return set_lookback_and_horizon(injected, model, None)


def apply_fixed_config(config: dict[str, Any], model: str, fixed: dict[str, Any], seed: int) -> dict[str, Any]:
    """A prepared config for a FIXED configuration (no search): ``fixed`` is ``{"model": {...}, "train": {...}, "label": str}``.

    The architecture and optimiser settings are overlaid on the dataset's base config; the lookback, the strides and the output
    length are the dataset's, as for tuned models (a direct multi-step model emits one step per forecast step up to the longest
    horizon, e.g. 48 at 15 minutes). The seed is set last; selection by validation MAE without early stopping, as in the searches.
    """
    from .config import inject_parameters
    from .powder15 import set_lookback_and_horizon

    config = inject_parameters(config, model, {}, seed=seed)
    section = config[model]
    section["model"].update({k: v for k, v in (fixed.get("model") or {}).items()
                             if k not in ("input_sequence_length", "prediction_horizon")})
    section["train"].update({k: v for k, v in (fixed.get("train") or {}).items()
                             if k not in ("seed", "train_stride", "val_stride", "test_stride", "selection_metric", "early_stopping")})
    direct = int((fixed.get("model") or {}).get("prediction_horizon", 1)) > 1
    config = set_lookback_and_horizon(config, model, max(config["windowing"]["horizons"]) if direct else None)
    config[model]["train"]["seed"] = int(seed)
    return config


# ------------------------------------------------------------------------------------------- workers
def train_prepared(config: dict[str, Any], model: str, trial_dir: Path) -> dict[str, Any]:
    """Train a fully prepared config; return the validation objective and the checkpoint path (no Ray Tune)."""
    from training.common.run_costs import read_run_costs
    from training.common.train_integrated import train_one_model
    from training.ray.trainable import TuneReporter

    run_dir = trial_dir / "integrated"
    run_dir.mkdir(parents=True, exist_ok=True)
    reporter = TuneReporter(report=lambda metrics, checkpoint=None: None)
    train_one_model(config, model, run_dir, callback=reporter)
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise RuntimeError("training produced no checkpoint")
    reporter.run_costs = read_run_costs(run_dir)
    reporter.complete(None)
    selected = reporter.selected_metrics
    return {"objective": float(selected["objective"]), "objective_name": selected["objective_name"],
            "checkpoint": checkpoints[0], "run_costs": reporter.run_costs}


def _put(client, bucket: str, key: str, record: dict[str, Any]) -> None:
    try:
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
    except Exception:
        pass


def trial_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one candidate with the search seed and record its validation objective."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.ray.candidate import candidate_id
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, rep, source, dataset, params = job["model"], job["rep"], job["source"], job["dataset"], job["parameters"]
    cid = candidate_id(model, params)
    record = {"kind": "trial", "source": source, "dataset": dataset, "model": model, "rep": rep,
              "candidate_id": cid, "parameters": params, "seed": SEARCH_SEED, "index": job["index"]}
    started = time.time()
    try:
        config, _ = base_config(source, dataset, model, rep, args["repo_root"], client, bucket)
        config = prepared(config, source, model, params, SEARCH_SEED)
        with tempfile.TemporaryDirectory(prefix="tune-") as scratch:
            result = train_prepared(config, model, Path(scratch) / "trial")
        record.update(ok=True, objective=result["objective"], objective_name=result["objective_name"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-700:])
    record["wall_seconds"] = round(time.time() - started, 1)
    _put(client, bucket, f"{job['prefix']}/trial-{cid}.json", record)
    return record


def final_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: retrain the winning candidate with one seed and score it on the test partition."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    model, rep, source, dataset, seed = job["model"], job["rep"], job["source"], job["dataset"], int(job["seed"])
    record = {"kind": "final", "source": source, "dataset": dataset, "model": model, "rep": rep, "seed": seed,
              "candidate_id": job["candidate_id"], "parameters": job["parameters"],
              "search_objective": job["search_objective"], "hyperparameters": f"tuned ({job['budget']} candidates)"}
    if job.get("fixed"):
        record["hyperparameters"] = job["fixed"].get("label", "fixed")        # e.g. inherited:1min-r2; no search, single seed
    started = time.time()
    try:
        config, metrics = base_config(source, dataset, model, rep, args["repo_root"], client, bucket)
        config = (apply_fixed_config(config, model, job["fixed"], seed) if job.get("fixed")
                  else prepared(config, source, model, job["parameters"], seed))
        with tempfile.TemporaryDirectory(prefix="tune-final-") as scratch:
            result = train_prepared(config, model, Path(scratch) / "trial")
            eval_dir = Path(scratch) / "eval"
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, result["checkpoint"], skip_plots=True)
            record.update(metrics(eval_dir))
            costs = result.get("run_costs") or {}
            record["run_costs"] = {k: v for k, v in costs.items() if isinstance(v, (int, float, str, bool))}
            from training.ray.artifacts import save_artifact

            record.update(save_artifact(client, bucket, f"{job['prefix']}/weights/seed-{seed}", result["checkpoint"],
                                        {"config": config, "parameters": job["parameters"], "seed": seed,
                                         "candidate_id": job["candidate_id"], "run_costs": record["run_costs"],
                                         "kind": "fixed_final" if job.get("fixed") else "tuned_final",
                                         **({"hyperparameters": job["fixed"].get("label", "fixed")} if job.get("fixed") else {})}))
        record.update(ok=True, objective=result["objective"], objective_name=result["objective_name"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-700:])
    record["wall_seconds"] = round(time.time() - started, 1)
    _put(client, bucket, f"{job['prefix']}/final/seed-{seed}.json", record)
    return record


# ------------------------------------------------------------------------------------------------ driver
def read_records(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                records.append(json.loads(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()))
    return records


def parse_models(tokens: list[str]) -> list[tuple[str, str]]:
    out = []
    for token in tokens:
        model, _, rep = token.partition(":")
        if rep not in ("1d", "2d", "4d"):
            raise ValueError(f"model token {token!r} must look like model:rep (rep 1d, 2d or 4d)")
        out.append((model, rep))
    return out


def launch(args: argparse.Namespace) -> None:
    import ray

    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket}

    def options(model: str) -> dict[str, Any]:
        resources = dict(get_model_spec(model).resources.as_ray())
        gpu = args.gpu_fraction if resources.get("gpu", 0) > 0 else 0
        opts = {"num_cpus": resources["cpu"], "num_gpus": gpu, "max_retries": 1}
        if args.memory_gb:
            opts["memory"] = int(args.memory_gb * 2 ** 30)           # reserved host memory, so Ray never oversubscribes a node
        return opts

    def run_bounded(calls: list) -> list:
        """Run (remote function, options, job) calls with at most --max-parallel in flight (0 = all at once, as before)."""
        if args.max_parallel <= 0:
            return ray.get([ray.remote(**opts)(fn).remote(job, worker_args) for fn, opts, job in calls])
        queue, pending, results = list(calls), [], []
        while queue or pending:
            while queue and len(pending) < args.max_parallel:
                fn, opts, job = queue.pop(0)
                pending.append(ray.remote(**opts)(fn).remote(job, worker_args))
            done_refs, pending = ray.wait(pending, num_returns=1)
            results.append(ray.get(done_refs[0]))
            row = results[-1]
            print(f"[tune] {row.get('kind')} {row.get('dataset')} {row.get('rep')}/{row.get('model')} ok={row.get('ok')} "
                  f"obj={row.get('objective')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
        return results

    fixed = json.loads(Path(args.fixed_config).read_text()) if getattr(args, "fixed_config", None) else None
    if fixed:
        launch_fixed(args, fixed, client, bucket, options, run_bounded)
        return
    groups = []
    trial_calls = []
    for dataset in args.datasets:
        for model, rep in parse_models(args.models):
            spec = get_model_spec(model, require_executable=True)
            prefix = group_prefix(args.output_prefix, args.source, dataset, rep, model)
            trials_prefix = group_prefix(args.trials_prefix, args.source, dataset, rep, model) if args.trials_prefix else prefix
            candidates = candidate_list(spec, args.budget, random.Random(f"{args.source}|{dataset}|{rep}|{model}"))
            from .candidate import candidate_id

            done = {r.get("candidate_id") for r in read_records(client, bucket, trials_prefix)
                    if r.get("kind") == "trial" and r.get("ok")}
            groups.append({"dataset": dataset, "model": model, "rep": rep, "prefix": prefix, "trials_prefix": trials_prefix, "n": len(candidates)})
            for index, params in enumerate(candidates):
                if args.trials_prefix or candidate_id(model, params) in done:        # --trials-prefix: finals only, the search is read, never run
                    continue
                job = {"source": args.source, "dataset": dataset, "model": model, "rep": rep, "prefix": prefix,
                       "parameters": params, "index": index}
                trial_calls.append((trial_job, options(model), job))
    print(f"[tune] {len(groups)} (dataset, model) groups; {len(trial_calls)} trials to run", flush=True)
    run_bounded(trial_calls)
    if getattr(args, "trials_only", False):
        print("[tune] --trials-only: finals skipped (rerun without it, with the full --budget, to pick the winner and train the finals)",
              flush=True)
        return

    final_calls = []
    winners = []
    for group in groups:
        trials = [r for r in read_records(client, bucket, group["trials_prefix"]) if r.get("kind") == "trial"]
        try:
            winner = pick_winner(trials)
        except ValueError:
            print(f"[tune] no completed trials for {group['dataset']}/{group['model']}", flush=True)
            continue
        winners.append({**{k: group[k] for k in ("dataset", "model", "rep")}, "candidate_id": winner["candidate_id"],
                        "objective": winner["objective"], "trials_ok": sum(1 for t in trials if t.get("ok")),
                        "budget": group["n"], "parameters": json.dumps(winner["parameters"], default=str)})
        # only finals trained from the CURRENT winner count; a rerun that finds a better candidate redoes them
        finished = {r["seed"] for r in read_records(client, bucket, group["prefix"] + "/final")
                    if r.get("ok") and r.get("candidate_id") == winner["candidate_id"]}
        for seed in (args.final_seeds or FINAL_SEEDS):
            if seed in finished:
                continue
            job = {"source": args.source, "dataset": group["dataset"], "model": group["model"], "rep": group["rep"],
                   "prefix": group["prefix"], "parameters": winner["parameters"], "seed": seed,
                   "candidate_id": winner["candidate_id"], "search_objective": winner["objective"],
                   "budget": group["n"]}
            final_calls.append((final_job, options(group["model"]), job))
    print(f"[tune] {len(final_calls)} final seed runs", flush=True)
    results = run_bounded(final_calls)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    for name, rows in (("winners", winners), ("finals", results)):
        buffer = io.StringIO()
        pd.DataFrame(rows).to_csv(buffer, index=False)
        key = f"{args.output_prefix.strip('/')}/{args.source}/summary_{name}_{stamp}.csv"
        client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
        print(f"[tune] wrote s3://{bucket}/{key} ({len(rows)} rows)", flush=True)
    failed = [r for r in results if not r.get("ok")]
    print(f"[tune] done; {len(failed)} failed final runs", flush=True)


def fixed_final_calls(args, fixed: dict[str, Any], client, bucket: str, options) -> list:
    """Final-run calls for a fixed configuration: one per (dataset, model, seed) whose record is not already ok (resumable)."""
    label = fixed.get("label", "fixed")
    calls = []
    for dataset in args.datasets:
        for model, rep in parse_models(args.models):
            prefix = group_prefix(args.output_prefix, args.source, dataset, rep, model)
            done = {r["seed"] for r in read_records(client, bucket, prefix + "/final")
                    if r.get("ok") and r.get("candidate_id") == label}
            for seed in args.fixed_seeds:
                if seed in done:
                    continue
                calls.append((final_job, options(model), {
                    "source": args.source, "dataset": dataset, "model": model, "rep": rep, "prefix": prefix, "parameters": {},
                    "seed": int(seed), "candidate_id": label, "search_objective": None, "budget": 0, "fixed": fixed}))
    return calls


def launch_fixed(args, fixed: dict[str, Any], client, bucket: str, options, run_bounded) -> None:
    calls = fixed_final_calls(args, fixed, client, bucket, options)
    print(f"[tune] --fixed-config {fixed.get('label', 'fixed')}: no search; {len(calls)} final runs", flush=True)
    results = run_bounded(calls)
    failed = [r for r in results if not r.get("ok")]
    print(f"[tune] done; {len(failed)} failed final runs", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCE_DATASETS), required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="model:rep tokens")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--trials-only", action="store_true",
                        help="run the search trials only (no winner, no finals); a later run with a larger --budget resumes from these records")
    parser.add_argument("--fixed-config", default=None,
                        help="JSON {model, train, label}: skip the search and train FINAL runs with this configuration (default off)")
    parser.add_argument("--fixed-seeds", type=int, nargs="+", default=[42], help="seeds for --fixed-config runs")
    parser.add_argument("--trials-prefix", default=None,
                        help="read the finished search (trials, winner) from this output prefix and run only the finals, which go to --output-prefix "
                             "(a second cluster can train finals of an already searched winner without repeating or overwriting anything)")
    parser.add_argument("--final-seeds", type=int, nargs="+", default=None, help="seeds of the finals of a tuned winner (default: all FINAL_SEEDS)")
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    parser.add_argument("--max-parallel", type=int, default=0, help="most tasks in flight at once (0 = all at once)")
    parser.add_argument("--memory-gb", type=float, default=0.0, help="host memory to reserve per task (0 = none)")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output-prefix", default="tuning")
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args(argv)
    args.datasets = args.datasets or list(SOURCE_DATASETS[args.source])
    return args


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
