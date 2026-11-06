"""Equal-budget hyperparameter search for Time-Series-Library baselines (PatchTST, DLinear, Informer, Autoformer, LightTS).

Same protocol as ``tune_fixed`` for our models: 16 candidates per (dataset, model) -- the library defaults first, then
random draws -- each trained with seed 42 and scored on *validation* data only (the lowest per-epoch ``Vali Loss``
the library prints, standardized MSE). TSL models are trained per forecast horizon, so a candidate's score is the mean
validation loss at a short and a long horizon (12 and 192). The winner per (dataset, model) is then trained for every
horizon with seeds 40-44 and scored on the test partition (raw and standardized MSE), exactly like the default-
hyperparameter library runs, and written under ``<prefix>/tsl/...`` with ``hyperparameters = "tuned"``.

PatchTST's patch length/stride, DLinear's ``individual`` switch, Informer's ``distil`` flag and LightTS's chunk size are
fixed inside the library's model constructors (not read from the command line), so they are not part of the search.

Records are written as they finish, so a restart skips completed work:

    python -m training.ray.tsl_runs setup
    python -m training.ray.tune_tsl --models PatchTST DLinear --output-prefix electrosense-results/tuned-tsl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from typing import Any

from .electrosense import DATASETS, HORIZONS, SEQ_LEN

SEARCH_SEED = 42
FINAL_SEEDS = (40, 41, 42, 43, 44)
TUNE_HORIZONS = (12, 192)
BUDGET = 16

# Library defaults (run.py) for the searched flags; candidate 0 is always this.
DEFAULTS = {
    "PatchTST": {"learning_rate": 1e-4, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_ff": 2048, "dropout": 0.1,
                 "batch_size": 32, "train_epochs": 10},
    "DLinear": {"learning_rate": 1e-4, "moving_avg": 25, "batch_size": 32, "train_epochs": 10},
    "Informer": {"learning_rate": 1e-4, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 2048,
                 "factor": 1, "dropout": 0.1, "batch_size": 32, "train_epochs": 10},
    "Autoformer": {"learning_rate": 1e-4, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 2048,
                   "factor": 1, "moving_avg": 25, "dropout": 0.1, "batch_size": 32, "train_epochs": 10},
    "LightTS": {"learning_rate": 1e-4, "d_model": 512, "dropout": 0.1, "batch_size": 32, "train_epochs": 10},
}
# ("choice", [..]) | ("loguniform", lo, hi) | ("uniform", lo, hi)
SPACES = {
    "PatchTST": {"learning_rate": ("loguniform", 1e-4, 3e-3), "d_model": ("choice", [32, 64, 128, 256, 512]),
                 "n_heads": ("choice", [4, 8]), "e_layers": ("choice", [1, 2, 3]),
                 "d_ff": ("choice", [128, 256, 512, 1024, 2048]), "dropout": ("uniform", 0.0, 0.3),
                 "batch_size": ("choice", [16, 32, 64]), "train_epochs": ("choice", [10, 20])},
    "DLinear": {"learning_rate": ("loguniform", 1e-4, 1e-2), "moving_avg": ("choice", [5, 13, 25, 49, 97]),
                "batch_size": ("choice", [16, 32, 64]), "train_epochs": ("choice", [10, 20])},
    "Informer": {"learning_rate": ("loguniform", 1e-4, 3e-3), "d_model": ("choice", [64, 128, 256, 512]),
                 "n_heads": ("choice", [4, 8]), "e_layers": ("choice", [1, 2, 3]), "d_layers": ("choice", [1, 2]),
                 "d_ff": ("choice", [256, 512, 1024, 2048]), "factor": ("choice", [1, 3, 5]),
                 "dropout": ("uniform", 0.0, 0.3), "batch_size": ("choice", [16, 32, 64]),
                 "train_epochs": ("choice", [10, 20])},
    "Autoformer": {"learning_rate": ("loguniform", 1e-4, 3e-3), "d_model": ("choice", [64, 128, 256, 512]),
                   "n_heads": ("choice", [4, 8]), "e_layers": ("choice", [1, 2, 3]), "d_layers": ("choice", [1, 2]),
                   "d_ff": ("choice", [256, 512, 1024, 2048]), "factor": ("choice", [1, 3, 5]),
                   "moving_avg": ("choice", [13, 25, 49]), "dropout": ("uniform", 0.0, 0.3),
                   "batch_size": ("choice", [16, 32, 64]), "train_epochs": ("choice", [10, 20])},
    # LightTS reads only d_model and dropout from the command line (its chunk size is fixed in the constructor)
    "LightTS": {"learning_rate": ("loguniform", 1e-4, 3e-3), "d_model": ("choice", [64, 128, 256, 512]),
                "dropout": ("uniform", 0.0, 0.3), "batch_size": ("choice", [16, 32, 64]),
                "train_epochs": ("choice", [10, 20])},
}
CHANNELS = {"spec4": 4, "spec25": 25, "spec151": 151}


def _draw(domain, rng: random.Random):
    kind = domain[0]
    if kind == "choice":
        return rng.choice(domain[1])
    if kind == "uniform":
        return round(rng.uniform(domain[1], domain[2]), 4)
    if kind == "loguniform":
        return float(f"{math.exp(rng.uniform(math.log(domain[1]), math.log(domain[2]))):.3g}")
    raise ValueError(kind)


def candidate_list(model: str, budget: int = BUDGET, seed: int = SEARCH_SEED) -> list[dict[str, Any]]:
    """Library defaults first, then distinct random draws until ``budget`` candidates."""
    rng = random.Random(f"{seed}-{model}")
    candidates = [dict(DEFAULTS[model])]
    seen = {json.dumps(candidates[0], sort_keys=True)}
    while len(candidates) < budget:
        draw = {name: _draw(domain, rng) for name, domain in SPACES[model].items()}
        key = json.dumps(draw, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(draw)
    return candidates


def flags(params: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for name, value in params.items():
        out += [f"--{name}", str(value)]
    return out


def pick_winner(scores: dict[int, dict[int, float]], horizons=TUNE_HORIZONS) -> tuple[int, float] | None:
    """scores[candidate][horizon] -> (candidate index, mean validation loss) of the best fully scored candidate."""
    complete = {c: sum(v[h] for h in horizons) / len(horizons)
                for c, v in scores.items() if all(h in v and math.isfinite(v[h]) for h in horizons)}
    if not complete:
        return None
    best = min(complete, key=lambda c: (complete[c], c))
    return best, complete[best]


def trial_key(prefix: str, dataset: str, model: str, index: int, horizon: int) -> str:
    return f"{prefix.strip('/')}/trials/{dataset}/{model}/c{index:02d}_h{horizon}.json"


def final_key(prefix: str, dataset: str, model: str, horizon: int, seed: int) -> str:
    return f"{prefix.strip('/')}/tsl/{dataset}/{model}/h{horizon}/seed-{seed}.json"


def trial_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one candidate at one horizon (seed 42), record its best validation loss."""
    import sys

    sys.path.insert(0, args["repo_root"])
    from training.ray.storage import spectrum_s3_client
    from training.ray.tsl_runs import TSL_DIR, parse_best_validation, run_command
    from .electrosense import DATASETS as FILES

    dataset, model, index, horizon = job["dataset"], job["model"], int(job["index"]), int(job["horizon"])
    record = {"dataset": dataset, "model": model, "index": index, "horizon": horizon, "params": job["params"],
              "kind": "tsl_trial"}
    started = time.time()
    try:
        root = f"{args['repo_root']}/data/electrosense_raw/"
        target = open(root + FILES[dataset]).readline().strip().split(",")[-1]
        command = run_command(model, dataset, horizon, SEARCH_SEED, root, target, training=True, inverse=False,
                              has_seed_flag=True, extra=flags(job["params"]) + (["--no_use_gpu"] if args.get("cpu") else []),
                              des=f"tune-{model}-c{index}")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if args.get("cpu"):                                  # one run per reserved-core group: do not oversubscribe the node
            env.update(OMP_NUM_THREADS=str(args["cpus"]), MKL_NUM_THREADS=str(args["cpus"]))
        run = subprocess.run(command, cwd=f"{TSL_DIR}/Time-Series-Library", capture_output=True, text=True,
                             env=env, timeout=args["timeout"])
        best = parse_best_validation(run.stdout + run.stderr)
        if best is None:
            raise RuntimeError("no validation loss in output: " + (run.stdout + run.stderr)[-300:])
        record.update(vali_loss=best, ok=True)
    except Exception as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400])
    record["wall_seconds"] = round(time.time() - started, 1)
    if record["ok"]:
        client = spectrum_s3_client()
        client.put_object(Bucket=args["bucket"], Key=trial_key(args["output_prefix"], dataset, model, index, horizon),
                          Body=json.dumps(record).encode())
    return record


def _exists(client, bucket: str, key: str, need_ok: bool = False) -> dict | None:
    try:
        body = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None
    return body if (body.get("ok") or not need_ok) else None


def run_bounded(ray, remote, jobs: list[dict[str, Any]], base: dict[str, Any], limit: int | None):
    """Yield (job, result) as runs finish, keeping at most ``limit`` runs in flight (None = submit everything at once)."""
    queue, inflight = list(jobs), {}
    while queue or inflight:
        while queue and (limit is None or len(inflight) < limit):
            job = queue.pop(0)
            inflight[remote.remote(job, base)] = job
        done, _ = ray.wait(list(inflight), num_returns=1)
        ref = done[0]
        yield inflight.pop(ref), ray.get(ref)


def launch(args: argparse.Namespace) -> None:
    import ray

    from .storage import spectrum_s3_client
    from .tsl_runs import tsl_job

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    base = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
            "timeout": args.timeout, "has_seed_flag": True, "cpu": args.cpu, "cpus": args.cpus}
    gpus = 0 if args.cpu else args.gpu_fraction
    trial_remote = ray.remote(num_cpus=args.cpus, num_gpus=gpus)(trial_job)
    final_remote = ray.remote(num_cpus=args.cpus, num_gpus=gpus)(tsl_job)

    candidates = {m: candidate_list(m, args.budget) for m in args.models}
    # 1. trials (skipping finished ones; failed ones retried, e.g. GPU memory contention)
    scores: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    todo = []
    for d in args.datasets:
        for m in args.models:
            scores[(d, m)] = {}
            for i, params in enumerate(candidates[m]):
                for h in TUNE_HORIZONS:
                    done = _exists(client, bucket, trial_key(args.output_prefix, d, m, i, h), need_ok=True)
                    if done:
                        scores[(d, m)].setdefault(i, {})[h] = float(done["vali_loss"])
                    else:
                        todo.append({"dataset": d, "model": m, "index": i, "horizon": h, "params": params})
    print(f"[tune-tsl] {len(todo)} trials to run", flush=True)
    for attempt in range(1, args.retries + 1):
        if not todo:
            break
        failed = []
        for job, r in run_bounded(ray, trial_remote, todo, base, args.max_parallel):
            if r["ok"]:
                scores[(r["dataset"], r["model"])].setdefault(r["index"], {})[r["horizon"]] = float(r["vali_loss"])
            else:
                failed.append(job)
                print(f"[tune-tsl] trial failed (attempt {attempt}): {r['dataset']}/{r['model']} c{r['index']} "
                      f"h{r['horizon']}: {r['error'][:120]}", flush=True)
        todo = failed
    # 2. winners
    winners = {}
    for (d, m), table in scores.items():
        won = pick_winner(table)
        if won is None:
            print(f"[tune-tsl] no complete candidate for {d}/{m}; skipping finals", flush=True)
            continue
        winners[(d, m)] = {"index": won[0], "vali_loss": won[1], "params": candidates[m][won[0]]}
        print(f"[tune-tsl] winner {d}/{m}: c{won[0]} val={won[1]:.4f} {candidates[m][won[0]]}", flush=True)
    client.put_object(Bucket=bucket, Key=f"{args.output_prefix.strip('/')}/winners.json",
                      Body=json.dumps({f"{d}/{m}": v for (d, m), v in winners.items()}).encode())
    if args.trials_only:
        return
    # 3. finals: winner x every horizon x seeds 40-44 (test scored, raw + standardized)
    jobs = []
    for (d, m), win in winners.items():
        for h in args.horizons:
            for s in FINAL_SEEDS:
                if not _exists(client, bucket, final_key(args.output_prefix, d, m, h, s), need_ok=True):
                    jobs.append({"model": m, "dataset": d, "horizon": h, "seed": s,
                                 "extra": flags(win["params"]) + (["--no_use_gpu"] if args.cpu else []),
                                 "threads": args.cpus if args.cpu else None,
                                 "des": f"tuned-{m}", "hp_label": "tuned", "params": win["params"]})
    print(f"[tune-tsl] {len(jobs)} final runs", flush=True)
    for attempt in range(1, args.retries + 1):
        if not jobs:
            break
        failed = [job for job, r in run_bounded(ray, final_remote, jobs, base, args.max_parallel) if not r.get("ok")]
        print(f"[tune-tsl] finals attempt {attempt}: {len(failed)} failed", flush=True)
        jobs = failed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["PatchTST", "DLinear"], choices=list(DEFAULTS))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--cpu", action="store_true", help="train on CPU (library --no_use_gpu) and reserve no GPU")
    p.add_argument("--cpus", type=int, default=2, help="CPU cores reserved per run (more helps with --cpu)")
    p.add_argument("--max-parallel", type=int, default=None, help="cap on runs in flight (default: all at once)")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--trials-only", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="electrosense-results/tuned-tsl")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return p.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
