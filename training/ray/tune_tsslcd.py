"""Equal-budget search for the TSS-LCD authors' model on POWDER 600-800 MHz (1 minute), then five tuned final seeds.

The model is the authors' released code (``models/tsslcd_authors``, wrapped by ``training.ray.tsslcd_authors``); nothing inside their
classes is changed. Only their constructor and optimiser settings are searched, at their full schedule (300/300/2000 epochs), so every
trial and every final costs the same:

    latent_dim {16, 32, 48, 64}, learning rate loguniform(3e-5, 3e-4) (all three stages), batch size {64, 128, 256},
    training-time context masking {0, 0.1, 0.2}.

Candidate 0 is the authors' defaults (32, 1e-4, 256, 0.1); the rest are random draws from an RNG seeded by the group name, so
``--budget 8`` is a prefix of ``--budget 16`` and a later larger-budget run reuses every finished trial.

Protocol: savgol off (the benchmark's raw series; see CAMPAIGN 22.2), search seed 42, objective = raw-target mean-over-horizons MAE
(t+1, t+15, t+60) on the validation partition (2026-10-25 13:55 to 10-27 13:55), no eval-time smoothing. Finals retrain the winner with
seeds 40-44 and score the test range with the same three eval-smoothing settings as the untuned runs.

    python -m training.ray.tune_tsslcd --budget 8 --trials-only      # round 1
    python -m training.ray.tune_tsslcd --budget 16                   # round 2 + winner + finals
    python -m training.ray.tune_tsslcd --resolution 15min --budget 16 --gpu-fraction 1.0   # same search at 15 minutes (benchmark windowing)

``--resolution 15min`` runs the identical search space and candidate draws on the 15-minute profile of ``training.ray.tsslcd_authors``
(lookback 100, future 48, horizons 1/4/12/24/48; objective = validation raw-target mean-horizon MAE, no smoothing; finals scored with
eval smoothing off), under separate prefixes (``tuning/tsslcd_authors/15min/600_800``, ``tsslcd-authors-15min-tuned``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any

BAND = "600_800"
SEARCH_SEED = 42
FINAL_SEEDS = (40, 41, 42, 43, 44)
DEFAULTS = {"latent_dim": 32, "learning_rate": 1e-4, "batch_size": 256, "train_mask_ratio": 0.1}
SPACE = {"latent_dim": ("choice", (16, 32, 48, 64)), "learning_rate": ("loguniform", (3e-5, 3e-4)),
         "batch_size": ("choice", (64, 128, 256)), "train_mask_ratio": ("choice", (0.0, 0.1, 0.2))}


def candidate_id(params: dict[str, Any]) -> str:
    blob = json.dumps({k: (round(v, 12) if isinstance(v, float) else v) for k, v in sorted(params.items())}, sort_keys=True)
    return "tsslcd-" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def draw(rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (kind, domain) in SPACE.items():
        out[name] = rng.choice(list(domain)) if kind == "choice" else math.exp(rng.uniform(math.log(domain[0]), math.log(domain[1])))
    return out


def candidate_list(budget: int) -> list[dict[str, Any]]:
    """The authors' defaults first, then random draws, de-duplicated, up to ``budget``."""
    rng = random.Random(f"tsslcd_authors|{BAND}")
    candidates, seen, attempts = [dict(DEFAULTS)], {candidate_id(DEFAULTS)}, 0
    while len(candidates) < budget and attempts < budget * 50:
        attempts += 1
        sample = draw(rng)
        if candidate_id(sample) not in seen:
            seen.add(candidate_id(sample))
            candidates.append(sample)
    return candidates


def pick_winner(records: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in records if r.get("ok") and r.get("objective") is not None and r["objective"] == r["objective"]]
    if not done:
        raise ValueError("no completed trials")
    return min(done, key=lambda r: (r["objective"], r["candidate_id"]))


def trial_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one candidate at the full schedule, scored on the validation partition."""
    os.chdir(spec["repo_root"])
    sys.path.insert(0, spec["repo_root"])
    from training.ray.storage import spectrum_s3_client
    from training.ray.tsslcd_authors import run_one

    params = spec["parameters"]
    record: dict[str, Any] = {"kind": "trial", "candidate_id": candidate_id(params), "parameters": params, "seed": SEARCH_SEED,
                              "index": spec["index"]}
    started = time.time()
    try:
        result = run_one({**spec["run"], "seed": SEARCH_SEED, "hparams": params, "score_partition": "validation",
                          "save_weights": False})
        record.update(ok=True, objective=result["objective"], objective_name="validation raw-target mean-horizon MAE (dB)",
                      metrics={k: v for k, v in result["metrics"].items() if "raw_target__off__" in k},
                      stages={k: v for k, v in result.items() if k.startswith("stage_")})
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    spectrum_s3_client().put_object(Bucket=spec["bucket"], Key=f"{spec['prefix']}/trial-{record['candidate_id']}.json",
                                    Body=json.dumps(record, default=str).encode())
    return record


def read_trials(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    out = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/trial-"):
        for obj in page.get("Contents", []):
            out.append(json.loads(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()))
    return out


def run_bounded(ray, fn, specs: list[dict[str, Any]], options: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    remote = ray.remote(**options)(fn)
    queue, pending, results = list(specs), [], []
    while queue or pending:
        while queue and (limit <= 0 or len(pending) < limit):
            pending.append(remote.remote(queue.pop(0)))
        done, pending = ray.wait(pending, num_returns=1)
        row = ray.get(done[0])
        results.append(row)
        print(f"[tune-tsslcd] {row.get('candidate_id') or row.get('seed')} ok={row.get('ok')} "
              f"obj={row.get('objective')} {row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    return results


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", choices=("1min", "15min"), default="1min")
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--trials-only", action="store_true", help="trials only: no winner, no finals (a later larger --budget resumes)")
    p.add_argument("--skip-trials", action="store_true",
                   help="do not run or wait for trials; pick the winner from the finished ones (a later run without this flag picks up "
                        "late trials and redoes the finals if the winner changes)")
    p.add_argument("--from-index", type=int, default=0,
                   help="run only candidates from this index on (a second cluster takes the other half of the budget while the first is in flight)")
    p.add_argument("--epochs-ctx", type=int, default=300)
    p.add_argument("--epochs-fut", type=int, default=300)
    p.add_argument("--epochs-diff", type=int, default=2000)
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--max-parallel", type=int, default=0)
    p.add_argument("--memory-gb", type=float, default=0.0)
    p.add_argument("--bucket", default=None)
    p.add_argument("--tuning-prefix", default=None, help=f"default tuning/tsslcd_authors/{BAND} (1min) or tuning/tsslcd_authors/15min/{BAND}")
    p.add_argument("--output-prefix", default=None, help="finals' result prefix (default tsslcd-authors-tuned, or tsslcd-authors-15min-tuned)")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    fifteen = args.resolution == "15min"
    args.tuning_prefix = args.tuning_prefix or (f"tuning/tsslcd_authors/15min/{BAND}" if fifteen else f"tuning/tsslcd_authors/{BAND}")
    args.output_prefix = args.output_prefix or ("tsslcd-authors-15min-tuned" if fifteen else "tsslcd-authors-tuned")
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client
    from training.ray.tsslcd_authors import DEFAULT_EVAL_SMOOTH, job, parse_eval_smooth

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    ray.init(address=args.address, ignore_reinit_error=True)
    options = {"num_cpus": 4, "num_gpus": args.gpu_fraction, "max_retries": 1}
    if args.memory_gb:
        options["memory"] = int(args.memory_gb * 2 ** 30)
    epochs = {"ctx": args.epochs_ctx, "fut": args.epochs_fut, "diff": args.epochs_diff}
    base = {"repo_root": args.repo_root, "bucket": bucket, "device": None, "epochs": epochs, "savgol": "off",
            "test_stride": 1, "eval_mask_ratio": 0.0, "output_prefix": args.output_prefix, "resolution": args.resolution}
    finals_smooth = ("1:1:centered",) if fifteen else DEFAULT_EVAL_SMOOTH

    candidates = candidate_list(args.budget)
    done = {r["candidate_id"] for r in read_trials(client, bucket, args.tuning_prefix) if r.get("ok")}
    trials = [{"prefix": args.tuning_prefix, "bucket": bucket, "repo_root": args.repo_root, "index": i, "parameters": c,
               "run": {**base, "eval_smooth": [parse_eval_smooth("1:1:centered")]}}
              for i, c in enumerate(candidates) if i >= args.from_index and candidate_id(c) not in done]
    print(f"[tune-tsslcd] {len(candidates)} candidates, {len(trials)} trials to run", flush=True)
    if not args.skip_trials:
        run_bounded(ray, trial_job, trials, options, args.max_parallel)
    if args.trials_only:
        print("[tune-tsslcd] --trials-only: finals skipped", flush=True)
        return

    winner = pick_winner(read_trials(client, bucket, args.tuning_prefix))
    print(f"[tune-tsslcd] winner {winner['candidate_id']} objective={winner['objective']:.4f} {winner['parameters']}", flush=True)
    finals = []
    for seed in FINAL_SEEDS:
        key = f"{args.output_prefix}/savgol-off/seed-{seed}.json"
        try:
            record = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
            if record.get("ok") and record.get("hparams", {}).get("latent_dim") == winner["parameters"]["latent_dim"] and \
                    abs(record["hparams"]["learning_rate"] - winner["parameters"]["learning_rate"]) < 1e-12 and \
                    record["hparams"]["batch_size"] == winner["parameters"]["batch_size"] and \
                    abs(record["hparams"]["train_mask_ratio"] - winner["parameters"]["train_mask_ratio"]) < 1e-12:
                continue
        except Exception:
            pass
        finals.append({**base, "seed": seed, "hparams": winner["parameters"], "score_partition": "test", "save_weights": True,
                       "eval_smooth": [parse_eval_smooth(s) for s in finals_smooth]})
    print(f"[tune-tsslcd] {len(finals)} final seed runs", flush=True)
    run_bounded(ray, job, finals, options, args.max_parallel)


if __name__ == "__main__":
    main()
