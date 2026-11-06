"""Equal-budget tuning of OFA and Time-LLM (GPT-2) on ElectroSense, matching the protocol used for every other model.

Protocol (same as ``tune_tsl.py`` and ``tune_fixed.py``): 16 candidates per (dataset, arm) -- the SpectrumLLM reproduction's
default settings plus 15 seeded random draws -- each trained with seed 42 and scored on *validation* data only (the best
per-epoch standardized validation MSE) at horizons 12 and 192; the candidate with the lowest mean of the two wins. The winner
is then trained for every horizon (12/24/48/96/192) with seeds 40-44 and scored on the test partition (raw and standardized
MSE). Test data is never used for selection. Settings follow the reproduction regime otherwise (channel-independent, tslib
70/10/20 split, lookback 192, batch 32 with micro-batches, 10 epochs, raw-unit loss, GPT-2 backbone).

The searched knobs are the ones the reproduction exposes: learning rate (log-uniform 5e-5..1e-3), backbone layers (3 or 6),
patch length (8/16/32) and stride (half or full patch), and for Time-LLM also ``d_model`` (16/32/64) and ``d_ff`` (32/64/128).
OFA has no reprogramming layer, so ``d_model`` and ``d_ff`` stay at the default for it.

    python -m training.ray.tune_llm_es --datasets 4 25 --arms ofa timellm
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEARCH_SEED, BUDGET = 42, 16
TUNE_HORIZONS = (12, 192)
HORIZONS = (12, 24, 48, 96, 192)
FINAL_SEEDS = (40, 41, 42, 43, 44)
MICRO_BATCH = {4: 8, 25: 4}
BACKBONE = "openai-community/gpt2"
SCRIPT = "training/SpectrumLLM/repro/run_paper_regime.py"
DEFAULT_PARAMS: dict[str, Any] = {"lr": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8, "d_model": 32, "d_ff": 32}


def candidate_list(arm: str, budget: int = BUDGET, seed: int = SEARCH_SEED) -> list[dict[str, Any]]:
    """Candidate 0 is the reproduction's default; the rest are distinct seeded draws."""
    rng = random.Random(f"{seed}-{arm}")
    out = [dict(DEFAULT_PARAMS)]
    seen = {tuple(sorted(out[0].items()))}
    while len(out) < budget:
        patch = rng.choice([8, 16, 32])
        params = {"lr": round(math.exp(rng.uniform(math.log(5e-5), math.log(1e-3))), 6),
                  "llm_layers": rng.choice([3, 6]), "patch_length": patch, "patch_stride": rng.choice([patch // 2, patch]),
                  "d_model": rng.choice([16, 32, 64]), "d_ff": rng.choice([32, 64, 128])}
        if arm == "ofa":
            params["d_model"], params["d_ff"] = DEFAULT_PARAMS["d_model"], DEFAULT_PARAMS["d_ff"]
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            out.append(params)
    return out


def flags(params: dict[str, Any]) -> list[str]:
    return ["--lr", str(params["lr"]), "--llm-layers", str(params["llm_layers"]), "--patch-length", str(params["patch_length"]),
            "--patch-stride", str(params["patch_stride"]), "--d-model", str(params["d_model"]), "--d-ff", str(params["d_ff"])]


def pick_winner(scores: dict[int, dict[int, float]], horizons=TUNE_HORIZONS) -> tuple[int, float] | None:
    """scores[candidate][horizon] -> (candidate, mean validation MSE) of the best candidate scored at every tuning horizon."""
    complete = {c: sum(v[h] for h in horizons) / len(horizons)
                for c, v in scores.items() if all(h in v and math.isfinite(v[h]) for h in horizons)}
    if not complete:
        return None
    best = min(complete, key=lambda c: (complete[c], c))
    return best, complete[best]


def trial_key(prefix: str, channels: int, arm: str, index: int, horizon: int) -> str:
    return f"{prefix.strip('/')}/trials/spec{channels}/{arm}/c{index:02d}_h{horizon}.json"


def final_key(prefix: str, channels: int, arm: str, horizon: int, seed: int) -> str:
    return f"{prefix.strip('/')}/finals/spec{channels}/{arm}/h{horizon}/seed-{seed}.json"


def job(spec: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one training run (a search trial or a final) of the reproduction's script."""
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.artifacts import save_artifact
    from training.ray.llm_powder import HF_HOME, PYLIBS, use_isolated_libs
    from training.ray.storage import spectrum_s3_client

    use_isolated_libs()
    client, bucket = spectrum_s3_client(), spec["bucket"]
    channels, arm, horizon, seed, params = int(spec["channels"]), spec["arm"], int(spec["horizon"]), int(spec["seed"]), spec["params"]
    record: dict[str, Any] = {"kind": spec["kind"], "channels": channels, "arm": arm, "horizon": horizon, "seed": seed,
                              "params": params, "index": spec.get("index"), "backbone": BACKBONE, "split": "tslib",
                              "hyperparameters": f"tuned ({spec.get('budget', BUDGET)} candidates)" if spec["kind"] == "final" else "search"}
    started = time.time()
    try:
        name = f"spec{channels}_data.csv"
        body = client.get_object(Bucket=bucket, Key=f"datasets/electrosense-madrid/{name}")["Body"].read()
        with tempfile.TemporaryDirectory(prefix="llm-tune-") as scratch:
            data_dir = Path(scratch) / "data"
            data_dir.mkdir()
            (data_dir / name).write_bytes(body)
            out, ckpt = Path(scratch) / "row.json", Path(scratch) / "ckpt"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": HF_HOME,
                   "PYTHONPATH": os.pathsep.join([f"{root}/models", PYLIBS, os.environ.get("PYTHONPATH", "")])}
            command = [sys.executable, "-u", SCRIPT, "--data-dir", str(data_dir), "--datasets", name, "--horizons", str(horizon),
                       "--seeds", str(seed), "--arm", arm, "--backbone", BACKBONE, "--channel-independent",
                       "--micro-batch-size", str(MICRO_BATCH[channels]), "--split", "tslib", *flags(params),
                       "--checkpoint-dir", str(ckpt) if spec["kind"] == "final" else "", "--out", str(out)]
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=spec["timeout"])
            if not out.exists():
                raise RuntimeError("training script wrote no result: " + (run.stdout + run.stderr)[-400:])
            row = json.loads(out.read_text())[-1]
            record.update({k: row.get(k) for k in ("test_mse_raw", "test_mse_std", "best_val_mse_std", "epochs_run", "train_wall_s",
                                                   "peak_gpu_mb", "references_std", "params_trainable")})
            if spec["kind"] == "final":
                for path in sorted(ckpt.rglob("*.pt")):
                    saved = save_artifact(client, bucket, f"{spec['output_prefix'].strip('/')}/weights/spec{channels}/{arm}/h{horizon}/seed-{seed}",
                                          path, {"command": command, "params": params, "seed": seed, "horizon": horizon,
                                                 "channels": channels, "arm": arm, "result_row": row}, repo_root=root)
                    record.setdefault("artifact_keys", []).append(saved.get("weights_key"))
                    if "artifact_error" in saved:
                        record["artifact_error"] = saved["artifact_error"]
        record["ok"] = bool(record.get("best_val_mse_std") is not None and math.isfinite(float(record["best_val_mse_std"])))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    key = (trial_key(spec["output_prefix"], channels, arm, int(spec["index"]), horizon) if spec["kind"] == "trial"
           else final_key(spec["output_prefix"], channels, arm, horizon, seed))
    try:
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def _done(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        record = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        return record if record.get("ok") else None
    except Exception:
        return None


def _run_all(ray, specs: list[dict[str, Any]], label: str, gpu_fraction: float) -> list[dict[str, Any]]:
    remote = ray.remote(num_cpus=4, num_gpus=gpu_fraction, max_retries=1)(job)
    pending, results = [remote.remote(s) for s in specs], []
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        results.append(r)
        print(f"[llm-tune] {label} spec{r['channels']} {r['arm']} c{r.get('index')} h{r['horizon']} seed{r['seed']} ok={r['ok']} "
              f"val={r.get('best_val_mse_std')} test={r.get('test_mse_raw')} {r.get('error', '')} ({r['wall_seconds']}s) "
              f"[{len(pending)} left]", flush=True)
    return results


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=int, nargs="+", default=[4, 25], choices=[4, 25])
    p.add_argument("--arms", nargs="+", default=["ofa", "timellm"], choices=["ofa", "timellm"])
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=6 * 3600)
    p.add_argument("--trials-only", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="llm-tuning-electrosense")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    client = spectrum_s3_client()
    base = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix, "timeout": args.timeout,
            "budget": args.budget}

    # 1. search trials (seed 42, tuning horizons, validation only)
    scores: dict[tuple[int, str], dict[int, dict[int, float]]] = {}
    todo = []
    for channels in args.datasets:
        for arm in args.arms:
            for index, params in enumerate(candidate_list(arm, args.budget)):
                for horizon in TUNE_HORIZONS:
                    record = _done(client, bucket, trial_key(args.output_prefix, channels, arm, index, horizon))
                    if record:
                        scores.setdefault((channels, arm), {}).setdefault(index, {})[horizon] = float(record["best_val_mse_std"])
                    else:
                        todo.append({**base, "kind": "trial", "channels": channels, "arm": arm, "index": index, "horizon": horizon,
                                     "seed": SEARCH_SEED, "params": params})
    print(f"[llm-tune] {len(todo)} trials to run", flush=True)
    for r in _run_all(ray, todo, "trial", args.gpu_fraction):
        if r["ok"]:
            scores.setdefault((r["channels"], r["arm"]), {}).setdefault(r["index"], {})[r["horizon"]] = float(r["best_val_mse_std"])
    if args.trials_only:
        return

    # 2. winners, written down before any test-scored run
    winners: dict[str, Any] = {}
    for (channels, arm), table in sorted(scores.items()):
        picked = pick_winner(table)
        if picked is None:
            print(f"[llm-tune] no complete candidate for spec{channels} {arm}", flush=True)
            continue
        index, value = picked
        winners[f"spec{channels}/{arm}"] = {"index": index, "mean_val_mse_std": value, "params": candidate_list(arm, args.budget)[index],
                                           "default_mean_val_mse_std": (sum(table[0][h] for h in TUNE_HORIZONS) / len(TUNE_HORIZONS)
                                                                        if all(h in table.get(0, {}) for h in TUNE_HORIZONS) else None)}
    client.put_object(Bucket=bucket, Key=f"{args.output_prefix.strip('/')}/winners.json", Body=json.dumps(winners, indent=1).encode())
    print("[llm-tune] winners:", json.dumps(winners)[:1500], flush=True)

    # 3. finals: winner x every horizon x seeds, test scored
    finals = []
    for key, win in winners.items():
        channels, arm = int(key.split("/")[0][4:]), key.split("/")[1]
        for horizon in args.horizons:
            for seed in args.seeds:
                if not _done(client, bucket, final_key(args.output_prefix, channels, arm, horizon, seed)):
                    finals.append({**base, "kind": "final", "channels": channels, "arm": arm, "index": win["index"], "horizon": horizon,
                                   "seed": seed, "params": win["params"]})
    print(f"[llm-tune] {len(finals)} finals to run", flush=True)
    _run_all(ray, finals, "final", args.gpu_fraction)


if __name__ == "__main__":
    main()
