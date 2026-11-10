"""Untuned multi-seed reproduction of OFA and Time-LLM on ElectroSense (the article's regime, Time-Series-Library split).

The appendix table that validates the reconstruction against the published numbers has single-seed (seed 0) Time-LLM and OFA rows, and
the reproduction README warns that single-seed cells can swing by 20%. This runs the same script with the same default (untuned)
settings for several seeds: OFA (GPT-2), Time-LLM (GPT-2) and Time-LLM (BERT), spec4 and spec25, a few horizons. Seed 0 is included, so
its values can be checked against the table's. The reported error is the window-averaged raw MSE (the article's convention); the
endpoint-only value of the same predictions is stored too. The data come from the durable bucket (no local copy needed).

    python -m training.ray.es_repro_seeds --arms ofa_gpt2 timellm_gpt2 timellm_bert --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ARMS = {"ofa_gpt2": ("ofa", "openai-community/gpt2"), "timellm_gpt2": ("timellm", "openai-community/gpt2"),
        "timellm_bert": ("timellm", "google-bert/bert-base-uncased")}
GPU_SHARE = {"ofa_gpt2": 0.125, "timellm_gpt2": 0.5, "timellm_bert": 0.5}


def run_one(spec: dict) -> dict:
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.llm_powder import HF_HOME, PYLIBS, use_isolated_libs
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_llm_es import DEFAULT_PARAMS, MICRO_BATCH, SCRIPT, flags

    use_isolated_libs()
    client, bucket = spectrum_s3_client(), spec["bucket"]
    arm, channels, horizon, seed = spec["arm"], int(spec["channels"]), int(spec["horizon"]), int(spec["seed"])
    model_arm, backbone = ARMS[arm]
    record = {"arm": arm, "channels": channels, "horizon": horizon, "seed": seed, "backbone": backbone, "params": DEFAULT_PARAMS,
              "split": "tslib", "regime": "paper defaults, untuned"}
    started = time.time()
    try:
        name = f"spec{channels}_data.csv"
        body = client.get_object(Bucket=bucket, Key=f"datasets/electrosense-madrid/{name}")["Body"].read()
        with tempfile.TemporaryDirectory(prefix="es-repro-") as scratch:
            data_dir = Path(scratch) / "data"
            data_dir.mkdir()
            (data_dir / name).write_bytes(body)
            out = Path(scratch) / "row.json"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": HF_HOME,
                   "PYTHONPATH": os.pathsep.join([f"{root}/models", PYLIBS, os.environ.get("PYTHONPATH", "")])}
            command = [sys.executable, "-u", SCRIPT, "--data-dir", str(data_dir), "--datasets", name, "--horizons", str(horizon),
                       "--seeds", str(seed), "--arm", model_arm, "--backbone", backbone, "--channel-independent",
                       "--micro-batch-size", str(MICRO_BATCH[channels]), "--split", "tslib", *flags(DEFAULT_PARAMS),
                       "--checkpoint-dir", "", "--out", str(out)]
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=10800)
            if not out.exists():
                raise RuntimeError("no result: " + (run.stdout + run.stderr)[-400:])
            row = json.loads(out.read_text())[-1]
            record.update({k: row.get(k) for k in ("test_mse_raw", "test_mse_std", "test_mse_raw_endpoint", "test_mse_std_endpoint",
                                                   "best_val_mse_std", "epochs_run", "train_wall_s", "peak_gpu_mb")})
        record["ok"] = True
    except Exception as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{spec['prefix']}/{arm}/spec{channels}/h{horizon}-seed-{seed}.json", Body=json.dumps(record).encode())
    return record


def main() -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--channels", type=int, nargs="+", default=[4, 25])
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 48, 192])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--prefix", default="es-repro-seeds")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = [{"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "arm": a, "channels": c, "horizon": h, "seed": s}
             for a in args.arms for c in args.channels for h in args.horizons for s in args.seeds]
    if args.skip_done:
        def done(spec):
            try:
                key = f"{spec['prefix']}/{spec['arm']}/spec{spec['channels']}/h{spec['horizon']}-seed-{spec['seed']}.json"
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        specs = [s for s in specs if not done(s)]
    specs.sort(key=lambda s: (-(s["channels"] if s["arm"] != "ofa_gpt2" else 0), s["arm"], s["horizon"], s["seed"]))   # heaviest first
    print(f"[es-repro] {len(specs)} runs", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    queue, pending, rows = list(specs), [], []
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            spec = queue.pop(0)
            pending.append(ray.remote(num_cpus=4, num_gpus=GPU_SHARE[spec["arm"]], memory=int(10 * 2**30), max_retries=1)(run_one).remote(spec))
        done, pending = ray.wait(pending, num_returns=1)
        rows.append(ray.get(done[0]))
        r = rows[-1]
        print(f"[es-repro] {r['arm']} {r['channels']}ch h{r['horizon']} seed{r['seed']} ok={r['ok']} window={r.get('test_mse_raw')} "
              f"endpoint={r.get('test_mse_raw_endpoint')} {r.get('error', '')} ({r['wall_seconds']}s) [{len(queue) + len(pending)} left]", flush=True)


if __name__ == "__main__":
    main()
