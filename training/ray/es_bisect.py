"""Bisect the ElectroSense pipeline gap inside the collaborator's script (``run_paper_regime.py``, as called by ``tune_llm_es``).

That script differs from ``tune_direct``'s ElectroSense mode in the units the model is trained and scored in (``--space raw``: raw PSD
input and loss; ours: z-scored) and in which windows are scored (``--split tslib``: the test span starts ``seq_len`` rows early, so about
twice as many windows; ours keeps inputs and targets inside the test partition). Running the same OFA settings with one switch changed
at a time shows which difference moves the number from 0.58 / 1.03 (spec25, raw MSE, horizons 12 / 192) toward ``tune_direct``'s
0.79 / 1.44. One Ray task per (variant, horizon, seed); results are printed and written to the durable bucket.

    python -m training.ray.es_bisect --variants baseline std_space literal_split
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

OFA_PARAMS = {"lr": 1.59e-4, "llm_layers": 3, "patch_length": 32, "patch_stride": 32, "d_model": 32, "d_ff": 32}
VARIANTS = {"baseline": [], "std_space": ["--space", "std"], "literal_split": ["--split", "literal"],
            "std_literal": ["--space", "std", "--split", "literal"]}


def run_one(spec: dict) -> dict:
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.llm_powder import HF_HOME, PYLIBS, use_isolated_libs
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_llm_es import BACKBONE, MICRO_BATCH, SCRIPT, flags

    use_isolated_libs()
    client, bucket = spectrum_s3_client(), spec["bucket"]
    channels, horizon, seed, variant = int(spec["channels"]), int(spec["horizon"]), int(spec["seed"]), spec["variant"]
    record = {"variant": variant, "channels": channels, "horizon": horizon, "seed": seed, "params": OFA_PARAMS}
    started = time.time()
    try:
        name = f"spec{channels}_data.csv"
        body = client.get_object(Bucket=bucket, Key=f"datasets/electrosense-madrid/{name}")["Body"].read()
        with tempfile.TemporaryDirectory(prefix="es-bisect-") as scratch:
            data_dir = Path(scratch) / "data"
            data_dir.mkdir()
            (data_dir / name).write_bytes(body)
            out = Path(scratch) / "row.json"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": HF_HOME,
                   "PYTHONPATH": os.pathsep.join([f"{root}/models", PYLIBS, os.environ.get("PYTHONPATH", "")])}
            base = ["--split", "tslib"]
            extra = VARIANTS[variant]
            command = [sys.executable, "-u", SCRIPT, "--data-dir", str(data_dir), "--datasets", name, "--horizons", str(horizon),
                       "--seeds", str(seed), "--arm", "ofa", "--backbone", BACKBONE, "--channel-independent",
                       "--micro-batch-size", str(MICRO_BATCH[channels]), *base, *flags(OFA_PARAMS), "--checkpoint-dir", "",
                       "--out", str(out), *extra]
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=7200)
            if not out.exists():
                raise RuntimeError("no result: " + (run.stdout + run.stderr)[-400:])
            row = json.loads(out.read_text())[-1]
            record.update({k: row.get(k) for k in ("test_mse_raw", "test_mse_std", "test_mse_raw_endpoint", "test_mse_std_endpoint", "best_val_mse_std", "epochs_run", "train_wall_s")})
            record["command_extra"] = extra
        record["ok"] = True
    except Exception as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{spec['prefix']}/{variant}/spec{channels}/h{horizon}-seed-{seed}.json",
                      Body=json.dumps(record).encode())
    return record


def main() -> None:
    import ray

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--channels", type=int, default=25)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 192])
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    parser.add_argument("--prefix", default="es-bisect")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args()
    ray.init(address=args.address, ignore_reinit_error=True)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    refs = [ray.remote(num_cpus=4, num_gpus=0.25, max_retries=1)(run_one).remote(
                {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "variant": v, "channels": args.channels,
                 "horizon": h, "seed": s}) for v in args.variants for h in args.horizons for s in args.seeds]
    rows = ray.get(refs)
    for v in args.variants:
        for h in args.horizons:
            ok = [r for r in rows if r["variant"] == v and r["horizon"] == h and r.get("ok")]
            if ok:
                end = [r["test_mse_raw_endpoint"] for r in ok if r.get("test_mse_raw_endpoint") is not None]
                print(f"[es-bisect] {v:14s} h{h:<3d} n={len(ok)} window-averaged raw {sum(r['test_mse_raw'] for r in ok) / len(ok):.3f} "
                      f"std {sum(r['test_mse_std'] for r in ok) / len(ok):.3f}"
                      + (f" | endpoint-only raw {sum(end) / len(end):.3f}" if end else ""), flush=True)
            else:
                print(f"[es-bisect] {v:14s} h{h:<3d} FAILED {[r.get('error') for r in rows if r['variant'] == v and r['horizon'] == h][:1]}", flush=True)


if __name__ == "__main__":
    main()
