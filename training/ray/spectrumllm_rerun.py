"""Rerun Time-LLM (BERT backbone) at horizon 192 on ElectroSense to measure and fix its collapse.

``training/ray/spectrumllm_diagnose.py`` found that the BERT runs that "diverged" at K=192 had collapsed to the lookback mean
(test MSE equal to the lookback-mean MSE, error spread evenly over windows; not a padding artifact). Whether a run collapsed
is visible on validation data alone, so the fix can be chosen without touching the test set. This runner trains the
collaborator's port (``training/SpectrumLLM/repro/run_paper_regime.py``, unchanged) for several seeds and learning rates and
records, next to the usual metrics, the validation lookback-mean MSE (``val_lookback_mean_mse_std``, train-standardized units,
same windows and units as ``best_val_mse_std``). A run is flagged ``collapsed`` when its best validation MSE is not at least
3% below that baseline.

Settings follow the reproduction regime (channel-independent, tslib 70/10/20 split, lookback 192, batch 32 with micro-batches
of 8 at 4 channels and 4 at 25, 10 epochs, raw-unit loss, ``google-bert/bert-base-uncased``). Only ``--lrs`` and the seeds vary.

    python -m training.ray.spectrumllm_rerun --channels 4 --seeds 1 2 3 4 5 --lrs 1e-4 3e-4
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEQ, HORIZON = 192, 192
BACKBONE = "google-bert/bert-base-uncased"
MICRO_BATCH = {4: 8, 25: 4}
COLLAPSE_MARGIN = 0.03
SCRIPT = "training/SpectrumLLM/repro/run_paper_regime.py"


def val_lookback_mean_mse_std(raw: np.ndarray, horizon: int = HORIZON, seq: int = SEQ) -> float:
    """Lookback-mean MSE on the tslib validation windows, in train-standardized units (matches ``best_val_mse_std``)."""
    n = len(raw)
    n_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - n_train - num_test
    starts = range(n_train - seq, n_train + num_vali - seq - horizon + 1)
    sd = raw[:n_train].std(axis=0)
    err = 0.0
    for s in starts:
        history = raw[s:s + seq]
        truth = raw[s + seq:s + seq + horizon]
        err += float((((history.mean(axis=0, keepdims=True) - truth) / sd) ** 2).mean())
    return err / len(starts)


def is_collapsed(best_val_mse_std: float, val_lookback_mean_mse_std_: float) -> bool:
    """True when validation is not at least ``COLLAPSE_MARGIN`` better than predicting the lookback mean."""
    return best_val_mse_std > (1.0 - COLLAPSE_MARGIN) * val_lookback_mean_mse_std_


def result_key(prefix: str, lr: str, channels: int, seed: int) -> str:
    return f"{prefix.strip('/')}/lr{lr}/n{channels}/seed-{seed}.json"


def job(spec: dict[str, Any]) -> dict[str, Any]:
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.llm_powder import HF_HOME, PYLIBS, use_isolated_libs
    from training.ray.storage import spectrum_s3_client

    use_isolated_libs()
    client, bucket = spectrum_s3_client(), spec["bucket"]
    channels, lr, seed = int(spec["channels"]), spec["lr"], int(spec["seed"])
    record: dict[str, Any] = {"kind": "spectrumllm_rerun", "channels": channels, "lr": float(lr), "seed": seed,
                              "horizon": HORIZON, "backbone": BACKBONE, "split": "tslib"}
    started = time.time()
    try:
        name = f"spec{channels}_data.csv"
        body = client.get_object(Bucket=bucket, Key=f"datasets/electrosense-madrid/{name}")["Body"].read()
        frame = pd.read_csv(io.BytesIO(body))
        record["val_lookback_mean_mse_std"] = val_lookback_mean_mse_std(frame.drop(columns=["date"]).to_numpy(np.float32))
        with tempfile.TemporaryDirectory(prefix="bert-rerun-") as scratch:
            data_dir = Path(scratch) / "data"
            data_dir.mkdir()
            (data_dir / name).write_bytes(body)
            out = Path(scratch) / "row.json"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": HF_HOME,
                   "PYTHONPATH": os.pathsep.join([f"{root}/models", PYLIBS, os.environ.get("PYTHONPATH", "")])}
            command = [sys.executable, "-u", SCRIPT, "--data-dir", str(data_dir), "--datasets", name,
                       "--horizons", str(HORIZON), "--seeds", str(seed), "--arm", "timellm", "--backbone", BACKBONE,
                       "--channel-independent", "--micro-batch-size", str(MICRO_BATCH[channels]), "--split", "tslib",
                       "--lr", str(lr), "--checkpoint-dir", str(Path(scratch) / "ckpt"), "--out", str(out)]
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=spec["timeout"])
            if not out.exists():
                raise RuntimeError("training script wrote no result: " + (run.stdout + run.stderr)[-400:])
            row = json.loads(out.read_text())[-1]
            from training.ray.artifacts import save_artifact

            for path in sorted((Path(scratch) / "ckpt").rglob("*.pt")):   # trainable weights, config and normalization inside
                saved = save_artifact(client, bucket, f"{spec['output_prefix'].strip('/')}/weights/lr{lr}/n{channels}/seed-{seed}",
                                      path, {"command": command, "lr": float(lr), "seed": seed, "channels": channels,
                                             "backbone": BACKBONE, "result_row": row}, repo_root=root)
                record.setdefault("artifact_keys", []).append(saved.get("weights_key"))
                if "artifact_error" in saved:
                    record["artifact_error"] = saved["artifact_error"]
        record.update({k: row.get(k) for k in ("test_mse_raw", "test_mse_std", "best_val_mse_std", "epochs_run",
                                               "train_wall_s", "peak_gpu_mb", "references_std", "params_trainable")})
        record["collapsed"] = is_collapsed(float(record["best_val_mse_std"]), record["val_lookback_mean_mse_std"])
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=result_key(spec["output_prefix"], lr, channels, seed),
                          Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, nargs="+", default=[4, 25], choices=[4, 25])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--lrs", nargs="+", default=["1e-4", "3e-4"])
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=6 * 3600)
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="spectrumllm-bert-reruns")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    client = spectrum_s3_client()
    specs = []
    for channels in args.channels:
        for lr in args.lrs:
            for seed in args.seeds:
                try:
                    if json.loads(client.get_object(Bucket=bucket, Key=result_key(args.output_prefix, lr, channels, seed))
                                  ["Body"].read()).get("ok"):
                        continue
                except Exception:
                    pass
                specs.append({"channels": channels, "lr": lr, "seed": seed, "repo_root": args.repo_root,
                              "bucket": bucket, "output_prefix": args.output_prefix, "timeout": args.timeout})
    print(f"[bert-rerun] {len(specs)} runs", flush=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpu_fraction, max_retries=1)(job)
    pending = [remote.remote(s) for s in specs]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        print(f"[bert-rerun] n{r['channels']} lr={r['lr']} seed={r['seed']} ok={r['ok']} "
              f"test={r.get('test_mse_raw')} val={r.get('best_val_mse_std')} base={r.get('val_lookback_mean_mse_std')} "
              f"collapsed={r.get('collapsed')} {r.get('error', '')} ({r['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
