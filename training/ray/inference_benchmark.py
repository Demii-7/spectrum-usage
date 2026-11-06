"""Clean inference-cost benchmark: one model per GPU, nothing else on that GPU, fixed batch sizes, warm-up, median.

For every selected checkpoint the worker builds the evaluation pipeline exactly as ``final_eval`` does, but wraps
``evaluation_integrated.forecast`` (one full forecast of every horizon for a batch of windows, including the roll-out for
one-step models) with CUDA-synchronized timers. Per (model, batch size) it reports the median seconds per forecast call,
milliseconds per window, windows per second, forward passes per call, parameter count and peak GPU memory.

Timings are only comparable if nothing else runs on the GPUs: each task reserves a whole GPU (``num_gpus=1``) and the
launcher refuses to start unless the cluster is idle (``--force`` overrides, for smoke tests only; the output is then marked
``idle=false``).

    python -m training.ray.inference_benchmark --manifest-only                      # which checkpoints exist, which models are missing
    python -m training.ray.inference_benchmark --batch-sizes 1 32 --output-prefix inference-benchmark/h100
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WARMUP_CALLS = 3
BATCH_SIZES = (1, 32)
PREFERRED_SEED = 42


def pick_checkpoints(entries: list[dict[str, Any]], preferred_seed: int = PREFERRED_SEED) -> list[dict[str, Any]]:
    """One checkpoint per (prefix, rep, model): the preferred seed if present, else the lowest seed. Timing does not depend on the seed."""
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: (e["prefix"], e["rep"], e["model"], e["seed"])):
        key = (entry["prefix"], entry["rep"], entry["model"])
        current = chosen.get(key)
        if current is None or (entry["seed"] == preferred_seed and current["seed"] != preferred_seed):
            chosen[key] = entry
    return list(chosen.values())


def summarize_calls(seconds: list[float], batch_rows: list[int], warmup: int = WARMUP_CALLS) -> dict[str, Any]:
    """Median over the calls after warm-up; ``batch_rows`` can differ for the last partial batch, which is excluded."""
    full = max(batch_rows) if batch_rows else 0
    timed = [s for s, b in zip(seconds[warmup:], batch_rows[warmup:]) if b == full]
    if not timed:
        return {"n_timed_calls": 0}
    median = statistics.median(timed)
    return {"n_timed_calls": len(timed), "batch": full, "median_s_per_call": median, "ms_per_window": 1000.0 * median / full,
            "windows_per_s": full / median, "p90_s_per_call": sorted(timed)[int(0.9 * (len(timed) - 1))]}


def benchmark_entry(entry: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker (whole GPU): download one checkpoint, time its forecast at each batch size."""
    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    import torch
    import yaml

    from training.common import evaluation_integrated as ei
    from training.ray.final_eval import build_eval_config, config_file_for
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    rows: list[dict[str, Any]] = []
    base_row = {k: entry[k] for k in ("prefix", "rep", "model", "seed")}
    started = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        with tempfile.TemporaryDirectory(prefix="infer-bench-") as scratch:
            checkpoint = Path(scratch) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(checkpoint))
            costs = json.loads(client.get_object(Bucket=bucket, Key=entry["costs_key"])["Body"].read())
            base = yaml.safe_load(Path(config_file_for(entry["model"], entry["rep"])).read_text())
            original = ei.forecast
            for batch in args["batch_sizes"]:
                config = build_eval_config(base, entry["model"], costs, test_start=args["test_start"], test_end=args["test_end"],
                                           test_stride=args["test_stride"], device=args["device"])
                config.setdefault(entry["model"], {}).setdefault("evaluation", {})["batch_size"] = int(batch)
                seconds: list[float] = []
                batch_rows: list[int] = []
                info: dict[str, Any] = {}
                forward_calls = {"n": 0}

                def count(_module, _inputs):
                    forward_calls["n"] += 1

                def timed(*a, **kw):
                    model, x = kw["model"], kw["x"]
                    if "params" not in info:
                        info["params"] = sum(p.numel() for p in model.parameters())
                        info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        info["handle"] = model.register_forward_pre_hook(count)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = original(*a, **kw)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - t0)
                    batch_rows.append(int(x.shape[0]))
                    return out

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                ei.forecast = timed
                try:
                    out_dir = Path(scratch) / f"eval-b{batch}"
                    out_dir.mkdir()
                    ei.evaluate_one_model(config, entry["model"], out_dir, checkpoint, skip_plots=True)
                finally:
                    ei.forecast = original
                    if "handle" in info:
                        info["handle"].remove()
                calls = max(len(seconds), 1)
                rows.append({**base_row, "ok": True, "gpu": gpu, "device": str(device), "requested_batch": int(batch),
                             "params": info.get("params"), "trainable_params": info.get("trainable_params"),
                             "forward_passes_per_call": round(forward_calls["n"] / calls, 2),
                             "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else None,
                             "idle": args["idle"], **summarize_calls(seconds, batch_rows)})
    except Exception as error:
        import traceback

        rows.append({**base_row, "ok": False, "error": f"{type(error).__name__}: {error}"[:300], "trace": traceback.format_exc()[-800:],
                     "idle": args["idle"]})
    for row in rows:
        row["task_seconds"] = round(time.time() - started, 1)
    return {"rows": rows}


def manifest(client, bucket: str, prefixes: list[str]) -> list[dict[str, Any]]:
    """Every (prefix, rep, model) with a checkpoint and cost record, plus the models of the campaign plan that have none."""
    from training.ray.campaign_all import PLAN
    from training.ray.final_eval import discover

    entries = discover(client, bucket, prefixes, None, None)
    have = {(e["rep"], e["model"]) for e in entries}
    rows = [{"prefix": e["prefix"], "rep": e["rep"], "model": e["model"], "seeds": [x["seed"] for x in entries
            if (x["prefix"], x["rep"], x["model"]) == (e["prefix"], e["rep"], e["model"])]} for e in pick_checkpoints(entries)]
    for model, rep, _config in PLAN:
        if (rep, model) not in have:
            rows.append({"prefix": "", "rep": rep, "model": model, "seeds": [], "missing": True})
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefixes", nargs="+", default=["checkpoints-oct2026", "checkpoints-oct2026-r2"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    p.add_argument("--test-start", default="2026-10-27T20:00:00Z")
    p.add_argument("--test-end", default="2026-11-03T20:00:00Z")
    p.add_argument("--test-stride", type=int, default=20, help="windows are sampled at this stride; 20 gives about 500 windows")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--force", action="store_true", help="run even if the cluster is not idle (smoke tests; results are marked idle=false)")
    p.add_argument("--gpus-per-task", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="inference-benchmark")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.final_eval import discover
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    if args.manifest_only:
        for row in manifest(client, bucket, args.prefixes):
            print(json.dumps(row))
        return
    import pandas as pd
    import ray

    ray.init(address=args.address, ignore_reinit_error=True)
    total, free = ray.cluster_resources().get("GPU", 0), ray.available_resources().get("GPU", 0)
    idle = free >= total
    if not idle and not args.force:
        raise SystemExit(f"cluster is not idle ({free}/{total} GPUs free); timings would be contaminated. Use --force only for a smoke test")
    entries = pick_checkpoints(discover(client, bucket, args.prefixes, args.models, None))
    print(f"[bench] {len(entries)} checkpoints, batch sizes {args.batch_sizes}, idle={idle}", flush=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpus_per_task, max_retries=0)(benchmark_entry)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "batch_sizes": args.batch_sizes, "test_start": args.test_start,
                   "test_end": args.test_end, "test_stride": args.test_stride, "device": args.device, "idle": idle}
    pending = [remote.remote(e, worker_args) for e in entries]
    rows: list[dict[str, Any]] = []
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        for row in ray.get(done[0])["rows"]:
            rows.append(row)
            print(f"[bench] {row['rep']}/{row['model']} b={row.get('requested_batch')} ok={row['ok']} "
                  f"ms/window={row.get('ms_per_window')} {row.get('error', '')} [{len(pending)} left]", flush=True)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    key = f"{args.output_prefix.strip('/')}/inference_benchmark_{stamp}.csv"
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    print(f"[bench] wrote s3://{bucket}/{key} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
