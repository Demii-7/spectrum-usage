"""Clean inference-cost benchmark: one model per GPU, nothing else on that GPU, fixed batch sizes, warm-up, median.

For every selected checkpoint the worker rebuilds the evaluation pipeline exactly as the original run did, but wraps
``evaluation_integrated.forecast`` (one full forecast of every horizon for a batch of windows, including the roll-out for
one-step models) with CUDA-synchronized timers. Per (model, batch size) it reports the median seconds per forecast call,
milliseconds per window, windows per second, forward passes per call, parameter count and peak GPU memory.

Checkpoint sources (all POWDER, all in the results bucket):

* ``oct``      1-minute 600-800 MHz campaign and round 2 (``checkpoints-oct2026``, ``checkpoints-oct2026-r2``): config from the base
               YAML plus the run's ``run_costs.json``.
* ``band``     1-minute 2400-2600 MHz runs (``oct2026-band-runs``): same, with the band swapped.
* ``p15``      15-minute pooled models, rolled out and direct (``powder15-results-w``): the exact resolved config in the ``.meta.json``.
* ``p15tuned`` 15-minute tuned finals (``tuning/powder15``): same.

Not covered: per-receiver and pooled-6 models (same architectures and shapes as the 2D models, so the same cost), lookback-mean
baselines (no parameters), TSS-LCD, Time-LLM/OFA (own pipelines).

Timings are only comparable if nothing else runs on the GPUs: each task reserves a whole GPU (``num_gpus=1``) and the launcher
refuses to start unless the cluster is idle (``--force`` overrides, for smoke tests only; the output is then marked ``idle=false``).

    python -m training.ray.inference_benchmark --manifest-only
    python -m training.ray.inference_benchmark --output-prefix inference-benchmark/h100
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WARMUP_CALLS = 3
BATCH_SIZES = (1, 32)
PREFERRED_SEED = 42
ALL_SOURCES = ("oct", "band", "p15", "p15tuned", "direct")
TEST_RANGE = ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z")
DEFAULT_STRIDE = {"oct": 20, "band": 20, "p15": 1, "p15tuned": 1, "direct": 1}   # about 500 windows either way (direct: per resolution, see stride_key)

META_KEY = re.compile(
    r"^(?P<prefix>.+?)/weights/(?P<band>\d+_\d+)/(?P<rep>[124]d)/(?P<model>[^/]+?)(?:/h(?P<horizon>\d+))?"
    r"/seed-(?P<seed>\d+)/(?P<file>[^/]+\.pt)$")
TUNED_KEY = re.compile(
    r"^(?P<prefix>tuning/powder15)/(?P<band>\d+_\d+)/(?P<rep>[124]d)/(?P<model>[^/]+)/weights/seed-(?P<seed>\d+)/(?P<file>[^/]+\.pt)$")
DIRECT_KEY = re.compile(
    r"^(?P<prefix>direct-tuning)/(?P<resolution>15min|1min)/(?P<band>\d+_\d+)/(?P<model>[^/]+)/weights/h(?P<horizon>\d+)"
    r"/seed-(?P<seed>\d+)/(?P<file>[^/]+\.pt)$")
BAND_PREFIX = re.compile(r"(\d+_\d+)$")
BINS = 200                  # 1 MHz bins in each 200 MHz band
ONE_D_STRIDE_FACTOR = 20    # a 1D "window" is one (receiver, bin) series, about 800 per time window, so thin the time windows 20x


def effective_stride(rep: str, stride: int) -> int:
    """1D models are scored one bin at a time (about 800 series windows per time window), so use a larger stride to keep about 20,000 windows."""
    return stride * ONE_D_STRIDE_FACTOR if rep == "1d" else stride


def spectra_per_window(rep: str) -> float:
    """How many receiver spectra one forecast window covers: 1D = one bin of one receiver, 2D = one receiver's 200 bins, 4D = the four core receivers."""
    return {"1d": 1.0 / BINS, "2d": 1.0, "4d": 4.0}[rep]


def parse_direct_key(key: str) -> dict[str, Any] | None:
    """Weights of ``tune_direct`` finals (``direct-tuning/<resolution>/<band>/<model>/weights/h<h>/seed-<s>/<file>.pt``)."""
    from training.ray.tune_direct import PIPELINE_NAME

    match = DIRECT_KEY.match(key)
    if not match:
        return None
    f = match.groupdict()
    if f["model"] not in PIPELINE_NAME:
        return None
    return {"prefix": f["prefix"], "resolution": f["resolution"], "band": f["band"], "rep": "2d", "model": f["model"],
            "pipeline": PIPELINE_NAME[f["model"]], "horizon": int(f["horizon"]), "seed": int(f["seed"]), "pt_key": key,
            "meta_key": key + ".meta.json"}


def stride_key(entry: dict[str, Any]) -> str:
    """Which stride setting applies: direct models follow their resolution, the others their source."""
    if entry["source"] == "direct":
        return "oct" if entry.get("resolution") == "1min" else "p15"
    return entry["source"]


def parse_meta_key(key: str) -> dict[str, Any] | None:
    """Weights uploaded by ``artifacts.save_run``/``save_artifact`` (15-minute runs) -> entry fields, else None."""
    match = TUNED_KEY.match(key) or META_KEY.match(key)
    if not match or "/site-" in key:
        return None
    fields = match.groupdict()
    return {"prefix": fields["prefix"], "band": fields["band"], "rep": fields["rep"], "model": fields["model"],
            "horizon": int(fields["horizon"]) if fields.get("horizon") else None, "seed": int(fields["seed"]), "pt_key": key,
            "meta_key": key + ".meta.json"}


def pick_checkpoints(entries: list[dict[str, Any]], preferred_seed: int = PREFERRED_SEED) -> list[dict[str, Any]]:
    """One checkpoint per (source, prefix, band, rep, model, horizon): the preferred seed if present, else the lowest seed.

    Timing does not depend on the seed. Direct models trained for different horizons are different models and are all kept.
    """
    chosen: dict[tuple, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: (e["source"], e["prefix"], e.get("band") or "", e["rep"], e["model"],
                                                 e.get("horizon") or 0, e["seed"])):
        key = (entry["source"], entry["prefix"], entry.get("band"), entry["rep"], entry["model"], entry.get("horizon"))
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


def discover_sources(client, bucket: str, sources: list[str], models: list[str] | None = None) -> list[dict[str, Any]]:
    """Every checkpoint of the requested sources that has what its config needs (a cost record or a sidecar)."""
    from training.ray.final_eval import discover
    from training.ray.fixed_train import REFERENCE_MODELS

    entries: list[dict[str, Any]] = []
    costs_prefixes = {"oct": ["checkpoints-oct2026", "checkpoints-oct2026-r2"], "band": ["oct2026-band-runs"]}
    for source in sources:
        if source in costs_prefixes:
            for entry in discover(client, bucket, costs_prefixes[source], models, None):
                band = BAND_PREFIX.search(entry["prefix"])
                entries.append({**entry, "source": source, "kind": "costs", "band": band.group(1) if band else "600_800",
                                "horizon": None})
        elif source == "direct":
            keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="direct-tuning/")
                    for o in page.get("Contents", [])]
            have = set(keys)
            for key in keys:
                entry = parse_direct_key(key) if key.endswith(".pt") else None
                if entry and entry["meta_key"] in have and (not models or entry["model"] in models):
                    entries.append({**entry, "source": "direct", "kind": "meta"})
        else:
            root = "powder15-results-w/weights/" if source == "p15" else "tuning/powder15/"
            keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root)
                    for o in page.get("Contents", [])]
            have = set(keys)
            for key in keys:
                if not key.endswith(".pt"):
                    continue
                entry = parse_meta_key(key)
                if entry and entry["meta_key"] in have and (not models or entry["model"] in models):
                    entries.append({**entry, "source": source, "kind": "meta"})
    return [e for e in entries if e["model"] not in REFERENCE_MODELS]


def build_config(entry: dict[str, Any], args: dict[str, Any], client, bucket: str, scratch: Path) -> tuple[dict[str, Any], Path]:
    """Download the checkpoint and return the evaluation config that rebuilds it, plus the local checkpoint path."""
    import yaml

    from training.ray.final_eval import build_eval_config, config_file_for
    from training.ray.fixed_train import swap_band

    checkpoint = scratch / Path(entry["pt_key"]).name
    client.download_file(bucket, entry["pt_key"], str(checkpoint))
    stride = effective_stride(entry["rep"], args["strides"][stride_key(entry)])
    name = entry.get("pipeline", entry["model"])
    if entry["kind"] == "costs":
        costs = json.loads(client.get_object(Bucket=bucket, Key=entry["costs_key"])["Body"].read())
        base = yaml.safe_load(Path(config_file_for(entry["model"], entry["rep"])).read_text())
        config = build_eval_config(base, entry["model"], costs, test_start=args["test_start"], test_end=args["test_end"],
                                   test_stride=stride, device=args["device"])
        if entry["band"] != "600_800":
            config = swap_band(config, entry["band"])
        return config, checkpoint
    meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
    config = meta["config"]
    config[name].setdefault("train", {})["test_stride"] = int(stride)
    config["training"] = {**config.get("training", {}), "models": [name]}
    if args["device"]:
        config["training"]["device"] = args["device"]
    if entry.get("resolution", "15min") == "15min":
        from training.ray.powder15 import ensure_powder15

        ensure_powder15(entry["band"], args["repo_root"])    # the 15-minute traces, converted once per worker
    return config, checkpoint


def benchmark_entry(entry: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker (whole GPU): download one checkpoint, time its forecast at each batch size."""
    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.ray.llm_powder import use_isolated_libs

    use_isolated_libs()                  # transformers lives in /workspace/pylibs on workers (OFA / Time-LLM checkpoints need it)
    import copy

    import torch

    from training.common import evaluation_integrated as ei
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    rows: list[dict[str, Any]] = []
    base_row = {k: entry.get(k) for k in ("source", "prefix", "resolution", "band", "rep", "model", "horizon", "seed")}
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="infer-bench-") as scratch_dir:
            scratch = Path(scratch_dir)
            base_config, checkpoint = build_config(entry, args, client, bucket, scratch)
            device = torch.device("cuda" if torch.cuda.is_available() and args["device"] != "cpu" else "cpu")
            gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            original = ei.forecast
            for batch in args["batch_sizes"]:
                config = copy.deepcopy(base_config)
                name = entry.get("pipeline", entry["model"])
                config.setdefault(name, {}).setdefault("evaluation", {})["batch_size"] = int(batch)
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
                    out_dir = scratch / f"eval-b{batch}"
                    out_dir.mkdir()
                    ei.evaluate_one_model(config, name, out_dir, checkpoint, skip_plots=True)
                finally:
                    ei.forecast = original
                    if "handle" in info:
                        info["handle"].remove()
                calls = max(len(seconds), 1)
                rows.append({**base_row, "ok": True, "gpu": gpu, "device": str(device), "requested_batch": int(batch),
                             "params": info.get("params"), "trainable_params": info.get("trainable_params"),
                             "forward_passes_per_call": round(forward_calls["n"] / calls, 2),
                             "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else None,
                             "idle": args["idle"], "spectra_per_window": spectra_per_window(entry["rep"]),
                             **summarize_calls(seconds, batch_rows)})
                if rows[-1].get("ms_per_window") is not None:
                    rows[-1]["ms_per_spectrum"] = rows[-1]["ms_per_window"] / rows[-1]["spectra_per_window"]
    except Exception as error:
        import traceback

        rows.append({**base_row, "ok": False, "error": f"{type(error).__name__}: {error}"[:300], "trace": traceback.format_exc()[-800:],
                     "idle": args["idle"]})
    for row in rows:
        row["task_seconds"] = round(time.time() - started, 1)
    return {"rows": rows}


def manifest(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model the benchmark will time (the chosen seed) with the seeds that exist."""
    seeds: dict[tuple, list[int]] = {}
    for e in entries:
        seeds.setdefault((e["source"], e["prefix"], e.get("band"), e["rep"], e["model"], e.get("horizon")), []).append(e["seed"])
    return [{**{k: e.get(k) for k in ("source", "prefix", "band", "rep", "model", "horizon", "seed")},
             "seeds": sorted(seeds[(e["source"], e["prefix"], e.get("band"), e["rep"], e["model"], e.get("horizon"))])}
            for e in pick_checkpoints(entries)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=list(ALL_SOURCES), choices=list(ALL_SOURCES))
    p.add_argument("--models", nargs="+")
    p.add_argument("--max-per-source", type=int, help="smoke tests: keep only this many models per source (spread across models)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    p.add_argument("--test-start", default=TEST_RANGE[0])
    p.add_argument("--test-end", default=TEST_RANGE[1])
    p.add_argument("--stride-1min", type=int, default=DEFAULT_STRIDE["oct"], help="window stride for the 1-minute sources")
    p.add_argument("--stride-15min", type=int, default=DEFAULT_STRIDE["p15"], help="window stride for the 15-minute sources")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--force", action="store_true", help="run even if the cluster is not idle (smoke tests; results are marked idle=false)")
    p.add_argument("--gpus-per-task", type=float, default=1.0)
    p.add_argument("--device", default=None, help="'cpu' for smoke tests without a GPU")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="inference-benchmark")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    entries = pick_checkpoints(discover_sources(client, bucket, args.sources, args.models))
    if args.max_per_source:
        kept: list[dict[str, Any]] = []
        for source in args.sources:
            group = [e for e in entries if e["source"] == source]
            step = max(len(group) // args.max_per_source, 1)
            kept += group[::step][:args.max_per_source]
        entries = kept
    if args.manifest_only:
        for row in manifest(entries):
            print(json.dumps(row))
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["source"]] = counts.get(e["source"], 0) + 1
        print("[bench] models per source:", json.dumps(counts), "total", len(entries), flush=True)
        return
    import pandas as pd
    import ray

    ray.init(address=args.address, ignore_reinit_error=True)
    total, free = ray.cluster_resources().get("GPU", 0), ray.available_resources().get("GPU", 0)
    idle = free >= total
    if not idle and not args.force:
        raise SystemExit(f"cluster is not idle ({free}/{total} GPUs free); timings would be contaminated. Use --force only for a smoke test")
    print(f"[bench] {len(entries)} checkpoints, batch sizes {args.batch_sizes}, idle={idle}, GPUs {total}", flush=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpus_per_task, max_retries=0)(benchmark_entry)
    strides = {"oct": args.stride_1min, "band": args.stride_1min, "p15": args.stride_15min, "p15tuned": args.stride_15min,
               "direct": args.stride_15min}
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "batch_sizes": args.batch_sizes, "test_start": args.test_start,
                   "test_end": args.test_end, "strides": strides, "device": args.device, "idle": idle}
    pending = [remote.remote(e, worker_args) for e in entries]
    rows: list[dict[str, Any]] = []
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        for row in ray.get(done[0])["rows"]:
            rows.append(row)
            print(f"[bench] {row['source']} {row['band']} {row['rep']}/{row['model']} h={row.get('horizon')} b={row.get('requested_batch')} "
                  f"ok={row['ok']} ms/window={row.get('ms_per_window')} {row.get('error', '')} [{len(pending)} left]", flush=True)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    key = f"{args.output_prefix.strip('/')}/inference_benchmark_{stamp}.csv"
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    print(f"[bench] wrote s3://{bucket}/{key} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
