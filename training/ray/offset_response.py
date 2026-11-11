"""Constant-offset response test: does a frozen model carry a change of signal level through to its forecast? (evaluation only)

For each saved 15-minute tuned checkpoint (1D and 2D, ``p15tuned``) and a core receiver's October test week, every input window is
shifted by a constant ``c`` dB (c = -20, -10, -5, +5, +10, +20) before the usual frozen pipeline (the checkpoint's own source
normalization, the shared autoregressive forecaster), and the forecast is compared with the forecast of the unshifted windows on the same
origins. The response is

    shift(c, h)  = mean over origins and bins of  forecast_c - forecast_0        (dB, at horizon h)
    gain(c, h)   = shift(c, h) / c

A model that carries the level exactly (a residual anchor, a per-window normalization, a decomposition) has gain 1 at every c; a model that
ignores the level has gain near 0; a model that follows small shifts and flattens for large ones shows a gain that falls with |c| or
differs between positive and negative c (saturation). Also recorded: the spread of the per-value gain and the share of values that move by
within 0.5 dB of c.

Nothing is trained or fitted. Targets are not used. Origins are shared across offsets (they depend only on timestamps).

Output: ``<prefix>/<band>/<receiver>/<rep>/<model>/seed-<n>/summary.json`` (one row per offset and horizon) and a combined CSV from
``evaluation/analysis/offset_response_table.py``.

    python -m training.ray.offset_response --bands 600_800 2400_2600 --receivers humanities moran --seeds 40 41 42 --skip-done
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

OFFSETS = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0)
HORIZONS = (1, 4, 12, 24, 48)
STRIDE = {"2d": 4, "1d": 8}
DIRECT_MODELS = ("revinmlp", "patchtst", "patchtst_noinorm", "patchtst_joint", "patchtst_nopatch", "dlinear", "nlinear",
                 "linearar2d", "residuallinearar2d", "residualnonlinear2d")
# Large direct models (isolated libraries, pretrained weights; GPU): the share of a GPU and host memory (GB) a task reserves, as in training.ray.transfer_all.
LLM_MODELS = ("ofa", "ofa_rand", "ofa_scratch", "timellm", "timeran_lp", "moment_lp")
LLM_GPU = {"timellm": 1.0, "ofa": 0.5, "ofa_rand": 0.5, "ofa_scratch": 0.5, "timeran_lp": 0.25, "moment_lp": 0.25}
LLM_MEMORY_GB = 24.0
DIRECT_INTRINSIC = {**{m: "full" for m in LLM_MODELS}, "revinmlp": "full", "patchtst": "full", "patchtst_joint": "full", "patchtst_nopatch": "full", "patchtst_noinorm": "none",
                    "dlinear": "decomposition (moving-average trend)", "nlinear": "last value subtracted", "linearar2d": "none",
                    "residuallinearar2d": "partial", "residualnonlinear2d": "partial"}                    # origins are thinned (the response does not need all 625 per receiver)


def resources_for(label: str, gpu_override: float | None, memory_gb: float) -> tuple[float, float]:
    """(GPU share, host memory in GB) of one task: the large models reserve a GPU share and 24 GB, every other model runs on CPU."""
    if label in LLM_MODELS:
        return (gpu_override if gpu_override is not None else LLM_GPU[label]), max(memory_gb, LLM_MEMORY_GB)
    return (gpu_override or 0.0), memory_gb


def response_rows(base: dict[int, np.ndarray], shifted: dict[int, np.ndarray], offset: float, horizons=HORIZONS) -> list[dict[str, Any]]:
    """Response of forecasts (dB, arrays of one shape per horizon) to a constant input offset."""
    rows = []
    for h in horizons:
        delta = np.asarray(shifted[h], dtype=np.float64) - np.asarray(base[h], dtype=np.float64)
        gain = delta / offset
        rows.append({"offset_db": float(offset), "horizon": int(h), "mean_shift_db": float(delta.mean()), "gain": float(gain.mean()),
                     "gain_sd": float(gain.std()), "share_within_0p5_db": float(np.mean(np.abs(delta - offset) < 0.5)),
                     "n_values": int(delta.size)})
    return rows


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one checkpoint on one core receiver, every offset."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry, receiver, band = spec["entry"], spec["receiver"], spec["band"]
    from training.ray import transfer15 as t15
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    label, seed = entry["model"], int(entry["seed"])
    name = entry.get("pipeline", label)                       # the pipeline's model name (several direct variants share one)
    direct = entry.get("source") == "direct"
    horizons = [int(entry["horizon"])] if direct else list(HORIZONS)          # a direct model is trained for one horizon
    base = (f"{args['prefix'].strip('/')}/{band}/{receiver}/{entry['rep']}/{label}" + (f"/h{horizons[0]}" if direct else "") + f"/seed-{seed}")
    intrinsic = DIRECT_INTRINSIC.get(label, "none") if direct else t15.INTRINSIC.get(label, "none")
    record = {"kind": "offset_response", "band": band, "receiver": receiver, "rep": entry["rep"], "model": label, "seed": seed,
              "trained_horizon": horizons[0] if direct else None, "checkpoint_key": entry["pt_key"], "intrinsic_window_normalization": intrinsic,
              "offsets_db": list(OFFSETS), "horizons": horizons, "origin_stride": STRIDE[entry["rep"]]}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(max(1, int(args.get("threads", 2))))
        if label in LLM_MODELS:
            from training.ray.llm_powder import use_isolated_libs

            use_isolated_libs()
            if label.startswith(("timeran", "moment")):
                from training.ray.tune_direct import ensure_timeran_weights

                ensure_timeran_weights(client, bucket)
        from training.common import transfer_1d, transfer_2d
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        ensure_powder15(band, root, SPATIAL_NODES)
        meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
        config = meta["config"]
        config["windowing"] = {"lookback": t15.LOOKBACK, "horizons": horizons}
        config[name].setdefault("train", {})["test_stride"] = STRIDE[entry["rep"]]
        module = transfer_2d if entry["rep"] == "2d" else transfer_1d
        task = t15.target_task(band, receiver)
        with tempfile.TemporaryDirectory(prefix="offset-") as scratch_dir:
            checkpoint = Path(scratch_dir) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(checkpoint))
            source = module.load_target(task, [t15.target_file(root, band, receiver)], impute=False, max_missing_gap=0)
            reference = t15.evaluate_frozen(module, entry["rep"], task, source, name, checkpoint, config, seed, center=False)
            rows = []
            for offset in OFFSETS:
                shifted_source = dataclasses.replace(source, data=np.asarray(source.data, dtype=np.float32) + np.float32(offset))
                out = t15.evaluate_frozen(module, entry["rep"], task, shifted_source, name, checkpoint, config, seed, center=False)
                if not all(np.array_equal(out["rows"][h], reference["rows"][h]) for h in horizons):
                    raise RuntimeError("origins changed with the offset")
                rows.extend(response_rows(reference["predictions"], out["predictions"], offset, horizons))
        record.update(rows=rows, n_origins=int(len(reference["rows"][horizons[0]]) // (200 if entry["rep"] == "1d" else 1)),
                      checkpoint_sha256=meta.get("sha256"), code_revision=args["code_revision"], ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=["600_800", "2400_2600"])
    p.add_argument("--receivers", nargs="+", default=["humanities", "moran"], choices=["humanities", "madsen", "moran", "sagepoint"])
    p.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--source", choices=("rolled", "direct"), default="rolled",
                   help="rolled: the tuned 1D/2D checkpoints (p15tuned); direct: the tune_direct finals (one checkpoint per trained horizon, 15 minutes)")
    p.add_argument("--horizons", type=int, nargs="+", default=[12, 48], help="trained horizons to test (direct source)")
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    p.add_argument("--gpu-fraction", type=float, default=None,
                   help="GPU share per task; default 0 (CPU) for the cheap models and 0.25-1.0 per large model (OFA 0.5, Time-LLM 1.0, TimeRAN/MOMENT 0.25)")
    p.add_argument("--threads", type=int, default=2, help="torch threads per task")
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--max-parallel", type=int, default=24)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="offset-response-oct2026")
    p.add_argument("--code-revision", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = args.code_revision
    if revision is None:
        try:
            revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    if args.source == "direct":
        if args.prefix == "offset-response-oct2026":
            args.prefix = "offset-response-direct-oct2026"
        entries = [e for e in discover_sources(client, bucket, ["direct"], args.models or list(DIRECT_MODELS))
                   if e["resolution"] == "15min" and e["band"] in args.bands and int(e["seed"]) in args.seeds
                   and int(e["horizon"]) in args.horizons]
    else:
        entries = [e for e in discover_sources(client, bucket, ["p15tuned"], args.models)
                   if e["band"] in args.bands and e["rep"] in args.reps and int(e["seed"]) in args.seeds]
    jobs = [{"entry": e, "receiver": r, "band": e["band"]} for e in entries for r in args.receivers]
    if args.skip_done:
        def done(j):
            e = j["entry"]
            key = (f"{args.prefix.strip('/')}/{j['band']}/{j['receiver']}/{e['rep']}/{e['model']}"
                   + (f"/h{e['horizon']}" if e.get("source") == "direct" else "") + f"/seed-{e['seed']}/summary.json")
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    print(f"[offset] {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision, "threads": args.threads}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, max_retries=1)(job)

    def launch(j):
        gpu, memory = resources_for(j["entry"]["model"], args.gpu_fraction, args.memory_gb)
        return remote.options(num_gpus=gpu, memory=int(memory * 2 ** 30)).remote(j, shared)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(launch(queue.pop(0)))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        g = [r["gain"] for r in (row.get("rows") or []) if r["offset_db"] == 10.0][:1]
        print(f"[offset] {row.get('band')} {row.get('receiver')} {row.get('rep')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} "
              f"gain(+10 dB, t+{row.get('trained_horizon') or 12})={g[0] if g else None} {row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[offset] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
