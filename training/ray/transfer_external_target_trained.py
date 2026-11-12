"""Target-trained RevIN-MLP at COSMOS / AERPAW / ARA: the reference for the zero-shot / transferred POWDER models of ``transfer_external``.

For every usable frozen target (``evaluation/analysis/transfer_external_intervals.csv``, same converted 15-minute series as the transfer
rows, ``data/ext15``) a direct RevIN-MLP (one model per horizon 4/12/24/48, lookback 96, the tuned POWDER winner
``direct-tuning/15min/<band>/revinmlp/winners.json``, so the comparison isolates the training data, not hyperparameters) is trained on the
target's own first 60% of blocks (normalization fitted there), validated on the next 15% and scored on the last 25% (chronological; the
validation and test partitions start 96 blocks early so their windows have full history, their targets are all after the split point).
Persistence and the 96-step lookback mean are scored on exactly the same origins. The record keeps the first test target time, so the transfer
rows (their per_origin.csv under ``transfer15-external-oct2026``) can be restricted to the same origins for a like-for-like comparison.

    python -m training.ray.transfer_external_target_trained run --skip-done
Records: ``<prefix>/<platform>/<site>/<run>/<band>/h<H>/seed-<S>.json`` (errors as ``...-error.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOOKBACK = 96
HORIZONS = (4, 12, 24, 48)
FRACTIONS = (0.60, 0.75)           # train ends at 60%, validation at 75%, test is the rest
PREFIX = "transfer-external-target-trained"
INTERVALS_CSV = "evaluation/analysis/transfer_external_intervals.csv"


def record_key(prefix: str, t: dict[str, Any], horizon: int, seed: int, failed: bool = False) -> str:
    return (f"{prefix.strip('/')}/{t['platform']}/{t['site']}/{t['run']}/{t['band']}/h{horizon}/seed-{seed}"
            + ("-error" if failed else "") + ".json")


def split_rows(n: int) -> tuple[int, int]:
    return int(FRACTIONS[0] * n), int(FRACTIONS[1] * n)


def apply_target(config: dict[str, Any], path: str, stamps: pd.Series, band: str, horizon: int, n: int) -> dict[str, Any]:
    import copy

    config = copy.deepcopy(config)
    a, b = split_rows(n)

    def stamp(i: int) -> str:
        return stamps.iloc[i]

    config["data"]["files"] = [{"path": path, "partition": p} for p in ("train", "validation", "test")]
    config["data"]["chunks"] = [{"id": f"external_{band}", "start_mhz": float(band.split("_")[0]), "end_mhz": float(band.split("_")[1])}]
    config["data"]["split"] = {"ranges": {"train": {"start": stamp(0), "end": stamp(a - 1)},
                                          "validation": {"start": stamp(a - LOOKBACK), "end": stamp(b - 1)},
                                          "test": {"start": stamp(b - LOOKBACK), "end": stamp(n - 1)}}}
    config["data"].pop("map", None)
    config["data"]["concat"] = "rows"
    config["windowing"] = {"lookback": LOOKBACK, "horizons": [int(horizon)]}
    config["preprocessing"] = {**config.get("preprocessing", {}), "normalize": True, "impute": False, "max_missing_gap": 0}
    return config


def cell_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    t, horizon, seed = job["target"], int(job["horizon"]), int(job["seed"])
    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.spectral_context_direct import tuned_params
    from training.ray.storage import spectrum_s3_client
    from training.ray.transfer_external import convert, converted_path
    from training.ray.tune_direct import build_config
    from training.ray.tune_fixed import train_prepared

    client, bucket = spectrum_s3_client(), args["bucket"]
    record = {"kind": "transfer_external_target_trained", "platform": t["platform"], "site": t["site"], "run": t["run"], "band": t["band"],
              "model": "revinmlp", "horizon": horizon, "seed": seed, "split_fractions": list(FRACTIONS), "lookback": LOOKBACK}
    started = time.time()
    try:
        convert(root, t)
        path = converted_path(root, t)
        frame = pd.read_csv(path)
        stamps = frame["timestamp_utc"]
        values = frame.drop(columns=["timestamp_utc"]).to_numpy(dtype=np.float64)
        n = len(frame)
        a, b = split_rows(n)
        found = tuned_params(client, bucket, t["band"], "revinmlp")
        if found is None:
            raise RuntimeError(f"no tuned revinmlp winner for {t['band']}")
        params, source = found
        config, name = build_config("revinmlp", "15min", t["band"], horizon, seed, params, root, train_stride=1, test_stride=1)
        config = apply_target(config, str(path.relative_to(root)), stamps, t["band"], horizon, n)
        starts = np.arange(b - LOOKBACK, n - LOOKBACK - horizon + 1)        # test-partition windows (history inside the partition)
        rows = starts + LOOKBACK + horizon - 1
        record.update(config_source=source, rows=n, train_rows=a, validation_end_row=b, n_test_origins=int(len(starts)),
                      first_test_target_time=stamps.iloc[int(rows[0])], last_test_target_time=stamps.iloc[int(rows[-1])],
                      target_file=str(path.relative_to(root)), parameters=params)
        if len(starts) < 20:
            raise RuntimeError(f"only {len(starts)} test origins")
        base = {"persistence": values[starts + LOOKBACK - 1], "lookbackmean": np.stack([values[s:s + LOOKBACK].mean(axis=0) for s in starts])}
        for k, pred in base.items():
            err = pred - values[rows]
            record[f"{k}_mae_db"], record[f"{k}_mse_db"] = float(np.mean(np.abs(err))), float(np.mean(err ** 2))
        with tempfile.TemporaryDirectory(prefix="ext-tt-") as scratch:
            result = train_prepared(config, name, Path(scratch) / "run")
            eval_dir = Path(scratch) / "eval"
            eval_dir.mkdir()
            evaluate_one_model(config, name, eval_dir, result["checkpoint"], skip_plots=True)
            meta = next((eval_dir / "forecasts").glob("*_metadata.json"))
            stem = meta.name.removesuffix("_metadata.json")
            with np.load(eval_dir / "forecasts" / f"{stem}_predictions.npz") as p, np.load(eval_dir / "forecasts" / f"{stem}_targets.npz") as q:
                pred, target = np.asarray(p[f"t_plus_{horizon}"], dtype=np.float64), np.asarray(q[f"t_plus_{horizon}"], dtype=np.float64)
        if len(pred) != len(starts):
            record["origin_count_mismatch"] = [int(len(pred)), int(len(starts))]
        err = pred - target
        record.update(model_mae_db=float(np.mean(np.abs(err))), model_mse_db=float(np.mean(err ** 2)), n_scored_origins=int(len(pred)),
                      objective=result.get("objective"))
        record["ok"] = bool(np.isfinite(record["model_mae_db"]))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1000:])
    record["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=record_key(args["prefix"], t, horizon, seed, failed=not record["ok"]),
                          Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    table = pd.read_csv(Path(a.repo_root) / a.intervals)
    targets = [r for r in table.to_dict("records") if r["use"] and r["band"] in a.bands and r["platform"] in a.platforms]
    done = set()
    if a.skip_done:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=a.prefix.strip("/") + "/"):
            done.update(o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("-error.json"))
    jobs = [{"target": t, "horizon": h, "seed": s} for s in a.seeds for t in targets for h in HORIZONS
            if record_key(a.prefix, t, h, s) not in done]
    print(f"[ext-target-trained] {len(targets)} targets, {len(jobs)} cells to run", flush=True)
    ray.init(address=a.address, ignore_reinit_error=True)
    task = ray.remote(num_cpus=1, num_gpus=a.gpu_fraction, memory=int(a.memory_gb * 2 ** 30), max_retries=1)(cell_job)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix}
    queue, pending, ok, failed = list(jobs), [], 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            pending.append(task.remote(queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        print(f"[ext-target-trained] {row.get('platform')} {row.get('site')} {row.get('run')} {row.get('band')} h{row.get('horizon')} "
              f"seed{row.get('seed')} ok={row.get('ok')} mae={row.get('model_mae_db')} lookbackmean={row.get('lookbackmean_mae_db')} "
              f"{row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[ext-target-trained] done: {ok} ok, {failed} failed", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    r.add_argument("--platforms", nargs="+", default=["cosmos", "aerpaw", "ara"])
    r.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    r.add_argument("--intervals", default=INTERVALS_CSV)
    r.add_argument("--gpu-fraction", type=float, default=0.125)
    r.add_argument("--memory-gb", type=float, default=6.0)
    r.add_argument("--max-parallel", type=int, default=6)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--bucket", default=None)
    r.add_argument("--prefix", default=PREFIX)
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
