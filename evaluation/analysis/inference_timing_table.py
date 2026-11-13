"""Merge every inference-benchmark run into one table (run on the cluster: reads and writes the results bucket).

    python -m evaluation.analysis.inference_timing_table --out inference_timing_h100.csv

Reads ``inference-benchmark/<run>/inference_benchmark_*.csv`` for the listed runs, adds ``run`` and ``file`` columns, keeps one row per
(model, checkpoint, batch size) per file, and writes the combined CSV locally and to ``analysis-outputs/``. Smoke runs are not read.
"""
from __future__ import annotations

import argparse
import io
import re

import pandas as pd

RUNS = ("h100", "final-h100", "h100three")
KEEP = ["run", "file", "source", "prefix", "resolution", "band", "rep", "model", "horizon", "seed", "requested_batch", "ok", "ms_per_window",
        "ms_per_spectrum", "spectra_per_window", "median_s_per_call", "forward_passes_per_call", "params", "trainable_params", "peak_gpu_mb",
        "gpu", "idle", "task_seconds", "error"]


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate run frames, make sure every KEEP column exists, order rows by model."""
    table = pd.concat(frames, ignore_index=True)
    for column in KEEP:
        if column not in table:
            table[column] = None
    table = table[KEEP]
    return table.sort_values(["model", "rep", "band", "horizon", "run", "requested_batch"], na_position="first").reset_index(drop=True)


def tsslcd_rows(record: dict, prefix: str) -> list[dict]:
    """Rows from a ``tsslcd_authors --eval-only --time-inference`` record (key ``timing``): one per batch size, sampler steps in ``error``-free ``prefix`` note."""
    rows = []
    for t in record.get("timing") or []:
        rows.append({"run": "h100three", "file": prefix, "source": "tsslcd", "prefix": prefix, "resolution": record.get("resolution"), "band": record.get("band"),
                     "rep": "4d", "model": "tsslcd_" + prefix.rsplit("-", 1)[-1], "horizon": None, "seed": record.get("seed"), "requested_batch": t["batch"],
                     "ok": bool(record.get("ok")), "ms_per_window": t["ms_per_window"], "ms_per_spectrum": t["ms_per_window"] / 4.0, "spectra_per_window": 4.0,
                     "median_s_per_call": t["median_s_per_call"], "forward_passes_per_call": t["sampling_steps"], "params": t["params"],
                     "peak_gpu_mb": t["peak_gpu_mb"], "gpu": t["gpu"], "idle": True, "task_seconds": record.get("wall_seconds")})
    return rows


def tsslcd_frames(client, bucket: str) -> list[pd.DataFrame]:
    import json

    rows = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="inference-benchmark/h100three-tsslcd-"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/weights/" not in obj["Key"]:
                record = json.loads(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
                rows += tsslcd_rows(record, obj["Key"].split("/")[1])
    return [pd.DataFrame(rows)] if rows else []


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=list(RUNS))
    p.add_argument("--out", default="inference_timing_h100.csv")
    p.add_argument("--bucket", default=None)
    args = p.parse_args(argv)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), None
    import os
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    frames = []
    for run in args.runs:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"inference-benchmark/{run}/"):
            for obj in page.get("Contents", []):
                if re.search(r"inference_benchmark_.*\.csv$", obj["Key"]):
                    frame = pd.read_csv(io.BytesIO(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()))
                    frame.insert(0, "run", run)
                    frame.insert(1, "file", obj["Key"].rsplit("/", 1)[-1])
                    frames.append(frame)
                    print(f"[table] {obj['Key']}: {len(frame)} rows", flush=True)
    frames += tsslcd_frames(client, bucket)
    table = combine(frames)
    table.to_csv(args.out, index=False)
    client.put_object(Bucket=bucket, Key=f"analysis-outputs/{args.out}", Body=table.to_csv(index=False).encode())
    print(f"[table] {len(table)} rows, ok={int(table.ok.astype(bool).sum())}, uploaded analysis-outputs/{args.out}")
    print(table.groupby(["run", "ok"]).size().to_string())


if __name__ == "__main__":
    main()
