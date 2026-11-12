"""Cluster-side collector for the neural-4D tables (task 5 of the neural-4D brief): rebuild the tables whenever new per-origin results have landed.

Polls S3 every --interval seconds; the signature is the number of objects under the prefixes the tables read (spatial15 cells and per-origin errors,
the neighbour ablation and its MSE re-run, the centred job, the direct per-origin errors with MSE). When the signature has changed and stayed
unchanged for two polls, the table jobs run one after the other (spatial15 tables + map-construction comparison, neighbour value, paired table,
centred table); each uploads its outputs to s3://spectrum/analysis-outputs/. Stops after --deadline-utc or --max-rebuilds. Never writes to a result prefix.

    python training/ray/scripts/neural4d_collect.py --deadline-utc 2026-11-13T14:00Z
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

PREFIXES = ("spatial15-tuned-oct2026/", "spatial15-origin-errors-oct2026/", "neighbor-ablation-oct2026/", "neighbor-ablation-oct2026-mse/",
            "spatial15-centered-oct2026/", "origin-errors-oct2026-mse/")
JOBS = (("spatial15 tables", "bash training/ray/scripts/run_spatial15_tables.sh"),
        ("neighbour value", "bash training/ray/scripts/run_neighbor_value.sh"),
        ("paired table", "bash training/ray/scripts/run_spatial_paired.sh"),
        ("centred table", "bash training/ray/scripts/run_spatial15_centered_table.sh"))


def signature(client, bucket: str) -> tuple:
    out = []
    for prefix in PREFIXES:
        n = 0
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            n += len(page.get("Contents", []))
        out.append(n)
    return tuple(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=900)
    p.add_argument("--deadline-utc", default="2026-11-13T14:00Z")
    p.add_argument("--max-rebuilds", type=int, default=12)
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args()
    sys.path.insert(0, a.repo_root)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), "spectrum"
    deadline = datetime.strptime(a.deadline_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    built, seen, stable, rebuilds = None, None, 0, 0
    while datetime.now(timezone.utc) < deadline and rebuilds < a.max_rebuilds:
        sig = signature(client, bucket)
        stable = stable + 1 if sig == seen else 0
        seen = sig
        print(f"[neural4d-collect] {time.strftime('%F %T', time.gmtime())} signature {sig} (built {built}, stable {stable})", flush=True)
        if sig != built and stable >= 2:
            for name, command in JOBS:
                code = subprocess.call(command, shell=True, cwd=a.repo_root)
                print(f"[neural4d-collect] {name}: exit {code}", flush=True)
            built, rebuilds = sig, rebuilds + 1
            print(f"[neural4d-collect] rebuild {rebuilds} done for signature {sig}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
