"""Cluster-side poller: rebuild the 36 h neighbour-shift table (nearest-cell readout) whenever the number of finished cells changes.

Counts ``summary.json`` objects under ``neighbor-ablation-oct2026-nearest/`` (52 cells expected: 6 models, both bands: 5 seeds each except lookbackmean4d
with 1); runs ``evaluation/analysis/neighbor_value_nearest.py --upload`` (outputs to s3://spectrum/analysis-outputs/neighbor_value_ablation_nearest.{csv,txt});
stops when all cells exist or at the deadline (one last build then).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "/workspace/spectrum-usage")
from training.ray.storage import spectrum_s3_client  # noqa: E402


def count(client) -> int:
    n = 0
    for pg in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix="neighbor-ablation-oct2026-nearest/"):
        n += sum(1 for o in pg.get("Contents", []) if o["Key"].endswith("summary.json"))
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected", type=int, default=52)
    p.add_argument("--deadline-utc", default="2026-11-13T14:30Z")
    p.add_argument("--interval", type=int, default=600)
    a = p.parse_args()
    deadline = datetime.strptime(a.deadline_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    client, last = spectrum_s3_client(), -1
    while True:
        n = count(client)
        late = datetime.now(timezone.utc) >= deadline
        print(f"[nbr36-poller] {time.strftime('%F %T', time.gmtime())} cells {n} of {a.expected}{' (deadline)' if late else ''}", flush=True)
        if n != last and n > 0:
            subprocess.run("cd /workspace/spectrum-usage && PYTHONPATH=evaluation/analysis:. python -u evaluation/analysis/neighbor_value_nearest.py --upload",
                           shell=True)
            last = n
        if n >= a.expected or late:
            break
        time.sleep(a.interval)
    print("[nbr36-poller] done", flush=True)


if __name__ == "__main__":
    main()
