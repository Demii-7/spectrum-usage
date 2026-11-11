"""Coverage of the four core receivers' raw 1-minute traces after the test week (task J, item 9): valid minutes per UTC day, per receiver and band.

A minute is valid when the raw row exists and all 200 bins are finite. Runs as a Ray job (the data live on the workers), prints a table and writes
``level-drift-oct2026/coverage.json`` to the bucket.

    python -m training.ray.level_diagnostics_coverage
"""
from __future__ import annotations

import json
import os
import sys

CORE = {"humanities": "humanities-nuc1", "madsen": "madsen-nuc1", "moran": "moran-nuc1", "sagepoint": "sagepoint-nuc1"}
BANDS = ("600_800", "2400_2600")
ROOT = os.environ.get("REPO_ROOT", "/workspace/spectrum-usage")


def coverage_job(receiver: str, band: str) -> dict:
    import numpy as np
    import pandas as pd

    out = {"receiver": receiver, "band": band, "runs": {}}
    base = os.path.join(ROOT, "data/powder", CORE[receiver])
    for run in sorted(os.listdir(base)):
        path = os.path.join(base, run, band, "power_1mhz_avg_per_minute.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
        valid = np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
        series = pd.Series(valid, index=stamps)
        series = series[~series.index.duplicated(keep="first")]
        per_day = series[series].groupby(series[series].index.strftime("%Y-%m-%d")).size()
        out["runs"][run] = {"first": str(stamps.min()), "last": str(stamps.max()), "rows": int(len(frame)), "valid_rows": int(valid.sum()),
                            "valid_minutes_per_day": {k: int(v) for k, v in per_day.items()}}
    return out


def main() -> None:
    import ray

    sys.path.insert(0, ROOT)
    from training.ray.storage import spectrum_s3_client

    ray.init(address="auto", ignore_reinit_error=True)
    remote = ray.remote(num_cpus=1, memory=4 * 2 ** 30)(coverage_job)
    rows = ray.get([remote.remote(r, b) for r in CORE for b in BANDS])
    spectrum_s3_client().put_object(Bucket="spectrum", Key="level-drift-oct2026/coverage.json", Body=json.dumps(rows).encode())
    days = [f"2026-10-{d}" for d in range(27, 32)] + [f"2026-11-{d:02d}" for d in range(1, 12)]
    for band in BANDS:
        print(f"\n{band}: valid minutes per UTC day (1,440 = full), run 20261025T1355Z")
        print("day         " + "  ".join(f"{r[:6]:>6}" for r in CORE))
        for d in days:
            vals = []
            for r in CORE:
                run = next(x for x in rows if x["receiver"] == r and x["band"] == band)["runs"].get("20261025T1355Z", {})
                vals.append(run.get("valid_minutes_per_day", {}).get(d, 0))
            print(f"{d}  " + "  ".join(f"{v:>6}" for v in vals))
    print("\nruns and last timestamps:")
    for x in rows:
        print(x["receiver"], x["band"], {k: (v["first"][:16], v["last"][:16]) for k, v in x["runs"].items()})


if __name__ == "__main__":
    main()
