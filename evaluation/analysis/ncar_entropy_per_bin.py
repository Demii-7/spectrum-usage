"""Item 4b of the NCAR transfer batch: power-difference entropy at 15 minutes, per bin and per receiver, at AERPAW/ARA/COSMOS and the core POWDER receivers.

Definition and code are those of ``evaluation/analysis/october_entropy.py`` (``diff_entropy``: first differences of the 15-minute dB series,
Q = 8 states by the common thresholds ``DEFAULT_DIFF_EDGES``, LZ78 entropy rate normalized by log2 Q; fewer than 50 differences give NaN),
used by ``entropy_advantage_october.py``. Applied per receiver (no averaging over receivers), per bin.

Series: the converted 15-minute file of each target (``transfer_all.prepare_target``), restricted to the target's SCORED interval
[target_start, target_end) (the interval the transfer metrics are computed on). For the core October receivers two rows per bin:
``interval = scored_test_week`` (2026-10-27T20:00Z to 11-03T20:00Z, the reference cells' scored interval) and ``interval = training_run``
(the October training run 20261015T1434Z, the interval of ``october_entropy.py``).

    python evaluation/analysis/ncar_entropy_per_bin.py --root /workspace/spectrum-usage --out-key transfer-ncar-2026-11/item4b/transfer_external_entropy_per_bin.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def target_rows(root: str, t: dict, interval: str, start, end, frame: pd.DataFrame) -> list[dict]:
    from evaluation.analysis.october_entropy import diff_entropy

    sub = frame.loc[(frame.index >= start) & (frame.index < end)]
    rows = []
    for index, col in enumerate(sub.columns):
        e, n = diff_entropy(sub[col].to_numpy(dtype=float))
        rows.append({"site": t["platform"], "receiver": t["site"].replace("-nuc1", ""), "target": f"{t['platform']}/{t['site']}/{t['run']}/{t['band']}",
                     "band": t["band"], "bin": index, "frequency_mhz": float(col), "entropy": e, "n_diffs": n, "interval": interval,
                     "interval_start": str(start), "interval_end": str(end)})
    return rows


def compute(root: str) -> pd.DataFrame:
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray import ncar_cells as nc
    from training.ray import transfer_all as ta
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, TRAIN_RUN

    bands = ["600_800", "2400_2600"]
    rows = []

    def load(path):
        f = pd.read_csv(path)
        f.index = pd.to_datetime(f.pop("timestamp_utc"), utc=True)
        return f

    for t in ta.targets(root, ["cosmos", "aerpaw", "ara"], bands, ta.EXTERNAL_INTERVALS, core=False):
        task, file = ta.prepare_target(root, t)
        rows += target_rows(root, t, "scored_interval", task.target_start, task.target_end, load(file))
    for t in nc.core_targets(bands, "oct"):
        task, file = ta.prepare_target(root, t)
        rows += target_rows(root, t, "scored_test_week", task.target_start, task.target_end, load(file))
        from training.ray import transfer15 as t15

        train = load(Path(root) / OUT_ROOT / t15.RECEIVERS[t["site"]] / TRAIN_RUN / t["band"] / OUT_NAME)
        rows += target_rows(root, {**t, "run": TRAIN_RUN}, "training_run", train.index[0], train.index[-1] + pd.Timedelta(minutes=15), train)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/workspace/spectrum-usage")
    p.add_argument("--out-key", default=None, help="S3 key in the source bucket (never overwritten)")
    p.add_argument("--out", default=None, help="local CSV path")
    a = p.parse_args()
    sys.path.insert(0, a.root)
    table = compute(a.root)
    text = table.to_csv(index=False)
    if a.out:
        Path(a.out).write_text(text)
    if a.out_key:
        from training.ray.storage import spectrum_s3_client

        client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
        try:
            client.head_object(Bucket=bucket, Key=a.out_key)
            raise SystemExit(f"refusing to overwrite s3://{bucket}/{a.out_key}")
        except client.exceptions.ClientError:
            pass
        client.put_object(Bucket=bucket, Key=a.out_key, Body=text.encode())
        print(f"wrote s3://{bucket}/{a.out_key}")
    pd.set_option("display.width", 200)
    print(table.groupby(["site", "band", "interval"]).entropy.describe().round(3).to_string())


if __name__ == "__main__":
    main()
