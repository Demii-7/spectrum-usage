"""Offset-response test of Residual ConvLSTM, DSwinLSTM-I and TSS-LCD (15-minute frozen checkpoints), from ``offset-extra-2026-11/``.

Reads every ``offset-extra-2026-11/<model>/<band>/<receiver>/seed-N/summary.json`` (written by ``training.ray.offset_response_sts --layout model-first`` and
``training.ray.tsslcd_authors --eval-only --input-offsets``) and writes

* ``offset_response_extra.csv``: one row per model x band x receiver x seed x offset x horizon, the columns of ``offset_response_sts.csv``
  (band, receiver, rep, model, seed, intrinsic, offset_db, horizon, mean_shift_db, gain, gain_sd, share_within_0p5_db, n_values);
* ``offset_response_extra_checks.csv``: the zero-offset reproduction check per model x band x seed (max |difference| of the unshifted forecast against the stored
  scores, dB; the requirement is <= 1e-4);
* per horizon (stdout; the runner redirects it to ``offset_response_extra_h<h>.txt``): per band and model the mean gain per offset, the mean over offsets, the min and
  max gain over receivers, seeds and offsets, and the number of seeds. Gain 1 = the level is carried through exactly, 0 = ignored.

    python evaluation/analysis/offset_response_extra_table.py --horizon 12 [--out evaluation/results/offset_response_extra.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

PREFIX = "offset-extra-2026-11/"


def load(bucket: str = "spectrum", prefix: str = PREFIX) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = spectrum_s3_client()
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
    with ThreadPoolExecutor(16) as pool:
        records = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
    rows, checks = [], {}
    for r in records:
        if not r.get("ok"):
            continue
        for row in r["rows"]:
            rows.append({"band": r["band"], "receiver": r["receiver"], "rep": r["rep"], "model": r["model"], "seed": r["seed"],
                         "intrinsic": r.get("intrinsic_window_normalization"), **row})
        checks[(r["model"], r["band"], r["seed"])] = {"model": r["model"], "band": r["band"], "seed": r["seed"],
                                                      "zero_offset_max_abs_diff_db": r.get("zero_offset_max_abs_diff_db"),
                                                      "zero_offset_values_compared": r.get("zero_offset_values_compared"),
                                                      "n_origins": r.get("n_origins")}
    return pd.DataFrame(rows), pd.DataFrame(list(checks.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--out", default="evaluation/results/offset_response_extra.csv")
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    frame, checks = load(prefix=args.prefix)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["model", "band", "receiver", "seed", "offset_db", "horizon"]).to_csv(args.out, index=False)
    if len(checks):
        checks.sort_values(["model", "band", "seed"]).to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_checks.csv")), index=False)
    pd.set_option("display.width", 250)
    print(f"{frame.groupby(['band', 'model', 'seed', 'receiver']).ngroups} runs (model x band x seed x receiver)")
    if len(checks):
        worst = checks.groupby(["model", "band"]).zero_offset_max_abs_diff_db.max()
        print("\nzero-offset reproduction check, max |difference| against the stored scores in dB (requirement <= 1e-4):")
        print(worst.to_string())
    for band in sorted(frame.band.unique()):
        d = frame[(frame.band == band) & (frame.horizon == args.horizon)]
        t = d.pivot_table(index=["model"], columns="offset_db", values="gain", aggfunc="mean").round(3)
        t.columns = [f"{c:+g}" for c in t.columns]
        t["mean"] = d.groupby("model").gain.mean().round(3)
        t["min"] = d.groupby("model").gain.min().round(3)
        t["max"] = d.groupby("model").gain.max().round(3)
        t["n_seeds"] = d.groupby("model").seed.nunique()
        print(f"\n{band}: mean gain of the forecast shift to an input offset of c dB at t+{args.horizon} (columns: c); mean / min / max over receivers, "
              f"seeds and offsets; 1 = level carried exactly")
        print(t.to_string())


if __name__ == "__main__":
    main()
