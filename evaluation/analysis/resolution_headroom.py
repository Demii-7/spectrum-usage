#!/usr/bin/env python3
"""How much predictable structure is left as the averaging window grows? (POWDER, one receiver-band at a time.)

For each time resolution r (minutes), per-minute dBm is averaged in LINEAR power (mW) over aligned r-minute blocks
(blocks with fewer than 80% valid minutes are dropped) and converted back to dB. On the test blocks we compare, for
every bin, a causal predictor (mean of the previous k blocks) with a two-sided oracle (median of the k blocks on each
side of the target, target excluded). Their gap ("headroom") estimates how much any forecaster could gain over a
simple past-only predictor at that resolution. Also reports the train-to-test level shift in training standard
deviations. Averaging is done within each partition's own rows, so nothing leaks across the split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START, TEST_END = "2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z"
TRAIN_RUN, TEST_RUN = "20261015T1434Z", "20261025T1355Z"


def block_average_db(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Per-minute dBm rows (indexed by UTC time) -> r-minute block means in dB (linear-power average)."""
    if minutes == 1:
        return frame
    power = np.power(10.0, frame / 10.0)
    block = power.index.floor(f"{minutes}min")
    grouped = power.groupby(block)
    mean = grouped.mean()
    count = grouped.count()
    mean = mean.where(count >= int(np.ceil(0.8 * minutes)))
    full = pd.date_range(mean.index.min(), mean.index.max(), freq=f"{minutes}min")
    return 10.0 * np.log10(mean.reindex(full))


def read_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
    frame.index = stamps
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="min"))


def headroom(values: np.ndarray, rows: np.ndarray, k: int) -> dict[str, float]:
    """Causal (mean of previous k) vs two-sided oracle (median of k each side) at the given target rows."""
    rows = rows[(rows - k >= 0) & (rows + k < len(values))]
    past = np.stack([values[rows - j] for j in range(1, k + 1)])
    future = np.stack([values[rows + j] for j in range(1, k + 1)])
    target = values[rows]
    valid = np.isfinite(past).all(axis=0) & np.isfinite(future).all(axis=0) & np.isfinite(target)
    causal = past.mean(axis=0)
    oracle = np.median(np.concatenate([past, future]), axis=0)
    out = {}
    for name, estimate in (("causal", causal), ("oracle", oracle)):
        error = np.where(valid, estimate - target, np.nan)
        out[f"{name}_mae"] = float(np.nanmean(np.abs(error)))
        out[f"{name}_mse"] = float(np.nanmean(error ** 2))
    return out


def run(args: argparse.Namespace) -> pd.DataFrame:
    start, end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
    rows = []
    for node in args.nodes:
        train = read_trace(args.root / node / TRAIN_RUN / args.band / "power_1mhz_avg_per_minute.csv")
        test = read_trace(args.root / node / TEST_RUN / args.band / "power_1mhz_avg_per_minute.csv")
        test = test[(test.index >= start) & (test.index < end)]
        for minutes in args.resolutions:
            tr, te = block_average_db(train, minutes), block_average_db(test, minutes)
            shift = float(np.nanmedian(np.abs(te.mean(axis=0).to_numpy() - tr.mean(axis=0).to_numpy())
                                       / tr.std(axis=0, ddof=0).to_numpy()))
            values = te.to_numpy(dtype=float)
            for k in args.ks:
                result = headroom(values, np.arange(len(values)), k)
                rows.append({"node": node, "minutes": minutes, "k": k, "n_blocks": len(values),
                             "level_shift_std": shift, **result,
                             "headroom_mae": 1 - result["oracle_mae"] / result["causal_mae"],
                             "headroom_mse": 1 - result["oracle_mse"] / result["causal_mse"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/powder"))
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--band", default="600_800")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1, 5, 15, 60])
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    frame = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
