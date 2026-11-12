#!/usr/bin/env python3
"""Non-causal 'oracle' error floor at horizon 1, per receiver and bin.

For each bin and minute t in the test range, estimate the local level from the minutes on BOTH
sides of t (excluding t) and score |x_t - level|. A forecaster only sees the past, so on bins whose
level is smooth over +-k minutes this approximates the irreducible per-minute noise. Also reports
the causal counterpart from the same k past minutes only (what a mean/median-of-last-k predictor
does), so the gap between the two is the part attributable to not seeing the future.

Windows: k in --ks. Estimators: median and mean. Missing values are ignored; minutes missing from the
trace are treated as NaN (the series is put on a 1-minute grid).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START = "2026-10-27T20:00:00Z"
TEST_END = "2026-11-03T20:00:00Z"


def on_minute_grid(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.set_index("timestamp_utc").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame.index = frame.index.floor("min")
    return frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="min"))


def shifted(values: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Stack values shifted by each offset (NaN where it falls off the ends): (len(offsets), T, F)."""
    out = np.full((len(offsets),) + values.shape, np.nan, dtype=np.float32)
    for i, offset in enumerate(offsets):
        if offset > 0:  # value from `offset` minutes in the past
            out[i, offset:] = values[:-offset]
        else:           # value from `-offset` minutes in the future
            out[i, :offset] = values[-offset:]
    return out


def floors(values: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Per-bin MAE of: two-sided median/mean (oracle), one-sided median/mean (causal), over rows with a full window."""
    past = shifted(values, list(range(1, k + 1)))
    future = shifted(values, list(range(-1, -k - 1, -1)))
    both = np.concatenate([past, future])
    out = {}
    for name, stack in (("oracle", both), ("causal", past)):
        for stat, fn in (("median", np.nanmedian), ("mean", np.nanmean)):
            with np.errstate(all="ignore"):
                level = fn(stack, axis=0)
            error = np.abs(values - level)
            out[f"{name}_{stat}_k{k}"] = error
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/powder"))
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--run", default="20261025T1355Z")
    parser.add_argument("--band", default="600_800")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10, 30])
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
    rows = []
    for node in args.nodes:
        path = args.input_root / node / args.run / args.band / "power_1mhz_avg_per_minute.csv"
        grid = on_minute_grid(pd.read_csv(path))
        freqs = [float(c) for c in grid.columns]
        values = grid.to_numpy(dtype=np.float32)
        in_test = (grid.index >= start) & (grid.index < end) & np.isfinite(values).all(axis=1)
        for k in args.ks:
            for name, error in floors(values, k).items():
                masked = np.where(in_test[:, None], error, np.nan)  # only test rows with a measured target
                with np.errstate(all="ignore"):
                    per_bin = np.nanmean(masked, axis=0)
                for freq, mae in zip(freqs, per_bin):
                    rows.append({"node": node, "frequency_mhz": freq, "estimator": name, "mae_db": float(mae),
                                 "n_rows": int(np.isfinite(masked[:, 0]).sum())})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
