"""Part 1: per-horizon 15-minute test metrics (MAE, MSE, RMSE, MAPE linear and dBm) for every model family on the October test week
(2026-10-27 20:00 to 11-03 20:00 UTC), both bands.

Sources (all on the common range, all test windows): direct finals ``direct-tuning/15min/<band>/<model>/finals/h<h>/seed-N.json``; tuned rolled-out
1D/2D models ``rescored/p15tuned/...`` (t+4..t+48) and ``rescored-h1/p15tuned/...`` (t+1); heuristics from ``baselines_all_horizons.csv`` (tuned per
horizon on validation, and untuned full 96-step window); TSS-LCD from ``tsslcd_all.csv``. Mean over seeds and seed sd.
Writes ``evaluation/results/part1_per_horizon_15min.csv`` (long: band, family, model, horizon, metric, mean, sd, n) and prints MAE and MSE.
"""
from __future__ import annotations

import pandas as pd

from part1_common import baseline_rows, client, listing, print_tables, read_all, record_rows, summarize

HORIZONS = (1, 4, 12, 24, 48)


def collect() -> pd.DataFrame:
    c = client()
    rows = []
    for k, d in read_all(c, [k for k in listing(c, "direct-tuning/15min/") if "/finals/" in k]):
        if d.get("ok"):
            p = k.split("/")
            rows += record_rows(d, int(p[5][1:]), {"family": "direct", "band": p[2], "model": p[3], "seed": int(d["seed"])})
    for prefix, hs in (("rescored/p15tuned/", (4, 12, 24, 48)), ("rescored-h1/p15tuned/", (1,))):
        for k, d in read_all(c, listing(c, prefix)):
            if d.get("ok"):
                for h in hs:
                    rows += record_rows(d, h, {"family": "rolled-out", "band": d["band"], "model": f"{d['model']} ({d['rep']})", "seed": int(d["seed"])})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "15min") & base.selection.isin(["per_horizon", "full_lookback"])]

    def make(r):
        if r.selection == "full_lookback":
            return {"family": "heuristic (untuned)", "band": r.band, "model": f"{r.family} (full lookback, 96)"}
        return {"family": "heuristic", "band": r.band, "model": r.family + (" (tuned)" if r.family in ("mean", "median") else "")}
    rows += baseline_rows(base, make)
    ts = pd.read_csv("evaluation/results/tsslcd_all.csv")
    ts = ts[(ts.resolution == "15min") & ts.horizon.isin([str(h) for h in HORIZONS])]
    for r in ts.itertuples():
        rows.append({"family": "TSS-LCD", "band": r.band, "model": f"tsslcd ({r.setting}, savgol {r.savgol})", "horizon": int(r.horizon), "metric": r.metric,
                     "value": float(r.mean), "seed": 0})
    return pd.DataFrame(rows)


def main() -> None:
    s = summarize(collect(), ["band", "family", "model"])
    s.to_csv("evaluation/results/part1_per_horizon_15min.csv", index=False)
    print_tables(s, ("600_800", "2400_2600"), HORIZONS, ["family", "model"], title="15-minute")


if __name__ == "__main__":
    main()
