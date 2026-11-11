"""Part 1, 1 minute, direct finals as trained (test windows thinned 15x): all five metrics at t+1, 15, 60 minutes, with the heuristics and TSS-LCD.

Direct finals ``direct-tuning/1min/<band>/<model>/finals/h<h>/seed-N.json``; heuristics ``baselines_all_horizons.csv``; TSS-LCD ``tsslcd_all.csv``
(savgol off). The stride-1 version is ``part1_per_horizon_1min_stride1.py``. Writes ``evaluation/results/part1_per_horizon_1min.csv`` (long).
"""
from __future__ import annotations

import pandas as pd

from part1_common import baseline_rows, client, listing, print_tables, read_all, record_rows, summarize

H = (1, 15, 60)


def main() -> None:
    c = client()
    rows = []
    for k, d in read_all(c, [k for k in listing(c, "direct-tuning/1min/") if "/finals/" in k]):
        if d.get("ok"):
            p = k.split("/")
            rows += record_rows(d, int(p[5][1:]), {"family": "direct", "band": p[2], "model": p[3], "seed": int(d["seed"])})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "1min") & base.selection.isin(["per_horizon", "full_lookback"]) & base.family.isin(["mean", "median", "persistence", "kalman", "ewma"])]
    rows += baseline_rows(base, lambda r: {"family": "heuristic", "band": r.band, "model": r.family + (" (full 60 min)" if r.selection == "full_lookback" else
                                                                                                          (" (tuned)" if r.family in ("mean", "median") else ""))})
    ts = pd.read_csv("evaluation/results/tsslcd_all.csv")
    ts = ts[(ts.resolution == "1min") & (ts.savgol == "off") & ts.horizon.isin([str(h) for h in H])]
    for r in ts.itertuples():
        rows.append({"family": "TSS-LCD", "band": r.band, "model": f"tsslcd ({r.setting}, savgol off)", "horizon": int(r.horizon), "metric": r.metric,
                     "value": float(r.mean), "seed": 0})
    s = summarize(pd.DataFrame(rows), ["band", "family", "model"])
    s.to_csv("evaluation/results/part1_per_horizon_1min.csv", index=False)
    print_tables(s, ("600_800", "2400_2600"), H, ["family", "model"], title="1-minute (direct finals, windows thinned 15x)")


if __name__ == "__main__":
    main()
