#!/usr/bin/env python3
"""TimeRAN (timeran_lp), 1 minute, every test window (stride 1): MAE, MSE, RMSE, MAPE (linear power and dBm), mean and SD over seeds.

Reads the stride-1 rescoring records ``rescored/direct/1min/<band>/2d/timeran_lp/h<h>/seed-N.json`` (``rescore.py --sources direct --resolution 1min
--models timeran_lp``: one checkpoint per trained horizon and seed, scored on every window of the October test week) and writes the long table in the format of
``part1_per_horizon_1min.csv`` (columns band, family, model, horizon, metric, mean, sd, n) to ``evaluation/results/part1_timeran_1min_stride1.csv`` and a text report
(.txt) that also lists missing seeds/horizons and compares with the finals' own (thinned, one window in 15) scores.

    python evaluation/analysis/part1_timeran_1min_stride1.py
"""
from __future__ import annotations

import pandas as pd

from part1_common import METRICS, client, listing, read_all, record_rows, summarize

H = (1, 15, 60)
BANDS = ("600_800", "2400_2600")
SEEDS = (40, 41, 42, 43, 44)


def main() -> None:
    c = client()
    rows, windows, thinned, seen = [], {}, [], set()
    for k, d in read_all(c, listing(c, "rescored/direct/1min/")):
        if d.get("model") != "timeran_lp":
            continue
        if not d.get("ok"):
            continue
        h, band, seed = int(d["horizon"]), d["band"], int(d["seed"])
        seen.add((band, h, seed))
        windows[(band, h, seed)] = d.get("n_test_windows")
        rows += record_rows(d, h, {"family": "direct", "band": band, "model": "timeran_lp", "seed": seed})
    for k, d in read_all(c, listing(c, "direct-tuning/1min/")):
        if d.get("model") == "timeran_lp" and d.get("kind") == "final" and d.get("ok"):
            thinned.append({"band": d["band"], "horizon": int(d["horizon"]), "seed": int(d["seed"]), "mae_thinned": d.get("test_mae_db_t%d" % int(d["horizon"])),
                            "mse_thinned": d.get("test_mse_db2_t%d" % int(d["horizon"]))})
    table = summarize(pd.DataFrame(rows), ["band", "family", "model"]) if rows else pd.DataFrame(columns=["band", "family", "model", "horizon", "metric", "mean", "sd", "n"])
    table.to_csv("evaluation/results/part1_timeran_1min_stride1.csv", index=False)
    pd.set_option("display.width", 220)
    out = ["TimeRAN (timeran_lp), 1 minute, every test window of 2026-10-27 20:00 to 11-03 20:00 UTC (stride 1); mean (SD) over seeds; one checkpoint per trained horizon and seed.", ""]
    for band in BANDS:
        out.append(f"== {band}")
        for m, (_, _, unit) in METRICS.items():
            d = table[(table.band == band) & (table.metric == m)].set_index("horizon")
            cells = [f"t+{h}: {d.loc[h, 'mean']:.4f} ({d.loc[h, 'sd']:.4f}, n={int(d.loc[h, 'n'])})" if h in d.index else f"t+{h}: missing" for h in H]
            out.append(f"  {m:12s} [{unit:5s}] " + "   ".join(cells))
        out.append("")
    missing = [(b, h, s) for b in BANDS for h in H for s in SEEDS if (b, h, s) not in seen]
    out.append("Missing (band, horizon, seed): " + (", ".join(f"{b} h{h} seed{s}" for b, h, s in missing) if missing else "none (30 of 30 checkpoints scored)"))
    counts = sorted({v for v in windows.values() if v})
    out.append("Test windows per checkpoint (stride 1): " + (", ".join(str(v) for v in counts) if counts else "n/a"))
    if thinned and rows:
        t = pd.DataFrame(thinned).groupby(["band", "horizon"]).agg(mae_thinned=("mae_thinned", "mean"), mse_thinned=("mse_thinned", "mean")).reset_index()
        s1 = table[table.metric.isin(["mae", "mse"])].pivot_table(index=["band", "horizon"], columns="metric", values="mean").reset_index()
        m = t.merge(s1, on=["band", "horizon"])
        m["mae_diff"] = m["mae"] - m["mae_thinned"]
        m["mse_diff"] = m["mse"] - m["mse_thinned"]
        out += ["", "Stride-1 minus the finals' own thinned scores (one window in 15), mean over the seeds present:", m.round(4).to_string(index=False)]
    text = "\n".join(out)
    open("evaluation/results/part1_timeran_1min_stride1.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
