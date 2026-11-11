"""Part 1, 1 minute, every test window (stride 1): all five metrics for all models rescored on the common range, with the heuristics.

Sources (``rescore.py`` records on 2026-10-27 20:00 to 11-03 20:00 UTC): ``rescored/checkpoints-oct2026*`` (600-800: the October 1D/2D/4D checkpoints, 5 seeds;
the 1D LSTMs are thinned to one window in ten, disclosed in each record) and ``rescored/direct/1min`` (both bands: the direct finals at stride 1). 4D rows are
map-level errors and are NOT comparable with the 1D/2D/direct rows (separate family). Heuristics: ``baselines_all_horizons.csv``; TSS-LCD: ``tsslcd_all.csv``.
Writes ``evaluation/results/part1_per_horizon_1min_stride1.csv`` (long).
"""
from __future__ import annotations

import pandas as pd

from part1_common import baseline_rows, client, listing, print_tables, read_all, record_rows, summarize

H = (1, 15, 60)


def main() -> None:
    c = client()
    rows = []
    for k, d in read_all(c, listing(c, "rescored/checkpoints-oct2026") + listing(c, "rescored/direct/1min")):
        if not d.get("ok"):
            continue
        if "/direct/1min/" in k:
            rows += record_rows(d, int(d["horizon"]), {"family": "direct", "band": d["band"], "model": d["model"], "seed": int(d["seed"])})
        else:
            fam = "4D (map-level, not comparable)" if d["rep"] == "4d" else "oct checkpoint"
            for h in H:
                rows += record_rows(d, h, {"family": fam, "band": d["band"], "model": f"{d['model']} ({d['rep']})", "seed": int(d["seed"])})
    base = pd.read_csv("evaluation/results/baselines_all_horizons.csv")
    base = base[(base.resolution == "1min") & base.selection.isin(["per_horizon", "full_lookback"]) & base.family.isin(["mean", "median", "persistence", "kalman"])]
    rows += baseline_rows(base, lambda r: {"family": "heuristic", "band": r.band, "model": r.family + (" (full 60 min)" if r.selection == "full_lookback" else
                                                                                                          (" (tuned)" if r.family in ("mean", "median") else ""))})
    ts = pd.read_csv("evaluation/results/tsslcd_all.csv")
    ts = ts[(ts.resolution == "1min") & (ts.savgol == "off") & ts.horizon.isin([str(h) for h in H])]
    for r in ts.itertuples():
        rows.append({"family": "TSS-LCD", "band": r.band, "model": f"tsslcd ({r.setting}, savgol off)", "horizon": int(r.horizon), "metric": r.metric,
                     "value": float(r.mean), "seed": 0})
    s = summarize(pd.DataFrame(rows), ["band", "family", "model"])
    s.to_csv("evaluation/results/part1_per_horizon_1min_stride1.csv", index=False)
    comparable = s[s.family != "4D (map-level, not comparable)"]
    print_tables(comparable, ("600_800", "2400_2600"), H, ["family", "model"], title="1-minute, every test window")
    print_tables(s[s.family == "4D (map-level, not comparable)"], ("600_800",), H, ["model"], title="4D map-level (not comparable)")


if __name__ == "__main__":
    main()
