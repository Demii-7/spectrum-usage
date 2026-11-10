"""How much of a left-out receiver's error is a static level offset? (zero-lag IDW estimate from the other receivers, 15-minute data)

For each band and each of the six POWDER receivers, estimate the receiver's value at every 15-minute step of the test week
(2026-10-27T20:00Z to 2026-11-03T20:00Z) by IDW (power 2, dB domain, as the maps) from the OTHER receivers of the six (the S3
leave-one-out set; for guest house and law 73 also the four-receiver core set the base models use). Then split the error
e = measured - estimate into
    * the static per-frequency-bin offset (mean over the week) and the scalar offset (mean over bins and week), and
    * what is left after removing each.
This is a nowcast (no forecasting), so it isolates how far apart the receivers' levels are from what interpolation gives; it does not use any model.

    python evaluation/analysis/loo_level_decomposition.py [--root /workspace/spectrum-usage] [--out loo_level_decomposition.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIX = {"humanities-nuc1": "humanities", "madsen-nuc1": "madsen", "moran-nuc1": "moran", "sagepoint-nuc1": "sage point",
       "guesthouse-nuc1": "guest house", "law73-nuc1": "law 73"}
CORE = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
RUN = "20261025T1355Z"
START, END = pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z")
BANDS = ("600_800", "2400_2600")


def xy(lon, lat, lon0, lat0):
    radius = 6_371_000.0
    return radius * np.radians(lon - lon0) * np.cos(np.radians(lat0)), radius * np.radians(lat - lat0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/workspace/spectrum-usage")
    parser.add_argument("--out", default="loo_level_decomposition.csv")
    args = parser.parse_args()
    root = Path(args.root)
    points = {e["name"]: (e["longitude"], e["latitude"]) for e in json.loads((root / "data/locations/powder.json").read_text())["endpoints"]}
    lon0 = float(np.mean([points[n][0] for n in SIX.values()]))
    lat0 = float(np.mean([points[n][1] for n in SIX.values()]))
    pos = {node: xy(*points[name], lon0, lat0) for node, name in SIX.items()}
    rows = []
    for band in BANDS:
        data = {}
        for node in SIX:
            frame = pd.read_csv(root / "data/powder15" / node / RUN / band / "power_15min.csv")
            frame.index = pd.to_datetime(frame.pop("timestamp_utc"), utc=True)
            data[node] = frame.loc[START:END - pd.Timedelta(minutes=15)]
        common = sorted(set.intersection(*(set(f.index) for f in data.values())))
        arrays = {node: data[node].loc[common].to_numpy(float) for node in SIX}
        for target in SIX:
            for label, sources in (("six minus target (S3)", [n for n in SIX if n != target]),
                                   ("core four", [n for n in CORE if n != target])):
                if label == "core four" and target in CORE:
                    continue
                d = np.array([np.hypot(pos[n][0] - pos[target][0], pos[n][1] - pos[target][1]) for n in sources])
                w = 1.0 / d ** 2
                w /= w.sum()
                estimate = sum(wi * arrays[n] for wi, n in zip(w, sources))
                error = arrays[target] - estimate                                   # (time, bin), dB
                per_bin = error.mean(axis=0, keepdims=True)
                scalar = error.mean()
                rows.append({"band": band, "target": SIX[target], "sources": label, "n_steps": len(common),
                             "mae_db": np.abs(error).mean(), "bias_db": scalar,
                             "mae_after_scalar_offset": np.abs(error - scalar).mean(),
                             "mae_after_per_bin_offset": np.abs(error - per_bin).mean(),
                             "share_removed_by_scalar": 1 - np.abs(error - scalar).mean() / np.abs(error).mean(),
                             "share_removed_by_per_bin": 1 - np.abs(error - per_bin).mean() / np.abs(error).mean()})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
