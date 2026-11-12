"""15-minute test-week MAE (dB) of the four strong direct models side by side: revinmlp, revinmlp_mix, itransformer, patchtst.

Reads the direct finals like ``part1_per_horizon.collect()`` (same conventions: October test week, all test windows, mean over seeds with seed sd), keeps the direct finals of
the four models and adds the mean over horizons t+4/12/24/48. Writes ``<out>`` (long CSV with a ``mean_4_12_24_48`` pseudo-horizon 0) and ``<out>.txt``; the optional
``--upload`` puts both in s3://spectrum/analysis-outputs/.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import part1_per_horizon as p1
from part1_common import listing, read_all, record_rows, summarize

MODELS = ("revinmlp", "revinmlp_mix", "itransformer", "patchtst")
HS = (4, 12, 24, 48)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/results/strong_models_15min.csv")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    c = p1.client()
    rows = []
    for k, d in read_all(c, [k for k in listing(c, "direct-tuning/15min/") if "/finals/" in k and k.split("/")[3] in MODELS]):
        if d.get("ok"):
            q = k.split("/")
            rows += record_rows(d, int(q[5][1:]), {"family": "direct", "band": q[2], "model": q[3], "seed": int(d["seed"])})
    f = pd.DataFrame(rows)
    s = summarize(f, ["band", "model"])
    mae = s[(s.metric == "mae") & s.horizon.isin(HS)]
    lines = []
    for band in ("600_800", "2400_2600"):
        w = mae[mae.band == band].pivot_table(index="model", columns="horizon", values="mean").reindex(list(MODELS))
        sd = mae[mae.band == band].pivot_table(index="model", columns="horizon", values="sd").reindex(list(MODELS))
        n = mae[mae.band == band].pivot_table(index="model", columns="horizon", values="n").reindex(list(MODELS))
        w["mean"] = w[[h for h in HS if h in w.columns]].mean(axis=1)
        lines.append(f"== {band}: MAE (dB), mean over seeds (sd), n seeds ==")
        for m in MODELS:
            if m in w.index and not w.loc[m].isna().all():
                lines.append(f"{m:14s} " + "  ".join(f"t+{h}: {w.loc[m, h]:.4f} ({sd.loc[m, h]:.4f}, n={int(n.loc[m, h])})" for h in HS if h in w.columns)
                             + f"  mean: {w.loc[m, 'mean']:.4f}")
    text = "\n".join(lines)
    print(text)
    extra = mae.groupby(["band", "model"]).apply(lambda d: d.groupby("horizon")["mean"].first().mean()).rename("mean").reset_index()
    extra["horizon"], extra["metric"] = 0, "mae_mean_4_12_24_48"
    pd.concat([s, extra.rename(columns={"mean": "mean"})], ignore_index=True).to_csv(a.out, index=False)
    open(a.out.replace(".csv", ".txt"), "w").write(text + "\n")
    if a.upload:
        c = p1.client()
        for path in (a.out, a.out.replace(".csv", ".txt")):
            c.put_object(Bucket="spectrum", Key="analysis-outputs/" + os.path.basename(path), Body=open(path, "rb").read())
            print("uploaded", path)


if __name__ == "__main__":
    main()
