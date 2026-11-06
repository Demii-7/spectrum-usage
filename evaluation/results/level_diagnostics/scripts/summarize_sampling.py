"""Summarize the sampling part: python summarize_sampling.py <dir>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
d = Path(sys.argv[1]); r = pd.read_csv(d / "sampling.csv")
print(pd.read_csv(d / "cells.csv")[["node", "band", "ok", "wall_seconds"]])
fl = r[r.part == "floor"]; s = r[(r.part == "sampling") & (r.split == "test")].copy()
s["rep2"] = np.where(s.rep.str.startswith("decim"), "decim(avg over phases)", s.rep)
agg = s.groupby(["band", "target", "horizon", "lookback_min", "rep2", "predictor"]).agg(
    mae=("mae", "mean"), mse=("mse", "mean"), skill=("skill_vs_1min_strong", "mean"),
    own=("skill_vs_own_strong", "mean")).reset_index()
for lb in (60, 240):
    print(f"\n==== lookback {lb} min: test MAE (mean over receivers; decim = mean over receivers and 15 phases)")
    print(agg[agg.lookback_min == lb].pivot_table(index=["target", "rep2", "predictor"], columns=["band", "horizon"], values="mae").round(3))
    print(f"\n---- lookback {lb}: skill vs the 1-min validation-selected simple baseline (same target/origins)")
    print(agg[agg.lookback_min == lb].pivot_table(index=["target", "rep2", "predictor"], columns=["band", "horizon"], values="skill").round(3))
    print(f"\n---- lookback {lb}: ridge skill vs its own representation's best simple baseline")
    print(agg[(agg.lookback_min == lb) & (agg.predictor == "ridge")].pivot_table(index=["target", "rep2"], columns=["band", "horizon"], values="own").round(3))
print("\n==== per-receiver ridge vs own-strong CIs (1min and block), lookback 60")
x = s[(s.lookback_min == 60) & (s.predictor == "ridge") & s.rep.isin(["1min", "block"])]
print(x[["band", "target", "horizon", "rep", "node", "skill_vs_own_strong", "own_ci_lo", "own_ci_hi"]].sort_values(["band", "target", "horizon", "rep", "node"]).round(3).to_string())
print("\n==== selected simple baselines (1min/block), lookback 240")
y = s[(s.lookback_min == 240) & (s.predictor == "strong") & s.rep.isin(["1min", "block"])]
print(y.pivot_table(index=["band", "target", "horizon", "node"], columns="rep", values="selected", aggfunc="first"))
print("\n==== decimation phase spread (test MAE sd over phases, mean over receivers)")
dz = s[s.rep.str.startswith("decim")].groupby(["band", "target", "horizon", "lookback_min", "predictor", "node"]).mae.std().groupby(["band", "target", "horizon", "lookback_min", "predictor"]).mean()
print(dz.unstack("predictor").round(4))
print("\n==== error floor (local-level model) vs achieved MSE: floor / achieved (1.0 = at the floor)")
best = s.groupby(["band", "node", "target", "horizon"]).mse.min().rename("best_mse")
f = fl.set_index(["band", "node", "target", "horizon"])[["floor_mse", "noise_only_mse", "level_est_mse", "drift_mse", "floor_mae_gaussian"]]
j = f.join(best)
one = s[(s.rep == "1min") & (s.lookback_min == 60) & (s.predictor == "strong")].set_index(["band", "node", "target", "horizon"]).mse.rename("strong1min_mse")
j = j.join(one); j["floor/best"] = j.floor_mse / j.best_mse; j["floor/strong1min"] = j.floor_mse / j.strong1min_mse
print(j.round(4).to_string())
print(j.groupby(["band", "target", "horizon"])[["floor/best", "floor/strong1min"]].mean().round(3))
