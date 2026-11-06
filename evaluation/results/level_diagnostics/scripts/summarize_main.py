"""Summarize level_diagnostics outputs: python summarize.py <dir>"""
import sys
from pathlib import Path
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
d = Path(sys.argv[1]); r = pd.read_csv(d / "results.csv"); t = r[r.split == "test"]
base = t[t.experiment == "baselines"]
keep = ["persistence", "train_mean", "mean60", "median60", "linmean60", "kalman_local_level", "kalman_moments", "strong"]
print("== Baselines, test MAE dB (mean over 4 receivers)")
print(base[base.predictor.isin(keep)].pivot_table(index="predictor", columns=["band", "horizon"], values="mae").round(3))
print("\n== strong (val-selected) per cell")
print(base[base.predictor == "strong"].pivot_table(index=["band", "node"], columns="horizon", values="selected", aggfunc="first"))
print(base[base.predictor == "strong"].pivot_table(index=["band", "node"], columns="horizon", values="skill_vs_mean60").round(3))
s = t[t.experiment == "seasonal"]
print("\n== seasonal naive (own origins)"); print(s.pivot_table(index="predictor", columns=["band", "horizon"], values="mae").round(3))
pr = r[r.experiment.isin(["probe_per_bin", "probe_pooled"])]
print("\n== probes: test skill vs mean60 (mean over receivers)")
print(pr[pr.split == "test"].pivot_table(index="predictor", columns=["band", "horizon"], values="skill_vs_mean60").round(4))
print("\n== probes: test skill vs strong baseline (per-bin probes only), per cell, with CI")
x = pr[(pr.split == "test") & pr.predictor.isin(["mean60:val_scaled", "last:val_scaled", "none:full"])]
print(x[["band", "horizon", "node", "predictor", "skill_vs_strong", "strong_ci_lo", "strong_ci_hi"]].sort_values(["band", "horizon", "predictor", "node"]).round(4).to_string())
print("\n== GBM(MAE loss) and pooled linear, per cell skill vs mean60 with CI")
g = pr[(pr.split == "test") & pr.predictor.isin(["pooled_gbm_mae:val_scaled", "pooled_linear:val_scaled", "pooled_rich_linear:val_scaled"])]
print(g[["band", "horizon", "node", "predictor", "skill_vs_mean60", "ci_lo", "ci_hi", "selected"]].sort_values(["band", "horizon", "predictor", "node"]).round(4).to_string())
pb = r[r.experiment == "probe_per_bin"]
w = pb.pivot_table(index=["band", "horizon", "split"], columns="predictor", values="mae")
for a in ("mean60", "last"):
    w[f"{a}_gain"] = w[f"{a}:anchor_only"] - w[f"{a}:val_scaled"]
print("\n== correction gain (anchor MAE - corrected MAE, dB) by split"); print(w[["mean60_gain", "last_gain"]].round(4))
print("\n== selected ridge/shrink (per-bin mean60)"); print(pb[(pb.split == "test") & (pb.predictor == "mean60:val_scaled")].groupby(["band", "horizon"]).selected.agg(lambda v: sorted(set(v))))
for name in ("statespace", "cheap", "equivalence"):
    print(f"\n== {name}"); print(pd.read_csv(d / f"{name}.csv").round(4).to_string())
cm = pd.read_csv(d / "condmedian.csv")
print("\n== conditional median of mean60 residual by decile of (last - mean60), median over cells")
print(cm.pivot_table(index=["horizon", "decile"], columns="split", values=["dev_median", "residual_median"], aggfunc="median").round(3))
dd = pd.read_csv(d / "daily.csv")
print("\n== per-day skill vs mean60 (mean over receivers)"); print(dd.pivot_table(index=["band", "horizon", "predictor"], columns="day", values="skill_vs_mean60").round(3))
ag = r[(r.experiment == "aggregation") & (r.split == "test")]
print("\n== aggregation: test skill vs dB mean of 1-min history (mean over receivers)")
print(ag.pivot_table(index=["arm", "predictor"], columns=["band", "horizon", "lookback_min"], values="skill_vs_mean_dB").round(4))
print("\n== aggregation: test MAE"); print(ag.pivot_table(index=["arm", "predictor"], columns=["band", "horizon", "lookback_min"], values="mae").round(3))
print("\n== aggregation decimation phase sd"); print(ag[ag.arm.str.startswith("D")].pivot_table(index="predictor", columns=["band", "horizon", "lookback_min"], values="skill_phase_sd").round(4))
