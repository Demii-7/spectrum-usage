"""Derived tables cited in README.md, computed from full-1/ and sampling-1/ (run from the repo root)."""
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[1]
r = pd.read_csv(D / "full-1/results.csv")
t = r[r.split == "test"]
key = ["node", "band", "horizon"]

# 1. GBM (MAE loss) and the best per-bin ridge vs the validation-selected strong baseline, per cell (point estimate)
strong = t[(t.experiment == "baselines") & (t.predictor == "strong")].set_index(key).mae.rename("strong_mae")
m60 = t[(t.experiment == "baselines") & (t.predictor == "mean60")].set_index(key).mae.rename("mean60_mae")
out = pd.concat([m60, strong], axis=1)
for p in ("pooled_gbm_mae:val_scaled", "pooled_linear:val_scaled", "mean60:val_scaled", "last:val_scaled", "none:full"):
    out[p] = t[t.predictor == p].set_index(key).mae
for c in ["strong_mae", "pooled_gbm_mae:val_scaled", "pooled_linear:val_scaled", "mean60:val_scaled", "last:val_scaled", "none:full"]:
    out[f"skill_vs_strong[{c}]"] = 1 - out[c] / out["strong_mae"] if c != "strong_mae" else 1 - out[c] / out["mean60_mae"]
out = out.rename(columns={"skill_vs_strong[strong_mae]": "strong_skill_vs_mean60"})
out.round(4).to_csv(D / "derived_probe_vs_strong.csv")

# 2. Error floors: white-noise share of the best achieved MSE
s = pd.read_csv(D / "sampling-1/sampling.csv")
fl = s[s.part == "floor"].set_index(["node", "band", "target", "horizon"])
st = s[(s.part == "sampling") & (s.split == "test")]
best = st.groupby(["node", "band", "target", "horizon"]).mse.min().rename("best_mse")
f = fl[["floor_mse", "noise_only_mse", "level_est_mse", "drift_mse"]].join(best)
f["noise_share_of_best"] = f.noise_only_mse / f.best_mse
f["noise_plus_levelest_share_of_best"] = (f.noise_only_mse + f.level_est_mse) / f.best_mse
f["floor_over_best"] = f.floor_mse / f.best_mse
f.round(4).to_csv(D / "derived_floors.csv")

# 3. Like-for-like 60-min-ahead: best of {ridge, strong} per representation (decim averaged over phases), mean over receivers
st = st.assign(rep2=st.rep.where(~st.rep.astype(str).str.startswith("decim"), "decim"))
g = st.groupby(["band", "target", "horizon", "lookback_min", "rep2", "node", "predictor"])[["mae", "mse"]].mean().reset_index()
b = g.loc[g.groupby(["band", "target", "horizon", "lookback_min", "rep2", "node"]).mae.idxmin()]
ll = b.groupby(["band", "target", "horizon", "lookback_min", "rep2"])[["mae"]].mean().unstack("rep2")
ll.round(3).to_csv(D / "derived_like_for_like_best_mae.csv")

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = [c for c in out.columns if c.startswith("skill") or c.startswith("strong_skill")]
    print(out[cols].groupby(level=["band", "horizon"]).agg(["min", "median", "max"]).round(3).T.to_string())
    print(f[["noise_share_of_best", "noise_plus_levelest_share_of_best", "floor_over_best"]].round(2).to_string())
    print(f.groupby(["target", "horizon"])[["noise_share_of_best", "noise_plus_levelest_share_of_best"]].median().round(2))
    print(ll[ll.index.get_level_values("horizon") == 60].round(3))
