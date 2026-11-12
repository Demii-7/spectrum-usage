"""Spectral-context ablation of the rolled-out 2D models, version 2 (15 minutes; seed 42 first pass).

Why v2: the original ablation (``spectral_context_fixed.py``, source ``2d_tuned15``) trained BOTH conditions with the loss on the region's bins only, so the models'
outputs for the other bins were never trained, and the autoregressive rollout fed those untrained outputs back as input (CAMPAIGN_STATUS 39.39). Here

* ``full_context``  = the tuned 15-minute final of the same seed (loss on all 200 bins; no training here), scored on the region's bins
                      (``rescored/per-bin/p15tuned/<band>/2d/<model>/seed-<s>.csv``, the mean over the region's bins of the per-bin test MAE, mean over t+4/12/24/48);
* ``region_only``   = a new run with the bins outside the region replaced by noise-floor values in the inputs AND the targets and the loss over all 200 bins
                      (condition ``region_only_fullloss`` of ``training.ray.spectral_context_fixed``; records in ``oct2026-spectral-context-15min-v2/<band>/``).

``delta`` = region_only - full_context (dB, > 0: the other bins' context helps). Source value ``2d_final_vs_region``. Extra columns per region: the per-bin RevIN-MLP
full-band reference (``ref_revinmlp_full``, the ``direct`` rows of spectral_ablation_strong_by_region.csv, mean over its seeds) and the model's 1D counterpart's tuned
final on the same region (``ref_1d_final``: LSTM 1D for vanillalstm, residual LSTM 1D for residualvanillalstm; none for lstmattn and TCN). Also writes a checks file:
the 600-800 full-band per-region values averaged over seeds 40-44 against ``entropy_advantage_october_regions.csv`` (largest difference), the control
(``residuallinearar2d``: separate weights per bin, no cross-bin terms, so region_only must equal the final up to checkpoint selection) and the out-of-region
output statistics of the new runs (the trained model's predictions for the masked bins against the noise-floor targets).

    python evaluation/analysis/spectral_ablation_2d_v2.py [--out evaluation/results/spectral_ablation_2d_v2.csv]   (needs the SPECTRUM_S3_* environment)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PLAN = {"600_800": HERE / "plan_regions_600_800.csv", "2400_2600": HERE / "plan_regions_2400_2600.csv"}
V2_PREFIX = "oct2026-spectral-context-15min-v2"
SOURCE = "2d_final_vs_region"
HORIZONS = (4, 12, 24, 48)
MODELS = ("vanillalstm", "residualvanillalstm", "lstmattn", "temporalconvnet")
CONTROL = "residuallinearar2d"
ONE_D = {"vanillalstm": "1d/vanillalstm1d", "residualvanillalstm": "1d/residualvanillalstm"}
SEEDS = (42,)
NOTE = ("v2 first pass, single seed: full_context = tuned 15-minute final (loss on all 200 bins) scored on the region's bins; region_only = new run with the other bins "
        "noise-floor in inputs and targets and the loss over all 200 bins (about 190 unpredictable noise targets enter the loss: if region_only looks implausibly bad it may "
        "reflect that, not the lost context)")
RNG = np.random.default_rng(0)


def regions(band: str) -> pd.DataFrame:
    plan = pd.read_csv(PLAN[band])
    return plan[~plan.is_noise_floor.astype(str).str.lower().isin({"true", "1"})].reset_index(drop=True)


def per_bin(client, key: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(client.get_object(Bucket="spectrum", Key=key)["Body"].read()))
    except Exception:
        return None


def region_mae(table: pd.DataFrame, region, horizons=HORIZONS) -> float:
    """Mean over the region's bins of the per-bin MAE, averaged over the horizons."""
    sel = table[table.horizon.isin(horizons) & (table.frequency_mhz >= region.start_mhz - 1e-6) & (table.frequency_mhz <= region.end_mhz + 1e-6)]
    per_h = sel.groupby("horizon").mae_db.mean()
    return float(per_h.mean()) if len(per_h) == len(horizons) else float("nan")


def final_side(client, band: str, model: str, seed: int, plan: pd.DataFrame, rep_dir: str = "2d") -> dict[str, float]:
    table = per_bin(client, f"rescored/per-bin/p15tuned/{band}/{rep_dir}/{model}/seed-{seed}.csv") if "/" not in rep_dir else per_bin(client, f"rescored/per-bin/p15tuned/{band}/{rep_dir}/seed-{seed}.csv")
    if table is None:
        return {}
    return {r.band_id: region_mae(table, r) for r in plan.itertuples()}


def v2_records(client, band: str, model: str, seed: int, plan: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for r in plan.itertuples():
        key = f"{V2_PREFIX}/{band}/{model}/{r.band_id}-region_only_fullloss-seed-{seed}.json"
        try:
            rec = json.loads(client.get_object(Bucket="spectrum", Key=key)["Body"].read())
        except Exception:
            continue
        if rec.get("ok"):
            out[r.band_id] = rec
    return out


def bootstrap(d: np.ndarray) -> tuple[float, float]:
    draws = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build(client, seeds=SEEDS):
    rows, checks = [], []
    strong = pd.read_csv(RESULTS / "spectral_ablation_strong_by_region.csv")
    revin = strong[(strong.source == "direct") & (strong.model == "revinmlp")].set_index(["band", "region"]).full_context
    for band in ("600_800", "2400_2600"):
        plan = regions(band)
        for model in MODELS + ((CONTROL,) if band == "600_800" else ()):
            fulls = {s: final_side(client, band, model, s, plan) for s in seeds}
            recs = {s: v2_records(client, band, model, s, plan) for s in seeds}
            one_d = {s: final_side(client, band, ONE_D[model], s, plan, rep_dir=ONE_D[model]) for s in seeds} if model in ONE_D else {}
            for r in plan.itertuples():
                vals = [(fulls[s].get(r.band_id), recs[s].get(r.band_id)) for s in seeds]
                vals = [(f, rec["test_mean_horizon_mae_db"]) for f, rec in vals if f is not None and rec is not None and np.isfinite(f)]
                if not vals:
                    continue
                full, only = float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
                ref1d = [one_d[s].get(r.band_id) for s in seeds if s in one_d and one_d[s].get(r.band_id) is not None]
                rows.append({"band": band, "source": SOURCE, "model": model, "region": r.band_id, "width_mhz": r.end_mhz - r.start_mhz + 1.0, "full_context": full,
                             "region_only": only, "delta": only - full, "n_seeds": len(vals),
                             "ref_revinmlp_full": float(revin.get((band, r.band_id), np.nan)),
                             "ref_1d_final": float(np.mean(ref1d)) if ref1d else np.nan, "note": NOTE if model != CONTROL else "control: " + NOTE})
        # out-of-region output statistics of the new runs
        for model in MODELS + ((CONTROL,) if band == "600_800" else ()):
            for s in seeds:
                for region, rec in v2_records(client, band, model, s, plan).items():
                    oor = {k: v for k, v in rec.items() if k.startswith("oor_")}
                    if oor:
                        checks.append({"check": "out_of_region_outputs", "band": band, "model": model, "region": region, "seed": s,
                                       **{f"{k}": v for k, v in oor.items()}, "peak_gpu_mb": rec.get("peak_gpu_mb"), "wall_seconds": rec.get("wall_seconds")})
    by_region = pd.DataFrame(rows)
    # consistency of the full-band side with entropy_advantage_october_regions.csv (600-800, seeds 40-44)
    entropy = pd.read_csv(RESULTS / "entropy_advantage_october_regions.csv")
    entropy = entropy[(entropy.resolution == "15min")]
    plan = regions("600_800")
    for model in MODELS:
        ref = entropy[entropy.model == f"{model} (2d)"].set_index("region")
        if ref.empty:
            continue
        horizons = tuple(int(h) for h in str(ref.horizons.iloc[0]).strip("[]").split(",") if h.strip())
        per_seed = [final_side_h(client, "600_800", model, s, plan, horizons) for s in (40, 41, 42, 43, 44)]
        diffs = []
        for r in plan.itertuples():
            vals = [p[r.band_id] for p in per_seed if p.get(r.band_id) is not None and np.isfinite(p[r.band_id])]
            if vals and r.band_id in ref.index:
                diffs.append(abs(float(np.mean(vals)) - float(ref.loc[r.band_id, "mae_db"])))
        if diffs:
            checks.append({"check": "full_band_vs_entropy_regions_600_800", "band": "600_800", "model": model, "max_abs_diff_db": max(diffs), "regions": len(diffs), "horizons": str(horizons)})
    return by_region, checks


def final_side_h(client, band, model, seed, plan, horizons):
    table = per_bin(client, f"rescored/per-bin/p15tuned/{band}/2d/{model}/seed-{seed}.csv")
    return {} if table is None else {r.band_id: region_mae(table, r, horizons) for r in plan.itertuples()}


def summarize(by_region: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, source, model), g in by_region.groupby(["band", "source", "model"]):
        d = g["delta"].to_numpy()
        lo, hi = bootstrap(d) if len(d) > 1 else (float("nan"), float("nan"))
        rows.append({"band": band, "source": source, "model": model, "regions": len(g), "seeds_per_cell": int(g["n_seeds"].max()), "mean_full": g.full_context.mean(),
                     "mean_region_only": g.region_only.mean(), "mean_delta": d.mean(), "ci_low": lo, "ci_high": hi,
                     "bin_weighted_delta": float(np.average(d, weights=g.width_mhz)), "regions_context_helps": int((d > 0).sum()), "regions_context_hurts": int((d < 0).sum()),
                     "delta_pct_of_full": 100 * d.mean() / g.full_context.mean(), "ref_revinmlp_full": g.ref_revinmlp_full.mean(),
                     "ref_1d_final": g.ref_1d_final.mean() if g.ref_1d_final.notna().any() else np.nan, "note": g.note.iloc[0]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(RESULTS / "spectral_ablation_2d_v2.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    client = spectrum_s3_client()
    by_region, checks = build(client, tuple(args.seeds))
    summary = summarize(by_region)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_region.to_csv(out.with_name(out.stem + "_by_region.csv"), index=False)
    summary.to_csv(out, index=False)
    pd.DataFrame(checks).to_csv(out.with_name(out.stem + "_checks.csv"), index=False)
    pd.set_option("display.width", 250)
    print("v2: region_only (noise bins in inputs and targets, loss on all bins) - tuned final, test MAE dB mean over horizons; > 0: context helps")
    print(summary.drop(columns=["note"]).round(3).to_string(index=False))
    print(pd.DataFrame(checks).round(4).to_string(index=False) if checks else "no checks yet")


if __name__ == "__main__":
    main()
