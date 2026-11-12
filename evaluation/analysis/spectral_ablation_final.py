"""Do models use spectral context? Full-band input against region-only input (other bins replaced by noise-floor values), October benchmark.

Reads the records of ``training.ray.spectral_context_fixed`` (``oct2026-spectral-context/`` at 1 minute, ``oct2026-spectral-context-15min/``
at 15 minutes; seeds 40-42; five 2D models; the 13 regions of ``plan_regions_600_800.csv`` minus the noise-floor regions are those the
driver ran). For every (model, region) the paired difference ``region_only - full_context`` in test MAE (dB, mean over the horizons,
averaged over seeds) is the effect of removing the out-of-region bins; positive = the models used spectral context. Per model: the mean
of the per-region differences (each region counted once, whatever its width), the bin-weighted mean, the number of regions where
removing the context hurts, and a bootstrap interval over regions (2,000 resamples). Needs the SPECTRUM_S3_* environment.

    python evaluation/analysis/spectral_ablation_final.py [--out evaluation/results/spectral_ablation_final.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

STRONG_PREFIX = "oct2026-spectral-context-15min-strong/"                      # direct models (training.ray.spectral_context_direct)
TUNED_2D = {"600_800": "oct2026-spectral-context-15min-tuned/", "2400_2600": "oct2026-spectral-context-15min-tuned-2400_2600/"}  # tuned 2D models
FINAL_HORIZONS = (4, 12, 24, 48)
PREFIXES = {"1min": "oct2026-spectral-context/", "15min": "oct2026-spectral-context-15min/"}
SEEDS = (40, 41, 42)
RNG = np.random.default_rng(0)


def load(bucket: str = "spectrum") -> pd.DataFrame:
    client = spectrum_s3_client()
    rows = []
    for resolution, prefix in PREFIXES.items():
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", []) if o["Key"].endswith(".json")]
        with ThreadPoolExecutor(16) as pool:
            records = list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))
        for r in records:
            if r.get("ok") and int(r["seed"]) in SEEDS:
                rows.append({"resolution": resolution, "model": r["model"], "region": r["band_id"], "condition": r["condition"],
                             "seed": int(r["seed"]), "start_mhz": float(r["start_mhz"]), "end_mhz": float(r["end_mhz"]),
                             **{k: float(v) for k, v in r.items() if k.startswith("test_mean_horizon_mae") or k.startswith("test_mae_db_t")}})
    return pd.DataFrame(rows)


def _records(client, bucket: str, prefix: str) -> list[dict]:
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", []) if o["Key"].endswith(".json") and not o["Key"].endswith(".error.json")
            and "/summary_" not in o["Key"]]
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(lambda k: json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()), keys))


def load_strong(bucket: str = "spectrum") -> pd.DataFrame:
    """Per-cell test MAE (mean over the horizons) of every ablation run at 15 minutes: the direct models (one training per horizon, so a
    (model, region, condition, seed) needs all four horizons) and the tuned 2D models (both bands)."""
    client = spectrum_s3_client()
    rows = []
    direct: dict[tuple, dict[int, float]] = {}
    for r in _records(client, bucket, STRONG_PREFIX):
        if r.get("ok") and int(r["horizon"]) in FINAL_HORIZONS:
            direct.setdefault((r["band"], r["model"], r["region"], r["condition"], int(r["seed"]), r["start_mhz"], r["end_mhz"]), {})[int(r["horizon"])] = \
                float(r[f"test_mae_db_t{int(r['horizon'])}"])
    for (band, model, region, condition, seed, start, end), by_h in direct.items():
        if set(by_h) == set(FINAL_HORIZONS):
            rows.append({"band": band, "source": "direct", "model": model, "region": region, "condition": condition, "seed": seed,
                         "start_mhz": start, "end_mhz": end, "mae": float(np.mean([by_h[h] for h in FINAL_HORIZONS]))})
    for band, prefix in TUNED_2D.items():
        for r in _records(client, bucket, prefix):
            if r.get("ok"):
                rows.append({"band": band, "source": "2d_tuned15", "model": r["model"], "region": r["band_id"], "condition": r["condition"],
                             "seed": int(r["seed"]), "start_mhz": float(r["start_mhz"]), "end_mhz": float(r["end_mhz"]),
                             "mae": float(r["test_mean_horizon_mae_db"])})
    return pd.DataFrame(rows)


def in_distribution(bucket: str = "spectrum") -> pd.DataFrame:
    """Full-band test MAE (mean over horizons 4/12/24/48 and over the finals' seeds) of the direct models: RevIN-MLP and PatchTST from
    ``part1_per_horizon_15min.csv`` when present, the rest (and any missing) from the tuned finals in ``direct-tuning/15min``."""
    out = {}
    part1 = Path(__file__).resolve().parents[1] / "results" / "part1_per_horizon_15min.csv"
    if part1.exists():
        t = pd.read_csv(part1)
        t = t[(t.family == "direct") & (t.metric == "mae") & t.horizon.isin(FINAL_HORIZONS) & t.model.isin(["revinmlp", "patchtst"])]
        for (band, model), g in t.groupby(["band", "model"]):
            if len(g) == len(FINAL_HORIZONS):
                out[(band, model)] = (float(g["mean"].mean()), "part1_per_horizon_15min.csv")
    client = spectrum_s3_client()
    for band in ("600_800", "2400_2600"):
        for model in ("revinmlp", "patchtst", "revinmlp_mix", "itransformer", "patchtst_joint"):
            if (band, model) in out:
                continue
            records = [r for r in _records(client, bucket, f"direct-tuning/15min/{band}/{model}/finals/") if r.get("ok")]
            by_h = {h: [r["test_mae_db_t%d" % h] for r in records if int(r["horizon"]) == h] for h in FINAL_HORIZONS}
            if all(by_h.values()):
                out[(band, model)] = (float(np.mean([np.mean(v) for v in by_h.values()])), "direct-tuning finals")
    return pd.DataFrame([{"band": b, "model": m, "in_dist_mae": v, "in_dist_source": src} for (b, m), (v, src) in out.items()])


def bootstrap(d: np.ndarray) -> tuple[float, float]:
    draws = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def strong_main(out_path: str) -> None:
    """spectral_ablation_strong.csv (per band and model) and _by_region.csv. delta = region_only - full_context (mean over horizons, seeds averaged,
    each region once; > 0 = removing the other bins' inputs made the region's MAE worse = the context helps. ``regions_context_helps`` / ``_hurts`` count the
    regions with delta > 0 / < 0; they replace the earlier ``regions_hurt``, which counted delta > 0, i.e. the regions where the context helped);
    for the matched pair also input_only and the difference input_only - region_only (> 0: the lost training signal, not
    the lost inputs, hurts)."""
    frame = load_strong()
    wide = frame.pivot_table(index=["band", "source", "model", "region", "seed", "start_mhz", "end_mhz"], columns="condition", values="mae").reset_index()
    for c in ("full_context", "region_only", "input_only"):
        if c not in wide:
            wide[c] = np.nan
    wide["width_mhz"] = wide["end_mhz"] - wide["start_mhz"] + 1.0
    wide["delta"] = wide["region_only"] - wide["full_context"]
    wide["input_only_minus_region_only"] = wide["input_only"] - wide["region_only"]
    keys = ["band", "source", "model", "region", "width_mhz"]
    by_region = (wide.groupby(keys).agg(full_context=("full_context", "mean"), region_only=("region_only", "mean"), input_only=("input_only", "mean"),
                                        delta=("delta", "mean"), input_only_minus_region_only=("input_only_minus_region_only", "mean"),
                                        n_seeds=("delta", "count")).reset_index())
    by_region = by_region[by_region["delta"].notna()]
    indist = in_distribution()
    rows = []
    for (band, source, model), g in by_region.groupby(["band", "source", "model"]):
        d = g["delta"].to_numpy()
        low, high = bootstrap(d)
        row = {"band": band, "source": source, "model": model, "regions": len(g), "seeds_per_cell": int(g["n_seeds"].max()),
               "mean_full": g.full_context.mean(), "mean_region_only": g.region_only.mean(), "mean_delta": d.mean(), "ci_low": low, "ci_high": high,
               "bin_weighted_delta": float(np.average(d, weights=g.width_mhz)), "regions_context_helps": int((d > 0).sum()), "regions_context_hurts": int((d < 0).sum()),
               "delta_pct_of_full": 100 * d.mean() / g.full_context.mean()}
        h = g[g["input_only_minus_region_only"].notna()]
        if len(h):
            dd = h["input_only_minus_region_only"].to_numpy()
            lo, hi = bootstrap(dd)
            row.update(mean_input_only=h.input_only.mean(), input_only_delta=float((h.input_only - h.full_context).mean()),
                       input_minus_region_only=dd.mean(), input_minus_region_ci_low=lo, input_minus_region_ci_high=hi,
                       regions_input_only_worse=int((dd > 0).sum()))
        rows.append(row)
    summary = pd.DataFrame(rows).merge(indist, on=["band", "model"], how="left")
    refs = indist[indist.model.isin(["revinmlp", "patchtst"])].pivot(index="band", columns="model", values="in_dist_mae").add_prefix("ref_in_dist_mae_")
    summary = summary.merge(refs.reset_index(), on="band", how="left").sort_values(["band", "source", "model"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    by_region.to_csv(Path(out_path).with_name(Path(out_path).stem + "_by_region.csv"), index=False)
    summary.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    print("region_only - full_context, test MAE dB (mean over horizons), per-region differences averaged over seeds; > 0: context helps")
    print(summary.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/spectral_ablation_final.csv")
    parser.add_argument("--strong", action="store_true",
                        help="15-minute ablation of the strong cross-frequency models and the 2400-2600 tuned 2D models; writes spectral_ablation_strong(.csv|_by_region.csv)")
    args = parser.parse_args()
    if args.strong:
        strong_main(args.out if args.out != "evaluation/results/spectral_ablation_final.csv" else "evaluation/results/spectral_ablation_strong.csv")
        return
    frame = load()
    metric = "test_mean_horizon_mae_db"
    wide = frame.pivot_table(index=["resolution", "model", "region", "seed", "start_mhz", "end_mhz"], columns="condition", values=metric).reset_index()
    wide["delta"] = wide["region_only"] - wide["full_context"]
    wide["width_mhz"] = wide["end_mhz"] - wide["start_mhz"] + 1.0
    per_region = (wide.groupby(["resolution", "model", "region", "width_mhz"])
                  .agg(full=("full_context", "mean"), region_only=("region_only", "mean"), delta=("delta", "mean"), n_seeds=("seed", "nunique"))
                  .reset_index())
    out = []
    for (resolution, model), g in per_region.groupby(["resolution", "model"]):
        d = g["delta"].to_numpy()
        draws = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        out.append({"resolution": resolution, "model": model, "regions": len(g), "mean_full": g.full.mean(), "mean_region_only": g.region_only.mean(),
                    "mean_delta": d.mean(), "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5),
                    "bin_weighted_delta": np.average(d, weights=g.width_mhz), "regions_context_helps": int((d > 0).sum()), "regions_context_hurts": int((d < 0).sum()),
                    "delta_pct_of_full": 100 * d.mean() / g.full.mean()})
    summary = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_region.to_csv(Path(args.out).with_name(Path(args.out).stem + "_by_region.csv"), index=False)
    summary.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print("region_only - full_context, MAE dB (mean over horizons), per-region differences averaged over seeds 40-42; > 0 means context helps")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
