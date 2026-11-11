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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/spectral_ablation_final.csv")
    args = parser.parse_args()
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
                    "bin_weighted_delta": np.average(d, weights=g.width_mhz), "regions_hurt": int((d > 0).sum()),
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
