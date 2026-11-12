"""Does model advantage over the simple baseline grow with a band's entropy? (October benchmark; per-bin and per-region)

Joins ``evaluation/results/october_entropy.csv`` (power-difference entropy of the October training run, ``october_entropy.py``) with
per-bin TEST errors of the tuned models and the lookback-mean baseline (whole-window mean) from the durable bucket, both resolutions:

* 1 minute, 600-800: ``oct2026-results/final-eval-v3/main/per-bin`` (round-2 table when a model has one); 2400-2600:
  ``rescored/per-bin/band/2400_2600`` (horizons 1/15/60 minutes);
* 15 minutes: ``rescored/per-bin/p15tuned/<band>`` (horizons that the baseline table also holds);
* baseline (``--baseline``): by default the lookback median with the validation-selected window per horizon
  (``median-study/selected_k.json`` + ``median-study/sweep/<resolution>/<band>/per-bin/median_k<k>.csv``), as in the paper's tables;
  ``mean-whole`` restores the earlier whole-window lookback mean (``mean_k60|mean_k96.csv``).

Per model and band: Spearman correlation of entropy with error, with the relative advantage (model - baseline) / baseline per bin, mean
relative advantage in entropy terciles, and the share of bins where the model beats the baseline. All errors are means over seeds and the
common horizons. Region level (600-800 only): the same on the annotated non-noise regions (region error = mean over its bins).

    python evaluation/analysis/entropy_advantage_october.py [--out evaluation/results/entropy_advantage_october.csv]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

B = "spectrum"
BASELINE_K = {"1min": 60, "15min": 96}
# Direct finals (one model trained per horizon): every horizon is scored from its OWN checkpoint, so a model's per-bin table set is the union over horizons (and seeds)
# of ``rescored/per-bin/direct/<res>/<band>/2d/<model>/h<h>/seed-N.csv`` (stride-1 rescoring); where the rescoring does not cover all horizons yet (Time-LLM at 1 minute) the
# finals' own tables ``direct-tuning/<res>/<band>/<model>/per-bin/h<h>/seed-N.csv`` are used (1-minute test windows thinned 15x).
DIRECT_MODELS = ("revinmlp", "patchtst", "ofa", "timellm", "nlinear", "dlinear", "deepspred2d", "timeran_lp")
# Collapsed finals left out, as in the per-horizon tables (seed MSE above 1.5x the cell median): 3D-SwinSTB, 15 minutes, 2400-2600, seed 43 at 3 h and 6 h.
EXCLUDED = ("direct-tuning/15min/2400_2600/deepspred2d/per-bin/h12/seed-43.csv", "direct-tuning/15min/2400_2600/deepspred2d/per-bin/h24/seed-43.csv")


def read_csv(client, key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(client.get_object(Bucket=B, Key=key)["Body"].read()))


def listing(client, prefix: str) -> list[str]:
    return [o["Key"] for p in client.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix) for o in p.get("Contents", [])]


def rolled_model_tables(client, resolution: str, band: str) -> dict[str, list[str]]:
    """model label -> list of per-bin table keys (one per seed) for the tuned rolled-out 1D and 2D models."""
    found: dict[str, dict[str, list[str]]] = {}
    if resolution == "1min" and band == "600_800":
        for key in listing(client, "oct2026-results/final-eval-v3/main/per-bin/"):
            parts = key.split("/")           # .../per-bin/<prefix>/<rep>/<model>/seed-N.csv
            prefix, rep, model = parts[4], parts[5], parts[6]
            found.setdefault(f"{rep} {model}", {}).setdefault(prefix, []).append(key)
        # round 2 where it exists, else round 1
        return {m: (v.get("checkpoints-oct2026-r2") or v.get("checkpoints-oct2026") or next(iter(v.values()))) for m, v in found.items()}
    prefix = "rescored/per-bin/band/2400_2600/" if resolution == "1min" else f"rescored/per-bin/p15tuned/{band}/"
    out: dict[str, list[str]] = {}
    for key in listing(client, prefix):
        parts = key.split("/")
        rep, model = parts[-3], parts[-2]
        if rep in ("1d", "2d"):
            out.setdefault(f"{rep} {model}", []).append(key)
    return out


def direct_model_tables(client, resolution: str, band: str) -> dict[str, list[str]]:
    """``direct <model>`` -> per-bin table keys of every (horizon, seed) of the direct finals (see ``DIRECT_MODELS``)."""
    out: dict[str, list[str]] = {}
    for model in DIRECT_MODELS:
        rescored = [k for k in listing(client, f"rescored/per-bin/direct/{resolution}/{band}/2d/{model}/") if k.endswith(".csv")]
        finals = [k for k in listing(client, f"direct-tuning/{resolution}/{band}/{model}/per-bin/") if k.endswith(".csv")]
        keys = [k for k in (rescored if len(rescored) >= len(finals) else finals) if k not in EXCLUDED]
        if keys:
            out[f"direct {model}"] = sorted(keys)
    return out


def tsslcd_model_tables(client, resolution: str, band: str) -> dict[str, list[str]]:
    """``tsslcd (tuned)`` -> per-bin tables (one per seed, all horizons, mean over the four receivers) of the TSS-LCD tuned finals, built by
    ``tsslcd_per_bin_tables.py`` from the eval-only rescoring (``tsslcd_authors.py --eval-only --per-bin-output``)."""
    keys = [k for k in listing(client, f"rescored/per-bin/tsslcd/{resolution}/{band}/tuned/") if k.endswith(".csv")]
    return {"tsslcd (tuned)": sorted(keys)} if keys else {}


def model_tables(client, resolution: str, band: str) -> dict[str, list[str]]:
    """model label -> per-bin table keys: the tuned 1D/2D models (one table per seed), the direct finals (one table per horizon and seed) and TSS-LCD."""
    return {**rolled_model_tables(client, resolution, band), **direct_model_tables(client, resolution, band), **tsslcd_model_tables(client, resolution, band)}


def probe_keys(label: str, keys: list[str]) -> list[str]:
    """The tables that tell which horizons a model has: one table holds all horizons of a rolled-out model, a direct model has one table per horizon."""
    return keys if label.startswith("direct ") else keys[:1]


def n_seeds_of(keys: list[str]) -> int:
    return len({k.rsplit("seed-", 1)[1].split(".")[0] for k in keys})


def per_bin_error(client, keys: list[str], horizons: set[int] | None) -> tuple[pd.Series, set[int]]:
    frames = [read_csv(client, k) for k in keys]
    table = pd.concat(frames)
    table = table[table["split"].astype(str).str.endswith("test")]
    if horizons is not None:
        table = table[table["horizon"].isin(horizons)]
    table["bin"] = table["frequency_mhz"].round(1)
    return table.groupby("bin").mae_db.mean(), set(int(h) for h in table["horizon"].unique())


def baseline_per_bin(client, resolution: str, band: str, horizons: set[int] | None, kind: str) -> tuple[pd.Series, set[int]]:
    """Per-bin TEST MAE of the simple baseline, averaged over ``horizons`` (in steps).

    ``median-tuned`` (default): lookback median with the window selected on validation for each horizon
    (``median-study/selected_k.json``), i.e. the same baseline as the paper's tables; the median is the MAE-optimal location estimate.
    ``mean-whole``: lookback mean over the whole input window (k = 60 / 96), the baseline used before November 2026.
    """
    if kind == "mean-whole":
        key = f"median-study/sweep/{resolution}/{band}/per-bin/mean_k{BASELINE_K[resolution]}.csv"
        return per_bin_error(client, [key], horizons)
    k_per_h = json.loads(client.get_object(Bucket=B, Key="median-study/selected_k.json")["Body"].read())[f"{resolution}/{band}"]["k_per_horizon"]
    available = {int(h) for h in k_per_h}
    use = sorted(available if horizons is None else available & set(horizons))
    parts = []
    for h in use:
        key = f"median-study/sweep/{resolution}/{band}/per-bin/median_k{k_per_h[str(h)]}.csv"
        err, _ = per_bin_error(client, [key], {h})
        parts.append(err)
    return pd.concat(parts, axis=1).mean(axis=1), set(use)


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(pd.concat([a, b], axis=1).dropna().rank().corr().iloc[0, 1])


def tiers(entropy: pd.Series, advantage: pd.Series) -> list[float]:
    frame = pd.concat([entropy.rename("e"), advantage.rename("a")], axis=1).dropna()
    frame["tier"] = pd.qcut(frame.e.rank(method="first"), 3, labels=False)
    return [float(frame[frame.tier == t].a.mean()) for t in range(3)]


def tier_ranges(entropy: pd.Series, advantage: pd.Series) -> dict[str, float]:
    """Entropy range (min, max) of each tercile used by ``tiers`` (1-minute entropy), for reporting the terciles in entropy units."""
    frame = pd.concat([entropy.rename("e"), advantage.rename("a")], axis=1).dropna()
    frame["tier"] = pd.qcut(frame.e.rank(method="first"), 3, labels=False)
    out = {}
    for t in range(3):
        e = frame[frame.tier == t].e
        out[f"entropy_tier{t + 1}_min"], out[f"entropy_tier{t + 1}_max"] = float(e.min()), float(e.max())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entropy", default="evaluation/results/october_entropy.csv")
    parser.add_argument("--out", default="evaluation/results/entropy_advantage_october.csv")
    parser.add_argument("--plan", default="evaluation/analysis/plan_regions_600_800.csv")
    parser.add_argument("--baseline", choices=("median-tuned", "mean-whole"), default="median-tuned")
    args = parser.parse_args()
    client = spectrum_s3_client()
    entropy = pd.read_csv(args.entropy)
    plan = pd.read_csv(args.plan)
    rows = []
    for resolution in ("1min", "15min"):
        for band in ("600_800", "2400_2600"):
            ent_bin = entropy[(entropy.resolution == resolution) & (entropy.band == band) & (entropy.level == "bin")]
            ent_bin = ent_bin.assign(bin=ent_bin.start_mhz.round(1)).set_index("bin").entropy
            # Terciles are defined on the 1-minute entropy at both resolutions, so the same bins (and regions)
            # fall in each tercile at 1 and 15 minutes; correlations use each resolution's own entropy.
            tier_bin = entropy[(entropy.resolution == "1min") & (entropy.band == band) & (entropy.level == "bin")]
            tier_bin = tier_bin.assign(bin=tier_bin.start_mhz.round(1)).set_index("bin").entropy
            tier_reg = entropy[(entropy.resolution == "1min") & (entropy.band == band) & (entropy.level == "region")].set_index("id").entropy
            _, base_h = baseline_per_bin(client, resolution, band, None, args.baseline)
            tables = model_tables(client, resolution, band)
            for label, keys in sorted(tables.items()):
                try:
                    probe, hs = per_bin_error(client, probe_keys(label, keys), None)
                    common = hs & base_h
                    if not common:
                        continue
                    err, _ = per_bin_error(client, keys, common)
                    base, _ = baseline_per_bin(client, resolution, band, common, args.baseline)
                except Exception as error:               # a table with a different layout is skipped, and reported
                    print(f"[skip] {resolution} {band} {label}: {type(error).__name__}: {error}", file=sys.stderr)
                    continue
                adv = (err - base) / base
                gap = err - base                                        # dB; the paper's figure plots this difference
                row = {"resolution": resolution, "band": band, "level": "bin", "model": label, "n_seeds": n_seeds_of(keys), "horizons": sorted(common),
                       "n_units": int(adv.notna().sum()), "rho_entropy_error": spearman(ent_bin, err),
                       "rho_entropy_baseline_error": spearman(ent_bin, base), "rho_entropy_advantage": spearman(ent_bin, adv),
                       "mean_rel_advantage": float(adv.mean()), "share_bins_better": float((adv < 0).mean())}
                row.update({f"adv_tier{t + 1}": v for t, v in enumerate(tiers(tier_bin, adv))})
                row.update({"rho_entropy_gap_db": spearman(ent_bin, gap), "mean_gap_db": float(gap.mean())})
                row.update({f"gap_db_tier{t + 1}": v for t, v in enumerate(tiers(tier_bin, gap))})
                row.update(tier_ranges(tier_bin, gap))
                rows.append(row)
                if band == "600_800":
                    keep = plan[~plan.is_noise_floor.astype(str).str.lower().eq("true")]
                    ent_reg = entropy[(entropy.resolution == resolution) & (entropy.band == band) & (entropy.level == "region")].set_index("id").entropy
                    reg_err, reg_base = {}, {}
                    for r in keep.itertuples():
                        sel = [b for b in err.index if r.start_mhz - 1e-6 <= b <= r.end_mhz + 1e-6]
                        if sel:
                            reg_err[r.band_id], reg_base[r.band_id] = err.loc[sel].mean(), base.loc[sel].mean()
                    re_, rb_ = pd.Series(reg_err), pd.Series(reg_base)
                    radv = (re_ - rb_) / rb_
                    rgap = re_ - rb_
                    rr = {"resolution": resolution, "band": band, "level": "region", "model": label, "n_seeds": n_seeds_of(keys), "horizons": sorted(common),
                          "n_units": int(len(radv)), "rho_entropy_error": spearman(ent_reg.reindex(re_.index), re_),
                          "rho_entropy_baseline_error": spearman(ent_reg.reindex(rb_.index), rb_),
                          "rho_entropy_advantage": spearman(ent_reg.reindex(radv.index), radv), "mean_rel_advantage": float(radv.mean()),
                          "share_bins_better": float((radv < 0).mean())}
                    rr.update({f"adv_tier{t + 1}": v for t, v in enumerate(tiers(tier_reg.reindex(radv.index), radv))})
                    rr.update({"rho_entropy_gap_db": spearman(ent_reg.reindex(rgap.index), rgap), "mean_gap_db": float(rgap.mean())})
                    rr.update({f"gap_db_tier{t + 1}": v for t, v in enumerate(tiers(tier_reg.reindex(rgap.index), rgap))})
                    rr.update(tier_ranges(tier_reg.reindex(rgap.index), rgap))
                    rows.append(rr)
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    for (res, band, level), d in table.groupby(["resolution", "band", "level"]):
        print(f"\n{res} {band} per {level} ({int(d.n_units.median())} units, {len(d)} models; horizons {d.horizons.iloc[0]}): "
              f"baseline rho(entropy, error) = {d.rho_entropy_baseline_error.iloc[0]:.2f}; "
              f"median over models: rho(entropy, error) {d.rho_entropy_error.median():.2f}, rho(entropy, advantage) {d.rho_entropy_advantage.median():.2f}; "
              f"mean relative advantage by entropy tercile (low/mid/high) "
              f"{d.adv_tier1.mean():+.3f} / {d.adv_tier2.mean():+.3f} / {d.adv_tier3.mean():+.3f}; tercile 1-min entropy ranges "
              + " / ".join(f"{d[f'entropy_tier{t}_min'].iloc[0]:.2f}-{d[f'entropy_tier{t}_max'].iloc[0]:.2f}" for t in (1, 2, 3)))
        print(d[["model", "n_seeds", "rho_entropy_error", "rho_entropy_gap_db", "mean_gap_db", "gap_db_tier1", "gap_db_tier2", "gap_db_tier3",
                 "adv_tier1", "adv_tier2", "adv_tier3", "share_bins_better"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
