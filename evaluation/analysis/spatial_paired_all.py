"""One paired table for Part 4: every 4D model and the RevIN-MLP +/- neighbours pair against the per-receiver 1D/2D models (15 minutes, October test week).

Models compared (base core-four IDW power-2 map for the 4D models), all on IDENTICAL forecast origins (paired on the origin time):
  4D: linearar4d, residuallinearar4d (seeds 40-44), stsprednet (40-44), residualconvlstm (40-44, when its finals exist), dswinlstm_i (seed 42 only)
  direct per-receiver with and without neighbour inputs: revinmlp, revinmlp_nbr (seeds 40-42; the same bins at the other three core receivers as extra input channels)
against: per-receiver 1D / 2D linear models, persistence, the 4D lookback mean (same map), and (for revinmlp_nbr) plain revinmlp.

Per-origin errors: 4D models from ``spatial15-origin-errors-oct2026`` (core4-p2), else from the ``real`` mode of the neighbour ablation
(``neighbor-ablation-oct2026``; same model, same geometry, same origins); revinmlp / revinmlp_nbr from ``origin-errors-oct2026``; per-receiver models from
``transfer15-oct2026``. Per origin the error is the mean over horizons t+4/12/24/48 of the MAE over the 200 bins, averaged over the seeds that exist
(columns ``n_seeds`` and ``expected_seeds``); deltas use a moving-block bootstrap over origins (block 96 = one day, 2,000 resamples), receivers resampled
independently and pooled as in ``map_construction_vs_per_receiver``. Rows with fewer seeds than expected, or one seed by design, are marked in
``seed_status`` / ``note``. Missing models are skipped (never an error), so the script can be run on whatever exists.

    python evaluation/analysis/spatial_paired_all.py --out evaluation/results/spatial_paired_all.csv [--upload]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import map_construction_vs_per_receiver as M  # noqa: E402
import neighbor_value as NV  # noqa: E402   (metric- and horizon-aware per-receiver / persistence loaders)
from training.ray.storage import spectrum_s3_client  # noqa: E402

BANDS, HORIZONS = M.BANDS, M.HORIZONS
NBR_PREFIX = "neighbor-ablation-oct2026"
DIRECT_PREFIX = "origin-errors-oct2026"
DIRECT_MSE_PREFIX = "origin-errors-oct2026-mse"            # the same direct finals re-scored with ``mse_db2`` stored
CENTERED_PREFIX = "spatial15-centered-oct2026"             # per_origin.npz of the base map with ``<receiver>|mse`` (training.ray.spatial15_centered)
MSE_ABLATION_PREFIX = "neighbor-ablation-oct2026-mse"     # real mode of the re-run neighbour ablation with ``|real|mse``
H3 = (4, 12, 24)                                          # 1, 3 and 6 hours at 15 minutes
MODELS_4D = ("linearar4d", "residuallinearar4d", "stsprednet", "residualconvlstm", "dswinlstm_i")
DIRECT = ("revinmlp", "revinmlp_nbr")
EXPECTED = {"linearar4d": (40, 41, 42, 43, 44), "residuallinearar4d": (40, 41, 42, 43, 44), "stsprednet": (40, 41, 42, 43, 44),
            "residualconvlstm": (40, 41, 42, 43, 44), "dswinlstm_i": (40, 41, 42, 43, 44), "lookbackmean4d": (42,),
            "revinmlp": (40, 41, 42), "revinmlp_nbr": (40, 41, 42)}
NOTES = {"dswinlstm_i": "inherited:1min-r2 configuration (no 15-minute search; a reduced search is running); seeds 40-44 of that configuration",
         "residualconvlstm": "weights of finals trained on the UC or final cluster (same winner configuration; see CAMPAIGN_STATUS)",
         "revinmlp_nbr": "RevIN-MLP with the same bin at the other three core receivers as extra input channels (matched control)"}
BASE_VARIANT = "core4-p2"


def npz_series(client, key: str, receiver: str, mode: str | None, metric: str = "mae", horizons: tuple = HORIZONS):
    """Per-origin mean-over-horizons error of one receiver from a per_origin.npz (``<receiver>|<metric>`` or ``<receiver>|<mode>|<metric>``)."""
    mid = f"|{mode}" if mode else ""
    try:
        with np.load(io.BytesIO(M.read(client, key)), allow_pickle=True) as npz:
            hs = list(npz["horizons"])
            pick = [hs.index(h) for h in horizons]
            return pd.Series(npz[f"{receiver}{mid}|{metric}"][:, pick].mean(axis=1), index=M.utc(npz[f"{receiver}{mid}|times"]))
    except Exception:
        return None


def series_4d(client, band: str, model: str, receiver: str, metric: str = "mae", horizons: tuple = HORIZONS):
    """MAE: spatial15 per-origin file, else the neighbour ablation's real mode. MSE (dB^2): the centred job's per-origin file, else the re-run ablation."""
    frames = []
    for seed in EXPECTED[model]:
        if metric == "mae":
            s = npz_series(client, f"{M.ORIGIN_PREFIX}/{band}/{BASE_VARIANT}/{model}/seed-{seed}/per_origin.npz", receiver, None, "mae", horizons)
            if s is None:
                s = npz_series(client, f"{NBR_PREFIX}/{band}/{model}/seed-{seed}/per_origin.npz", receiver, "real", "mae", horizons)
        else:
            s = npz_series(client, f"{CENTERED_PREFIX}/{band}/{BASE_VARIANT}/{model}/seed-{seed}/per_origin.npz", receiver, None, "mse", horizons)
            if s is None:
                s = npz_series(client, f"{MSE_ABLATION_PREFIX}/{band}/{model}/seed-{seed}/per_origin.npz", receiver, "real", "mse", horizons)
        if s is not None:
            frames.append(s)
    if not frames:
        return None, 0
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame(), len(frames)


def series_direct(client, band: str, model: str, receiver: str, metric: str = "mae", horizons: tuple = HORIZONS):
    """revinmlp / revinmlp_nbr: per seed the horizons (one model each) averaged per origin time, then the seed mean (MSE from the re-scored prefix)."""
    prefix, column = (DIRECT_PREFIX, "mae_db") if metric == "mae" else (DIRECT_MSE_PREFIX, "mse_db2")
    frames = []
    for seed in EXPECTED[model]:
        per_h = []
        for h in horizons:
            try:
                raw = M.read(client, f"{prefix}/15min/{band}/{receiver}/direct/{model}/h{h}/seed-{seed}/per_origin.csv")
                per_h.append(pd.read_csv(io.BytesIO(raw)).set_index("time")[column])
            except Exception:
                per_h = []
                break
        if per_h:
            s = pd.concat(per_h, axis=1).mean(axis=1, skipna=False).dropna()
            s.index = M.utc(s.index.to_numpy())
            frames.append(s)
    if not frames:
        return None, 0
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame(), len(frames)


def load(client, band: str, model: str, receiver_4d: str, receiver: str, metric: str = "mae", horizons: tuple = HORIZONS):
    if model in DIRECT:
        return series_direct(client, band, model, receiver, metric, horizons)
    return series_4d(client, band, model, receiver_4d, metric, horizons)


def status(model: str, n_seeds: int) -> str:
    exp = len(EXPECTED[model])
    if exp == 1:
        return "single seed by design"
    return "all seeds" if n_seeds >= exp else f"partial seeds ({n_seeds} of {exp})"


def compare(client, metric: str = "mae", horizons: tuple = HORIZONS) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        refs = {}
        for r4, r in M.RECEIVERS.items():
            for rep, name in M.PER_RECEIVER:
                refs[(r4, f"{rep} {name}")] = NV.per_receiver_series(client, band, r, rep, name, metric, horizons)
            refs[(r4, "persistence")] = NV.persistence_series(client, band, r, metric, horizons)
            refs[(r4, "4D lookback mean (same map)")] = (series_4d(client, band, "lookbackmean4d", r4, metric, horizons)[0] if metric == "mae" else
                                                         NV.four_d_series(client, band, "lookbackmean4d", "real", r4, metric, horizons))
        for model in (*MODELS_4D, *DIRECT):
            ours = {r4: load(client, band, model, r4, r, metric, horizons) for r4, r in M.RECEIVERS.items()}
            if any(v[0] is None for v in ours.values()):
                print(f"[paired] {band} {model} ({metric}, {len(horizons)} horizons): not available yet, skipped", flush=True)
                continue
            versus = {name: {r4: refs[(r4, name)] for r4 in M.RECEIVERS} for name in
                      [f"{rep} {n}" for rep, n in M.PER_RECEIVER] + ["persistence", "4D lookback mean (same map)"]}
            if model == "revinmlp_nbr":
                versus["revinmlp (no neighbours)"] = {r4: load(client, band, "revinmlp", r4, r, metric, horizons)[0] for r4, r in M.RECEIVERS.items()}
            if model in MODELS_4D:
                for d in DIRECT:
                    versus[f"{d} (per receiver, direct)"] = {r4: load(client, band, d, r4, r, metric, horizons)[0] for r4, r in M.RECEIVERS.items()}
            for name, others in versus.items():
                entries = []
                for r4 in M.RECEIVERS:
                    other = others.get(r4)
                    if other is None:
                        break
                    delta, n, mean_a, mean_b = M.paired(ours[r4][0], other)
                    entries.append((n, mean_a, mean_b, float(delta.mean()), M.block_bootstrap(delta)))
                if len(entries) != len(M.RECEIVERS):
                    continue
                draws = np.mean(np.stack([e[4] for e in entries]), axis=0)
                n_seeds = min(v[1] for v in ours.values())
                rows.append({"band": band, "model": model, "versus": name, "receiver": "pooled", "n_origins": int(sum(e[0] for e in entries)),
                             "n_seeds": int(n_seeds), "expected_seeds": len(EXPECTED[model]), "seed_status": status(model, n_seeds),
                             "note": NOTES.get(model, ""), "mae_model": float(np.mean([e[1] for e in entries])),
                             "mae_other": float(np.mean([e[2] for e in entries])), "delta_model_minus_other": float(np.mean([e[3] for e in entries])),
                             "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


EXTRA = (("mae", H3, "_h3"), ("mse", HORIZONS, ""), ("mse", H3, "_h3"))     # (metric, horizons, horizon suffix of the new columns)
KEYS = ["band", "model", "versus"]


def add_extra(table: pd.DataFrame, client) -> pd.DataFrame:
    """Next to the existing columns (MAE, t+4/12/24/48; unchanged): MAE over t+4/12/24 (1, 3, 6 h) ``*_h3``; squared error (dB^2) ``mse_*`` and ``*_mse``; both ``*_mse_h3``."""
    for metric, horizons, hsuffix in EXTRA:
        extra = compare(client, metric, horizons)
        if extra.empty:
            continue
        mae = metric == "mae"
        tag = "" if mae else "_mse"
        names = {"mae_model": "mae_model" if mae else "mse_model", "mae_other": "mae_other" if mae else "mse_other",
                 "delta_model_minus_other": "delta_model_minus_other" if mae else "delta_mse"}
        cols = {old: new + hsuffix for old, new in names.items()}
        cols.update({"ci_low": f"ci_low{tag}{hsuffix}", "ci_high": f"ci_high{tag}{hsuffix}", "n_origins": f"n_origins{tag}{hsuffix}"})
        table = table.merge(extra[KEYS + list(cols)].rename(columns=cols), on=KEYS, how="left")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="evaluation/results/spatial_paired_all.csv")
    ap.add_argument("--upload", action="store_true", help="also put the CSV in s3://spectrum/analysis-outputs/")
    a = ap.parse_args()
    client = spectrum_s3_client()
    table = compare(client)
    if len(table):
        table = add_extra(table, client)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    for band in BANDS:
        d = table[table.band == band] if len(table) else table
        for model in (*MODELS_4D, *DIRECT):
            m = d[d.model == model] if len(d) else d
            if m.empty:
                continue
            print(f"\n{band} {model}: {m.seed_status.iloc[0]}; delta = model minus other (dB, negative = model better), 95% block-bootstrap interval")
            for _, r in m.iterrows():
                print(f"  vs {r.versus:34s} {r.mae_model:.3f} vs {r.mae_other:.3f}  {r.delta_model_minus_other:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  n={r.n_origins}")
    if a.upload:
        key = "analysis-outputs/spatial_paired_all.csv"
        client.put_object(Bucket=M.BUCKET, Key=key, Body=out.read_bytes())
        print("uploaded", key)


if __name__ == "__main__":
    main()
