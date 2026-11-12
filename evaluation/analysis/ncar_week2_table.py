"""NCAR transfer batch, item 2: the week-2 table (later interval: the four core receivers, 2026-11-03 20:00 to 11-10 20:00 UTC).

Sources (existing records, nothing rescored):
  week 2     ``rescored-week2full/p15tuned/<band>/<rep>/<model>/seed-N.json`` (tuned rolled-out 1D/2D, t+4..t+48) and
             ``rescored-week2full/direct/15min/<band>/<rep>/<model>/h<h>/seed-N.json`` (direct finals, one record per trained horizon);
             TSS-LCD ``tsslcd-authors-15min-{tuned,tuned-2400,default,default-2400}-week2full/savgol-<s>/seed-N.json``.
  October    the same rescore code on the October test week: ``rescored/p15tuned/...`` and ``rescored/direct/15min/...``; TSS-LCD
             ``tsslcd-authors-15min-{tuned,...}/savgol-<s>/seed-N.json`` (the records the Part 1 table uses).
  baselines  item 1 (``evaluation/results/transfer_baselines.csv``, group ``core4_week2``: the transfer windows, history taken from the 24 h
             before the week) and, recomputed here from the item-1 arrays (``transfer-ncar-2026-11/item1/powder/<receiver>/20261025T1355Z-week2/
             <band>/<baseline>/forecasts.npz``), the same baselines restricted to the rescore windows (origins whose 96-step history lies inside
             the week, i.e. origin >= 11-04 19:45; the rescore keeps history inside its range).
Metrics: MAE, MSE, RMSE (= sqrt of the pooled MSE, as the records), MAPE linear and dBm; TSS-LCD records have MAE and MSE only (RMSE = sqrt(MSE),
MAPE NA). Seed mean, SE = sd / sqrt(n_seeds). ``mean`` row = mean over t+4/12/24/48 per seed, then over seeds. 4D models and STS-PredNet: NA rows
(no week-2 source). Per-bin files: the rescore records' ``per_bin_key`` (pooled over the four receivers, per frequency and horizon).

    python evaluation/analysis/ncar_week2_table.py --out-dir /tmp/item2 --s3-tag v1
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

B = "spectrum"
H4 = (4, 12, 24, 48)
METRICS = {"mae": "mae_db", "mse": "mse_db2", "rmse": "rmse_db", "mape_linear": "mape_linear_pct", "mape_dbm": "mape_dbm_pct"}
CORE4 = ("humanities", "madsen", "moran", "sagepoint")
BASELINES = ("tuned_median", "same_time_yesterday", "lookbackmean24h", "persistence")
NA_MODELS = ("linearar4d", "residuallinearar4d", "convlstm", "residualconvlstm", "dswinlstm_i", "stsprednet")
TSSLCD = {"tuned": ("tsslcd-authors-15min-tuned", "600_800", ("off",)), "tuned-2400": ("tsslcd-authors-15min-tuned-2400", "2400_2600", ("off",)),
          "default": ("tsslcd-authors-15min-default", "600_800", ("off", "causal")),
          "default-2400": ("tsslcd-authors-15min-default-2400", "2400_2600", ("off", "causal"))}
WEEK2_START = pd.Timestamp("2026-11-03T20:00:00Z")


def keys(c, prefix: str, suffix: str = ".json") -> list[str]:
    return [o["Key"] for pg in c.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix) for o in pg.get("Contents", [])
            if o["Key"].endswith(suffix)]


def read_json(c, ks: list[str]) -> list[tuple[str, dict]]:
    def one(k):
        try:
            return k, json.loads(c.get_object(Bucket=B, Key=k)["Body"].read())
        except Exception:
            return k, {}
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, ks))


def rescore_rows(c, root: str, week: str) -> tuple[list[dict], list[dict]]:
    """Long rows (one per metric) and per-bin index rows from one rescore prefix (``rescored`` or ``rescored-week2full``)."""
    rows, index = [], []
    for kind, prefix in (("rolled", f"{root}/p15tuned/"), ("direct", f"{root}/direct/15min/")):
        for k, d in read_json(c, keys(c, prefix)):
            if not d.get("ok"):
                continue
            hs = [int(d["horizon"])] if kind == "direct" else [h for h in H4 if d.get(f"test_mae_db_t{h}") is not None]
            if kind == "direct" and hs[0] not in H4:
                continue
            base = {"week": week, "source": kind, "band": d["band"], "rep": d["rep"], "model": d["model"], "seed": int(d["seed"]),
                    "n_windows": d.get("n_test_windows")}
            for h in hs:
                for m, suffix in METRICS.items():
                    v = d.get(f"test_{suffix}_t{h}")
                    if v is not None:
                        rows.append({**base, "horizon": h, "metric": m, "value": float(v)})
            index.append({**{k2: base[k2] for k2 in ("week", "source", "band", "rep", "model", "seed")},
                          "trained_horizon": d.get("horizon"), "record_key": k, "per_bin_key": d.get("per_bin_key"),
                          "test_start": d.get("test_start"), "test_end": d.get("test_end"), "n_windows": d.get("n_test_windows")})
    return rows, index


def tsslcd_rows(c, week: str) -> list[dict]:
    rows = []
    for setting, (prefix, band, savgols) in TSSLCD.items():
        p = prefix + ("-week2full" if week == "week2" else "")
        for s in savgols:
            for k, d in read_json(c, keys(c, f"{p}/savgol-{s}/")):
                if not d.get("ok"):
                    continue
                m = d["metrics"]
                for h in H4:
                    mae, mse = m.get(f"raw_target__off__mae_db_t{h}"), m.get(f"raw_target__off__mse_db2_t{h}")
                    base = {"week": week, "source": "tsslcd", "band": band, "rep": "4-receiver", "model": f"tsslcd ({setting}, savgol {s})",
                            "seed": int(d["seed"]), "n_windows": d.get("test_windows"), "horizon": h}
                    if mae is not None:
                        rows += [{**base, "metric": "mae", "value": float(mae)}, {**base, "metric": "mse", "value": float(mse)},
                                 {**base, "metric": "rmse", "value": float(np.sqrt(mse))}]
    return rows


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """Seed mean and SE per (week, source, band, rep, model, horizon, metric), plus a ``mean`` horizon (per-seed mean over t+4..t+48)."""
    keys_ = ["week", "source", "band", "rep", "model", "metric"]
    per_seed_mean = (long.groupby(keys_ + ["seed"]).filter(lambda g: g.horizon.nunique() == 4)
                     .groupby(keys_ + ["seed"]).agg(value=("value", "mean"), n_windows=("n_windows", "min")).reset_index().assign(horizon="mean"))
    both = pd.concat([long.assign(horizon=long.horizon.astype(str)), per_seed_mean], ignore_index=True)
    s = both.groupby(keys_ + ["horizon"]).agg(mean=("value", "mean"), sd=("value", "std"), n_seeds=("seed", "nunique"),
                                              seeds=("seed", lambda x: " ".join(map(str, sorted(set(x))))),
                                              n_windows=("n_windows", "min")).reset_index()
    s["se"] = s["sd"] / np.sqrt(s["n_seeds"])
    return s


def restricted_baselines(c) -> pd.DataFrame:
    """Item-1 week-2 baselines restricted to origins >= week start + 95 steps (the rescore's in-range history), pooled over receivers."""
    from training.ray.ncar_cells import load_npz, per_bin_table

    cutoff = int((WEEK2_START + 95 * pd.Timedelta(minutes=15)).timestamp())
    rows = []
    for band in ("600_800", "2400_2600"):
        for name in BASELINES:
            acc: dict[int, list] = {}
            for r in CORE4:
                key = f"transfer-ncar-2026-11/item1/powder/{r}/20261025T1355Z-week2/{band}/{name}/forecasts.npz"
                try:
                    mats = load_npz(c.get_object(Bucket=B, Key=key)["Body"].read())
                except Exception:
                    continue
                for h, m in mats.items():
                    if h not in H4:
                        continue
                    keep = m["origin"] >= cutoff
                    acc.setdefault(h, []).append((m["pred"][keep], m["target"][keep]))
            for h, parts in acc.items():
                P = np.concatenate([p for p, _ in parts]).astype(np.float64)
                T = np.concatenate([t for _, t in parts]).astype(np.float64)
                e = P - T
                e32 = e.astype(np.float32)
                lin = np.abs(np.power(np.float32(10.0), e32 / np.float32(10.0)) - np.float32(1.0)) * np.float32(100.0)
                dbm = np.abs(e32) / np.maximum(np.abs(T.astype(np.float32)), np.float32(1e-6)) * np.float32(100.0)
                rows.append({"band": band, "baseline": name, "horizon": str(h), "n_windows": int(P.shape[0]), "n_receivers": len(parts),
                             "mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                             "mape_linear": float(lin.mean(dtype=np.float64)), "mape_dbm": float(dbm.mean(dtype=np.float64))})
    out = pd.DataFrame(rows)
    if len(out):
        mean = (out.groupby(["band", "baseline"]).filter(lambda g: g.horizon.nunique() == 4).groupby(["band", "baseline"])
                .agg(**{m: (m, "mean") for m in METRICS}, n_windows=("n_windows", "min"), n_receivers=("n_receivers", "min")).reset_index()
                .assign(horizon="mean"))
        out = pd.concat([out, mean], ignore_index=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="evaluation/results")
    p.add_argument("--baselines", default="evaluation/results/transfer_baselines.csv")
    p.add_argument("--part1", default="evaluation/results/part1_per_horizon_15min.csv")
    p.add_argument("--s3-tag", default=None, help="also upload the outputs to transfer-ncar-2026-11/item2/tables/<tag>/ (never overwritten)")
    a = p.parse_args()
    from training.ray.ncar_cells import tags
    from training.ray.storage import spectrum_s3_client

    c = spectrum_s3_client()
    w2, idx2 = rescore_rows(c, "rescored-week2full", "week2")
    oc, idxo = rescore_rows(c, "rescored", "oct")
    long = pd.DataFrame(w2 + oc + tsslcd_rows(c, "week2") + tsslcd_rows(c, "oct"))
    s = summarize(long)
    wide = s.pivot_table(index=["source", "band", "rep", "model", "horizon"], columns=["week", "metric"],
                         values=["mean", "se"], aggfunc="first")
    wide.columns = [f"{wk}_{m}" + ("" if stat == "mean" else "_se") for stat, wk, m in wide.columns]
    meta = s.pivot_table(index=["source", "band", "rep", "model", "horizon"], columns="week", values=["n_seeds", "n_windows"], aggfunc="max")
    meta.columns = [f"{wk}_{stat}" for stat, wk in meta.columns]
    seeds = s[s.metric == "mae"].pivot_table(index=["source", "band", "rep", "model", "horizon"], columns="week", values="seeds", aggfunc="first")
    seeds.columns = [f"{wk}_seeds" for wk in seeds.columns]
    table = wide.join(meta).join(seeds).reset_index()
    # baselines: item 1 (transfer windows) and restricted to the rescore windows
    base = pd.read_csv(a.baselines)
    base = base[(base.level == "group") & (base.group == "core4_week2")]
    for name in BASELINES:
        b = base[base.baseline == name].assign(horizon=lambda d: d.horizon.astype(str).replace({"mean_4h": "mean"}))
        b = b[["band", "horizon", "n_windows", *METRICS.values()]].rename(columns={v: f"base_{name}_{k}" for k, v in METRICS.items()} | {
            "n_windows": f"base_{name}_n_windows"})
        table = table.merge(b, on=["band", "horizon"], how="left")
    rb = restricted_baselines(c)
    for name in BASELINES:
        b = rb[rb.baseline == name][["band", "horizon", "n_windows", *METRICS]].rename(
            columns={m: f"base_{name}_rescorewin_{m}" for m in METRICS} | {"n_windows": f"base_{name}_rescorewin_n_windows"})
        table = table.merge(b, on=["band", "horizon"], how="left")
    t = table.apply(lambda r: tags(r["model"].split(" ")[0], r["rep"] if r["rep"] in ("1d", "2d") else "2d"), axis=1, result_type="expand")
    table[["spectral_context", "spatial_context", "level_keeping"]] = t
    table.loc[table.source == "tsslcd", ["spectral_context", "spatial_context"]] = "yes"
    na = [{"source": "4d", "band": b, "rep": "4d", "model": m, "horizon": h, "note": "no week-2 source (no rescore of 4D/STS-PredNet on week 2)",
           **tags(m)} for b in ("600_800", "2400_2600") for m in NA_MODELS for h in (*map(str, H4), "mean")]
    table = pd.concat([table, pd.DataFrame(na)], ignore_index=True)
    table["week2_minus_oct_mae"] = table.get("week2_mae") - table.get("oct_mae")
    table["horizon_sort"] = table.horizon.map(lambda h: 99 if h == "mean" else int(h))
    table = table.sort_values(["band", "source", "model", "rep", "horizon_sort"]).drop(columns="horizon_sort")
    index = pd.DataFrame(idx2 + idxo)
    # October check against the Part 1 table (rolled: rescored/p15tuned is its source; direct: Part 1 reads direct-tuning finals)
    p1 = pd.read_csv(a.part1)
    p1 = p1[(p1.metric == "mae") & p1.horizon.isin(H4)]
    chk = []
    for r in table[(table.source.isin(["rolled", "direct"])) & (table.horizon != "mean")].itertuples():
        name = f"{r.model} ({r.rep})" if r.source == "rolled" else r.model
        fam = "rolled-out" if r.source == "rolled" else "direct"
        m = p1[(p1.band == r.band) & (p1.family == fam) & (p1.model == name) & (p1.horizon == int(r.horizon))]
        if len(m) and getattr(r, "oct_mae", None) == getattr(r, "oct_mae", None):
            chk.append({"source": r.source, "band": r.band, "model": name, "horizon": r.horizon, "oct_mae_here": r.oct_mae,
                        "part1_mae": float(m["mean"].iloc[0]), "part1_n": int(m["n"].iloc[0]), "n_seeds_here": r.oct_n_seeds,
                        "diff": r.oct_mae - float(m["mean"].iloc[0])})
    chk = pd.DataFrame(chk)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"transfer_week2.csv": table, "transfer_week2_per_bin_index.csv": index, "transfer_week2_oct_vs_part1.csv": chk,
             "transfer_week2_baselines_rescore_windows.csv": rb}
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
        if a.s3_tag:
            k = f"transfer-ncar-2026-11/item2/tables/{a.s3_tag}/{name}"
            try:
                c.head_object(Bucket=B, Key=k)
                raise SystemExit(f"refusing to overwrite s3://{B}/{k}")
            except c.exceptions.ClientError:
                pass
            c.put_object(Bucket=B, Key=k, Body=frame.to_csv(index=False).encode())
    print(f"rows: table {len(table)}, per-bin index {len(index)} (missing per_bin_key: {int(index.per_bin_key.isna().sum())}), "
          f"oct check {len(chk)}, restricted baselines {len(rb)}")
    if len(chk):
        print("October |diff| > 0.01 dB vs Part 1:")
        print(chk[chk["diff"].abs() > 0.01].to_string())
        print("max |diff|:", chk["diff"].abs().max())


if __name__ == "__main__":
    main()
