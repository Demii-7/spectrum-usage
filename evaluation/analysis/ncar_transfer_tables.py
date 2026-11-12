"""NCAR transfer batch: tables for brief items 3, 4, 4b, 5, 9 and 10 from the saved per-window arrays and per-bin metrics.

Inputs (``s3://spectrum``):
  ``transfer-ncar-2026-11/cells/<regime>/<platform>/<site>/<run>[-week2]/<band>/<rep>/<model>[/h<h>]/seed-<n>/``  (``training.ray.ncar_tx_cells``)
  ``transfer-ncar-2026-11/item5/from-<source band>/<regime>/powder/<node>/<run>/<band>/<rep>/<model>[/h<h>]/seed-<n>/``  (cross-band)
  ``transfer-ncar-2026-11/item1/<target key>/<baseline>/``  (item-1 baselines, identical windows to the model cells)
  ``transfer-ncar-2026-11/item3/target-trained/<platform>/<site>/<run>/<band>/h<h>/seed-<n>/``  (``training.ray.ncar_target_trained``)
each with ``per_bin.csv`` and ``forecasts.npz`` (``training.ray.ncar_cells``). A ``<model>/h<h>`` directory is a direct model trained for
horizon h (scored at h only); a model directory without ``h<h>`` is a rolled-out one-step model (scored at every horizon; we keep 4/12/24/48).

Levels (brief): per bin (one row per cell, bin, horizon, seed); per receiver = mean over bins (and over a receiver's runs, e.g. the two
June-July runs), per seed; per target group = mean over receivers, per seed; summary rows = mean over seeds and the standard error across
seeds (sd / sqrt(n)). Metrics MAE, MSE, RMSE (mean over bins of the per-bin RMSE, as the brief's "mean over bins"; ``rmse_pooled_db`` =
sqrt(MSE) also given), MAPE = MAPE of linear power (``mape_linear_pct``, the scoring code's primary MAPE) and MAPE of dBm values.
Conditions: frozen (A), centered (B), refit (C), mean-only refit (Cm). Seeds: those of the existing transfer runs (40-44; TCN 40/41/43/44;
Time-LLM 40-42); other seeds present in S3 are ignored.

    # on NCAR (reads S3, writes s3://spectrum/transfer-ncar-2026-11/tables/<stamp>/ and runs/ncar-tables/<stamp>/):
    python evaluation/analysis/ncar_transfer_tables.py build [--items 3 4 4b 5 9 10] [--workers 16]
    # locally, after copying the build outputs into evaluation/results/: add TSS-LCD (item 6) and 4D (item 7) rows to transfer_combined.csv
    python evaluation/analysis/ncar_transfer_tables.py merge
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BUCKET = "spectrum"
ROOT = "transfer-ncar-2026-11"
HORIZONS = (4, 12, 24, 48)
SEEDS = {"temporalconvnet": (40, 41, 43, 44), "timellm": (40, 41, 42)}
DEFAULT_SEEDS = (40, 41, 42, 43, 44)
CORE4 = ("humanities", "madsen", "moran", "sagepoint")
CONDITION = {"A": "frozen", "B": "centered", "C": "refit", "Cm": "mean-only refit"}
METRICS = ("mae_db", "mse_db2", "rmse_db", "rmse_pooled_db", "mape_linear_pct", "mape_dbm_pct", "bias_db",
           "mae_after_per_bin_offset", "rmse_after_per_bin_offset")
CORE_METRICS = ("mae_db", "mse_db2", "rmse_db", "mape_linear_pct")
BASELINES = ("tuned_median", "same_time_yesterday", "lookbackmean24h", "persistence")
LEVEL_KEEPING = ("revinmlp", "patchtst", "ofa", "timellm", "moment_lp", "timeran_lp", "nlinear", "residuallinearar1d", "residuallinearar2d")
SHIFT = {"core4_week2": "later interval", "heldout_oct": "new collection points", "july_core4": "separate collection",
         "july_other": "separate collection", "aerpaw": "AERPAW", "ara": "ARA", "cosmos": "COSMOS", "core4_oct": "in distribution (reference)",
         "crossband_800_1000": "new frequency", "crossband_5725_5925": "new frequency"}
PAIRS = [(("linearar1d", "1d", "rolled"), ("linearar2d", "2d", "rolled")), (("residuallinearar1d", "1d", "rolled"), ("residuallinearar2d", "2d", "rolled")),
         (("vanillalstm1d", "1d", "rolled"), ("vanillalstm", "2d", "rolled")), (("revinmlp", "2d", "direct"), ("revinmlp_mix", "2d", "direct")),
         (("patchtst", "2d", "direct"), ("patchtst_joint", "2d", "direct"))]


# ------------------------------------------------------------------------------------------------ keys
def receiver_of(site: str) -> str:
    return site.removesuffix("-nuc1")


def group_of(platform: str, site: str, run: str, band: str) -> str:
    if platform != "powder":
        return platform
    if band in ("800_1000", "5725_5925"):
        return f"crossband_{band}"
    if run.endswith("-week2"):
        return "core4_week2"
    if run.startswith("20261025"):
        return "heldout_oct" if receiver_of(site) in ("guesthouse", "law73") else "core4_oct"
    return "july_core4" if receiver_of(site) in CORE4 else "july_other"


def parse_cell(key: str) -> dict | None:
    """Metadata of a model cell's per_bin.csv key (cells or item5)."""
    parts = key.split("/")
    if parts[1] == "cells":
        source_band, rest = None, parts[2:]
    elif parts[1] == "item5":
        source_band, rest = parts[2].removeprefix("from-"), parts[3:]
    else:
        return None
    regime, platform, site, run, band, rep, model = rest[:7]
    tail = rest[7:]
    horizon = int(tail[0][1:]) if tail[0].startswith("h") and tail[0][1:].isdigit() else None
    seed = int(tail[-2].removeprefix("seed-"))
    if seed not in SEEDS.get(model, DEFAULT_SEEDS):
        return None
    group = group_of(platform, site, run, band)
    return {"key": key, "model": model, "rep": rep, "source": "direct" if horizon else "rolled", "trained_horizon": horizon,
            "condition": CONDITION[regime], "regime": regime, "source_band": source_band or band, "target_band": band,
            "platform": platform, "site": site, "receiver": receiver_of(site), "run": run, "group": group,
            "shift": SHIFT[group], "target": f"{platform}/{site}/{run}/{band}", "seed": seed}


def parse_baseline(key: str) -> dict | None:
    parts = key.split("/")                                     # transfer-ncar-2026-11/item1/<platform>/<site>/<run>/<band>/<name>/per_bin.csv
    if len(parts) != 8 or parts[1] != "item1":
        return None
    platform, site, run, band, name = parts[2:7]
    return {"key": key, "baseline": name, "platform": platform, "site": site, "receiver": receiver_of(site), "run": run, "target_band": band,
            "group": group_of(platform, site, run, band), "target": f"{platform}/{site}/{run}/{band}"}


def parse_tt(key: str) -> dict | None:
    parts = key.split("/")                                     # .../item3/target-trained/<platform>/<site>/<run>/<band>/h<h>/seed-<n>/per_bin.csv
    platform, site, run, band, h, seed = parts[3:9]
    return {"key": key, "model": "revinmlp_target_trained", "rep": "2d", "source": "direct", "trained_horizon": int(h[1:]),
            "condition": "target-trained", "regime": "TT", "source_band": band, "target_band": band, "platform": platform, "site": site,
            "receiver": site, "run": run, "group": platform, "shift": SHIFT[platform], "target": f"{platform}/{site}/{run}/{band}",
            "seed": int(seed.removeprefix("seed-"))}


def list_keys(client, prefix: str, suffix: str = "/per_bin.csv") -> list[str]:
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith(suffix) and "/tables/" not in o["Key"]]


def tags(model: str, rep: str) -> dict:
    from training.ray.ncar_cells import tags as t

    out = t(model, rep)
    if model == "revinmlp_target_trained":
        out = t("revinmlp", rep)
    return out


# --------------------------------------------------------------------------------------------- loading
def read_per_bin(client, metas: list[dict], workers: int) -> pd.DataFrame:
    def one(m):
        frame = pd.read_csv(io.BytesIO(client.get_object(Bucket=BUCKET, Key=m["key"])["Body"].read()))
        frame = frame[frame.horizon.isin(HORIZONS)]
        if m.get("trained_horizon"):
            frame = frame[frame.horizon == m["trained_horizon"]]
        for k, v in m.items():
            if k != "key":
                frame[k] = v
        frame["per_bin_key"] = m["key"]
        return frame

    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(one, metas))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["rmse_pooled_db"] = out["rmse_db"]          # per bin they are the same; differs only after averaging over bins
    return out


def read_npz(client, key: str):
    from training.ray.ncar_cells import load_npz

    return load_npz(client.get_object(Bucket=BUCKET, Key=key)["Body"].read())


# ------------------------------------------------------------------------------------------ aggregation
def receiver_level(per_bin: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean over bins per cell, then over a receiver's targets (runs); one row per (keys, receiver, horizon, seed)."""
    cell_keys = keys + ["receiver", "target", "horizon", "seed"]
    agg = {m: "mean" for m in METRICS if m in per_bin}
    agg["n_windows"] = "first"
    cell = per_bin.groupby(cell_keys, dropna=False).agg(agg).reset_index()
    cell["rmse_pooled_db"] = np.sqrt(cell["mse_db2"])
    agg = {m: "mean" for m in METRICS if m in cell}
    agg.update(n_windows="sum", target="nunique")
    rec = cell.groupby(keys + ["receiver", "horizon", "seed"], dropna=False).agg(agg).reset_index().rename(columns={"target": "n_targets"})
    rec["share_removed"] = 1 - rec["mae_after_per_bin_offset"] / rec["mae_db"]
    return rec


def seed_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean over seeds and SE across seeds of every metric."""
    metrics = [m for m in list(METRICS) + ["share_removed"] if m in frame]
    g = frame.groupby(keys, dropna=False)
    mean = g[metrics].mean()
    se = g[metrics].sem(ddof=1).rename(columns=lambda c: f"{c}_se")
    extra = g.agg(n_seeds=("seed", "nunique"), n_windows=("n_windows", "mean"))
    if "n_receivers" in frame:
        extra = extra.join(g.n_receivers.max())
    return mean.join(se).join(extra).reset_index()


def group_level(rec: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean over receivers per seed (receiver-weighted), then seed summary."""
    metrics = [m for m in list(METRICS) + ["share_removed"] if m in rec]
    g = rec.groupby(keys + ["horizon", "seed"], dropna=False)
    per_seed = g[metrics].mean().join(g.agg(n_receivers=("receiver", "nunique"), n_windows=("n_windows", "sum"))).reset_index()
    per_seed["share_removed"] = 1 - per_seed["mae_after_per_bin_offset"] / per_seed["mae_db"] if "mae_after_per_bin_offset" in per_seed else np.nan
    return seed_summary(per_seed, keys + ["horizon"])


def add_mean_row(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """A ``horizon = 'mean'`` row (mean over 4/12/24/48) wherever all four horizons exist."""
    num = [c for c in table.columns if c not in keys + ["horizon"] and pd.api.types.is_numeric_dtype(table[c])]
    full = table.groupby(keys, dropna=False).filter(lambda g: set(g.horizon) >= set(HORIZONS))
    if not len(full):
        return table
    mean = full.groupby(keys, dropna=False)[num].mean().reset_index()
    mean["horizon"] = "mean"
    return pd.concat([table.astype({"horizon": object}), mean], ignore_index=True)


def baseline_tables(per_bin_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Receiver-level and per-bin baseline metrics (no seeds)."""
    b = per_bin_base.copy()
    b["seed"] = 0
    rec = receiver_level(b, ["baseline", "group", "target_band"])
    return rec, b


def group_baselines(model_rec: pd.DataFrame, base_rec: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Baseline metrics averaged over exactly the receivers each model row covers (receiver-weighted), per horizon."""
    covered = model_rec[keys + ["receiver", "horizon"]].drop_duplicates()
    rows = []
    for name in BASELINES:
        sub = base_rec[base_rec.baseline.str.startswith(name)].copy()
        if sub.empty:
            continue
        sub["source_band"] = sub.baseline.str.extract(r"_k(\d+_\d+)")[0].fillna(sub.target_band)
        if name != "tuned_median":
            sub = sub[sub.baseline == name]
        join = ["group", "target_band", "receiver", "horizon"] + (["source_band"] if name == "tuned_median" else [])
        m = covered.merge(sub[join + list(CORE_METRICS)], on=join, how="left")
        g = m.groupby(keys + ["horizon"], dropna=False)[list(CORE_METRICS)].mean().rename(columns={c: f"{name}_{c}" for c in CORE_METRICS})
        rows.append(g)
    out = pd.concat(rows, axis=1).reset_index() if rows else covered[keys + ["horizon"]].drop_duplicates()
    return out


# ------------------------------------------------------------------------------------------------ items
MODEL_KEYS = ["model", "rep", "source", "condition", "source_band", "target_band", "shift", "group"]


def finish(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    if "model" in t:
        tg = t.apply(lambda r: tags(r["model"], r["rep"]), axis=1, result_type="expand")
        for c in ("spectral_context", "spatial_context"):
            t[c] = tg[c]
    return t


def restrict_windows(mats: dict, h: int, first_target: pd.Timestamp, last_target: pd.Timestamp | None):
    origin = pd.to_datetime(mats[h]["origin"], unit="s", utc=True)
    target_time = origin + pd.Timedelta(minutes=15 * h)
    keep = target_time >= first_target
    if last_target is not None:
        keep &= target_time <= last_target
    return np.asarray(keep)


def item3(client, cells_meta, base_meta, tt_meta, workers) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen transfer of every model at AERPAW/ARA/COSMOS, item-1 baselines and the target-trained RevIN-MLP, all on the target-trained
    test windows (target time >= first_test_target_time, per target and horizon)."""
    from training.ray.ncar_cells import per_bin_table

    tt_sum = []
    for m in tt_meta:
        try:
            s = json.loads(client.get_object(Bucket=BUCKET, Key=m["key"].replace("per_bin.csv", "summary.json"))["Body"].read())
            tt_sum.append({"target": m["target"], "horizon": m["trained_horizon"], "first": pd.Timestamp(s["first_test_target_time"]),
                           "n_tt": s.get("n_windows")})
        except Exception:
            pass
    if not tt_sum:
        return pd.DataFrame(), pd.DataFrame()
    windows = pd.DataFrame(tt_sum).groupby(["target", "horizon"]).agg(first=("first", "first"), n_tt=("n_tt", "first")).reset_index()
    wmap = {(r.target, int(r.horizon)): (r.first, r.n_tt) for r in windows.itertuples()}
    ext = [m for m in cells_meta if m["platform"] in ("aerpaw", "ara", "cosmos") and m["regime"] == "A"]
    ext_base = [{**m, "model": m["baseline"], "rep": "baseline", "source": "baseline", "condition": "baseline", "source_band": m["target_band"],
                 "shift": SHIFT[m["group"]], "seed": 0, "trained_horizon": None}
                for m in base_meta if m["platform"] in ("aerpaw", "ara", "cosmos")]

    def one(m):
        mats = read_npz(client, m["key"].replace("per_bin.csv", "forecasts.npz"))
        frames = []
        for h in mats:
            if h not in HORIZONS or (m.get("trained_horizon") and h != m["trained_horizon"]) or (m["target"], h) not in wmap:
                continue
            first, n_tt = wmap[(m["target"], h)]
            keep = restrict_windows(mats, h, first, None)
            t = per_bin_table({h: mats[h]}, {h: keep})
            if len(t):
                t["n_windows_target_trained"] = n_tt
                for k, v in m.items():
                    if k != "key":
                        t[k] = v
                frames.append(t)
        return pd.concat(frames) if frames else None

    with ThreadPoolExecutor(workers) as pool:
        frames = [f for f in pool.map(one, ext + ext_base + tt_meta) if f is not None]
    per_bin = pd.concat(frames, ignore_index=True)
    per_bin["rmse_pooled_db"] = per_bin["rmse_db"]
    rec = receiver_level(per_bin, MODEL_KEYS)
    grp = group_level(rec, MODEL_KEYS)
    rec_s = seed_summary(rec, MODEL_KEYS + ["receiver", "horizon"])
    rec_s["level"], grp["level"] = "receiver", "group"
    table = pd.concat([rec_s, grp], ignore_index=True)
    return finish(table), per_bin


def coverage(cells_meta, base_meta) -> pd.DataFrame:
    """Present vs expected (target, seed, condition) cells per model and group. Expected targets = the item-1 targets of the group."""
    from training.common.transfer_regimes import intrinsic_class

    targets = pd.DataFrame(base_meta)[["group", "target_band", "target"]].drop_duplicates() if base_meta else pd.DataFrame()
    present = pd.DataFrame(cells_meta)
    if present.empty:
        return present
    present = present.drop_duplicates(["model", "rep", "source", "regime", "source_band", "target", "seed"])
    rows = []
    for (model, rep, source, group, sband), g in present.groupby(["model", "rep", "source", "group", "source_band"]):
        tband = g.target_band.iloc[0] if group.startswith("crossband") else sband
        n_t = int(((targets.group == group) & (targets.target_band == tband)).sum()) if len(targets) else 0
        seeds = len(SEEDS.get(model, DEFAULT_SEEDS))
        full = intrinsic_class(model) == "full"
        regimes = 1 if (full or group in ("core4_oct", "core4_week2")) else 4
        rows.append({"model": model, "rep": rep, "source": source, "group": group, "source_band": sband, "target_band": tband,
                     "present_cells": len(g), "expected_cells": n_t * seeds * regimes, "targets_present": g.target.nunique(), "targets_expected": n_t})
    out = pd.DataFrame(rows)
    out["complete"] = out.present_cells >= out.expected_cells
    return out


def build(a) -> None:
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    stamp = a.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(a.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    s3_out = f"{ROOT}/tables/{stamp}"

    def put(name: str, frame: pd.DataFrame, local: bool = True) -> None:
        body = frame.to_csv(index=False).encode()
        if name.endswith(".gz"):
            import gzip

            body = gzip.compress(body)
        client.put_object(Bucket=BUCKET, Key=f"{s3_out}/{name}", Body=body)
        if local:
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / name).write_bytes(body)
        print(f"[tables] {name}: {len(frame)} rows", flush=True)

    t0 = time.time()
    cells_meta = [m for m in (parse_cell(k) for k in list_keys(client, f"{ROOT}/cells/") + list_keys(client, f"{ROOT}/item5/")) if m]
    base_meta = [m for m in (parse_baseline(k) for k in list_keys(client, f"{ROOT}/item1/")) if m]
    tt_meta = [parse_tt(k) for k in list_keys(client, f"{ROOT}/item3/target-trained/")]
    print(f"[tables] {len(cells_meta)} model cells, {len(base_meta)} baseline cells, {len(tt_meta)} target-trained cells "
          f"({time.time() - t0:.0f}s)", flush=True)
    cov = coverage(cells_meta, base_meta)
    put("ncar_transfer_coverage.csv", cov)

    per_bin = read_per_bin(client, cells_meta, a.workers)
    base_pb = read_per_bin(client, base_meta, a.workers)
    print(f"[tables] read per-bin tables ({time.time() - t0:.0f}s): {len(per_bin)} model rows, {len(base_pb)} baseline rows", flush=True)
    base_rec, _ = baseline_tables(base_pb)
    rec = receiver_level(per_bin, MODEL_KEYS) if len(per_bin) else pd.DataFrame()
    rec_s = seed_summary(rec, MODEL_KEYS + ["receiver", "horizon"])
    rec_s["level"] = "receiver"
    grp = group_level(rec, MODEL_KEYS)
    grp["level"] = "group"
    gb = group_baselines(rec, base_rec, MODEL_KEYS)
    grp = grp.merge(gb, on=MODEL_KEYS + ["horizon"], how="left")
    grp["mae_ratio_tuned_median"] = grp["mae_db"] / grp["tuned_median_mae_db"]
    rb = group_baselines(rec, base_rec, MODEL_KEYS + ["receiver"])
    rec_s = rec_s.merge(rb, on=MODEL_KEYS + ["receiver", "horizon"], how="left")
    rec_s["mae_ratio_tuned_median"] = rec_s["mae_db"] / rec_s["tuned_median_mae_db"]
    rec_s["mae_ratio_same_time_yesterday"] = rec_s["mae_db"] / rec_s["same_time_yesterday_mae_db"]
    both = finish(pd.concat([rec_s, grp], ignore_index=True))
    put("ncar_transfer_all_levels.csv", both)

    # per-bin files (S3 only) per shift and model, with an index
    index = []
    for (shift, model), g in per_bin.groupby(["shift", "model"]):
        name = f"per_bin/{shift.replace(' ', '_').replace('(', '').replace(')', '')}/{model}.csv.gz"
        put(name, g.drop(columns=["per_bin_key"]), local=False)
        index.append({"shift": shift, "model": model, "rows": len(g), "s3_key": f"{s3_out}/{name}"})
    cell_index = per_bin[MODEL_KEYS + ["target", "receiver", "seed", "per_bin_key"]].drop_duplicates()
    cell_index["forecasts_key"] = cell_index.per_bin_key.str.replace("per_bin.csv", "forecasts.npz", regex=False)

    items = set(a.items)
    if "4" in items:                                         # level / dynamics split
        cols = MODEL_KEYS + ["level", "receiver", "horizon", "n_seeds", "n_windows", "mae_db", "mae_after_per_bin_offset", "rmse_db",
                             "rmse_after_per_bin_offset", "share_removed", "mae_db_se", "mae_after_per_bin_offset_se", "share_removed_se"]
        split = both[[c for c in cols if c in both]].rename(columns={"mae_db": "mae_raw", "rmse_db": "rmse_raw"})
        split = split[split.group != "core4_week2"]
        put("transfer_error_split.csv", finish(split))
    if "3" in items and tt_meta:
        t3, pb3 = item3(client, cells_meta, base_meta, tt_meta, a.workers)
        put("transfer_vs_target_trained.csv", t3)
        put("per_bin/item3_target_trained_windows.csv.gz", pb3, local=False)
        index.append({"shift": "item3 (target-trained windows)", "model": "all", "rows": len(pb3), "s3_key": f"{s3_out}/per_bin/item3_target_trained_windows.csv.gz"})
        tt_split = t3[t3.model == "revinmlp_target_trained"][[c for c in ["model", "rep", "source", "condition", "source_band", "target_band",
                                                                          "shift", "group", "level", "receiver", "horizon", "n_seeds",
                                                                          "n_windows", "mae_db", "mae_after_per_bin_offset", "rmse_db",
                                                                          "rmse_after_per_bin_offset", "share_removed"] if c in t3]]
        put("transfer_error_split_target_trained.csv", tt_split.rename(columns={"mae_db": "mae_raw", "rmse_db": "rmse_raw"}))
    if "4b" in items:
        ext = both[(both.model.isin(LEVEL_KEEPING)) & (both.condition == "frozen") & both.group.isin(["aerpaw", "ara", "cosmos", "core4_oct"])]
        put("transfer_external_per_receiver.csv", ext)
        pb = per_bin[(per_bin.model.isin(LEVEL_KEEPING)) & (per_bin.condition == "frozen") & per_bin.group.isin(["aerpaw", "ara", "cosmos", "core4_oct"])]
        pbs = pb.groupby(MODEL_KEYS + ["receiver", "target", "horizon", "bin", "frequency_mhz"], dropna=False)[list(CORE_METRICS)].agg(["mean", "sem"])
        pbs.columns = [f"{m}{'' if s == 'mean' else '_se'}" for m, s in pbs.columns]
        pbs = pbs.reset_index()
        bb = base_pb[base_pb.baseline.isin(["tuned_median", "same_time_yesterday"])].pivot_table(
            index=["target", "horizon", "bin"], columns="baseline", values="mae_db").reset_index()
        pbs = pbs.merge(bb, on=["target", "horizon", "bin"], how="left")
        pbs["mae_ratio_tuned_median"] = pbs["mae_db"] / pbs["tuned_median"]
        pbs["mae_ratio_same_time_yesterday"] = pbs["mae_db"] / pbs["same_time_yesterday"]
        pbs = pbs.rename(columns={"tuned_median": "tuned_median_mae_db", "same_time_yesterday": "same_time_yesterday_mae_db"})
        put("per_bin/item4b_external_per_bin.csv.gz", finish(pbs), local=False)
        index.append({"shift": "item4b external per bin (seed mean)", "model": "level-keeping", "rows": len(pbs),
                      "s3_key": f"{s3_out}/per_bin/item4b_external_per_bin.csv.gz"})
    if "5" in items:
        cb = both[both.group.str.startswith("crossband")]
        put("transfer_crossband.csv", cb)
        put("transfer_crossband_pairs.csv", pairs(rec))
    if "9" in items:
        put("transfer_july_all_regimes_v2.csv", july_v2(rec, base_rec))
    if "10" in items:
        comb = both[(both.level == "group") & (both.group != "core4_oct")].copy()
        put("transfer_combined.csv", comb)
        put("transfer_combined_per_bin_index.csv", cell_index)
    put("transfer_per_bin_files_index.csv", pd.DataFrame(index))
    print(f"[tables] done in {time.time() - t0:.0f}s -> s3://{BUCKET}/{s3_out}/ and {out_dir}", flush=True)


def pairs(rec: pd.DataFrame) -> pd.DataFrame:
    """Paired gaps (2D/joint minus 1D/per-bin), receiver-weighted: per receiver the seed-mean MAE of each model, difference, mean over receivers;
    SE from the two models' seed-level group means (independent seeds): sqrt(se1^2 + se2^2). In-band: core4 October, frozen; cross-band:
    800_1000 centered and refit, from each source band."""
    rows = []
    settings = [("in-band (core4 October)", "core4_oct", "frozen"), ("cross-band 800_1000", "crossband_800_1000", "centered"),
                ("cross-band 800_1000", "crossband_800_1000", "refit")]
    for label, group, condition in settings:
        sub = rec[(rec.group == group) & (rec.condition == condition)]
        for (m1, r1, s1), (m2, r2, s2) in PAIRS:
            for (sb, tb), g in sub.groupby(["source_band", "target_band"]):
                a = g[(g.model == m1) & (g.rep == r1) & (g.source == s1)]
                b = g[(g.model == m2) & (g.rep == r2) & (g.source == s2)]
                for h in HORIZONS:
                    ah, bh = a[a.horizon == h], b[b.horizon == h]
                    if ah.empty or bh.empty:
                        continue
                    ar, br = ah.groupby("receiver").mae_db.mean(), bh.groupby("receiver").mae_db.mean()
                    common = ar.index.intersection(br.index)
                    if not len(common):
                        continue
                    sa = ah[ah.receiver.isin(common)].groupby("seed").mae_db.mean()
                    sbb = bh[bh.receiver.isin(common)].groupby("seed").mae_db.mean()
                    se = float(np.sqrt((sa.sem() if len(sa) > 1 else np.nan) ** 2 + (sbb.sem() if len(sbb) > 1 else np.nan) ** 2))
                    rows.append({"setting": label, "condition": condition, "source_band": sb, "target_band": tb, "horizon": h,
                                 "model_per_bin": f"{m1} ({r1}, {s1})", "model_mixing": f"{m2} ({r2}, {s2})",
                                 "mae_per_bin": float(ar[common].mean()), "mae_mixing": float(br[common].mean()),
                                 "gap_mixing_minus_per_bin": float((br[common] - ar[common]).mean()), "gap_se": se,
                                 "n_receivers": int(len(common)), "n_seeds_per_bin": int(len(sa)), "n_seeds_mixing": int(len(sbb))})
    out = pd.DataFrame(rows)
    if len(out):
        keys = ["setting", "condition", "source_band", "target_band", "model_per_bin", "model_mixing"]
        out = add_mean_row(out, keys)
    return out


def july_v2(rec: pd.DataFrame, base_rec: pd.DataFrame) -> pd.DataFrame:
    """Layout of ``transfer_july_all_regimes.csv``: group (core4/other), band, regime, rep, model (+ source), mae_t12, n_targets, n_expected,
    mae_mean_4h, complete; seed-averaged per target, mean over the group's targets (here: over receivers of their target means)."""
    jr = rec[rec.group.isin(["july_core4", "july_other"])].copy()
    jr["regime"] = jr.condition.map({v: k for k, v in CONDITION.items()})
    per_target = jr.groupby(["group", "target_band", "regime", "rep", "model", "source", "receiver", "horizon"]).agg(
        mae=("mae_db", "mean"), n_targets=("n_targets", "max")).reset_index()
    jb = base_rec[base_rec.group.isin(["july_core4", "july_other"])].copy()
    jb = jb.assign(regime="baseline", rep="baseline", model=jb.baseline, source="baseline", mae=jb.mae_db)
    expected = jb.groupby(["group", "target_band"]).agg(n_expected=("n_targets", "sum")).reset_index()
    both = pd.concat([per_target, jb[["group", "target_band", "regime", "rep", "model", "source", "receiver", "horizon", "mae", "n_targets"]]])
    g = both.groupby(["group", "target_band", "regime", "rep", "model", "source", "horizon"]).agg(
        mae=("mae", "mean"), n_targets=("n_targets", "sum")).reset_index()
    t12 = g[g.horizon == 12].rename(columns={"mae": "mae_t12"}).drop(columns="horizon")
    full = g[g.horizon.isin(HORIZONS)].groupby(["group", "target_band", "regime", "rep", "model", "source"]).filter(lambda x: x.horizon.nunique() == 4)
    m4 = full.groupby(["group", "target_band", "regime", "rep", "model", "source"]).mae.mean().rename("mae_mean_4h").reset_index()
    out = t12.merge(m4, how="outer").merge(expected, how="left")
    out["complete"] = out.n_targets >= out.n_expected
    out["group"] = out.group.str.removeprefix("july_")
    return out.rename(columns={"target_band": "band"})[["group", "band", "regime", "rep", "model", "source", "mae_t12", "n_targets",
                                                       "n_expected", "mae_mean_4h", "complete"]]


def merge(a) -> None:
    """Local: append TSS-LCD (item 6) and neural/linear 4D (item 7) group rows to transfer_combined.csv when their files exist."""
    res = Path("evaluation/results")
    comb = pd.read_csv(res / "transfer_combined.csv")
    added = []
    ts = res / "transfer_tsslcd.csv"
    if ts.exists():
        t = pd.read_csv(ts)
        if "level" in t:
            t = t[t.level == "group"]
        t = t.rename(columns={"case": "group_case"})
        t["shift"] = t.get("group_case", pd.Series(dtype=str)).map({"6a": "new collection points", "6b": "separate collection",
                                                                     "6c": "new frequency"})
        t["model"], t["rep"], t["source"] = "tsslcd", "4d", "eval-only"
        t["spectral_context"], t["spatial_context"] = "yes", "yes"
        added.append(t)
    sp = res / "spatial15_s3_neural.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        s["shift"] = "new collection points (held-out receiver, spatial)"
        s["spatial_context"] = "yes"
        added.append(s)
    out = pd.concat([comb] + added, ignore_index=True, sort=False) if added else comb
    out.to_csv(res / "transfer_combined.csv", index=False)
    print(f"[merge] transfer_combined.csv: {len(comb)} model rows + {sum(len(x) for x in added)} rows from items 6/7")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--items", nargs="+", default=["3", "4", "4b", "5", "9", "10"])
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--stamp", default=None)
    b.add_argument("--out-dir", default="/workspace/spectrum-usage/runs/ncar-tables")
    sub.add_parser("merge")
    a = p.parse_args()
    {"build": build, "merge": merge}[a.command](a)


if __name__ == "__main__":
    main()
