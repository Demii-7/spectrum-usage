"""Shared helpers for the Part 1 per-horizon tables: all five metrics (MAE, MSE, RMSE, MAPE in linear power and in dBm), long CSV, one printed table per metric."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# short name -> (record suffix in ``test_<suffix>_t<h>``, baselines column, unit)
METRICS = {"mae": ("mae_db", "mae_db", "dB"), "mse": ("mse_db2", "mse_db2", "dB^2"), "rmse": ("rmse_db", "rmse_db", "dB"),
           "mape_linear": ("mape_linear_pct", "mape_linear_pct", "%"), "mape_dbm": ("mape_dbm_pct", "mape_dbm_pct", "%")}


def client():
    from training.ray.storage import spectrum_s3_client

    return spectrum_s3_client()


def listing(c, prefix: str) -> list[str]:
    return [o["Key"] for pg in c.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith(".json")]


def read_all(c, keys: list[str]) -> list[tuple[str, dict]]:
    def one(k):
        try:
            return k, json.loads(c.get_object(Bucket="spectrum", Key=k)["Body"].read())
        except Exception:
            return k, {}
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, keys))


def record_rows(d: dict, h: int, base: dict) -> list[dict]:
    """One row per metric of a ``test_<metric>_t<h>`` record."""
    out = []
    for name, (suffix, _, _) in METRICS.items():
        v = d.get(f"test_{suffix}_t{h}")
        if v is not None:
            out.append({**base, "horizon": h, "metric": name, "value": float(v)})
    return out


def baseline_rows(base: pd.DataFrame, make) -> list[dict]:
    rows = []
    for r in base.itertuples():
        for name, (_, col, _) in METRICS.items():
            v = getattr(r, col, None)
            if v is not None and v == v:
                rows.append({**make(r), "horizon": int(r.horizon), "metric": name, "value": float(v), "seed": 0})
    return rows


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    s = frame.groupby(keys + ["horizon", "metric"]).value.agg(mean="mean", sd="std", n="count").reset_index()
    s["sd"] = s["sd"].fillna(0.0)
    return s


def print_tables(s: pd.DataFrame, bands, horizons, index, metrics=("mae", "mse"), title="") -> None:
    pd.set_option("display.width", 220)
    for band in bands:
        for m in metrics:
            d = s[(s.band == band) & (s.metric == m) & s.horizon.isin(horizons)]
            if d.empty:
                continue
            w = d.pivot_table(index=index, columns="horizon", values="mean")
            cols = [h for h in horizons if h in w.columns]
            w = w[cols]
            digits = 3 if m in ("mae", "rmse") else 3
            w["mean_all"] = w.mean(axis=1, skipna=False)
            print(f"\n{band}: {title} {m.upper()} ({METRICS[m][2]}), mean over seeds")
            print(w.round(digits).sort_values("mean_all").to_string())
