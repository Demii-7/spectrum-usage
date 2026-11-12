"""Item 6 tables: TSS-LCD transfer (``training.ray.ncar_tsslcd``) per receiver, per target group, and paired with baselines/models on the same windows.

Reads the run records under ``transfer-ncar-2026-11/item6/records/`` and writes
  ``transfer_tsslcd.csv``               level = run (one seed, one variant, one receiver), receiver (mean over seeds; for 6a also over the
                                         12 placements, with the range over placements), group (mean over receivers); MAE, MSE, RMSE, MAPE
                                         (linear; dBm as an extra column), SE across seeds, n_windows, the context tags;
  ``transfer_tsslcd_same_windows.csv``   (``--compare-prefixes``) every comparator cell (``forecasts.npz`` in the ``ncar_cells`` format, e.g.
                                         the item-1 baselines and the frozen per-receiver models) restricted to the windows it shares with
                                         TSS-LCD for the same target and horizon, next to TSS-LCD on exactly those windows (seed mean).
Both go to S3 under ``transfer-ncar-2026-11/item6/tables/<tag>/`` and, with ``--out-dir``, to local files.

    python evaluation/analysis/transfer_tsslcd_table.py --tag v1 --compare-prefixes transfer-ncar-2026-11/item1/ transfer-ncar-2026-11/<scoring prefix>/A/
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray import ncar_cells  # noqa: E402
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET = "spectrum"
PREFIX = "transfer-ncar-2026-11/item6"
METRICS = {"mae_db": "mae", "mse_db2": "mse", "rmse_db": "rmse", "mape_linear_pct": "mape", "mape_dbm_pct": "mape_dbm"}
TARGET_RE = re.compile(r"((?:powder|aerpaw|ara|cosmos)/[^/]+/[^/]+/\d+_\d+)/")


def list_keys(client, prefix: str, suffix: str) -> list[str]:
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", []) if o["Key"].endswith(suffix)]


def read_json(client, key):
    return json.loads(client.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def run_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if not r.get("ok"):
            continue
        for i, pos in enumerate(r["positions"]):
            swapped_in = r["case"] == "6a" and r["variant"] != "unswapped" and pos["receiver"] in ("guesthouse", "law73")
            role = "held-out (swapped in)" if swapped_in else ("core (remaining, swapped run)" if r["case"] == "6a" and r["variant"] != "unswapped"
                                                              else "core (unswapped)")
            for h, m in r["metrics"][f"p{i}:{pos['receiver']}"].items():
                rows.append({"model": "tsslcd", "case": r["case"], "condition": r["condition"], "source_band": r["source_band"],
                             "target_band": r["target_band"], "variant": r["variant"], "position": i, "receiver": pos["receiver"],
                             "target_key": pos["target_key"], "role": role, "seed": int(r["seed"]), "horizon": int(h),
                             "n_windows": m["n_windows"], "windows_lost_to_gaps": r.get("windows_lost_to_gaps"),
                             "frozen_n_origins_receiver": pos.get("frozen_n_origins"),
                             **{short: m[k] for k, short in METRICS.items()}})
    out = pd.DataFrame(rows)
    tags = ncar_cells.tags("tsslcd")
    out["spectral_context"], out["spatial_context"] = tags["spectral_context"], tags["spatial_context"]
    return out


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Receiver level (6a swapped: placement-mean per seed, range over placements of the seed-mean) and group level (mean over receivers)."""
    keys = ["model", "case", "condition", "source_band", "target_band", "role", "receiver", "target_key", "horizon"]
    m = list(METRICS.values())
    runs = runs.assign(variant_class=np.where(runs.role == "held-out (swapped in)", "12 placements",
                                              np.where(runs.role.str.startswith("core (remaining"), "12 placements", runs.variant)))
    per_seed = runs.groupby(keys + ["variant_class", "seed"]).agg(**{c: (c, "mean") for c in m + ["n_windows"]}).reset_index()
    per_var = runs.groupby(keys + ["variant_class", "variant"])[m].mean().reset_index()       # placement means over seeds (range)
    rng = per_var.groupby(keys + ["variant_class"]).agg(mae_min_over_placements=("mae", "min"), mae_max_over_placements=("mae", "max"),
                                                       n_placements=("variant", "nunique")).reset_index()
    g = per_seed.groupby(keys + ["variant_class"])
    rec = g[m + ["n_windows"]].mean().reset_index()
    se = g[m].sem().reset_index().rename(columns={c: f"{c}_se" for c in m})
    rec = rec.merge(se, on=keys + ["variant_class"]).merge(g.seed.nunique().rename("n_seeds").reset_index(), on=keys + ["variant_class"])
    rec = rec.merge(rng, on=keys + ["variant_class"], how="left").assign(level="receiver")
    # group: mean over receivers per seed, then over seeds
    gkeys = ["model", "case", "condition", "source_band", "target_band", "role", "variant_class", "horizon"]
    gs = per_seed.groupby(gkeys + ["seed"]).agg(**{c: (c, "mean") for c in m}, n_windows=("n_windows", "sum"),
                                                 n_receivers=("receiver", "nunique")).reset_index()
    gg = gs.groupby(gkeys)
    grp = gg[m + ["n_windows", "n_receivers"]].mean().reset_index().merge(
        gg[m].sem().reset_index().rename(columns={c: f"{c}_se" for c in m}), on=gkeys).merge(gg.seed.nunique().rename("n_seeds").reset_index(), on=gkeys)
    grp["level"], grp["receiver"] = "group", "mean over receivers"
    out = pd.concat([rec, grp], ignore_index=True)
    tags = ncar_cells.tags("tsslcd")
    out["spectral_context"], out["spatial_context"] = tags["spectral_context"], tags["spatial_context"]
    return out


def metrics_of(m: dict, keep: np.ndarray) -> dict:
    e = m["pred"][keep].astype(np.float64) - m["target"][keep].astype(np.float64)
    e32 = e.astype(np.float32)
    return {"mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": float((np.abs(np.power(np.float32(10), e32 / np.float32(10)) - 1) * 100).mean(dtype=np.float64))}


def same_windows(client, records: list[dict], prefixes: list[str]) -> pd.DataFrame:
    load = lambda k: ncar_cells.load_npz(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())     # noqa: E731
    # TSS-LCD cells by (case, condition, target_key): keys of all runs (6a: placements where the receiver is swapped in, or unswapped core)
    cells: dict[tuple, list[str]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        for pos in r["positions"]:
            if r["case"] == "6a" and r["variant"] != "unswapped" and pos["receiver"] not in ("guesthouse", "law73"):
                continue
            base = f"{PREFIX}/{r['case']}/{r['condition']}/{pos['target_key']}/2d/tsslcd/{r['variant']}/seed-{r['seed']}"
            cells.setdefault((r["case"], r["condition"], pos["target_key"]), []).append(f"{base}/forecasts.npz")
    targets = {k[2] for k in cells}
    comps: dict[str, list[str]] = {}
    for p in prefixes:
        for k in list_keys(client, p, "forecasts.npz"):
            hit = TARGET_RE.search(k)
            if hit and hit.group(1) in targets:
                comps.setdefault(hit.group(1), []).append(k)
    rows = []

    def one(item):
        (case, cond, tk), keys = item
        mine = [load(k) for k in keys]
        out = []
        for ck in comps.get(tk, []):
            other = load(ck)
            label = re.sub(r"/seed-\d+/forecasts\.npz$", "", ck.replace(tk + "/", ""))
            seed = re.search(r"/seed-(\d+)/", ck)
            for h in sorted(set(other) & set(mine[0])):
                common = np.intersect1d(other[h]["origin"], mine[0][h]["origin"])
                if not len(common):
                    continue
                mt = [metrics_of(x[h], np.isin(x[h]["origin"], common)) for x in mine]
                ot = metrics_of(other[h], np.isin(other[h]["origin"], common))
                out.append({"case": case, "condition": cond, "target_key": tk, "horizon": h, "comparator": label,
                            "comparator_seed": int(seed.group(1)) if seed else None, "n_common_windows": int(len(common)),
                            "tsslcd_n_runs": len(mt), **{f"tsslcd_{k}": float(np.mean([x[k] for x in mt])) for k in mt[0]},
                            **{f"comparator_{k}": v for k, v in ot.items()}})
        return out

    with ThreadPoolExecutor(16) as pool:
        for chunk in pool.map(one, cells.items()):
            rows += chunk
    table = pd.DataFrame(rows)
    if len(table):                                   # comparator seeds averaged (TSS-LCD already the mean of its runs)
        keys = ["case", "condition", "target_key", "horizon", "comparator"]
        num = [c for c in table.columns if c.startswith(("tsslcd_", "comparator_")) and c != "comparator_seed"]
        table = table.groupby(keys).agg(**{c: (c, "mean") for c in num}, n_common_windows=("n_common_windows", "min"),
                                        comparator_n_seeds=("comparator_seed", "nunique")).reset_index()
    return table


def put(client, key: str, frame: pd.DataFrame) -> None:
    client.put_object(Bucket=BUCKET, Key=key, Body=frame.to_csv(index=False).encode())
    print(f"wrote s3://{BUCKET}/{key} ({len(frame)} rows)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="table version (S3 sub-prefix, never overwritten)")
    p.add_argument("--compare-prefixes", nargs="*", default=[])
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    client = spectrum_s3_client()
    out_prefix = f"{PREFIX}/tables/{a.tag}"
    if list_keys(client, out_prefix + "/", ".csv"):
        raise SystemExit(f"{out_prefix} exists; use a new --tag")
    with ThreadPoolExecutor(32) as pool:
        records = list(pool.map(lambda k: read_json(client, k), list_keys(client, f"{PREFIX}/records/", ".json")))
    runs = run_rows(records)
    table = pd.concat([runs.assign(level="run"), summarise(runs)], ignore_index=True)
    put(client, f"{out_prefix}/transfer_tsslcd.csv", table)
    frames = {"transfer_tsslcd.csv": table}
    if a.compare_prefixes:
        sw = same_windows(client, records, a.compare_prefixes)
        put(client, f"{out_prefix}/transfer_tsslcd_same_windows.csv", sw)
        frames["transfer_tsslcd_same_windows.csv"] = sw
    if a.out_dir:
        for name, f in frames.items():
            f.to_csv(Path(a.out_dir) / name, index=False)
    pd.set_option("display.width", 250)
    s = table[(table.level == "group") & (table.horizon == 12)]
    print(s[["case", "condition", "source_band", "target_band", "role", "variant_class", "mae", "mae_se", "n_seeds"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
