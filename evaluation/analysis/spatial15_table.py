"""15-minute spatial (4D) results: map sensitivity (S2), held-out receivers (S3), receiver count (S5) and permuted coordinates (S4).

Reads ``spatial15-tuned-oct2026/<band>/<variant>/<model>/seed-N/result.json`` (written by ``training.ray.spatial15``) and
``.../permuted/<model>/seed-N/p<perm>/result.json``, from the durable bucket (read only; needs the usual SPECTRUM_S3_* environment).

Two kinds of numbers are kept apart and must not be mixed with each other or with the 1D/2D tables:

* **map MAE** (``map__test_mean_horizon_mae_db``): the model's forecast of its OWN interpolated 10 x 10 map, scored on all 100 grid cells
  against the map built from the measurements. Most cells are interpolated, so this is the easiest of the three numbers.
* **receiver MAE** (``receivers.<condition>.mean_horizon_mae_db``): the forecast sampled at a physical receiver and scored against that
  receiver's own measurement, the same unit as the 1D/2D models. Conditions (spatial15 labels ``six_to_guesthouse-<receiver>`` and
  ``seven_to_guesthouse-<receiver>`` are legacy names): ``insample`` = every receiver in the input set (``geometry_correct``),
  ``absent`` = the receiver was left out of the input map (``six_to_...``), ``added`` = the receiver was added to the input map without
  retraining (``seven_to_...``).

Both are the mean over t+4/12/24/48 (15-minute steps) in dB, test week 2026-10-27 20:00 to 11-03 20:00 UTC. Means are over seeds with the
seed standard deviation; a model's coverage (cells finished of cells expected) is printed first because STS-PredNet variants are still running
and ResidualConvLSTM has none yet. ``lookbackmean4d`` is parameter-free (one seed, 42). Own-history references for S3 and S5 come from the
per-receiver 1D/2D tuned models and persistence scored by ``training.ray.transfer15`` (``neighbor_value.py`` helpers): guest house and law 73
are frozen transfer (the models were trained on the core four), the core receivers are in distribution.

    python evaluation/analysis/spatial15_table.py [--out evaluation/results/spatial15] [--cache cells.pkl]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BUCKET = "spectrum"
PREFIX = "spatial15-tuned-oct2026/"
BANDS = ("600_800", "2400_2600")
MODELS = ("lookbackmean4d", "linearar4d", "residuallinearar4d", "stsprednet", "residualconvlstm")
SEEDS = {"lookbackmean4d": (42,)}                 # every other model: 40, 41, 42
DEFAULT_SEEDS = (40, 41, 42)
S2 = ("core4-p1", "core4-p2", "core4-p4", "core4-p16", "core4-p2-linear", "core4-p2-g5", "core4-p2-g20")
S3 = tuple(f"loo-{n}" for n in ("humanities", "madsen", "moran", "sagepoint", "guesthouse", "law73"))
S5 = tuple(f"k{k}-{i}" for k in (2, 3) for i in range(3))
ALL_VARIANTS = S2 + S3 + S5
CONDITION = re.compile(r"^(geometry_correct|six_to_guesthouse|seven_to_guesthouse)(?:-(.+))?$")
COND_NAME = {"geometry_correct": "insample", "six_to_guesthouse": "absent", "seven_to_guesthouse": "added"}


# ------------------------------------------------------------------------------------------------------------ loading
def parse_condition(label: str) -> tuple[str, str | None]:
    """('absent', 'guesthouse') for ``six_to_guesthouse-guesthouse``; ('insample', None) for ``geometry_correct``."""
    match = CONDITION.match(label)
    if not match:
        raise ValueError(f"unknown condition label {label!r}")
    return COND_NAME[match.group(1)], match.group(2)


def parse_key(key: str) -> dict | None:
    """Fields of a result key, or None for files that are not cell results."""
    parts = key[len(PREFIX):].split("/")
    if parts[-1] != "result.json" or "eval" in parts:
        return None
    if len(parts) == 6 and parts[1] == "permuted":                      # <band>/permuted/<model>/seed-N/p<perm>/result.json
        return {"band": parts[0], "variant": "permuted", "model": parts[2], "seed": int(parts[3].split("-")[1]),
                "perm": int(parts[4][1:])}
    if len(parts) == 5:                                                  # <band>/<variant>/<model>/seed-N/result.json
        return {"band": parts[0], "variant": parts[1], "model": parts[2], "seed": int(parts[3].split("-")[1]), "perm": None}
    return None


def fetch(client, cache: Path | None) -> dict[str, dict]:
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX)
            for o in page.get("Contents", []) if o["Key"].endswith("/result.json")]
    keys = [k for k in keys if parse_key(k)]
    with ThreadPoolExecutor(16) as pool:
        out = dict(pool.map(lambda k: (k, json.loads(client.get_object(Bucket=BUCKET, Key=k)["Body"].read())), keys))
    if cache:
        cache.write_bytes(pickle.dumps(out))
    return out


def flatten(raw: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per cell with its map metrics, one row per cell x physical-receiver condition)."""
    cells, receivers = [], []
    for key, d in raw.items():
        info = parse_key(key)
        if info is None or not d.get("ok", True):
            continue
        base = {**info}
        if info["variant"] == "permuted":
            receivers.append({**base, "condition": "permuted", "receiver": "core four", "mae": float(d["mean_horizon_mae_db"]),
                              "n_origins": d.get("n_origins")})
            continue
        cells.append({**base, "map_mae": d.get("map__test_mean_horizon_mae_db"),
                      **{f"map_mae_t{h}": d.get(f"map__test_mae_db_t{h}") for h in (4, 12, 24, 48)}})
        for label, r in (d.get("receivers") or {}).items():
            condition, receiver = parse_condition(label)
            receivers.append({**base, "condition": condition, "receiver": receiver or "input set", "mae": r["mean_horizon_mae_db"],
                              "n_origins": r.get("n_origins"), **{f"mae_t{h}": r.get(f"mae_db_t{h}") for h in (4, 12, 24, 48)}})
    return pd.DataFrame(cells), pd.DataFrame(receivers)


# ------------------------------------------------------------------------------------------------------------ helpers
def mean_sd(values: pd.Series) -> tuple[float, float, int]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return (float(v.mean()) if len(v) else np.nan, float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))


def fmt(mean: float, sd: float, n: int) -> str:
    return "-" if n == 0 or np.isnan(mean) else (f"{mean:.3f}" if n == 1 else f"{mean:.3f}±{sd:.3f}")


def seed_mean_then_group(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Per-seed values averaged over the rows of a group first (variants or receivers), then mean and sd over seeds."""
    per_seed = frame.groupby(keys + ["seed"], dropna=False)[value].mean().reset_index()
    out = []
    for group, sub in per_seed.groupby(keys, dropna=False):
        m, s, n = mean_sd(sub[value])
        out.append({**dict(zip(keys, group if isinstance(group, tuple) else (group,))), "mean": m, "sd": s, "n_seeds": n})
    return pd.DataFrame(out)


def coverage(cells: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        for model in MODELS:
            expected_seeds = SEEDS.get(model, DEFAULT_SEEDS)
            expected = len(ALL_VARIANTS) * len(expected_seeds)
            sub = cells[(cells.band == band) & (cells.model == model)] if len(cells) else cells
            done = len(sub[sub.variant.isin(ALL_VARIANTS) & sub.seed.isin(expected_seeds)]) if len(sub) else 0
            p = perm[(perm.band == band) & (perm.model == model)] if len(perm) else perm
            rows.append({"band": band, "model": model, "cells_done": done, "cells_expected": expected,
                         "variants_started": int(sub.variant.nunique()) if len(sub) else 0,
                         "permuted_cells": len(p), "permutation_seeds": int(p.perm.nunique()) if len(p) else 0})
    return pd.DataFrame(rows)


def own_history_references(client, receivers: list[str]) -> pd.DataFrame:
    """Mean MAE over t+4/12/24/48 and the available origins of the per-receiver 1D/2D tuned models and persistence at each receiver."""
    import neighbor_value as nv

    rows = []
    for band in BANDS:
        for receiver in receivers:
            for label, series in (("persistence", nv.persistence_series(client, band, receiver)),
                                  *[(f"{rep} {name}", nv.per_receiver_series(client, band, receiver, rep, name))
                                    for rep, name in (("2d", "residuallinearar2d"), ("2d", "linearar2d"), ("1d", "residuallinearar1d"))]):
                if series is not None and len(series):
                    rows.append({"band": band, "receiver": receiver, "reference": label, "mae": float(series["mae"].mean()),
                                 "n_origins": int(len(series))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ tables
def table_s2(cells: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Map sensitivity: map MAE, in-sample receiver MAE and the guest house / law 73 MAE absent from and added to the map, per variant."""
    rows = []
    for band in BANDS:
        for model in MODELS:
            for variant in S2:
                c = cells[(cells.band == band) & (cells.model == model) & (cells.variant == variant)]
                r = rec[(rec.band == band) & (rec.model == model) & (rec.variant == variant)]
                if c.empty:
                    continue
                row = {"band": band, "model": model, "variant": variant, "n_seeds": c.seed.nunique()}
                row["map_mae"] = mean_sd(c.groupby("seed").map_mae.mean())[0]
                row["map_mae_sd"] = mean_sd(c.groupby("seed").map_mae.mean())[1]
                for cond in ("insample", "absent", "added"):
                    sub = r[(r.condition == cond) & r.receiver.isin(["input set", "guesthouse", "law73"])]
                    m, s, n = mean_sd(sub.groupby("seed").mae.mean())
                    row[f"{cond}_mae"], row[f"{cond}_mae_sd"] = m, s
                rows.append(row)
    return pd.DataFrame(rows)


def table_s3(rec: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out: the held-out receiver absent from / added to the map, the in-sample receivers, lookback mean and own-history references."""
    rows = []
    for band in BANDS:
        for variant in S3:
            held = variant.removeprefix("loo-")
            for model in MODELS:
                r = rec[(rec.band == band) & (rec.model == model) & (rec.variant == variant)]
                if r.empty:
                    continue
                row = {"band": band, "held_out": held, "model": model, "n_seeds": r.seed.nunique()}
                for cond, recv in (("insample", "input set"), ("absent", held), ("added", held)):
                    sub = r[(r.condition == cond) & (r.receiver == recv)]
                    row[f"{cond}_mae"], row[f"{cond}_mae_sd"], _ = mean_sd(sub.groupby("seed").mae.mean())
                rows.append(row)
    table = pd.DataFrame(rows)
    if len(refs) and len(table):
        wide = refs.pivot_table(index=["band", "receiver"], columns="reference", values="mae").reset_index().rename(columns={"receiver": "held_out"})
        table = table.merge(wide, on=["band", "held_out"], how="left")
    return table


def table_s5(rec: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Receiver count: guest house and law 73 absent from / added to the map for 2, 3 (and 4 = core4-p2) input receivers, plus own-history refs."""
    rows = []
    for band in BANDS:
        for model in MODELS:
            for k, variants in ((2, [v for v in S5 if v.startswith("k2-")]), (3, [v for v in S5 if v.startswith("k3-")]), (4, ["core4-p2"])):
                r = rec[(rec.band == band) & (rec.model == model) & rec.variant.isin(variants)]
                if r.empty:
                    continue
                row = {"band": band, "model": model, "input_receivers": k, "subsets": len(variants), "n_cells": r.groupby(["variant", "seed"]).ngroups}
                for receiver in ("guesthouse", "law73"):
                    for cond in ("absent", "added"):
                        sub = r[(r.condition == cond) & (r.receiver == receiver)]
                        row[f"{receiver}_{cond}"] = mean_sd(sub.groupby("seed").mae.mean())[0]
                sub = r[r.condition == "insample"]
                row["insample"] = mean_sd(sub.groupby("seed").mae.mean())[0]
                rows.append(row)
    table = pd.DataFrame(rows)
    if len(refs) and len(table):
        for receiver in ("guesthouse", "law73"):
            ref = refs[(refs.receiver == receiver) & (refs.reference == "2d residuallinearar2d")].set_index("band").mae
            table[f"{receiver}_own_history_2d"] = table.band.map(ref)
    return table


def table_s4(rec: pd.DataFrame) -> pd.DataFrame:
    """Permuted coordinates: the base (correct-geometry) error at the four core receivers against the permuted-coordinate error."""
    rows = []
    for band in BANDS:
        for model in MODELS:
            p = rec[(rec.band == band) & (rec.model == model) & (rec.condition == "permuted")]
            base = rec[(rec.band == band) & (rec.model == model) & (rec.variant == "core4-p2") & (rec.condition == "insample")]
            if p.empty:
                continue
            bm = mean_sd(base.groupby("seed").mae.mean())[0] if len(base) else np.nan
            pm, ps, _ = mean_sd(p.groupby("seed").mae.mean())
            rows.append({"band": band, "model": model, "base_insample_mae": bm, "permuted_mae": pm, "permuted_sd_over_seeds": ps,
                         "delta": pm - bm, "permutations": p.perm.nunique(), "seeds": p.seed.nunique()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ main
def show(title: str, table: pd.DataFrame, cols: list[str] | None = None) -> None:
    print(f"\n{title}")
    print("(no cells yet)" if table.empty else (table[cols] if cols else table).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/spatial15", help="prefix of the result CSVs")
    parser.add_argument("--cache", type=Path, default=None, help="pickle of the downloaded result files (reused if it exists)")
    parser.add_argument("--no-references", action="store_true", help="skip the per-receiver 1D/2D reference lookups")
    args = parser.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    cells, rec = flatten(fetch(client, args.cache))
    perm_rows = rec[rec.condition == "permuted"]
    cov = coverage(cells, perm_rows)
    refs = pd.DataFrame() if args.no_references else own_history_references(client, ["humanities", "madsen", "moran", "sagepoint", "guesthouse", "law73"])
    s2, s3, s5, s4 = table_s2(cells, rec), table_s3(rec, refs), table_s5(rec, refs), table_s4(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cells", cells), ("receivers", rec), ("coverage", cov), ("s2", s2), ("s3", s3), ("s4", s4), ("s5", s5), ("references", refs)):
        frame.to_csv(f"{out}_{name}.csv", index=False)

    pd.set_option("display.width", 250)
    show("COVERAGE (cells finished of expected: 19 variants x seeds 40-42; lookback mean 19 variants x seed 42; permuted = cells found)", cov)
    show("S2 map sensitivity (MAE dB mean over t+4/12/24/48, mean over seeds). map_mae = own interpolated map (NOT comparable with 1D/2D); "
         "insample/absent/added = physical receivers (core four in the input / guest house and law 73 left out of / added to the map)", s2,
         ["band", "model", "variant", "n_seeds", "map_mae", "insample_mae", "absent_mae", "added_mae"])
    show("S3 held-out receivers (physical-receiver MAE dB): absent = left out of the map, added = added without retraining, insample = the "
         "other receivers in the input; own-history references (frozen transfer for guesthouse and law73, in distribution for core receivers)", s3)
    show("S5 receiver count (physical-receiver MAE dB at guest house and law 73; 4 = the core-four model)", s5)
    show("S4 permuted core-receiver coordinates (physical-receiver MAE dB at the four core receivers)", s4)
    print(f"\nwrote {out}_*.csv")


if __name__ == "__main__":
    main()
