"""Level drift or local level variation? Tables for items 5-9 of task J from the per-origin outputs of ``training.ray.level_diagnostics_origins``.

Reads ``level-drift-oct2026/`` (level summaries, per-origin model files) and ``level-drift-oct2026/coverage.json`` from the bucket and writes
``evaluation/results/level_drift_*.csv`` (+ a text report on stdout). Meant to run as a Ray job on the cluster (``training/ray/scripts/run_level_drift.sh``).

(5) gap by range: pooled MAE of each model over the tuned lookback median's MAE minus one, per band, range and horizon (train = in-sample for the models).
(6) regression of the model's excess error over the median on the level distance d (per (window, bin)), with a day-block bootstrap, against the mechanism's
    prediction slope = 1 - gain (offset-response gain: the 1-minute test of task I when it exists, else the 15-minute one at t+12).
(7) quantiles of d and of d / training standard deviation per band, receiver and range.
(8) per day of the test week (and week2): level distance (median and 90th percentile of d), model MAE, median MAE, relative excess; slope per day.
(9) coverage of the raw traces after the test week.
"""
from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PREFIX = "level-drift-oct2026"
BANDS = ("600_800", "2400_2600")
RANGES = ("train", "validation", "test", "week2")
DAY_START = {"test": "2026-10-27T20:00:00Z", "week2": "2026-11-03T20:00:00Z", "validation": "2026-10-25T13:55:00Z", "train": "2026-10-15T14:34:00Z"}
HORIZONS = (1, 15, 60)
STAT = ["n", "sx", "sy", "sxy", "sxx", "syy"]
RNG = np.random.default_rng(0)
OUT = Path("evaluation/results")


# ----------------------------------------------------------------------------------------------- pure helpers
def ols(n, sx, sy, sxy, sxx, syy) -> tuple[float, float, float]:
    vx, cov, vy = n * sxx - sx * sx, n * sxy - sx * sy, n * syy - sy * sy
    if n < 3 or vx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = cov / vx
    return float(slope), float((sy - slope * sx) / n), float(cov * cov / (vx * vy)) if vy > 0 else float("nan")


def bootstrap_slope(block_sums: np.ndarray, draws: int = 2000, rng=RNG) -> tuple[float, float]:
    """95% interval of the pooled OLS slope when whole blocks (rows of ``block_sums``: n, sx, sy, sxy, sxx, syy) are resampled with replacement."""
    b = len(block_sums)
    if b < 2:
        return float("nan"), float("nan")
    slopes = np.array([ols(*block_sums[rng.integers(0, b, b)].sum(axis=0))[0] for _ in range(draws)])
    slopes = slopes[np.isfinite(slopes)]
    return (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))) if len(slopes) else (float("nan"), float("nan"))


def line_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x and its standard error (nan with fewer than 3 points)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum()))
    return float(slope), se


def gap(model_mae: float, median_mae: float) -> float:
    return float(model_mae / median_mae - 1.0)


# ------------------------------------------------------------------------------------------------------- loading
def keys(client, prefix: str, suffix: str) -> list[str]:
    return [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix) for o in pg.get("Contents", []) if o["Key"].endswith(suffix)]


def load_models(client) -> pd.DataFrame:
    ks = keys(client, f"{PREFIX}/model/", "per_origin.csv")

    def one(k):
        p = k.split("/")                                        # level-drift-oct2026/model/<band>/<receiver>/<tag>/<model>/seed-N/per_origin.csv
        df = pd.read_csv(io.BytesIO(client.get_object(Bucket="spectrum", Key=k)["Body"].read()))
        df["band"], df["receiver"], df["tag"], df["model"], df["seed"] = p[2], p[3], p[4], p[5], int(p[6].split("-")[1])
        return df
    with ThreadPoolExecutor(16) as pool:
        frames = list(pool.map(one, ks))
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["ckpt"] = frame["tag"] + "/seed" + frame["seed"].astype(str)
    return frame


def add_day(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    origin = frame["range"].map(DAY_START)
    frame["day"] = np.floor((frame["time"] - pd.to_datetime(origin, utc=True)) / pd.Timedelta(days=1)).astype(int)
    frame.loc[frame["range"] == "week2", "day"] += 7                # test week = days 0-6, week2 = days 7-12
    return frame


def load_gains(client) -> tuple[dict[tuple[str, str], float], str]:
    """{(band, model): gain}: the 1-minute offset test of task I when it exists (horizon 15, mean over offsets, receivers, seeds), else the 15-minute one (t+12)."""
    ks = keys(client, "offset-response-1min-oct2026/", "summary.json")
    rows = []
    for k in ks:
        d = json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())
        if d.get("ok") and d.get("rep", "2d") == "2d":
            for r in d.get("rows", []):
                if int(r["horizon"]) == 15:
                    rows.append({"band": d["band"], "model": d["model"], "gain": r["gain"]})
    if rows:
        g = pd.DataFrame(rows).groupby(["band", "model"]).gain.mean()
        return {k: float(v) for k, v in g.items()}, "1-minute offset test (t+15)"
    d = pd.read_csv(io.BytesIO(client.get_object(Bucket="spectrum", Key="analysis-inputs/offset_response.csv")["Body"].read()))
    g = d[(d.horizon == 12) & (d.rep == "2d")].groupby(["band", "model"]).gain.mean()
    return {k: float(v) for k, v in g.items()}, "15-minute offset test (t+12)"


# --------------------------------------------------------------------------------------------------------- items
def item5(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, model, rng, h), g in frame.groupby(["band", "model", "range", "horizon"]):
        per_ckpt = g.groupby("ckpt").apply(lambda x: gap(x.mae_model.mean(), x.mae_median.mean()), include_groups=False)
        rows.append({"band": band, "model": model, "range": rng, "horizon": h, "n_origin_rows": len(g), "n_checkpoints": g.ckpt.nunique(),
                     "mae_model": g.mae_model.mean(), "mae_median": g.mae_median.mean(), "mae_mean": g.mae_mean.mean(), "gap_vs_median": gap(g.mae_model.mean(), g.mae_median.mean()),
                     "gap_sd_over_checkpoints": float(per_ckpt.std()) if len(per_ckpt) > 1 else float("nan")})
    return pd.DataFrame(rows)


def item6(frame: pd.DataFrame, gains: dict, gain_source: str) -> pd.DataFrame:
    rows = []
    frame = add_day(frame)
    for (band, model, rng), g in frame.groupby(["band", "model", "range"]):
        for h in list(HORIZONS) + ["all"]:
            sel = g if h == "all" else g[g.horizon == h]
            tot = sel[STAT].sum().to_numpy()
            slope, intercept, r2 = ols(*tot)
            blocks = sel.groupby("day")[STAT].sum().to_numpy()
            lo, hi = bootstrap_slope(blocks)
            gain = gains.get((band, model), float("nan"))
            rows.append({"band": band, "model": model, "range": rng, "horizon": h, "slope": slope, "slope_lo": lo, "slope_hi": hi, "intercept": intercept, "r2": r2,
                         "n_points": tot[0], "n_day_blocks": len(blocks), "gain": gain, "predicted_slope_1_minus_gain": 1 - gain, "gain_source": gain_source})
    return pd.DataFrame(rows)


def item7(client) -> pd.DataFrame:
    rows = []
    for k in keys(client, f"{PREFIX}/level/", "summary.json"):
        d = json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())
        if not d.get("ok"):
            continue
        for rng, r in d["ranges"].items():
            for kind in ("d_db", "d_sd"):
                rows.append({"band": d["band"], "receiver": d["receiver"], "range": rng, "measure": kind, "n_origins": r["n_origins"],
                             **{q: r[kind][q] for q in ("q10", "q50", "q90", "q99", "mean")}})
    return pd.DataFrame(rows)


def level_by_day(client) -> pd.DataFrame:
    rows = []
    for k in keys(client, f"{PREFIX}/level/", "summary.json"):
        d = json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())
        if not d.get("ok"):
            continue
        for rng in ("test", "week2"):
            for day, r in d["ranges"][rng].get("per_day", {}).items():
                rows.append({"band": d["band"], "receiver": d["receiver"], "range": rng, "day": int(day), "n_origins": r["n_origins"],
                             "d_q50": r["d_db"]["q50"], "d_q90": r["d_db"]["q90"], "dsd_q50": r["d_sd"]["q50"], "dsd_q90": r["d_sd"]["q90"]})
    return pd.DataFrame(rows)


def item8(frame: pd.DataFrame, level_day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_day(frame)
    frame = frame[frame["range"].isin(["test", "week2"])]
    rows = []
    for (band, model, day), g in frame.groupby(["band", "model", "day"]):
        rows.append({"band": band, "model": model, "day": day, "n_origin_rows": len(g), "mae_model": g.mae_model.mean(), "mae_median": g.mae_median.mean(),
                     "rel_excess": gap(g.mae_model.mean(), g.mae_median.mean())})
    daily = pd.DataFrame(rows)
    lv = level_day.groupby(["band", "day"])[["d_q50", "d_q90", "dsd_q50", "dsd_q90"]].mean().reset_index()
    daily = daily.merge(lv, on=["band", "day"], how="left")
    slopes = []
    for (band, model), g in daily.groupby(["band", "model"]):
        for label, days in (("test week (days 0-6)", range(0, 7)), ("week2 (days 7-12)", range(7, 13)), ("both (days 0-12)", range(0, 13))):
            s = g[g.day.isin(days)]
            for col in ("rel_excess", "d_q50", "d_q90"):
                slope, se = line_slope(s.day.to_numpy(), s[col].to_numpy())
                slopes.append({"band": band, "model": model, "span": label, "quantity": col, "slope_per_day": slope, "se": se, "n_days": len(s)})
    return daily, pd.DataFrame(slopes)


def coverage(client) -> pd.DataFrame:
    rows = json.loads(client.get_object(Bucket="spectrum", Key=f"{PREFIX}/coverage.json")["Body"].read())
    out = []
    for r in rows:
        run = r["runs"].get("20261025T1355Z", {})
        for day, n in sorted(run.get("valid_minutes_per_day", {}).items()):
            out.append({"band": r["band"], "receiver": r["receiver"], "day": day, "valid_minutes": n})
    return pd.DataFrame(out)


def main() -> None:
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_models(client)
    gains, gsrc = load_gains(client)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(f"{len(frame)} origin rows, models {sorted(frame.model.unique())}, checkpoints {frame.groupby('model').ckpt.nunique().to_dict()}, gains from {gsrc}: {gains}")
    t5 = item5(frame)
    t5.to_csv(OUT / "level_drift_gap_by_range.csv", index=False)
    print("\n(5) gap of the model MAE over the tuned median, per range and horizon (train is in-sample for the models):")
    print(t5.pivot_table(index=["band", "model", "range"], columns="horizon", values="gap_vs_median").round(3).to_string())
    t6 = item6(frame, gains, gsrc)
    t6.to_csv(OUT / "level_drift_regression.csv", index=False)
    print("\n(6) slope of the excess error over the median on the level distance d (dB per dB), vs the mechanism's 1 - gain:")
    print(t6[t6.horizon == "all"].round(3).to_string(index=False))
    t7 = item7(client)
    t7.to_csv(OUT / "level_drift_level_quantiles.csv", index=False)
    print("\n(7) level distance d = |window mean - training mean| (dB): quantiles, mean over the four receivers:")
    print(t7[t7.measure == "d_db"].groupby(["band", "range"])[["q10", "q50", "q90", "q99", "mean"]].mean().round(3).to_string())
    print("\n(7b) the same divided by the bin's training standard deviation:")
    print(t7[t7.measure == "d_sd"].groupby(["band", "range"])[["q10", "q50", "q90", "q99", "mean"]].mean().round(3).to_string())
    lday = level_by_day(client)
    daily, slopes = item8(frame, lday)
    daily.to_csv(OUT / "level_drift_by_day.csv", index=False)
    slopes.to_csv(OUT / "level_drift_day_slopes.csv", index=False)
    print("\n(8) per day (day 0 = 2026-10-27 20:00; days 0-6 test week, 7-12 week2): level distance median / p90 (dB), MAE, relative excess:")
    for band in BANDS:
        d = daily[daily.band == band].pivot_table(index="day", columns="model", values="rel_excess").round(3)
        lv = daily[daily.band == band].drop_duplicates("day").set_index("day")[["d_q50", "d_q90"]].round(2)
        print(f"\n{band}:"); print(pd.concat([lv, d], axis=1).to_string())
    print("\nslopes per day:"); print(slopes.round(4).to_string(index=False))
    cov = coverage(client)
    cov.to_csv(OUT / "level_drift_coverage.csv", index=False)
    print("\n(9) valid minutes per day after the test week (1,440 = full):")
    print(cov[cov.day >= "2026-11-03"].pivot_table(index="day", columns=["band", "receiver"], values="valid_minutes").to_string())
    for f in sorted(OUT.glob("level_drift_*.csv")):
        client.put_object(Bucket="spectrum", Key="analysis-outputs/" + f.name, Body=f.read_bytes())


if __name__ == "__main__":
    main()
