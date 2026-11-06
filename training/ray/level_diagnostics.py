"""CPU-only diagnostics: why do learned corrections add little over simple baselines on 1-minute POWDER?

Everything here is closed-form or small (numpy / scikit-learn), runs on CPU workers, and needs no checkpoints or GPUs.
Split = the October 2026 protocol of the one-minute models: fit on the train run (2026-10-15T14:34Z to 10-22T13:13Z),
select on validation (10-25T13:55Z to 10-27T13:55Z), score on test (10-27T20:00Z to 11-03T20:00Z). Four core
receivers, both bands, 200 one-MHz bins. Values are dB(m); MAE in dB. Forecast origin t is the last observed minute;
the point target at horizon H is the value at t + H (the models' endpoint convention, lookback 60).

One Ray task per (receiver, band) computes:

* ``baselines``: train-mean, persistence, lookback mean / median / linear-power mean for L in L_GRID, EWMA (lambda grid),
  a local-level Kalman forecast with per-bin parameters from train moments, seasonal naive (24 h, own origin subset).
  The "strong" baseline is chosen on validation (per receiver/band/horizon, shared across bins) and frozen.
* ``probes``: ridge corrections ŷ = anchor + g(window - anchor) on the 60-minute window, per bin and pooled over bins,
  for anchors {mean60, last, none}. With centred inputs both anchored forms are affine maps whose lag weights sum
  to one (level-shift equivariant), so mean60 and last span the same class and differ only through regularization;
  "none" (a plain affine AR with intercept) is the unconstrained superset (it can revert toward a fixed level); ridge strength and a correction scale s chosen on validation;
  train / validation / test MAE reported for anchor alone, full correction and the validation-scaled correction.
  A pooled "rich" linear probe (lags + window std + slope + time of day) and a pooled gradient-boosting probe trained
  with absolute-error loss test for nonlinear or covariate signal.
* ``cheap``: for the last-value anchor, how much of the learned correction is just "pull toward the window mean"
  (slope and R^2 of the correction on mean60 - last).
* ``condmedian``: median of the mean60 residual by decile of (last - mean60): is any mean reversion left?
* ``statespace``: per-bin local-level moments from first differences (sigma_eps^2 = -gamma_1,
  sigma_eta^2 = gamma_0 + 2 gamma_1), lag-2/3 checks, and calibration of the implied forecast variance on test.
* ``aggregation``: same-target arms at 15-minute block-aligned origins with a 15-minute average target
  (W = 15, H in {15, 60}): 1-minute history, 15-minute block history (aligned and rolling training origins), and
  decimated history (every 15th minute, all 15 phases), each with its own baselines and ridge probe.
* ``daily``: per-test-day skill of the strong baseline and the best probe (temporal stability).

Skill = 1 - MAE / MAE(reference) with a paired moving-block bootstrap over origins (bin-averaged per origin).

    python -m training.ray.level_diagnostics --output-dir runs/claude/level-diagnostics/<id>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
BANDS = ("600_800", "2400_2600")
RAW_ROOT = "data/powder"
RAW_NAME = "power_1mhz_avg_per_minute.csv"
TRAIN_RUN, TEST_RUN = "20261015T1434Z", "20261025T1355Z"
RANGES = {
    "train": ("2026-10-15T14:34:00Z", "2026-10-22T13:13:00Z", TRAIN_RUN),
    "validation": ("2026-10-25T13:55:00Z", "2026-10-27T13:55:00Z", TEST_RUN),
    "test": ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z", TEST_RUN),
}
LOOKBACK = 60
HORIZONS = (1, 15, 60)
L_GRID = (5, 15, 30, 60, 120, 240)
MAX_HISTORY = max(L_GRID)
SEASON = 1440
EWMA_GRID = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01)
RIDGE_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)        # relative to the mean diagonal of X'X
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
ANCHORS = ("mean60", "last", "none")
BLOCK = 15
AGG_HORIZONS = (15, 60)
AGG_LOOKBACKS = (60, 240)
BOOT_BLOCK = 240
BOOT_REPS = 1000
EPSILON = 0.02
GBM_TRAIN_ROWS = 200_000


# ------------------------------------------------------------------------------------------------- data
def load_run(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """CSV -> (T, F) dB array on a regular 1-minute grid (NaN for missing minutes) and its minute index."""
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
    frame.index = pd.DatetimeIndex(stamps)
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    grid = pd.date_range(frame.index[0], frame.index[-1], freq="min")
    frame = frame.reindex(grid)
    return frame.to_numpy(dtype=np.float64), grid


class Split:
    """One run's array plus the [lo, hi) row range whose rows are the split's *targets*. History may come from
    earlier rows of the same run (causal; never from a later split)."""

    def __init__(self, values: np.ndarray, index: pd.DatetimeIndex, start: str, end: str):
        self.values, self.index = values, index
        self.lo = int(index.searchsorted(pd.Timestamp(start)))
        self.hi = int(index.searchsorted(pd.Timestamp(end)))
        self.cum = np.vstack([np.zeros((1, values.shape[1])), np.nancumsum(values, axis=0)])
        self.lin = np.power(10.0, values / 10.0)
        self.lincum = np.vstack([np.zeros((1, values.shape[1])), np.nancumsum(self.lin, axis=0)])
        valid = np.isfinite(values).all(axis=1).astype(np.int64)
        self.validcum = np.concatenate([[0], np.cumsum(valid)])

    def all_valid(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Rows [a, b] (inclusive) all finite, per element of the index arrays."""
        return (self.validcum[b + 1] - self.validcum[a]) == (b - a + 1)

    def origins(self, horizon: int, history: int = MAX_HISTORY, extra_lag: int = 0, aligned: bool = False,
                width: int = 1) -> np.ndarray:
        """Origins t with targets t+H-width+1..t+H inside the split and history rows t-history+1..t (and the
        seasonal lag) finite and inside the run."""
        first = max(self.lo - horizon + width - 1, history - 1, extra_lag - horizon if extra_lag else 0)
        last = self.hi - 1 - horizon
        if last < first:
            return np.zeros(0, dtype=np.int64)
        t = np.arange(first, last + 1)
        ok = self.all_valid(t - history + 1, t) & self.all_valid(t + horizon - width + 1, t + horizon)
        if extra_lag:
            ok &= self.all_valid(t + horizon - extra_lag, t + horizon - extra_lag)
        if aligned:
            ok &= (self.index[t].minute.to_numpy() % BLOCK) == BLOCK - 1
        return t[ok]


def load_cell(repo_root: str, node: str, band: str) -> dict[str, Split]:
    cache: dict[str, tuple[np.ndarray, pd.DatetimeIndex]] = {}
    splits = {}
    for name, (start, end, run) in RANGES.items():
        if run not in cache:
            cache[run] = load_run(Path(repo_root) / RAW_ROOT / node / run / band / RAW_NAME)
        splits[name] = Split(*cache[run], start, end)
    return splits


# ------------------------------------------------------------------------------------------- predictors
def window_mean(s: Split, t: np.ndarray, length: int) -> np.ndarray:
    return (s.cum[t + 1] - s.cum[t + 1 - length]) / length


def window_linmean(s: Split, t: np.ndarray, length: int) -> np.ndarray:
    return 10.0 * np.log10((s.lincum[t + 1] - s.lincum[t + 1 - length]) / length)


def windows(s: Split, t: np.ndarray, length: int) -> np.ndarray:
    """(n, F, length) window values, oldest first."""
    view = np.lib.stride_tricks.sliding_window_view(s.values, length, axis=0)   # (T-length+1, F, length)
    return view[t - length + 1]


def window_median(s: Split, t: np.ndarray, length: int, chunk: int = 1000) -> np.ndarray:
    out = np.empty((len(t), s.values.shape[1]))
    for i in range(0, len(t), chunk):
        out[i:i + chunk] = np.median(windows(s, t[i:i + chunk], length), axis=-1)
    return out


def ewma_series(values: np.ndarray, lam: np.ndarray | float) -> np.ndarray:
    """Causal EWMA level per bin; lam scalar or per-bin (F,). NaN rows carry the previous level."""
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), (values.shape[1],))
    level = np.empty_like(values)
    current = values[0].copy()
    for i, row in enumerate(values):
        ok = np.isfinite(row)
        current = np.where(ok & np.isfinite(current), current + lam * (row - current), np.where(ok, row, current))
        level[i] = current
    return level


def kalman_gain(q: np.ndarray) -> np.ndarray:
    """Steady-state gain of the local-level filter for signal-to-noise ratio q = sigma_eta^2 / sigma_eps^2."""
    q = np.maximum(q, 1e-12)
    p = (q + np.sqrt(q * q + 4.0 * q)) / 2.0
    return p / (p + 1.0)


GAIN_GRID = tuple(np.round(np.geomspace(0.002, 1.0, 30), 5))


def fit_local_level(values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bin steady-state local-level fit by minimizing the one-step prediction error of the equivalent EWMA
    (the concentrated steady-state likelihood). Returns gain K, q, sigma_eps^2, sigma_eta^2, innovation variance."""
    best_err = np.full(values.shape[1], np.inf)
    best_gain = np.full(values.shape[1], np.nan)
    for gain in GAIN_GRID:
        level = ewma_series(values, gain)
        err = np.nanmean((values[1:] - level[:-1]) ** 2, axis=0)
        better = err < best_err
        best_err[better], best_gain[better] = err[better], gain
    p = best_gain / np.maximum(1.0 - best_gain, 1e-9)
    eps = best_err / (p + 1.0)
    q = p * p / (p + 1.0)
    return {"gain": best_gain, "q": q, "sigma_eps2": eps, "sigma_eta2": q * eps, "innovation_var": best_err}


def diff_moments(values: np.ndarray) -> dict[str, np.ndarray]:
    d = np.diff(values, axis=0)
    d = d - np.nanmean(d, axis=0)

    def acov(lag: int) -> np.ndarray:
        return np.nanmean(d[lag:] * d[:len(d) - lag], axis=0) if lag else np.nanmean(d * d, axis=0)

    g0, g1, g2, g3 = (acov(k) for k in range(4))
    eps = np.maximum(-g1, 0.0)
    eta = np.maximum(g0 + 2.0 * g1, 0.0)
    return {"gamma0": g0, "gamma1": g1, "rho1": g1 / g0, "rho2": g2 / g0, "rho3": g3 / g0, "sigma_eps2": eps,
            "sigma_eta2": eta, "q": eta / np.maximum(eps, 1e-12)}


# ------------------------------------------------------------------------------------------- metrics
def per_origin_abs(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Bin-averaged absolute error per origin (n,)."""
    return np.nanmean(np.abs(pred - target), axis=1)


def block_bootstrap_skill(err: np.ndarray, ref: np.ndarray, rng: np.random.Generator,
                          block: int = BOOT_BLOCK, reps: int = BOOT_REPS) -> tuple[float, float, float]:
    """Skill 1 - sum(err)/sum(ref) with a moving-block bootstrap CI over the (time-ordered) origin sequence."""
    n = len(err)
    skill = 1.0 - err.sum() / ref.sum()
    if n < 2 * block:
        block = max(1, n // 10)
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    boot = 1.0 - err[idx].sum(axis=1) / ref[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(skill), float(lo), float(hi)


# ------------------------------------------------------------------------------------------- ridge
def ridge_fit(xtx: np.ndarray, xty: np.ndarray, rel_alpha: float) -> np.ndarray:
    """Ridge with an unpenalized intercept (last column of the design is the constant)."""
    k = xtx.shape[0]
    penalty = np.eye(k) * rel_alpha * float(np.mean(np.diag(xtx)[:-1]))
    penalty[-1, -1] = 0.0
    return np.linalg.lstsq(xtx + penalty, xty, rcond=None)[0]


def design(win: np.ndarray, anchor: np.ndarray | None) -> np.ndarray:
    """(n, L+1) design for one bin: centred window plus constant."""
    x = win - anchor[:, None] if anchor is not None else win
    return np.hstack([x, np.ones((len(x), 1))])


def anchor_values(kind: str, s: Split, t: np.ndarray) -> np.ndarray | None:
    if kind == "mean60":
        return window_mean(s, t, LOOKBACK)
    if kind == "last":
        return s.values[t]
    return None


def per_bin_probe(splits: dict[str, Split], origins: dict[str, np.ndarray], horizon: int, kind: str,
                  target_fn=None) -> dict[str, Any]:
    """Per-bin ridge correction for one anchor. Returns per-split predictions for the anchor alone, the full
    correction (validation-chosen ridge strength) and the validation-scaled correction, plus diagnostics."""
    target_fn = target_fn or (lambda s, t: s.values[t + horizon])
    data = {}
    for name, s in splits.items():
        t = origins[name]
        a = anchor_values(kind, s, t)
        data[name] = (windows(s, t, LOOKBACK), a, target_fn(s, t))
    nbins = data["train"][0].shape[1]
    preds = {name: {r: np.empty((len(origins[name]), nbins)) for r in RIDGE_GRID} for name in splits}
    for b in range(nbins):
        xs, offs = {}, {}
        for name in splits:
            wn, an, _ = data[name]
            offs[name] = an[:, b] if an is not None else 0.0
            xs[name] = design(wn[:, b, :], an[:, b] if an is not None else None)
        r = data["train"][2][:, b] - offs["train"]
        xtx, xty = xs["train"].T @ xs["train"], xs["train"].T @ r
        for rel in RIDGE_GRID:
            beta = ridge_fit(xtx, xty, rel)
            for name in splits:
                preds[name][rel][:, b] = xs[name] @ beta + offs[name]
    val_target = data["validation"][2]
    best_rel = min(RIDGE_GRID, key=lambda r: np.nanmean(np.abs(preds["validation"][r] - val_target)))
    anchor_pred = {name: (data[name][1] if data[name][1] is not None else None) for name in splits}
    out: dict[str, Any] = {"best_ridge": best_rel, "anchor": anchor_pred, "full": {}, "scaled": {}, "target": {},
                           "unreg": {n: preds[n][0.0] for n in splits}}
    for name in splits:
        out["full"][name] = preds[name][best_rel]
        out["target"][name] = data[name][2]
    if kind != "none":
        corr_val = preds["validation"][best_rel] - data["validation"][1]
        best_s = min(SHRINK_GRID, key=lambda sc: np.nanmean(np.abs(data["validation"][1] + sc * corr_val
                                                                   - val_target)))
        out["best_shrink"] = best_s
        for name in splits:
            out["scaled"][name] = data[name][1] + best_s * (preds[name][best_rel] - data[name][1])
    return out


def pooled_features(s: Split, t: np.ndarray, rich: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pooled-over-bins design (n*F, k): centred 60-minute window (mean60 anchor), plus optional covariates."""
    win = windows(s, t, LOOKBACK)                       # (n, F, L)
    anchor = win.mean(axis=-1)
    cent = win - anchor[..., None]
    feats = [cent]
    if rich:
        std = cent.std(axis=-1, keepdims=True)
        lags = np.arange(LOOKBACK) - (LOOKBACK - 1) / 2.0
        slope = (cent * lags).sum(axis=-1, keepdims=True) / (lags ** 2).sum()
        minute = (s.index[t].hour.to_numpy() * 60 + s.index[t].minute.to_numpy()) / 1440.0
        tod = np.stack([np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute)], axis=-1)   # (n, 2)
        tod = np.broadcast_to(tod[:, None, :], (len(t), win.shape[1], 2))
        feats += [std, slope, tod]
    x = np.concatenate(feats, axis=-1).reshape(-1, sum(f.shape[-1] for f in feats))
    return x, anchor


def gbm_features(s: Split, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compact pooled features for the nonlinear probe: last 15 centred lags, means of lags 16-30 and 31-60,
    window std, slope, time of day."""
    x, anchor = pooled_features(s, t, rich=True)
    cent = x[:, :LOOKBACK]
    compact = np.hstack([cent[:, -15:], cent[:, 30:45].mean(axis=1, keepdims=True),
                         cent[:, :30].mean(axis=1, keepdims=True), x[:, LOOKBACK:]])
    return compact, anchor


# ------------------------------------------------------------------------------------------- cell job
def cell_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    import sys

    sys.path.insert(0, args["repo_root"])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    started = time.time()
    node, band = job["node"], job["band"]
    rows: list[dict[str, Any]] = []
    tables: dict[str, list[dict[str, Any]]] = {k: [] for k in
                                                ("statespace", "condmedian", "cheap", "daily", "equivalence")}
    rng = np.random.default_rng(12345)
    meta = {"node": node, "band": band}
    try:
        splits = load_cell(args["repo_root"], node, band)
        train = splits["train"]

        # ---------------------------------------------------------------- state-space moments
        moments = {name: diff_moments(s.values[max(s.lo - 1, 0):s.hi]) for name, s in splits.items()}
        for name, m in moments.items():
            row = {**meta, "split": name}
            for key, v in m.items():
                row[f"{key}_median"] = float(np.nanmedian(v))
                row[f"{key}_p25"] = float(np.nanpercentile(v, 25))
                row[f"{key}_p75"] = float(np.nanpercentile(v, 75))
            row["share_rho1_negative"] = float(np.mean(m["rho1"] < 0))
            tables["statespace"].append(row)
        fit = fit_local_level(train.values[train.lo:train.hi])
        gain = fit["gain"]
        pred_var_base = gain * fit["sigma_eps2"] + fit["sigma_eps2"]
        row = {**meta, "split": "train", "fit": "prediction_error"}
        for key, v in fit.items():
            row[f"{key}_median"] = float(np.nanmedian(v))
            row[f"{key}_p25"] = float(np.nanpercentile(v, 25))
            row[f"{key}_p75"] = float(np.nanpercentile(v, 75))
        tables["statespace"].append(row)
        moment_gain = kalman_gain(moments["train"]["q"])

        # ---------------------------------------------------------------- baselines
        levels = {name: {lam: ewma_series(s.values, lam) for lam in EWMA_GRID} for name, s in splits.items()}
        kf_levels = {name: ewma_series(s.values, gain) for name, s in splits.items()}
        kf_moment_levels = {name: ewma_series(s.values, moment_gain) for name, s in splits.items()}
        base_preds: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        origins_by_h: dict[int, dict[str, np.ndarray]] = {}
        for h in HORIZONS:
            origins = {name: s.origins(h) for name, s in splits.items()}
            origins_by_h[h] = origins
            preds: dict[str, dict[str, np.ndarray]] = {}
            for name, s in splits.items():
                t = origins[name]
                p = {"train_mean": np.broadcast_to(np.nanmean(train.values[train.lo:train.hi], axis=0),
                                                   (len(t), s.values.shape[1])),
                     "train_median": np.broadcast_to(np.nanmedian(train.values[train.lo:train.hi], axis=0),
                                                     (len(t), s.values.shape[1])),
                     "persistence": s.values[t],
                     "kalman_local_level": kf_levels[name][t],
                     "kalman_moments": kf_moment_levels[name][t]}
                for length in L_GRID:
                    p[f"mean{length}"] = window_mean(s, t, length)
                    p[f"median{length}"] = window_median(s, t, length)
                    p[f"linmean{length}"] = window_linmean(s, t, length)
                for lam in EWMA_GRID:
                    p[f"ewma{lam}"] = levels[name][lam][t]
                p["_target"] = s.values[t + h]
                preds[name] = p
            selectable = [k for k in preds["validation"] if k.startswith(("mean", "median", "linmean", "ewma"))]
            strong = min(selectable, key=lambda k: np.nanmean(np.abs(preds["validation"][k]
                                                                     - preds["validation"]["_target"])))
            for name in splits:
                preds[name]["strong"] = preds[name][strong]
            base_preds[h] = preds
            ref_test = per_origin_abs(preds["test"]["mean60"], preds["test"]["_target"])
            for name in splits:
                target = preds[name]["_target"]
                for k, v in preds[name].items():
                    if k == "_target":
                        continue
                    e = per_origin_abs(v, target)
                    row = {**meta, "experiment": "baselines", "horizon": h, "predictor": k, "split": name,
                           "mae": float(e.mean()), "n_origins": len(e),
                           "selected": strong if k == "strong" else ""}
                    if name == "test":
                        row["skill_vs_mean60"], row["ci_lo"], row["ci_hi"] = block_bootstrap_skill(e, ref_test, rng)
                    rows.append(row)
            # forecast-variance calibration of the local-level model on test
            t = origins["test"]
            err2 = (kf_levels["test"][t] - splits["test"].values[t + h]) ** 2
            pred_var = pred_var_base + h * fit["sigma_eta2"]
            tables["statespace"].append({**meta, "split": "test", "horizon": h,
                                         "calibration_mse_over_predvar_median": float(
                                             np.nanmedian(np.nanmean(err2, axis=0) / pred_var))})

        # seasonal naive on its own origin subset (needs the same minute one day earlier)
        for h in HORIZONS:
            for name, s in splits.items():
                t = s.origins(h, extra_lag=SEASON)
                if len(t) == 0:
                    continue
                target = s.values[t + h]
                seasonal = s.values[t + h - SEASON]
                for k, v in (("seasonal_naive_24h", seasonal), ("mean60_same_origins", window_mean(s, t, 60))):
                    rows.append({**meta, "experiment": "seasonal", "horizon": h, "predictor": k, "split": name,
                                 "mae": float(per_origin_abs(v, target).mean()), "n_origins": len(t)})

        # ---------------------------------------------------------------- ridge probes
        best_probe_test: dict[int, np.ndarray] = {}
        for h in HORIZONS:
            origins = origins_by_h[h]
            ref_test = per_origin_abs(base_preds[h]["test"]["mean60"], base_preds[h]["test"]["_target"])
            strong_test = per_origin_abs(base_preds[h]["test"]["strong"], base_preds[h]["test"]["_target"])
            results = {}
            for kind in ANCHORS:
                res = per_bin_probe(splits, origins, h, kind)
                results[kind] = res
                for name in splits:
                    target = res["target"][name]
                    variants = {"full": res["full"][name]}
                    if kind != "none":
                        variants["anchor_only"] = res["anchor"][name]
                        variants["val_scaled"] = res["scaled"][name]
                    for variant, v in variants.items():
                        e = per_origin_abs(v, target)
                        row = {**meta, "experiment": "probe_per_bin", "horizon": h, "predictor": f"{kind}:{variant}",
                               "split": name, "mae": float(e.mean()), "n_origins": len(e),
                               "selected": json.dumps({"ridge": res["best_ridge"],
                                                       "shrink": res.get("best_shrink")})}
                        if name == "test":
                            row["skill_vs_mean60"], row["ci_lo"], row["ci_hi"] = block_bootstrap_skill(e, ref_test, rng)
                            sk, lo, hi = block_bootstrap_skill(e, strong_test, rng)
                            row.update(skill_vs_strong=sk, strong_ci_lo=lo, strong_ci_hi=hi)
                        rows.append(row)
            # identical affine class without regularization: the three anchors should agree
            for a, b in (("mean60", "last"), ("mean60", "none")):
                diff = np.nanmax(np.abs(results[a]["unreg"]["test"] - results[b]["unreg"]["test"]))
                tables["equivalence"].append({**meta, "horizon": h, "pair": f"{a}-{b}", "max_abs_diff_db": float(diff)})
            # cheap-correction test on the last-value anchor
            t = origins["test"]
            last = splits["test"].values[t]
            c_theta = (results["last"]["full"]["test"] - last).ravel()
            c_mean = (window_mean(splits["test"], t, LOOKBACK) - last).ravel()
            ok = np.isfinite(c_theta) & np.isfinite(c_mean)
            slope = float(np.dot(c_theta[ok], c_mean[ok]) / np.dot(c_mean[ok], c_mean[ok]))
            resid = c_theta[ok] - slope * c_mean[ok]
            r2 = float(1.0 - resid.var() / c_theta[ok].var())
            tables["cheap"].append({**meta, "horizon": h, "slope_on_mean_pull": slope, "r2_mean_pull": r2,
                                    "corr_std_db": float(c_theta[ok].std()), "mean_pull_std_db": float(c_mean[ok].std())})
            # conditional median of the mean60 residual by decile of last - mean60
            for name in ("train", "test"):
                s, tt = splits[name], origins[name]
                m = window_mean(s, tt, LOOKBACK)
                dev = (s.values[tt] - m).ravel()
                res_ = (s.values[tt + h] - m).ravel()
                if name == "train":
                    edges = np.nanpercentile(dev, np.arange(0, 101, 10))
                bins_ = np.clip(np.searchsorted(edges, dev, side="right") - 1, 0, 9)
                for dcl in range(10):
                    sel = (bins_ == dcl) & np.isfinite(res_)
                    tables["condmedian"].append({**meta, "split": name, "horizon": h, "decile": dcl,
                                                 "dev_median": float(np.median(dev[sel])),
                                                 "residual_median": float(np.median(res_[sel])),
                                                 "n": int(sel.sum())})
            best_kind = min(("mean60", "last"), key=lambda k: np.nanmean(
                np.abs(results[k]["scaled"]["validation"] - results[k]["target"]["validation"])))
            best_probe_test[h] = results[best_kind]["scaled"]["test"]

            # pooled probes: linear (lags only / rich) and gradient boosting with absolute-error loss
            for rich in (False, True):
                x, a = {}, {}
                for name, s in splits.items():
                    x[name], a[name] = pooled_features(s, origins[name], rich)
                y = {name: (s.values[origins[name] + h] - a[name]).ravel() for name, s in splits.items()}
                xt = np.hstack([x["train"], np.ones((len(x["train"]), 1))])
                ok = np.isfinite(y["train"]) & np.isfinite(xt).all(axis=1)
                xtx, xty = xt[ok].T @ xt[ok], xt[ok].T @ y["train"][ok]
                fits = {rel: ridge_fit(xtx, xty, rel) for rel in RIDGE_GRID}

                def pooled_pred(name, beta):
                    xx = np.hstack([x[name], np.ones((len(x[name]), 1))])
                    return (xx @ beta).reshape(a[name].shape)

                best = min(RIDGE_GRID, key=lambda r: np.nanmean(np.abs(pooled_pred("validation", fits[r])
                                                                         - y["validation"].reshape(a["validation"].shape))))
                corr_val = pooled_pred("validation", fits[best])
                yv = y["validation"].reshape(a["validation"].shape)
                shrink = min(SHRINK_GRID, key=lambda sc: np.nanmean(np.abs(sc * corr_val - yv)))
                label = "pooled_rich_linear" if rich else "pooled_linear"
                for name in splits:
                    target = splits[name].values[origins[name] + h]
                    e = per_origin_abs(a[name] + shrink * pooled_pred(name, fits[best]), target)
                    row = {**meta, "experiment": "probe_pooled", "horizon": h, "predictor": f"{label}:val_scaled",
                           "split": name, "mae": float(e.mean()), "n_origins": len(e),
                           "selected": json.dumps({"ridge": best, "shrink": shrink})}
                    if name == "test":
                        row["skill_vs_mean60"], row["ci_lo"], row["ci_hi"] = block_bootstrap_skill(e, ref_test, rng)
                    rows.append(row)
            if args.get("gbm", True):
                from sklearn.ensemble import HistGradientBoostingRegressor

                x, a = {}, {}
                for name, s in splits.items():
                    x[name], a[name] = gbm_features(s, origins[name])
                y = (splits["train"].values[origins["train"] + h] - a["train"]).ravel()
                ok = np.flatnonzero(np.isfinite(y) & np.isfinite(x["train"]).all(axis=1))
                pick = rng.choice(ok, size=min(GBM_TRAIN_ROWS, len(ok)), replace=False)
                gbm = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                    max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
                                                    random_state=0)
                gbm.fit(x["train"][pick], y[pick])
                corr = {name: np.nan_to_num(gbm.predict(np.nan_to_num(x[name]))).reshape(a[name].shape)
                        for name in splits}
                yv = splits["validation"].values[origins["validation"] + h] - a["validation"]
                shrink = min(SHRINK_GRID, key=lambda sc: np.nanmean(np.abs(sc * corr["validation"] - yv)))
                for name in splits:
                    target = splits[name].values[origins[name] + h]
                    for variant, sc in (("full", 1.0), ("val_scaled", shrink)):
                        e = per_origin_abs(a[name] + sc * corr[name], target)
                        row = {**meta, "experiment": "probe_pooled", "horizon": h,
                               "predictor": f"pooled_gbm_mae:{variant}", "split": name, "mae": float(e.mean()),
                               "n_origins": len(e), "selected": json.dumps({"shrink": shrink,
                                                                            "iters": int(gbm.n_iter_)})}
                        if name == "test":
                            row["skill_vs_mean60"], row["ci_lo"], row["ci_hi"] = block_bootstrap_skill(e, ref_test, rng)
                        rows.append(row)

        # ---------------------------------------------------------------- per-day stability on test
        for h in HORIZONS:
            t = origins_by_h[h]["test"]
            target = base_preds[h]["test"]["_target"]
            day = pd.Series(splits["test"].index[t + h].floor("D"))
            ref = per_origin_abs(base_preds[h]["test"]["mean60"], target)
            for label, pred in (("strong", base_preds[h]["test"]["strong"]), ("best_probe", best_probe_test[h]),
                                ("persistence", base_preds[h]["test"]["persistence"])):
                e = per_origin_abs(pred, target)
                for d in day.unique():
                    sel = (day == d).to_numpy()
                    tables["daily"].append({**meta, "horizon": h, "predictor": label, "day": str(d.date()),
                                            "skill_vs_mean60": float(1 - e[sel].sum() / ref[sel].sum()),
                                            "n_origins": int(sel.sum())})

        # ---------------------------------------------------------------- aggregation arms
        rows += aggregation_rows(splits, meta, rng)
        status = {"ok": True}
    except Exception as error:  # report, never crash the driver
        import traceback

        status = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300], "trace": traceback.format_exc()[-1500:]}
    return {"meta": {**meta, **status, "wall_seconds": round(time.time() - started, 1)}, "rows": rows,
            "tables": tables}


def avg_target(s: Split, t: np.ndarray, horizon: int, width: int = BLOCK) -> np.ndarray:
    """Mean over (t+H-W, t+H] in linear power, in dB."""
    return 10.0 * np.log10((s.lincum[t + horizon + 1] - s.lincum[t + horizon + 1 - width]) / width)


def block_history(s: Split, t: np.ndarray, nblocks: int, phase: int | None = None) -> np.ndarray:
    """(n, F, nblocks) history oldest first: 15-minute linear-power block means ending at t (phase None), or the
    decimated minutes t - phase - 15k (phase given)."""
    if phase is None:
        ends = t[:, None] - BLOCK * np.arange(nblocks)[::-1][None, :]                   # (n, K) block end rows
        lin = (s.lincum[ends + 1] - s.lincum[ends + 1 - BLOCK]) / BLOCK                 # (n, K, F)
        return np.transpose(10.0 * np.log10(lin), (0, 2, 1))
    rows_ = t[:, None] - phase - BLOCK * np.arange(nblocks)[::-1][None, :]
    return np.transpose(s.values[rows_], (0, 2, 1))


def ridge_on_history(hist: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-bin ridge correction around the history mean; ridge and shrink chosen on validation. Returns test/val preds."""
    nbins = hist["train"].shape[1]
    anchor = {n: h.mean(axis=-1) for n, h in hist.items()}
    preds = {n: {r: np.empty_like(anchor[n]) for r in RIDGE_GRID} for n in hist}
    for b in range(nbins):
        x = design(hist["train"][:, b, :], anchor["train"][:, b])
        y = target["train"][:, b] - anchor["train"][:, b]
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        xtx, xty = x[ok].T @ x[ok], x[ok].T @ y[ok]
        for rel in RIDGE_GRID:
            beta = ridge_fit(xtx, xty, rel)
            for n in hist:
                preds[n][rel][:, b] = design(hist[n][:, b, :], anchor[n][:, b]) @ beta
    best = min(RIDGE_GRID, key=lambda r: np.nanmean(np.abs(anchor["validation"] + preds["validation"][r]
                                                           - target["validation"])))
    shrink = min(SHRINK_GRID, key=lambda sc: np.nanmean(np.abs(anchor["validation"] + sc * preds["validation"][best]
                                                               - target["validation"])))
    return {n: anchor[n] + shrink * preds[n][best] for n in hist} | {"_anchor": anchor, "_sel": (best, shrink)}


def aggregation_rows(splits: dict[str, Split], meta: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in AGG_HORIZONS:
        aligned = {n: s.origins(h, history=MAX_HISTORY, aligned=True, width=BLOCK) for n, s in splits.items()}
        rolling = {n: s.origins(h, history=MAX_HISTORY, width=BLOCK) for n, s in splits.items()}
        tgt_avg = {n: avg_target(s, aligned[n], h) for n, s in splits.items()}
        tgt_point = {n: s.values[aligned[n] + h] for n, s in splits.items()}
        test_ref = {}

        def emit(arm, target_kind, predictor, preds, targets, lookback, selected=""):
            for n in splits:
                e = per_origin_abs(preds[n], targets[n])
                row = {**meta, "experiment": "aggregation", "arm": arm, "target": target_kind, "horizon": h,
                       "lookback_min": lookback, "predictor": predictor, "split": n, "mae": float(e.mean()),
                       "n_origins": len(e), "selected": selected}
                if n == "test":
                    ref = test_ref[(target_kind, lookback)]
                    row["skill_vs_mean_dB"], row["ci_lo"], row["ci_hi"] = block_bootstrap_skill(e, ref, rng)
                rows.append(row)

        for lookback in AGG_LOOKBACKS:
            k = lookback // BLOCK
            # reference for each target kind: the dB lookback mean of the 1-minute history, same origins
            mean_db = {n: window_mean(s, aligned[n], lookback) for n, s in splits.items()}
            for kind, tg in (("point", tgt_point), ("avg15", tgt_avg)):
                test_ref[(kind, lookback)] = per_origin_abs(mean_db["test"], tg["test"])
            # arm A: 1-minute history, point target (the original task, at aligned origins)
            emit("A_1min_to_point", "point", "mean_dB", mean_db, tgt_point, lookback)
            emit("A_1min_to_point", "point", "persistence", {n: s.values[aligned[n]] for n, s in splits.items()},
                 tgt_point, lookback)
            # arm B: 1-minute history, 15-minute-average target
            emit("B_1min_to_avg15", "avg15", "mean_dB", mean_db, tgt_avg, lookback)
            emit("B_1min_to_avg15", "avg15", "linmean", {n: window_linmean(s, aligned[n], lookback)
                                                         for n, s in splits.items()}, tgt_avg, lookback)
            emit("B_1min_to_avg15", "avg15", "last15_linmean", {n: window_linmean(s, aligned[n], BLOCK)
                                                                for n, s in splits.items()}, tgt_avg, lookback)
            if lookback == LOOKBACK:
                for origin_kind, org in (("aligned", aligned), ("rolling", rolling)):
                    train_org = {**aligned, "train": org["train"]}
                    hist = {n: windows(s, train_org[n], lookback) for n, s in splits.items()}
                    targ = {n: avg_target(s, train_org[n], h) for n, s in splits.items()}
                    out = ridge_on_history(hist, targ)
                    emit("B_1min_to_avg15", "avg15", f"ridge_1min_train_{origin_kind}", out, targ, lookback,
                         json.dumps(out["_sel"]))
                    if origin_kind == "aligned":
                        targ_p = {n: s.values[train_org[n] + h] for n, s in splits.items()}
                        out_p = ridge_on_history(hist, targ_p)
                        emit("A_1min_to_point", "point", "ridge_1min_train_aligned", out_p, targ_p, lookback,
                             json.dumps(out_p["_sel"]))
            # arm C: 15-minute block history (aligned blocks), same avg target; training aligned or rolling
            for origin_kind, org in (("aligned", aligned), ("rolling", rolling)):
                train_org = {**aligned, "train": org["train"]}
                hist = {n: block_history(s, train_org[n], k) for n, s in splits.items()}
                targ = {n: avg_target(s, train_org[n], h) for n, s in splits.items()}
                out = ridge_on_history(hist, targ)
                if origin_kind == "aligned":
                    emit("C_15min_to_avg15", "avg15", "block_mean_dB", out["_anchor"], tgt_avg, lookback)
                    emit("C_15min_to_avg15", "avg15", "last_block",
                         {n: hist[n][:, :, -1] for n in splits}, tgt_avg, lookback)
                emit("C_15min_to_avg15", "avg15", f"ridge_blocks_train_{origin_kind}", out, targ, lookback,
                     json.dumps(out["_sel"]))
            # arm D: decimated history (every 15th minute), all phases; summarized as mean and sd over phases
            per_phase: dict[str, list[float]] = {"dec_mean": [], "dec_last": [], "dec_ridge": []}
            for phase in range(BLOCK):
                hist = {n: block_history(s, aligned[n], k, phase=phase) for n, s in splits.items()}
                out = ridge_on_history(hist, tgt_avg)
                e_ref = test_ref[("avg15", lookback)]
                for key, pred in (("dec_mean", out["_anchor"]["test"]), ("dec_last", hist["test"][:, :, -1]),
                                  ("dec_ridge", out["test"])):
                    e = per_origin_abs(pred, tgt_avg["test"])
                    per_phase[key].append(1.0 - e.sum() / e_ref.sum())
            for key, vals in per_phase.items():
                rows.append({**meta, "experiment": "aggregation", "arm": "D_decimated_to_avg15", "target": "avg15",
                             "horizon": h, "lookback_min": lookback, "predictor": key, "split": "test",
                             "skill_vs_mean_dB": float(np.mean(vals)), "skill_phase_sd": float(np.std(vals)),
                             "skill_phase_min": float(np.min(vals)), "skill_phase_max": float(np.max(vals))})
    return rows


# ------------------------------------------------------------------------------------------------ driver
def launch(args: argparse.Namespace) -> None:
    import ray

    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "gbm": not args.no_gbm}
    remote = ray.remote(num_cpus=args.cpus_per_task, num_gpus=0, max_retries=0)(cell_job)
    jobs = [{"node": n, "band": b} for n in args.nodes for b in args.bands]
    print(f"[level-diag] {len(jobs)} cells, {args.cpus_per_task} CPUs each, no GPUs", flush=True)
    results = ray.get([remote.remote(job, worker_args) for job in jobs])
    out = Path(args.repo_root) / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r["meta"] for r in results]).to_csv(out / "cells.csv", index=False)
    pd.DataFrame([row for r in results for row in r["rows"]]).to_csv(out / "results.csv", index=False)
    for name in ("statespace", "condmedian", "cheap", "daily", "equivalence"):
        pd.DataFrame([row for r in results for row in r["tables"][name]]).to_csv(out / f"{name}.csv", index=False)
    (out / "run.json").write_text(json.dumps({"args": vars(args), "epsilon": EPSILON, "ranges": RANGES,
                                              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                             default=str, indent=1))
    failed = [r["meta"] for r in results if not r["meta"]["ok"]]
    print(f"[level-diag] wrote {out}; {len(results)} cells, {len(failed)} failed", flush=True)
    for f in failed:
        print(f"[level-diag] FAILED {f['node']}/{f['band']}: {f['error']}\n{f.get('trace', '')}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", nargs="+", default=list(NODES))
    p.add_argument("--bands", nargs="+", default=list(BANDS))
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    p.add_argument("--output-dir", required=True, help="relative to --repo-root, e.g. runs/claude/level-diagnostics/x")
    p.add_argument("--address", default="auto")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--no-gbm", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    launch(parse_args())
