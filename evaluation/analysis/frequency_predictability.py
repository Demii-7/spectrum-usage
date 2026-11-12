#!/usr/bin/env python3
"""Is there usable linear information about a bin's future in the OTHER bins at all? (model-free complement to the frozen-input ablation; CPU only)

The frequency counterpart of ``neighbor_predictability.py``. Per band, receiver, horizon h (t+4/12/24/48, 15-minute steps) and target bin b (every bin of a
non-noise plan region):

* Reference: the tuned lookback median of the receiver and bin ending at the origin t (window K = ``TUNED_K["15min|<band>|median"][h]``).
  Target: y = x_b(t + h) - median_b(t), the signed error of the tuned median forecast.
* Own history, two versions: ``short`` = deviations x_b(t - l) - median_b(t) for l = 0..7 (last 2 h); ``day`` = short plus l = 92..96 (the values around 24 h back).
* Context (low dimensional because the training week has only a few hundred origins): for a set S of OTHER bins, the deviations x_s(t - l) - median_s(t),
  l = 0..3, projected on the first 3 principal components (fit on the training week, lag-0 deviations of S). Sets: ``P`` = the P-class bins outside the target's
  region, ``N`` = the N-class bins outside it, ``all`` = every bin outside it, ``region`` = the other bins of the target's own region (a variant; the main sets exclude
  the target's own region). Classes: bin_periodicity_classes.csv.
* Models, one ridge regression per target bin (own only versus own + context), fit on the TRAINING run (20261015T1434Z), penalty chosen on the validation range (the first
  192 blocks of run 20261025T1355Z), scored on the TEST week 2026-10-27T20:00Z..2026-11-03T20:00Z.

Output: ``frequency_predictability_by_bin.csv`` (one row per band, receiver, bin, horizon, own version, context set: test MAE of own only and of own + context, R^2, MAE of the
median forecast, chosen penalties) and ``frequency_predictability.csv`` (by target class P / N, context set, own version, horizon; per receiver and pooled over receivers;
mean gain = own - (own + context), dB, > 0: the context helps; moving-block bootstrap (block 96, 1,000 draws) over test origins; R^2 gain).
Linear and low rank: a LOWER BOUND on the information that is there.

    python evaluation/analysis/frequency_predictability.py [--root /workspace/spectrum-usage] [--out-dir evaluation/results] [--bands 600_800] [--max-regions N]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent

CORE = ("humanities", "madsen", "moran", "sagepoint")
NODE = {name: f"{name}-nuc1" for name in CORE}
BANDS = ("600_800", "2400_2600")
HORIZONS = (4, 12, 24, 48)
STEP = pd.Timedelta(minutes=15)
TRAIN_RUN, TEST_RUN = "20261015T1434Z", "20261025T1355Z"
TEST_RANGE = (pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z"))
N_VALID_BLOCKS = 192
OWN_LAGS = {"short": list(range(0, 8)), "day": list(range(0, 8)) + list(range(92, 97))}
CTX_LAGS = 4
N_PC = 3
CTX_SETS = ("P", "N", "all", "region")
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
BLOCK, DRAWS = 96, 1000
MAX_LAG = 96


def load_aligned(root: Path, band: str, run: str):
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, ensure_powder15

    ensure_powder15(band, str(root))
    frames = []
    for receiver in CORE:
        frame = pd.read_csv(root / OUT_ROOT / NODE[receiver] / run / band / OUT_NAME)
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp_utc"), utc=True))
        frame.index = index
        frames.append(frame)
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    return common, np.stack([f.loc[common].to_numpy(dtype=np.float64) for f in frames], axis=1)       # (T, 4, bins)


def origins(stamps: pd.DatetimeIndex, k: int, h: int, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Rows t with target t + h inside [start, end], the K-row median window and the 96 lags inside one gap-free stretch."""
    n = len(stamps)
    gap = np.concatenate([[0], np.cumsum(np.diff(stamps.asi8) != STEP.value)])
    rows = np.arange(max(k - 1, MAX_LAG), n - h)
    target_time = stamps[rows + h]
    keep = (target_time >= start) & (target_time <= end) & (gap[rows - max(k - 1, MAX_LAG)] == gap[rows + h])
    return rows[np.asarray(keep)]


def deviations(x: np.ndarray, rows: np.ndarray, k: int, h: int):
    """Median reference (N, bins), target y (N, bins) and a function lag -> (N, bins) deviations of one receiver's series x (T, bins)."""
    windows = sliding_window_view(x, k, axis=0)                  # (T-k+1, bins, k)
    med = np.median(windows[rows - (k - 1)], axis=-1)
    y = x[rows + h] - med
    return med, y, (lambda lag: x[rows - lag] - med)


def ridge_fit(a: np.ndarray, y: np.ndarray, lam: float):
    """Ridge with intercept (features centred, penalty on the slopes): returns (coef, mean_a, mean_y)."""
    ma, my = a.mean(axis=0), y.mean()
    ac = a - ma
    coef = np.linalg.solve(ac.T @ ac + lam * np.eye(a.shape[1]), ac.T @ (y - my))
    return coef, ma, my


def choose_and_score(train_a, train_y, valid_a, valid_y, test_a, test_y):
    """Fit on train for every penalty, pick by validation MAE, return the test residuals and the chosen penalty."""
    best, best_mae = None, np.inf
    for lam in LAMBDAS:
        coef, ma, my = ridge_fit(train_a, train_y, lam)
        mae = np.abs(valid_y - ((valid_a - ma) @ coef + my)).mean()
        if mae < best_mae:
            best, best_mae = (coef, ma, my, lam), mae
    coef, ma, my, lam = best
    return test_y - ((test_a - ma) @ coef + my), lam


def pca_basis(train_vectors: np.ndarray):
    """Mean and first N_PC principal directions of (N, n) training vectors."""
    mean = train_vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(train_vectors - mean, full_matrices=False)
    return mean, vt[:min(N_PC, vt.shape[0])].T                  # (n, <=3)


def boot(per_origin: np.ndarray, rng: np.random.Generator):
    n = len(per_origin)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    starts = rng.integers(0, n - block + 1, size=(DRAWS, int(np.ceil(n / block))))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    draws = per_origin[index].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="evaluation/results")
    ap.add_argument("--bands", nargs="+", default=list(BANDS))
    ap.add_argument("--max-regions", type=int, default=0, help="smoke tests only")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    root = Path(a.root)
    from training.ray.origin_errors import TUNED_K

    classes_all = pd.read_csv(HERE.parent / "results" / "bin_periodicity_classes.csv") if (HERE.parent / "results" / "bin_periodicity_classes.csv").exists() else None
    if classes_all is None:
        from training.ray.storage import spectrum_s3_client

        classes_all = pd.read_csv(io.BytesIO(spectrum_s3_client().get_object(Bucket="spectrum", Key="analysis-inputs/bin_periodicity_classes.csv")["Body"].read()))
    rng = np.random.default_rng(0)
    by_bin_rows, per_origin_store = [], {}
    for band in a.bands:
        plan = pd.read_csv(HERE / f"plan_regions_{band}.csv")
        plan = plan[~plan["is_noise_floor"].astype(str).str.lower().isin({"true", "1"})]
        if a.max_regions:
            plan = plan.head(a.max_regions)
        freqs = 600.5 + np.arange(200) if band == "600_800" else 2400.5 + np.arange(200)
        stamps_tr, x_tr = load_aligned(root, band, TRAIN_RUN)
        stamps_te, x_te = load_aligned(root, band, TEST_RUN)
        valid_end = stamps_te[min(N_VALID_BLOCKS, len(stamps_te)) - 1]
        for ri, receiver in enumerate(CORE):
            cls = classes_all[(classes_all["band"] == band) & (classes_all["receiver"] == receiver)].sort_values("bin")["class"].to_numpy(dtype=object)
            for h in HORIZONS:
                k = TUNED_K[f"15min|{band}|median"][h]
                rows_tr = origins(stamps_tr, k, h, stamps_tr[0], stamps_tr[-1])
                rows_va = origins(stamps_te, k, h, stamps_te[0], valid_end)
                rows_te = origins(stamps_te, k, h, *TEST_RANGE)
                if min(len(rows_tr), len(rows_va), len(rows_te)) < 20:
                    continue
                med_tr, y_tr, dev_tr = deviations(x_tr[:, ri], rows_tr, k, h)
                med_va, y_va, dev_va = deviations(x_te[:, ri], rows_va, k, h)
                med_te, y_te, dev_te = deviations(x_te[:, ri], rows_te, k, h)
                mae_median = np.abs(y_te).mean(axis=0)
                sst = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
                for region in plan.itertuples():
                    in_region = (freqs >= region.start_mhz) & (freqs <= region.end_mhz)
                    ctx_bins = {"P": (~in_region) & (cls == "P"), "N": (~in_region) & (cls == "N"), "all": ~in_region, "region": in_region}
                    ctx_feats = {}
                    for cname, mask in ctx_bins.items():
                        if mask.sum() < 2:
                            continue
                        basis = None
                        feats = []
                        for dev in (dev_tr, dev_va, dev_te):
                            lag_blocks = [dev(l)[:, mask] for l in range(CTX_LAGS)]
                            if basis is None:
                                basis = pca_basis(lag_blocks[0])
                            mean, vt = basis
                            feats.append(np.concatenate([(blk - mean) @ vt for blk in lag_blocks], axis=1))
                        ctx_feats[cname] = feats
                    for b in np.flatnonzero(in_region):
                        if cls[b] == "noise":
                            continue
                        for own_name, lags in OWN_LAGS.items():
                            own = [np.stack([d(l)[:, b] for l in lags], axis=1) for d in (dev_tr, dev_va, dev_te)]
                            res_own, lam_own = choose_and_score(own[0], y_tr[:, b], own[1], y_va[:, b], own[2], y_te[:, b])
                            for cname, feats in ctx_feats.items():
                                if cname == "region":                              # the target bin itself must not be in its own context
                                    keep_mask = in_region & (np.arange(200) != b)
                                    if keep_mask.sum() < 2:
                                        continue
                                    basis = None
                                    rf = []
                                    for dev in (dev_tr, dev_va, dev_te):
                                        blocks = [dev(l)[:, keep_mask] for l in range(CTX_LAGS)]
                                        if basis is None:
                                            basis = pca_basis(blocks[0])
                                        rf.append(np.concatenate([(blk - basis[0]) @ basis[1] for blk in blocks], axis=1))
                                    feats = rf
                                full = [np.concatenate([o, f], axis=1) for o, f in zip(own, feats)]
                                res_ctx, lam_ctx = choose_and_score(full[0], y_tr[:, b], full[1], y_va[:, b], full[2], y_te[:, b])
                                by_bin_rows.append({"band": band, "receiver": receiver, "bin": int(b), "frequency_mhz": float(freqs[b]), "region": region.band_id,
                                                    "class": cls[b], "horizon": h, "own": own_name, "context": cname, "mae_median": float(mae_median[b]),
                                                    "mae_own": float(np.abs(res_own).mean()), "mae_ctx": float(np.abs(res_ctx).mean()),
                                                    "r2_own": float(1 - (res_own ** 2).sum() / sst[b]), "r2_ctx": float(1 - (res_ctx ** 2).sum() / sst[b]),
                                                    "lambda_own": lam_own, "lambda_ctx": lam_ctx, "n_train": len(rows_tr), "n_valid": len(rows_va), "n_test": len(rows_te)})
                                key = (band, receiver, cls[b], h, own_name, cname)
                                gain = np.abs(res_own) - np.abs(res_ctx)
                                store = per_origin_store.setdefault(key, [np.zeros(len(rows_te)), 0])
                                store[0] += gain
                                store[1] += 1
        print(f"[freq-pred] {band} done", flush=True)
    by_bin = pd.DataFrame(by_bin_rows)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bin.to_csv(out_dir / f"frequency_predictability_by_bin{a.suffix}.csv", index=False)
    summary = []
    for (band, receiver, tclass, h, own, ctx), (series, n_bins) in sorted(per_origin_store.items()):
        per_origin = series / n_bins
        lo, hi = boot(per_origin, rng)
        sel = by_bin[(by_bin.band == band) & (by_bin.receiver == receiver) & (by_bin["class"] == tclass) & (by_bin.horizon == h) & (by_bin.own == own) & (by_bin.context == ctx)]
        summary.append({"band": band, "receiver": receiver, "target_class": tclass, "horizon": h, "own": own, "context": ctx, "mae_own": sel.mae_own.mean(),
                        "mae_ctx": sel.mae_ctx.mean(), "mae_gain": float(per_origin.mean()), "ci_low": lo, "ci_high": hi,
                        "r2_own": sel.r2_own.mean(), "r2_ctx": sel.r2_ctx.mean(), "r2_gain": sel.r2_ctx.mean() - sel.r2_own.mean(),
                        "n_bins": n_bins, "n_test": int(len(per_origin))})
    pd.DataFrame(summary).to_csv(out_dir / f"frequency_predictability{a.suffix}.csv", index=False)
    print(f"{len(by_bin)} by-bin rows, {len(summary)} summary rows")
    if a.upload:
        from training.ray.storage import spectrum_s3_client

        client = spectrum_s3_client()
        for name in (f"frequency_predictability{a.suffix}.csv", f"frequency_predictability_by_bin{a.suffix}.csv"):
            client.put_object(Bucket="spectrum", Key="analysis-outputs/" + name, Body=(out_dir / name).read_bytes())
            print("uploaded", name)


if __name__ == "__main__":
    main()
