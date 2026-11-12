"""Is there shared information between the core receivers? (15-minute POWDER, four core receivers, both bands, CPU only)

For every receiver r the deviation of its current and recent values from its own tuned lookback median is the quantity a one-receiver
baseline leaves unexplained. The question: do the OTHER receivers' recent deviations predict the target receiver's future deviation, beyond
what the target's own recent deviations already predict?

Definitions (per band, per frequency bin, per horizon h in 4 / 12 / 24 steps = 1 / 3 / 6 h, 15-minute steps)
* Reference: the median of the receiver's last K values ending at the forecast origin t (K = the validation-selected window of the tuned median
  baseline for that horizon and band, ``training.ray.origin_errors.TUNED_K`` = ``evaluation/results/baselines_all_horizons.csv``, selection
  ``per_horizon``). So the reference IS the tuned median forecast.
* Target: y(t) = x_target(t + h) - median_target(t)   (the error of the tuned median forecast, signed).
* Features: d_r(t - l) = x_r(t - l) - median_r(t) for lags l = 0..3, r = the target itself ("own") and/or one neighbour (or all three: "all3").
* Models, one linear regression per bin (OLS with intercept, ``numpy.linalg.lstsq``), fit on the TRAINING week(s) and scored on the TEST week:
  own only (4 features)  versus  own + neighbour (8 features), and own + all three neighbours (16 features).
* Training data: run 20261015T1434Z, range 2026-10-15T14:34Z..2026-10-22T13:13Z. Test: run 20261025T1355Z, range 2026-10-27T20:00Z..2026-11-03T20:00Z
  (the paper's test week). An origin is used when its target row t + h lies inside the range and the K-row median window and the lags lie in the
  same gap-free stretch of the four receivers' common timestamps.
* Gain = (own-only) minus (own + neighbour) for the test MAE (dB; positive = the neighbour helps) and for R^2 (test R^2 relative to the test-week
  mean of y, per bin; pooled R^2 = 1 - sum SSE / sum SST over bins). ``mae_median`` is the MAE of the median forecast itself (mean |y|).
* Pooled values average the 200 bins. A moving-block bootstrap over test origins (block 96 = one day, 1,000 draws, seed 0) gives the 95% interval of
  the pooled MAE gain (origins resampled, bins pooled).

Outputs
* ``neighbor_predictability.csv``: ``scope`` = ``pair`` (ordered pair, pooled over bins, with interval and bin shares), ``all3`` (own + three
  neighbours), ``band_horizon`` (mean over the 12 ordered pairs) and ``distance`` (pairs grouped at the three distance classes of the 6 receiver
  pairs: nearest two / middle two / farthest two).
* ``neighbor_predictability_by_pair.csv``: the same per bin (one row per band, horizon, ordered pair or ``all3``, bin).

    python evaluation/analysis/neighbor_predictability.py [--root /workspace/spectrum-usage] [--out-dir evaluation/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CORE = ("humanities", "madsen", "moran", "sagepoint")
NODE = {name: f"{name}-nuc1" for name in CORE}
LOCATION_NAME = {"humanities": "humanities", "madsen": "madsen", "moran": "moran", "sagepoint": "sage point"}
BANDS = ("600_800", "2400_2600")
HORIZONS = (4, 12, 24)
LAGS = 4
STEP = pd.Timedelta(minutes=15)
TRAIN = ("20261015T1434Z", pd.Timestamp("2026-10-15T14:34:00Z"), pd.Timestamp("2026-10-22T13:13:00Z"))
TEST = ("20261025T1355Z", pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z"))
BLOCK, DRAWS = 96, 1000


def load_aligned(root: Path, band: str, run: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(timestamps, array (T, 4 receivers, bins)) on the timestamps all four receivers share (15-minute blocks, dB)."""
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
    arrays = [frame.loc[common].to_numpy(dtype=np.float64) for frame in frames]
    return common, np.stack(arrays, axis=1)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = map(lambda p: tuple(np.radians(p)), (a, b))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


def distances(root: Path) -> dict[tuple[str, str], float]:
    places = {e["name"]: (e["latitude"], e["longitude"]) for e in json.loads((root / "data/locations/powder.json").read_text())["endpoints"]}
    return {(a, b): haversine_km(places[LOCATION_NAME[a]], places[LOCATION_NAME[b]]) for a in CORE for b in CORE if a != b}


def design(stamps: pd.DatetimeIndex, x: np.ndarray, k: int, h: int, start: pd.Timestamp, end: pd.Timestamp):
    """Origins, reference medians and the (T', 4, bins) deviation arrays for lags 0..3 of every receiver, plus the target deviations.

    Returns (origin rows, lag deviations D[l] of shape (N, 4, bins) for l = 0..3, target deviation Y of shape (N, 4, bins))."""
    n = len(stamps)
    gap = np.concatenate([[0], np.cumsum(np.diff(stamps.asi8) != STEP.value)])
    rows = np.arange(max(k - 1, LAGS - 1), n - h)
    target_time = stamps[rows + h]
    keep = (target_time >= start) & (target_time <= end) & (gap[rows - (k - 1)] == gap[rows + h])
    rows = rows[np.asarray(keep)]
    if not len(rows):
        raise ValueError("no origins")
    # median of x[t-k+1 .. t] for every origin t, per receiver and bin
    windows = sliding_window_view(x, k, axis=0)                      # (T-k+1, 4, bins, k)
    med = np.median(windows[rows - (k - 1)], axis=-1)                # (N, 4, bins)
    lags = np.stack([x[rows - l] - med for l in range(LAGS)], axis=0)   # (LAGS, N, 4, bins)
    target = x[rows + h] - med
    return rows, lags, target


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """OLS with intercept per bin. train_x (N, p, bins), train_y (N, bins), test_x (M, p, bins) -> predictions (M, bins)."""
    bins = train_y.shape[1]
    out = np.empty((test_x.shape[0], bins))
    for b in range(bins):
        a = np.concatenate([train_x[:, :, b], np.ones((len(train_x), 1))], axis=1)
        coef = np.linalg.lstsq(a, train_y[:, b], rcond=None)[0]
        out[:, b] = np.concatenate([test_x[:, :, b], np.ones((len(test_x), 1))], axis=1) @ coef
    return out


def block_bootstrap(per_origin: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(per_origin)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    starts = rng.integers(0, n - block + 1, size=(DRAWS, int(np.ceil(n / block))))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    draws = per_origin[index].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def features(lags: np.ndarray, receivers: list[int]) -> np.ndarray:
    """(N, p, bins) with the lags 0..3 of the listed receivers (receiver-major)."""
    return np.concatenate([lags[l, :, r, :][:, None, :] for r in receivers for l in range(LAGS)], axis=1)


def analyse(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from training.ray.origin_errors import TUNED_K

    dist = distances(root)
    rng = np.random.default_rng(0)
    per_bin_rows, pooled_rows = [], []
    for band in BANDS:
        train_stamps, train_x = load_aligned(root, band, TRAIN[0])
        test_stamps, test_x = load_aligned(root, band, TEST[0])
        header = pd.read_csv(root / "data/powder15" / NODE[CORE[0]] / TEST[0] / band / "power_15min.csv", nrows=1).columns[1:]
        try:
            frequencies = header.astype(float).to_numpy()
        except ValueError:
            frequencies = np.arange(len(header), dtype=float)
        for h in HORIZONS:
            k = TUNED_K[f"15min|{band}|median"][h]
            _, tr_lags, tr_y = design(train_stamps, train_x, k, h, TRAIN[1], TRAIN[2])
            te_rows, te_lags, te_y = design(test_stamps, test_x, k, h, TEST[1], TEST[2])
            print(f"[neighbor-pred] {band} h={h} K={k}: {len(tr_y)} train origins, {len(te_y)} test origins", flush=True)
            for target, name in enumerate(CORE):
                y_tr, y_te = tr_y[:, target, :], te_y[:, target, :]
                sst = ((y_te - y_te.mean(axis=0)) ** 2)
                own_pred = fit_predict(features(tr_lags, [target]), y_tr, features(te_lags, [target]))
                own_err = y_te - own_pred
                cases = [(other, [target, other]) for other in range(len(CORE)) if other != target]
                cases.append(("all3", [target] + [o for o in range(len(CORE)) if o != target]))
                for other, receivers in cases:
                    pred = fit_predict(features(tr_lags, receivers), y_tr, features(te_lags, receivers))
                    err = y_te - pred
                    label = other if other == "all3" else CORE[other]
                    km = float("nan") if other == "all3" else dist[(name, label)]
                    mae_own, mae_nb = np.abs(own_err).mean(axis=0), np.abs(err).mean(axis=0)
                    sse_own, sse_nb = (own_err ** 2).sum(axis=0), (err ** 2).sum(axis=0)
                    sst_b = sst.sum(axis=0)
                    r2_own, r2_nb = 1 - sse_own / sst_b, 1 - sse_nb / sst_b
                    base = {"band": band, "horizon_steps": h, "horizon_hours": h / 4, "median_window_k": k, "target": name,
                            "neighbor": label, "distance_km": km, "n_train": len(y_tr), "n_test": len(y_te)}
                    per_bin_rows.append(pd.DataFrame({**base, "bin_mhz": frequencies, "mae_median": np.abs(y_te).mean(axis=0),
                                                      "mae_own": mae_own, "mae_with_neighbor": mae_nb, "delta_mae": mae_own - mae_nb,
                                                      "r2_own": r2_own, "r2_with_neighbor": r2_nb, "delta_r2": r2_nb - r2_own}))
                    per_origin_gain = np.abs(own_err).mean(axis=1) - np.abs(err).mean(axis=1)
                    lo, hi = block_bootstrap(per_origin_gain, rng)
                    pooled_rows.append({**base, "scope": "all3" if other == "all3" else "pair", "mae_median": float(np.abs(y_te).mean()),
                                        "mae_own": float(mae_own.mean()), "mae_with_neighbor": float(mae_nb.mean()),
                                        "delta_mae": float(mae_own.mean() - mae_nb.mean()), "delta_mae_ci_low": lo, "delta_mae_ci_high": hi,
                                        "r2_own": float(1 - sse_own.sum() / sst_b.sum()), "r2_with_neighbor": float(1 - sse_nb.sum() / sst_b.sum()),
                                        "delta_r2": float((1 - sse_nb.sum() / sst_b.sum()) - (1 - sse_own.sum() / sst_b.sum())),
                                        "share_bins_improved": float((mae_own - mae_nb > 0).mean())})
    per_bin = pd.concat(per_bin_rows, ignore_index=True)
    pooled = pd.DataFrame(pooled_rows)
    pairs = pooled[pooled.scope == "pair"]
    mean_cols = ["mae_median", "mae_own", "mae_with_neighbor", "delta_mae", "r2_own", "r2_with_neighbor", "delta_r2", "share_bins_improved"]
    summary = [pooled]
    by_h = pairs.groupby(["band", "horizon_steps", "horizon_hours", "median_window_k"])[mean_cols + ["distance_km"]].mean().reset_index()
    by_h.insert(0, "scope", "band_horizon")
    summary.append(by_h)
    # three distance classes of the six unordered pairs: nearest two, middle two, farthest two (each in both directions)
    unordered = sorted({tuple(sorted(p)) for p in dist}, key=lambda p: dist[(p[0], p[1])])
    cls = {}
    for i, pair in enumerate(unordered):
        cls[pair] = ("near", "middle", "far")[i // 2]
    pairs = pairs.assign(distance_class=[cls[tuple(sorted((t, n)))] for t, n in zip(pairs.target, pairs.neighbor)])
    by_d = pairs.groupby(["band", "horizon_steps", "horizon_hours", "median_window_k", "distance_class"])[mean_cols + ["distance_km"]].mean().reset_index()
    by_d.insert(0, "scope", "distance")
    summary.append(by_d)
    return pd.concat(summary, ignore_index=True), per_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()
    summary, per_bin = analyse(Path(args.root))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "neighbor_predictability.csv", index=False)
    per_bin.to_csv(out / "neighbor_predictability_by_pair.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["band", "horizon_hours", "mae_median", "mae_own", "mae_with_neighbor", "delta_mae", "r2_own", "r2_with_neighbor", "delta_r2", "share_bins_improved"]
    print("\nmean over the 12 ordered pairs (delta > 0: the neighbour helps)")
    print(summary[summary.scope == "band_horizon"][cols].round(4).to_string(index=False))
    print("\nown + all three neighbours, mean over the four targets")
    a3 = summary[summary.scope == "all3"].groupby(["band", "horizon_hours"])[["mae_own", "mae_with_neighbor", "delta_mae", "r2_own", "r2_with_neighbor", "delta_r2"]].mean().reset_index()
    print(a3.round(4).to_string(index=False))
    print("\nby distance class")
    print(summary[summary.scope == "distance"][["band", "horizon_hours", "distance_class", "distance_km", "delta_mae", "delta_r2"]].round(4).to_string(index=False))
    print(f"\n{len(summary)} summary rows, {len(per_bin)} per-bin rows -> {out}")


if __name__ == "__main__":
    main()
