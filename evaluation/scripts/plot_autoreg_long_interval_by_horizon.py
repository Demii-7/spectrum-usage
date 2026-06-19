from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectrum_eval.aerpaw_loader import load_aerpaw_data  # noqa: E402


OUT = ROOT / "results" / "figures" / "long_interval_forecasts"
LOOKBACK = 60
HORIZONS = (1, 5, 15, 60)
WINDOW = 720
CHUNKS = (
    ("chunk_600_800", 600.0, 800.0),
    ("chunk_2400_2600", 2400.0, 2600.0),
    ("chunk_3500_3700", 3500.0, 3700.0),
)


def lagged_matrix(series: np.ndarray, target_rows: np.ndarray, horizon: int) -> np.ndarray:
    starts = target_rows - horizon - LOOKBACK + 1
    return np.stack([series[start : start + LOOKBACK] for start in starts], axis=0)


def fit_autoreg(series: np.ndarray) -> tuple[float, np.ndarray]:
    target_rows = np.arange(LOOKBACK, len(series))
    x = lagged_matrix(series, target_rows, horizon=1)
    y = series[target_rows]
    design = np.column_stack([np.ones(len(x), dtype=np.float32), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), beta[1:].astype(np.float32)


def fit_lar(series: np.ndarray, horizon: int) -> tuple[float, np.ndarray]:
    target_rows = np.arange(horizon + LOOKBACK - 1, len(series))
    x = lagged_matrix(series, target_rows, horizon=horizon)
    y = series[target_rows]
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return float(model.intercept_), model.coef_.astype(np.float32)


def predict_autoreg(series: np.ndarray, target_rows: np.ndarray, horizon: int, intercept: float, coef: np.ndarray) -> np.ndarray:
    states = np.stack([series[row - horizon - LOOKBACK + 1 : row - horizon + 1] for row in target_rows], axis=0).astype(np.float32)
    pred = np.zeros(len(target_rows), dtype=np.float32)
    for _ in range(horizon):
        pred = (intercept + states @ coef).astype(np.float32)
        states = np.concatenate([states[:, 1:], pred[:, None]], axis=1)
    return pred


def predict_lookback_mean(series: np.ndarray, target_rows: np.ndarray, horizon: int) -> np.ndarray:
    return np.array(
        [np.mean(series[row - horizon - LOOKBACK + 1 : row - horizon + 1]) for row in target_rows],
        dtype=np.float32,
    )


def predict_lar(series: np.ndarray, target_rows: np.ndarray, horizon: int, intercept: float, coef: np.ndarray) -> np.ndarray:
    x = lagged_matrix(series, target_rows, horizon=horizon)
    return (intercept + x @ coef).astype(np.float32)


def choose_variable_frequency() -> tuple[str, float, float, float, int | None]:
    best = None
    for chunk_id, start_mhz, end_mhz in CHUNKS:
        data = load_aerpaw_data(ROOT / "aerpaw", start_mhz, end_mhz, normalize=False)
        test = data.splits["CC2_test"].raw_dbm
        spread = np.percentile(test, 95, axis=0) - np.percentile(test, 5, axis=0)
        idx = int(np.argmax(spread))
        candidate = (chunk_id, start_mhz, end_mhz, data.shared_frequencies[idx], float(spread[idx]))
        if best is None or candidate[-1] > best[-1]:
            best = candidate
    if best is None or best[-1] <= 5.0:
        raise ValueError("No CC2_test frequency bin has p95-p5 variation greater than 5 dB.")
    chunk_id, start_mhz, end_mhz, freq, variation = best
    return chunk_id, start_mhz, end_mhz, freq, variation, None


def choose_transition_frequency(direction: str) -> tuple[str, float, float, float, float, int]:
    best = None
    origins = np.arange(LOOKBACK, 2880 - LOOKBACK)
    for chunk_id, start_mhz, end_mhz in CHUNKS:
        data = load_aerpaw_data(ROOT / "aerpaw", start_mhz, end_mhz, normalize=False)
        test = data.splits["CC2_test"].raw_dbm
        spread = np.percentile(test, 95, axis=0) - np.percentile(test, 5, axis=0)
        valid = spread > 5.0
        if not np.any(valid):
            continue

        csum = np.vstack([np.zeros((1, test.shape[1]), dtype=np.float32), np.cumsum(test, axis=0)])
        before = (csum[origins] - csum[origins - LOOKBACK]) / np.float32(LOOKBACK)
        after = (csum[origins + LOOKBACK] - csum[origins]) / np.float32(LOOKBACK)
        score = before - after if direction == "falling" else after - before
        score[:, ~valid] = -np.inf
        origin_idx, freq_idx = np.unravel_index(int(np.argmax(score)), score.shape)
        candidate = (
            chunk_id,
            start_mhz,
            end_mhz,
            data.shared_frequencies[freq_idx],
            float(spread[freq_idx]),
            int(origins[origin_idx]),
            float(score[origin_idx, freq_idx]),
        )
        if best is None or candidate[-1] > best[-1]:
            best = candidate
    if best is None:
        raise ValueError(f"No CC2_test frequency bin has p95-p5 variation greater than 5 dB for {direction} search.")
    chunk_id, start_mhz, end_mhz, freq, variation, origin, _ = best
    return chunk_id, start_mhz, end_mhz, freq, variation, origin


def choose_window(series: np.ndarray, transition_origin: int | None) -> tuple[int, int, float]:
    if transition_origin is not None:
        start = min(max(transition_origin - WINDOW // 3, 0), len(series) - WINDOW)
        window = series[start : start + WINDOW]
        variation = float(np.percentile(window, 95) - np.percentile(window, 5))
        return start, start + WINDOW - 1, variation

    best_start = 0
    best_variation = -np.inf
    for start in range(0, len(series) - WINDOW + 1, 60):
        window = series[start : start + WINDOW]
        variation = float(np.percentile(window, 95) - np.percentile(window, 5))
        if variation > best_variation:
            best_variation = variation
            best_start = start
    return best_start, best_start + WINDOW - 1, best_variation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition",
        choices=("variable", "rising", "falling"),
        default="variable",
        help="Select the most variable bin/window, or a large rising/falling transition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.transition == "variable":
        chunk_id, start_mhz, end_mhz, freq, test_variation, transition_origin = choose_variable_frequency()
    else:
        chunk_id, start_mhz, end_mhz, freq, test_variation, transition_origin = choose_transition_frequency(args.transition)
    data = load_aerpaw_data(ROOT / "aerpaw", start_mhz, end_mhz, normalize=False)
    freq_idx = int(np.where(np.isclose(np.array(data.shared_frequencies), freq))[0][0])

    train = data.splits["CC2_train"].raw_dbm[:, freq_idx]
    test = data.splits["CC2_test"].raw_dbm[:, freq_idx]
    full = np.concatenate([train, test]).astype(np.float32)
    history_offset = len(train)
    intercept, coef = fit_autoreg(train)
    lar_models = {horizon: fit_lar(train, horizon) for horizon in HORIZONS}

    window_start, window_end, window_variation = choose_window(test, transition_origin)
    test_minutes = np.arange(window_start, window_end + 1)
    target_rows = history_offset + test_minutes
    actual = full[target_rows]

    rows = []
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(14, 11), sharex=True)
    for ax, horizon in zip(axes, HORIZONS):
        autoreg = predict_autoreg(full, target_rows, horizon, intercept, coef)
        lar_intercept, lar_coef = lar_models[horizon]
        lar = predict_lar(full, target_rows, horizon, lar_intercept, lar_coef)
        lookback = predict_lookback_mean(full, target_rows, horizon)

        ax.plot(test_minutes, actual, color="0.55", linewidth=1.0, alpha=0.8, label="actual")
        ax.plot(test_minutes, autoreg, color="tab:blue", linewidth=1.6, label=f"AutoReg H={horizon}")
        ax.plot(test_minutes, lar, color="tab:green", linewidth=1.6, label=f"LAR direct H={horizon}")
        ax.plot(test_minutes, lookback, color="tab:orange", linestyle="--", linewidth=1.4, label=f"lookback mean H={horizon}")
        ax.set_title(f"H={horizon}: prediction made {horizon} minute(s) before target")
        ax.set_ylabel("Power (dBm)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, frameon=False)

        series_by_model = {
            "actual": actual,
            "autoreg": autoreg,
            "lar": lar,
            "lookback_mean": lookback,
        }
        for model, values in series_by_model.items():
            for minute, value in zip(test_minutes, values):
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "frequency_mhz": freq,
                        "test_p95_p5_db": test_variation,
                        "window_p95_p5_db": window_variation,
                        "horizon": horizon,
                        "cc2_test_minute": int(minute),
                        "model": model,
                        "power_dbm": float(value),
                    }
                )

    axes[-1].set_xlabel("CC2_test minute")
    fig.suptitle(
        f"{args.transition.title()} long-interval forecasts at {freq:g} MHz ({chunk_id}); "
        f"test p95-p5={test_variation:.1f} dB, window p95-p5={window_variation:.1f} dB",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png_path = OUT / f"cc2_{args.transition}_by_horizon_long_interval.png"
    csv_path = OUT / f"cc2_{args.transition}_by_horizon_long_interval.csv"
    fig.savefig(png_path, dpi=170)
    if args.transition == "variable":
        fig.savefig(OUT / "cc2_autoreg_by_horizon_long_interval.png", dpi=170)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if args.transition == "variable":
        pd.DataFrame(rows).to_csv(OUT / "cc2_autoreg_by_horizon_long_interval.csv", index=False)
    print(f"Selected {freq:g} MHz from {chunk_id}; CC2_test p95-p5={test_variation:.3f} dB")
    if transition_origin is not None:
        print(f"Selected {args.transition} transition origin {transition_origin}")
    print(f"Window {window_start}-{window_end}; p95-p5={window_variation:.3f} dB")
    print(png_path)
    print(csv_path)
    if args.transition == "variable":
        print(OUT / "cc2_autoreg_by_horizon_long_interval.png")
        print(OUT / "cc2_autoreg_by_horizon_long_interval.csv")


if __name__ == "__main__":
    main()
