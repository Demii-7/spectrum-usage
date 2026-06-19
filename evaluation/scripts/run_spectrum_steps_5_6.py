from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectrum_eval.aerpaw_loader import load_aerpaw_data  # noqa: E402


RESULTS = ROOT / "results"
STEP2 = RESULTS / "step2"
OUT = RESULTS / "step5_6"
MODEL_DIR = OUT / "models"

LOOKBACK = 60
HORIZONS = (1, 5, 15, 60)
CHUNKS = (
    ("chunk_600_800", 600.0, 800.0),
    ("chunk_2400_2600", 2400.0, 2600.0),
    ("chunk_3500_3700", 3500.0, 3700.0),
)
TEST_SPLITS = ("CC2_test",)
BASELINES = ("persistence", "historical_mean", "lookback_mean", "same_time_last3day_mean", "autoreg", "lar")


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def denormalize(x: np.ndarray, normalization: dict[str, object] | None) -> np.ndarray:
    if normalization is None:
        return x
    return (x * np.float32(normalization["std_dbm"]) + np.float32(normalization["mean_dbm"])).astype(np.float32)


def lagged_matrix(series: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int = LOOKBACK) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    return np.stack([series[start : start + lookback] for start in starts], axis=0)


def target_rows_for(split_length: int, history_offset: int, horizon: int) -> np.ndarray:
    start = max(history_offset, horizon + LOOKBACK - 1, 4320)
    end = history_offset + split_length
    if start >= end:
        raise ValueError(f"Split length {split_length} is too short for horizon {horizon} and aligned evaluation.")
    return np.arange(start, end)


def fit_autoreg(train: np.ndarray) -> dict[str, np.ndarray]:
    coefs = np.zeros((LOOKBACK, train.shape[1]), dtype=np.float32)
    intercepts = np.zeros(train.shape[1], dtype=np.float32)
    target_rows = np.arange(LOOKBACK, len(train))
    for f in range(train.shape[1]):
        x = lagged_matrix(train[:, f], target_rows, horizon=1)
        y = train[target_rows, f]
        design = np.column_stack([np.ones(len(x), dtype=np.float32), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercepts[f] = beta[0]
        coefs[:, f] = beta[1:]
    return {"intercept": intercepts, "coef": coefs}


def predict_autoreg(x: np.ndarray, target_rows: np.ndarray, horizon: int, params: dict[str, np.ndarray]) -> np.ndarray:
    states = np.stack([x[row - horizon - LOOKBACK + 1 : row - horizon + 1] for row in target_rows], axis=0).astype(np.float32)
    coefs = params["coef"]
    intercept = params["intercept"]
    pred = None
    for _ in range(horizon):
        pred = intercept[None, :] + np.sum(states * coefs[None, :, :], axis=1)
        states = np.concatenate([states[:, 1:, :], pred[:, None, :]], axis=1)
    return pred.astype(np.float32)


def fit_lar(train: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    coefs = np.zeros((LOOKBACK, train.shape[1]), dtype=np.float32)
    intercepts = np.zeros(train.shape[1], dtype=np.float32)
    target_rows = np.arange(horizon + LOOKBACK - 1, len(train))
    for f in range(train.shape[1]):
        x = lagged_matrix(train[:, f], target_rows, horizon=horizon)
        y = train[target_rows, f]
        model = Ridge(alpha=1.0)
        model.fit(x, y)
        intercepts[f] = model.intercept_
        coefs[:, f] = model.coef_
    return {"intercept": intercepts, "coef": coefs}


def predict_lar(x: np.ndarray, target_rows: np.ndarray, horizon: int, params: dict[str, np.ndarray]) -> np.ndarray:
    preds = np.zeros((len(target_rows), x.shape[1]), dtype=np.float32)
    for f in range(x.shape[1]):
        features = lagged_matrix(x[:, f], target_rows, horizon=horizon)
        preds[:, f] = params["intercept"][f] + features @ params["coef"][:, f]
    return preds


def predictions_for_model(
    model: str,
    full_x: np.ndarray,
    target_rows: np.ndarray,
    horizon: int,
    hist_mean: np.ndarray,
    autoreg_params: dict[str, np.ndarray],
    lar_params: dict[str, np.ndarray],
) -> np.ndarray:
    if model == "persistence":
        return full_x[target_rows - horizon]
    if model == "historical_mean":
        return np.tile(hist_mean[None, :], (len(target_rows), 1)).astype(np.float32)
    if model == "lookback_mean":
        preds = np.zeros((len(target_rows), full_x.shape[1]), dtype=np.float32)
        for i, row in enumerate(target_rows):
            preds[i] = np.mean(full_x[row - horizon - LOOKBACK + 1 : row - horizon + 1], axis=0)
        return preds
    if model == "same_time_last3day_mean":
        return (
            full_x[target_rows - 1440] + full_x[target_rows - 2880] + full_x[target_rows - 4320]
        ) / np.float32(3.0)
    if model == "autoreg":
        return predict_autoreg(full_x, target_rows, horizon, autoreg_params)
    if model == "lar":
        return predict_lar(full_x, target_rows, horizon, lar_params)
    raise ValueError(f"Unknown model: {model}")


def metric_values(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def split_site(split_name: str) -> str:
    return split_name.split("_", 1)[0]


def band_indices(band: pd.Series, freqs: list[float]) -> list[int]:
    freq_to_idx = {round(freq, 6): idx for idx, freq in enumerate(freqs)}
    return [freq_to_idx[round(float(value), 6)] for value in str(band["included_frequency_mhz"]).split()]


def evaluate_chunk(chunk_id: str, start_mhz: float, end_mhz: float, bands: pd.DataFrame, normalize: bool):
    data = load_aerpaw_data(ROOT / "aerpaw", start_mhz, end_mhz, normalize=normalize)
    train_raw = data.splits["CC2_train"].raw_dbm
    train = data.splits["CC2_train"].model_input
    hist_mean = np.mean(train, axis=0).astype(np.float32)
    autoreg_params = fit_autoreg(train)
    lar_by_horizon = {horizon: fit_lar(train, horizon) for horizon in HORIZONS}

    aggregate_rows = []
    frequency_rows = []
    band_rows = []
    chunk_bands = bands[bands["chunk_id"] == chunk_id].copy()

    for horizon in HORIZONS:
        lar_params = lar_by_horizon[horizon]
        for split_name in TEST_SPLITS:
            split = data.splits[split_name]
            x_raw = split.raw_dbm
            x = split.model_input
            full_raw = np.vstack([train_raw, x_raw]).astype(np.float32)
            full_x = np.vstack([train, x]).astype(np.float32)
            history_offset = len(train)
            target_rows = target_rows_for(len(x_raw), history_offset, horizon)
            target = full_raw[target_rows]

            preds = {
                model: denormalize(
                    predictions_for_model(model, full_x, target_rows, horizon, hist_mean, autoreg_params, lar_params),
                    data.normalization,
                )
                for model in BASELINES
            }
            baseline_mae = {model: metric_values(pred, target)[0] for model, pred in preds.items()}

            for model, pred in preds.items():
                mae, rmse = metric_values(pred, target)
                aggregate_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "start_mhz": start_mhz,
                        "end_mhz": end_mhz,
                        "site": split_site(split_name),
                        "split": split_name,
                        "horizon": horizon,
                        "model": model,
                        "target_row_start": int(target_rows[0] - history_offset),
                        "target_row_end": int(target_rows[-1] - history_offset),
                        "n_targets": int(len(target_rows)),
                        "mae_db": mae,
                        "rmse_db": rmse,
                        "skill_vs_persistence": 1.0 - mae / baseline_mae["persistence"],
                        "skill_vs_hist_mean": 1.0 - mae / baseline_mae["historical_mean"],
                        "skill_vs_same_time_last3day_mean": 1.0 - mae / baseline_mae["same_time_last3day_mean"],
                    }
                )

                abs_err = np.abs(pred - target)
                sq_err = (pred - target) ** 2
                for idx, freq in enumerate(data.shared_frequencies):
                    frequency_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "frequency_mhz": freq,
                            "site": split_site(split_name),
                            "split": split_name,
                            "horizon": horizon,
                            "model": model,
                            "mae_db": float(np.mean(abs_err[:, idx])),
                            "rmse_db": float(np.sqrt(np.mean(sq_err[:, idx]))),
                        }
                    )

                for _, band in chunk_bands.iterrows():
                    indices = band_indices(band, data.shared_frequencies)
                    band_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "band_id": band["band_id"],
                            "start_mhz": band["start_mhz"],
                            "end_mhz": band["end_mhz"],
                            "behavior_category": band["behavior_category"],
                            "site": split_site(split_name),
                            "split": split_name,
                            "horizon": horizon,
                            "model": model,
                            "mae_db": float(np.mean(abs_err[:, indices])),
                            "rmse_db": float(np.sqrt(np.mean(sq_err[:, indices]))),
                        }
                    )

    return aggregate_rows, frequency_rows, band_rows, {"autoreg": autoreg_params, "lar": lar_by_horizon}


def write_summary(aggregate: pd.DataFrame, normalize: bool) -> None:
    lines = ["# Step 5 and 6 Evaluation Summary", ""]
    lines.append(f"Normalization: {'enabled' if normalize else 'disabled'}.")
    lines.append("")
    lines.append("This run evaluates persistence, historical mean, lookback mean, same-time last-3-days mean, AutoReg(60), and LAR on power estimation.")
    lines.append("All models use aligned target rows and may use CC2 training rows as prior history for the first CC2 test targets.")
    lines.append("")
    lines.append("## Aggregate MAE, H=1")
    lines.append("| Chunk | Split | Model | MAE dB | Skill vs same-time last-3-days mean |")
    lines.append("|---|---|---|---:|---:|")
    view = aggregate[aggregate["horizon"] == 1].sort_values(["chunk_id", "split", "mae_db"])
    for _, row in view.iterrows():
        lines.append(
            f"| {row['chunk_id']} | {row['split']} | {row['model']} | {row['mae_db']:.3f} | {row['skill_vs_same_time_last3day_mean']:.3f} |"
        )
    lines.append("")
    lines.append("## Best Model by Chunk/Split/Horizon")
    lines.append("| Chunk | Split | Horizon | Best model | MAE dB |")
    lines.append("|---|---|---:|---|---:|")
    idx = aggregate.groupby(["chunk_id", "split", "horizon"])["mae_db"].idxmin()
    for _, row in aggregate.loc[idx].sort_values(["chunk_id", "split", "horizon"]).iterrows():
        lines.append(f"| {row['chunk_id']} | {row['split']} | {row['horizon']} | {row['model']} | {row['mae_db']:.3f} |")
    (OUT / "metrics_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true", help="Train/evaluate models on normalized inputs, then report metrics in dB.")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Directory for generated Step 5/6 artifacts.")
    return parser.parse_args()


def main() -> None:
    global OUT, MODEL_DIR
    args = parse_args()
    OUT = args.output_dir
    MODEL_DIR = OUT / "models"
    ensure_dirs()
    bands = pd.read_csv(STEP2 / "band_definitions.csv").fillna("")
    all_aggregate = []
    all_frequency = []
    all_band = []
    model_store = {}

    for chunk_id, start_mhz, end_mhz in CHUNKS:
        print(f"Evaluating {chunk_id} ({start_mhz:g}-{end_mhz:g} MHz)")
        aggregate, frequency, band, models = evaluate_chunk(chunk_id, start_mhz, end_mhz, bands, normalize=args.normalize)
        all_aggregate.extend(aggregate)
        all_frequency.extend(frequency)
        all_band.extend(band)
        model_store[chunk_id] = models

    aggregate_df = pd.DataFrame(all_aggregate)
    frequency_df = pd.DataFrame(all_frequency)
    band_df = pd.DataFrame(all_band)
    aggregate_df.to_csv(OUT / "aggregate_metrics.csv", index=False)
    frequency_df.to_csv(OUT / "per_frequency_metrics.csv", index=False)
    band_df.to_csv(OUT / "per_band_metrics.csv", index=False)

    with (MODEL_DIR / "autoreg_models.pkl").open("wb") as f:
        pickle.dump({chunk: models["autoreg"] for chunk, models in model_store.items()}, f)
    with (MODEL_DIR / "lar_models.pkl").open("wb") as f:
        pickle.dump({chunk: models["lar"] for chunk, models in model_store.items()}, f)

    write_summary(aggregate_df, normalize=args.normalize)
    print(f"Wrote {len(aggregate_df)} aggregate metric rows to {OUT / 'aggregate_metrics.csv'}")
    print(f"Wrote {len(frequency_df)} per-frequency metric rows to {OUT / 'per_frequency_metrics.csv'}")
    print(f"Wrote {len(band_df)} per-band metric rows to {OUT / 'per_band_metrics.csv'}")


if __name__ == "__main__":
    main()
