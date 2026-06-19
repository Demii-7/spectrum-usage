from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.config import load_config  # noqa: E402
from training.common.data import chunk_specs, load_chunk  # noqa: E402
from training.common.metrics import absolute_and_squared_errors_dbm  # noqa: E402
from training.common.results import append_metric_rows, load_band_definitions, output_dir  # noqa: E402
from training.common.windowing import lagged_matrix, target_rows_for  # noqa: E402


MODEL_NAME = "linear_autoregressive"


def fit_lar(train: np.ndarray, horizon: int, lookback: int, alpha: float) -> dict[str, np.ndarray]:
    coefs = np.zeros((lookback, train.shape[1]), dtype=np.float32)
    intercepts = np.zeros(train.shape[1], dtype=np.float32)
    target_rows = np.arange(horizon + lookback - 1, len(train))
    for f in range(train.shape[1]):
        x = lagged_matrix(train[:, f], target_rows, horizon=horizon, lookback=lookback)
        y = train[target_rows, f]
        model = Ridge(alpha=alpha)
        model.fit(x, y)
        intercepts[f] = model.intercept_
        coefs[:, f] = model.coef_
    return {"intercept": intercepts, "coef": coefs}


def predict_lar(x: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int, params: dict[str, np.ndarray]) -> np.ndarray:
    preds = np.zeros((len(target_rows), x.shape[1]), dtype=np.float32)
    for f in range(x.shape[1]):
        features = lagged_matrix(x[:, f], target_rows, horizon=horizon, lookback=lookback)
        preds[:, f] = params["intercept"][f] + features @ params["coef"][:, f]
    return preds


def evaluate_chunk(config: dict[str, Any], chunk, bands: pd.DataFrame):
    lookback = int(config["windowing"]["lookback"])
    min_history = int(config["windowing"].get("min_history", 4320))
    horizons = [int(h) for h in config["windowing"]["horizons"]]
    alpha = float(config.get("linear_autoregressive", {}).get("ridge_alpha", 1.0))
    test_splits = config["data"].get("test_splits", ["CC2_test"])

    data = load_chunk(config, chunk)
    train = data.splits["CC2_train"].model_input
    train_raw = data.splits["CC2_train"].raw_dbm
    models = {horizon: fit_lar(train, horizon, lookback, alpha) for horizon in horizons}

    aggregate_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        params = models[horizon]
        for split_name in test_splits:
            split = data.splits[split_name]
            full_x = np.vstack([train, split.model_input]).astype(np.float32)
            full_raw = np.vstack([train_raw, split.raw_dbm]).astype(np.float32)
            history_offset = len(train)
            target_rows = target_rows_for(len(split.raw_dbm), history_offset, horizon, lookback, min_history)
            pred = predict_lar(full_x, target_rows, horizon, lookback, params)
            target = full_raw[target_rows]
            _, abs_err, sq_err = absolute_and_squared_errors_dbm(pred, target, data.normalization)
            append_metric_rows(
                aggregate_rows,
                frequency_rows,
                band_rows,
                chunk_id=chunk.chunk_id,
                start_mhz=chunk.start_mhz,
                end_mhz=chunk.end_mhz,
                split_name=split_name,
                horizon=horizon,
                model=MODEL_NAME,
                target_rows=target_rows,
                history_offset=history_offset,
                freqs=data.shared_frequencies,
                abs_err=abs_err,
                sq_err=sq_err,
                bands=bands,
            )

    return aggregate_rows, frequency_rows, band_rows, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = args.output_dir or output_dir(config, "LinearAutoRegressive")
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bands = load_band_definitions(config)

    all_aggregate: list[dict[str, Any]] = []
    all_frequency: list[dict[str, Any]] = []
    all_band: list[dict[str, Any]] = []
    model_store = {}

    for chunk in chunk_specs(config):
        print(f"Training LinearAutoRegressive for {chunk.chunk_id} ({chunk.start_mhz:g}-{chunk.end_mhz:g} MHz)")
        aggregate, frequency, band, models = evaluate_chunk(config, chunk, bands)
        all_aggregate.extend(aggregate)
        all_frequency.extend(frequency)
        all_band.extend(band)
        model_store[chunk.chunk_id] = models

    pd.DataFrame(all_aggregate).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(all_frequency).to_csv(out / "per_frequency_metrics.csv", index=False)
    pd.DataFrame(all_band).to_csv(out / "per_band_metrics.csv", index=False)
    with (model_dir / "linear_autoregressive_models.pkl").open("wb") as f:
        pickle.dump(model_store, f)

    print(f"Wrote {len(all_aggregate)} aggregate metric rows to {out / 'aggregate_metrics.csv'}")
    print(f"Wrote {len(all_frequency)} per-frequency metric rows to {out / 'per_frequency_metrics.csv'}")
    print(f"Wrote {len(all_band)} per-band metric rows to {out / 'per_band_metrics.csv'}")


if __name__ == "__main__":
    main()
