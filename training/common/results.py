from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.common.config import resolve_path


def split_site(split_name: str) -> str:
    return split_name.split("_", 1)[0]


def band_indices(band: pd.Series, freqs: list[float]) -> list[int]:
    freq_to_idx = {round(freq, 6): idx for idx, freq in enumerate(freqs)}
    return [freq_to_idx[round(float(value), 6)] for value in str(band["included_frequency_mhz"]).split()]


def load_band_definitions(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["data"]["band_definitions_path"])
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def output_dir(config: dict[str, Any], model_name: str) -> Path:
    path = resolve_path(config["outputs"]["root_dir"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_metric_rows(
    aggregate_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    *,
    chunk_id: str,
    start_mhz: float,
    end_mhz: float,
    split_name: str,
    horizon: int,
    model: str,
    target_rows: np.ndarray,
    history_offset: int,
    freqs: list[float],
    abs_err: np.ndarray,
    sq_err: np.ndarray,
    bands: pd.DataFrame,
) -> None:
    aggregate_rows.append(
        {
            "chunk_id": chunk_id,
            "start_mhz": start_mhz,
            "end_mhz": end_mhz,
            "site": split_site(split_name),
            "split": split_name,
            "horizon": int(horizon),
            "model": model,
            "target_row_start": int(target_rows[0] - history_offset),
            "target_row_end": int(target_rows[-1] - history_offset),
            "n_targets": int(len(target_rows)),
            "mae_db": float(np.mean(abs_err)),
            "rmse_db": float(np.sqrt(np.mean(sq_err))),
        }
    )

    for idx, freq in enumerate(freqs):
        frequency_rows.append(
            {
                "chunk_id": chunk_id,
                "frequency_mhz": freq,
                "site": split_site(split_name),
                "split": split_name,
                "horizon": int(horizon),
                "model": model,
                "mae_db": float(np.mean(abs_err[:, idx])),
                "rmse_db": float(np.sqrt(np.mean(sq_err[:, idx]))),
            }
        )

    if bands.empty:
        return
    chunk_bands = bands[bands["chunk_id"] == chunk_id].copy()
    for _, band in chunk_bands.iterrows():
        indices = band_indices(band, freqs)
        band_rows.append(
            {
                "chunk_id": chunk_id,
                "band_id": band["band_id"],
                "start_mhz": band["start_mhz"],
                "end_mhz": band["end_mhz"],
                "behavior_category": band["behavior_category"],
                "site": split_site(split_name),
                "split": split_name,
                "horizon": int(horizon),
                "model": model,
                "mae_db": float(np.mean(abs_err[:, indices])),
                "rmse_db": float(np.sqrt(np.mean(sq_err[:, indices]))),
            }
        )
