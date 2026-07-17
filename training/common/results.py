"""
Metric aggregation, result-table generation, and output-directory utilities.

This module provides the shared result-management functions used by integrated
training and evaluation. It converts elementwise forecast errors into
aggregate, per-frequency, and frequency-band records and writes finalized result
tables and execution summaries.

Primary responsibilities include:

- creating standard output and checkpoint directories;
- loading configured frequency-band definitions;
- validating frequency-band metadata;
- aggregating absolute and squared errors across samples and dimensions;
- calculating summary measures such as MAE, MSE, and RMSE;
- producing one record per model, chunk, split, and forecast horizon;
- producing per-frequency metric records;
- assigning frequencies to configured bands;
- producing frequency-band metric records;
- preserving target-row ranges and split-local indexing metadata;
- collecting metric rows across chunks;
- writing aggregate, per-frequency, and band-level CSV files;
- writing human-readable execution summaries; and
- maintaining stable result schemas for plotting and later assembly.

Expected error layouts supplied to metric aggregation:

    Vector models:
        (N, F)

    Map models:
        Spatial dimensions reduced before aggregation, producing (N, F).

This module does not run models or calculate raw elementwise errors. It receives
prepared error arrays and converts them into persistent summary results.
"""

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
    raw = config["data"].get("band_definitions_path")
    if not raw:
        return pd.DataFrame()
    path = resolve_path(raw)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def output_dir(config: dict[str, Any], model_name: str) -> Path:
    path = resolve_path(config["outputs"]["root_dir"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(config: dict[str, Any], model_name: str) -> Path:
    path = output_dir(config, model_name) / "checkpoints"
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


def prepare_output_dirs( config: dict[str, Any], model_name: str,) -> tuple[Path, Path]:
    out = output_dir(
        config,
        model_name,
    )

    checkpoints = checkpoints_dir(
        config,
        model_name,
    )

    return out, checkpoints


def finalize_results(
    out: Path,
    model_name: str,
    aggregate_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    extra_lines: Iterable[str] = (),
) -> None:
    aggregate = pd.DataFrame(aggregate_rows)
    frequency = pd.DataFrame(frequency_rows)
    band = pd.DataFrame(band_rows)
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    frequency.to_csv(out / "per_frequency_metrics.csv", index=False)
    band.to_csv(out / "per_band_metrics.csv", index=False)

    lines = [f"Model: {model_name}"]
    if aggregate.empty:
        lines.append("No aggregate metrics produced.")
    else:
        lines.append(f"Aggregate rows: {len(aggregate)}")
        lines.append(f"Per-frequency rows: {len(frequency)}")
        lines.append(f"Per-band rows: {len(band)}")
        best = aggregate.sort_values("mae_db").iloc[0]
        lines.append(
            "Best aggregate MAE: "
            f"{best['mae_db']:.4f} dB on {best['chunk_id']} {best['split']} h={int(best['horizon'])}"
        )
    lines.extend(extra_lines)
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
