from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from training.common.aerpaw_loader import LoadedSpectrumData, SplitArrays, frequency_value, interpolate_missing


DEFAULT_REFERENCE_SITE = "POWDER"


def _powder_numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp_utc" in df.columns:
        df = df.drop(columns=["timestamp_utc"])
    df.columns = [str(frequency_value(col)) for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _selected_columns(columns: Iterable[str], chunk_start_mhz: float, chunk_end_mhz: float) -> list[str]:
    return [
        col
        for col in sorted(columns, key=frequency_value)
        if chunk_start_mhz <= frequency_value(col) <= chunk_end_mhz
    ]


def _load_split_frame(paths: list[Path], chunk_start_mhz: float, chunk_end_mhz: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not paths:
        raise ValueError("POWDER loader requires at least one CSV path per split.")

    frames: list[pd.DataFrame] = []
    per_file_frames: dict[str, pd.DataFrame] = {}
    expected_cols: list[str] | None = None

    for path in paths:
        raw = _powder_numeric_frame(path)
        selected_cols = _selected_columns(raw.columns, chunk_start_mhz, chunk_end_mhz)
        if not selected_cols:
            raise ValueError(
                f"No POWDER frequencies in requested chunk {chunk_start_mhz}-{chunk_end_mhz} MHz for {path}."
            )

        chunk_frame = raw.loc[:, selected_cols].copy()
        current_cols = list(chunk_frame.columns)
        if expected_cols is None:
            expected_cols = current_cols
        elif current_cols != expected_cols:
            raise ValueError(
                f"POWDER files for the same split must share the same frequency columns. Mismatch at {path}."
            )

        frames.append(chunk_frame)
        per_file_frames[path.stem] = chunk_frame

    return pd.concat(frames, axis=0, ignore_index=True), per_file_frames


def load_powder_data(
    train_files: list[Path],
    test_files: list[Path],
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    normalize: bool = False,
    reference_site: str = DEFAULT_REFERENCE_SITE,
    max_rows: int | None = None,
) -> LoadedSpectrumData:
    train_raw, train_frames = _load_split_frame(train_files, chunk_start_mhz, chunk_end_mhz)
    test_raw, test_frames = _load_split_frame(test_files, chunk_start_mhz, chunk_end_mhz)

    if max_rows is not None:
        train_raw = train_raw.iloc[: int(max_rows)].copy()
        test_raw = test_raw.iloc[: int(max_rows)].copy()

    train_filled = interpolate_missing(train_raw)
    test_filled = interpolate_missing(test_raw)
    if train_filled.empty or test_filled.empty:
        raise ValueError("POWDER train/test splits must both contain at least one row.")

    train_array = train_filled.to_numpy(dtype=np.float32)
    test_array = test_filled.to_numpy(dtype=np.float32)
    train_model = train_array
    test_model = test_array
    normalization = None
    if normalize:
        mean = float(np.mean(train_array))
        std = float(np.std(train_array))
        if std == 0.0:
            raise ValueError("Cannot normalize a zero-variance POWDER training split.")
        train_model = (train_array - mean) / std
        test_model = (test_array - mean) / std
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train",
        }

    train_len = len(train_array)
    total_len = train_len + len(test_array)
    raw_frames = {f"train:{name}": frame for name, frame in train_frames.items()}
    raw_frames.update({f"test:{name}": frame for name, frame in test_frames.items()})
    filled_frames = {"train": train_filled, "test": test_filled}
    files = {f"train:{path.stem}": path for path in train_files}
    files.update({f"test:{path.stem}": path for path in test_files})

    return LoadedSpectrumData(
        files=files,
        raw_frames=raw_frames,
        filled_frames=filled_frames,
        reference_site=reference_site,
        frequencies=[frequency_value(col) for col in train_filled.columns],
        splits={
            f"{reference_site}_train": SplitArrays(
                raw_dbm=train_array,
                model_input=train_model,
                row_start=0,
                row_end=train_len - 1,
            ),
            f"{reference_site}_test": SplitArrays(
                raw_dbm=test_array,
                model_input=test_model,
                row_start=train_len,
                row_end=total_len - 1,
            ),
        },
        normalization=normalization,
    )
