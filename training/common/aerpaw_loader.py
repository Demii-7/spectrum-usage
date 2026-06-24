from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


SITES = ("CC1", "CC2", "LW1")
DEFAULT_REFERENCE_SITE = "CC2"


@dataclass(frozen=True)
class SplitArrays:
    raw_dbm: np.ndarray
    model_input: np.ndarray
    row_start: int
    row_end: int


@dataclass(frozen=True)
class LoadedSpectrumData:
    files: dict[str, Path]
    raw_frames: dict[str, pd.DataFrame]
    filled_frames: dict[str, pd.DataFrame]
    reference_site: str
    frequencies: list[float]
    splits: dict[str, SplitArrays]
    normalization: dict[str, object] | None

    @property
    def train_split(self) -> str:
        return f"{self.reference_site}_train"

    @property
    def test_split(self) -> str:
        return f"{self.reference_site}_test"


def discover_aerpaw_file(data_dir: Path, site: str) -> Path:
    files = sorted(data_dir.glob("*.csv"))
    for path in files:
        if re.search(site, path.name.upper()):
            return path

    listed = "\n".join(f"- {path.name}" for path in files) or "- none"
    raise FileNotFoundError(
        f"Missing required AERPAW site CSV: {site}\nFiles found in {data_dir}:\n{listed}"
    )


def frequency_value(column: str) -> float:
    return float(column)


def read_site_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(frequency_value(col)) for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()


def load_aerpaw_data(
    data_dir: Path,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    normalize: bool = False,
    reference_site: str = DEFAULT_REFERENCE_SITE,
) -> LoadedSpectrumData:
    if reference_site not in SITES:
        raise ValueError(f"reference_site must be one of {SITES}, got {reference_site!r}.")

    path = discover_aerpaw_file(data_dir, reference_site)
    raw = read_site_frame(path)
    selected_cols = [
        col
        for col in sorted(raw.columns, key=frequency_value)
        if chunk_start_mhz <= frequency_value(col) <= chunk_end_mhz
    ]
    if not selected_cols:
        raise ValueError(
            f"No {reference_site} frequencies in requested chunk "
            f"{chunk_start_mhz}-{chunk_end_mhz} MHz."
        )

    raw = raw.loc[:, selected_cols].copy()
    filled = interpolate_missing(raw)
    if len(filled) <= 2880:
        raise ValueError(f"{reference_site} must have more than 2880 rows for the chronological split.")

    train_end = len(filled) - 2880
    array = filled.to_numpy(dtype=np.float32)
    model_array = array
    normalization = None
    if normalize:
        train_chunk = array[:train_end]
        mean = float(np.mean(train_chunk))
        std = float(np.std(train_chunk))
        if std == 0.0:
            raise ValueError("Cannot normalize a zero-variance chunk.")
        model_array = (array - mean) / std
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train",
        }

    splits = {
        f"{reference_site}_train": SplitArrays(
            raw_dbm=array[:train_end],
            model_input=model_array[:train_end],
            row_start=0,
            row_end=train_end - 1,
        ),
        f"{reference_site}_test": SplitArrays(
            raw_dbm=array[train_end:],
            model_input=model_array[train_end:],
            row_start=train_end,
            row_end=len(array) - 1,
        ),
    }

    return LoadedSpectrumData(
        files={reference_site: path},
        raw_frames={reference_site: raw},
        filled_frames={reference_site: filled},
        reference_site=reference_site,
        frequencies=[frequency_value(col) for col in selected_cols],
        splits=splits,
        normalization=normalization,
    )
