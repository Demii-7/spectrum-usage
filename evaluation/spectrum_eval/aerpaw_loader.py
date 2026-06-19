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
    shared_frequencies: list[float]
    splits: dict[str, SplitArrays]
    normalization: dict[str, object] | None


def discover_aerpaw_files(data_dir: Path) -> dict[str, Path]:
    files = sorted(data_dir.glob("*.csv"))
    found: dict[str, Path] = {}
    for path in files:
        name = path.name.upper()
        for site in SITES:
            if re.search(site, name):
                found[site] = path

    missing = [site for site in SITES if site not in found]
    if missing:
        listed = "\n".join(f"- {path.name}" for path in files) or "- none"
        raise FileNotFoundError(
            "Missing required AERPAW site CSV(s): "
            + ", ".join(missing)
            + f"\nFiles found in {data_dir}:\n{listed}"
        )
    return found


def frequency_value(column: str) -> float:
    return float(column)


def read_site_frames(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for site, path in files.items():
        df = pd.read_csv(path)
        df.columns = [str(frequency_value(col)) for col in df.columns]
        frames[site] = df.apply(pd.to_numeric, errors="coerce")
    return frames


def shared_frequency_columns(frames: dict[str, pd.DataFrame]) -> list[str]:
    shared = set(frames[SITES[0]].columns)
    for site in SITES[1:]:
        shared &= set(frames[site].columns)
    return sorted(shared, key=frequency_value)


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()


def missing_runs(mask: pd.Series) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in mask.to_numpy(dtype=bool):
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def missing_report(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for site, df in frames.items():
        per_bin = df.isna().sum().astype(int)
        runs: list[int] = []
        for col in df.columns:
            runs.extend(missing_runs(df[col].isna()))
        total = int(per_bin.sum())
        sample_count = int(df.shape[0] * df.shape[1])
        if runs:
            run_stats = {
                "count": int(len(runs)),
                "min": int(np.min(runs)),
                "median": float(np.median(runs)),
                "max": int(np.max(runs)),
                "fraction_samples_affected": float(total / sample_count),
            }
        else:
            run_stats = {
                "count": 0,
                "min": 0,
                "median": 0.0,
                "max": 0,
                "fraction_samples_affected": 0.0,
            }
        report[site] = {"total": total, "per_bin": per_bin, "runs": run_stats}
    return report


def load_aerpaw_data(
    data_dir: Path,
    chunk_start_mhz: float | None = None,
    chunk_end_mhz: float | None = None,
    normalize: bool = False,
    reference_site: str = DEFAULT_REFERENCE_SITE,
    normalization_site: str | None = None,
) -> LoadedSpectrumData:
    if reference_site not in SITES:
        raise ValueError(f"reference_site must be one of {SITES}, got {reference_site!r}.")
    if normalization_site is None:
        normalization_site = reference_site
    if normalization_site not in SITES:
        raise ValueError(f"normalization_site must be one of {SITES}, got {normalization_site!r}.")

    files = discover_aerpaw_files(data_dir)
    raw_frames = read_site_frames(files)
    shared_cols = shared_frequency_columns(raw_frames)
    if not shared_cols:
        raise ValueError("No shared frequency columns across AERPAW sites.")

    if chunk_start_mhz is not None and chunk_end_mhz is not None:
        shared_cols = [
            col
            for col in shared_cols
            if chunk_start_mhz <= frequency_value(col) <= chunk_end_mhz
        ]
        if not shared_cols:
            raise ValueError(
                f"No shared frequencies in requested chunk {chunk_start_mhz}-{chunk_end_mhz} MHz."
            )

    raw_frames = {site: df.loc[:, shared_cols].copy() for site, df in raw_frames.items()}
    filled_frames = {site: interpolate_missing(df) for site, df in raw_frames.items()}

    train_ends: dict[str, int] = {}
    for site, df in filled_frames.items():
        if len(df) <= 2880:
            raise ValueError(f"{site} must have more than 2880 rows for the chronological split.")
        train_ends[site] = len(df) - 2880

    arrays = {site: df.to_numpy(dtype=np.float32) for site, df in filled_frames.items()}
    normalization = None
    model_arrays = arrays
    if normalize:
        train_chunk = arrays[normalization_site][: train_ends[normalization_site]]
        mean = float(np.mean(train_chunk))
        std = float(np.std(train_chunk))
        if std == 0.0:
            raise ValueError("Cannot normalize a zero-variance chunk.")
        model_arrays = {site: (arr - mean) / std for site, arr in arrays.items()}
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": normalization_site,
            "source_split": "train" if normalization_site == reference_site else "adaptation",
        }

    splits = {}
    for site in SITES:
        train_name = "train" if site == reference_site else "adaptation"
        train_end_exclusive = train_ends[site]
        splits[f"{site}_{train_name}"] = SplitArrays(
            raw_dbm=arrays[site][:train_end_exclusive],
            model_input=model_arrays[site][:train_end_exclusive],
            row_start=0,
            row_end=train_end_exclusive - 1,
        )
        splits[f"{site}_test"] = SplitArrays(
            raw_dbm=arrays[site][train_end_exclusive:],
            model_input=model_arrays[site][train_end_exclusive:],
            row_start=train_end_exclusive,
            row_end=len(arrays[site]) - 1,
        )

    return LoadedSpectrumData(
        files=files,
        raw_frames=raw_frames,
        filled_frames=filled_frames,
        shared_frequencies=[frequency_value(col) for col in shared_cols],
        splits=splits,
        normalization=normalization,
    )
