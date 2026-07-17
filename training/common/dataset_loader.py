"""
Dataset loading and chronological split construction for integrated spectrum
prediction.

This module loads supported AERPAW and POWDER spectrum datasets, selects the
requested frequency chunk, handles missing values, constructs leakage-safe
training and testing splits, and applies optional per-frequency normalization.

Supported input representations:

AERPAW CSV:
    A site-specific spectrum recording that is split chronologically into
    training and final testing portions.

POWDER CSV:
    Separate short-training and long-test recordings containing frequency
    columns and, when chronological training extension is enabled,
    timestamp_utc metadata.

POWDER map:
    Separate NumPy archives containing map tensors shaped (T, H, W, F),
    frequency metadata, and optional timestamp metadata.

POWDER chronological extension behavior:

When data.prediction_start_row is configured, the original long test recording
is divided into two logical portions. Timestamp comparison identifies the first
test row strictly after the final short-training timestamp. Rows from that point
up to prediction_start_row are appended to training. Final evaluation receives
only rows beginning at prediction_start_row.

This construction:

- detects overlap from timestamps rather than recording lengths;
- excludes duplicated overlapping timestamps;
- prevents any row from appearing in both training and final testing;
- interprets prediction_start_row against the original long test recording; and
- performs extension before optional max-row truncation, map cleaning, or
  normalization.

Additional responsibilities include:

- discovering required AERPAW source files;
- parsing and validating CSV timestamps;
- validating map-archive timestamps and frequency metadata;
- ensuring timestamp order and uniqueness;
- requiring consistent frequency columns across files in the same split;
- requiring consistent timestamp availability across concatenated CSV files;
- interpolating each CSV source file independently;
- preventing interpolation across recording boundaries;
- validating frequency equality between train and test maps;
- removing completely missing map timesteps;
- applying identical row masks to map tensors and timestamps;
- fitting map fallback values from the training portion;
- fitting normalization statistics on training data before validation;
- applying training-derived preprocessing to final testing data; and
- returning all outputs in a common LoadedSpectrumData structure.

Loader output layouts:

    CSV:
        raw_dbm:     (T, F)
        model_input: (T, F)

    Map:
        raw_dbm:     (T, H, W, F)
        model_input: (T, H, W, F)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

from training.common.preprocessing import (
    LoadedSpectrumData,
    SplitArrays,
    apply_per_frequency_normalization,
    clean_interpolated_map,
    fit_per_frequency_normalization,
    frequency_value,
    interpolate_missing,
    read_site_frame,
)


#-----  Aerpaw Loader ------
SITES = ("CC1", "CC2", "LW1") # Aerpaw sites 

def discover_aerpaw_file(data_dir: Path, site: str) -> Path:
    files = sorted(data_dir.glob("*.csv"))
    for path in files:
        if re.search(site, path.name.upper()):
            return path

    listed = "\n".join(f"- {path.name}" for path in files) or "- none"
    raise FileNotFoundError(
        f"Missing required AERPAW site CSV: {site}\nFiles found in {data_dir}:\n{listed}"
    )
    
def load_aerpaw_data(
    data_dir: Path,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    normalize: bool = False,
    reference_site: str = "CC2",
    max_rows: int | None = None,
    test_rows: int = 2880,
    val_fraction: float = 0.1,
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
    
    if max_rows is not None:
        raw = raw.iloc[: int(max_rows)].copy()

    filled = interpolate_missing(raw)

    if filled.isna().any().any():
        bad_columns = filled.columns[
            filled.isna().any()
        ].tolist()
    
        raise ValueError(
            "Caution! NaN values remain after AERPAW interpolation in "
            f"frequencies: {bad_columns}"
        )
        
    if len(filled) <= test_rows:
        raise ValueError(f"{reference_site} must have more than {test_rows} rows for the chronological split.")

    train_end = len(filled) - test_rows
    array = filled.to_numpy(dtype=np.float32)
    model_array = array
    normalization = None
    
    if normalize:
        fit_end = int(
            train_end * (1.0 - val_fraction)
        )
    
        if fit_end <= 0:
            raise ValueError(
                "AERPAW normalization training portion is empty."
            )
    
        mean, std = fit_per_frequency_normalization(
            array[:fit_end]
        )
    
        model_array = apply_per_frequency_normalization(
            array,
            mean,
            std,
        ).astype(np.float32)

        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train_before_validation",
            "frequency_axis": -1,
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

#------ Powder Loader  --------

def _powder_numeric_frame(path: Path,require_timestamps: bool, ) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    
    df = pd.read_csv(path)

    timestamps: pd.DatetimeIndex | None = None
    
    if "timestamp_utc" in df.columns:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                df.pop("timestamp_utc"),
                utc=True,
                errors="raise",
            )
        )
    elif require_timestamps:
        raise ValueError(
            f"{path} must contain timestamp_utc when "
            "data.prediction_start_row is configured."
        )
        
    if timestamps is not None:
        if timestamps.hasnans:
            raise ValueError(
                f"{path} contains missing timestamps."
            )
    
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"{path} timestamps are not chronologically ordered."
            )
    
        if timestamps.has_duplicates:
            raise ValueError(
                f"{path} contains duplicate timestamps."
            )
            
    df.columns = [
        str(frequency_value(col))
        for col in df.columns
    ]

    numeric = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    return numeric, timestamps


def _selected_columns(columns: Iterable[str], chunk_start_mhz: float, chunk_end_mhz: float) -> list[str]:
    return [
        col
        for col in sorted(columns, key=frequency_value)
        if chunk_start_mhz <= frequency_value(col) <= chunk_end_mhz
    ]


def _load_split_frame(
    paths: list[Path],
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    require_timestamps: bool,
) -> tuple[ pd.DataFrame, pd.DatetimeIndex | None, dict[str, pd.DataFrame],]:
    
    if not paths:
        raise ValueError(
            "POWDER loader requires at least one "
            "CSV path per split."
        )

    frames: list[pd.DataFrame] = []
    timestamp_parts: list[
        pd.DatetimeIndex
    ] = []

    per_file_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    expected_cols: list[str] | None = None
    timestamp_presence: bool | None = None

    for path in paths:
        raw, timestamps = _powder_numeric_frame(
            path,
            require_timestamps=require_timestamps,
        )
        current_has_timestamps = (
            timestamps is not None
        )
        
        if timestamp_presence is None:
            timestamp_presence = current_has_timestamps
        elif current_has_timestamps != timestamp_presence:
            raise ValueError(
                "POWDER files inside the same split must either "
                "all contain timestamp_utc or all omit it. "
                f"Timestamp availability differs at {path}."
            )

        selected_cols = _selected_columns(
            raw.columns,
            chunk_start_mhz,
            chunk_end_mhz,
        )

        if not selected_cols:
            raise ValueError(
                f"No POWDER frequencies in requested chunk "
                f"{chunk_start_mhz}-{chunk_end_mhz} MHz "
                f"for {path}."
            )

        chunk_frame = raw.loc[
            :,
            selected_cols,
        ].copy()

        current_cols = list(
            chunk_frame.columns
        )

        if expected_cols is None:
            expected_cols = current_cols
        elif current_cols != expected_cols:
            raise ValueError(
                "POWDER files for the same split must "
                "share the same frequency columns. "
                f"Mismatch at {path}."
            )

        # Interpolate inside the source recording so values are never
        # interpolated across separate file boundaries.
        chunk_filled = interpolate_missing(
            chunk_frame
        )

        if chunk_filled.isna().any().any():
            raise ValueError(
                f"NaN values remain in {path} after "
                "temporal interpolation."
            )

        frames.append(chunk_filled)
        if timestamps is not None:
            timestamp_parts.append(timestamps)
        per_file_frames[path.stem] = (
            chunk_filled
        )

    combined_frame = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    combined_timestamps: pd.DatetimeIndex | None = None

    if timestamp_parts:
        combined_timestamps = pd.DatetimeIndex(
            np.concatenate(
                [
                    timestamps.to_numpy()
                    for timestamps
                    in timestamp_parts
                ]
            )
        )
    
        if not combined_timestamps.is_monotonic_increasing:
            raise ValueError(
                "POWDER files inside the same split are not "
                "ordered as one chronological sequence."
            )
    
        if combined_timestamps.has_duplicates:
            raise ValueError(
                "POWDER files inside the same split contain "
                "duplicate timestamps."
            )

    return (
        combined_frame,
        combined_timestamps,
        per_file_frames,
    )

#--- Powder CSV Loader ---
def load_powder_data(
    train_files: list[Path],
    test_files: list[Path],
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    normalize: bool = False,
    reference_site: str = "POWDER",
    val_fraction: float = 0.1,
    max_rows: int | None = None,
    prediction_start_row: int | None = None,
) -> LoadedSpectrumData:
    
    require_timestamps = (
        prediction_start_row is not None
    )
    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError(
            "data.max_rows must be positive when provided."
        )
    
    train_filled, train_timestamps, train_frames = (
        _load_split_frame(
            train_files,
            chunk_start_mhz,
            chunk_end_mhz,
            require_timestamps=require_timestamps,
        )
    )
    
    test_filled, test_timestamps, test_frames = (
        _load_split_frame(
            test_files,
            chunk_start_mhz,
            chunk_end_mhz,
            require_timestamps=require_timestamps,
        )
    )

    extension_indices = None

    if prediction_start_row is not None:
        if (
            train_timestamps is None
            or test_timestamps is None
        ):
            raise ValueError(
                "Training extension requires timestamps "
                "for both training and test recordings."
            )
    
        extension_indices = (
            _prediction_extension_indices(
                train_timestamps=train_timestamps,
                test_timestamps=test_timestamps,
                prediction_start_row=prediction_start_row,
            )
        )
    
    if extension_indices is not None:
        (
            continuation_start_index,
            prediction_start_index,
        ) = extension_indices
    
        borrowed_test_rows = test_filled.iloc[
            continuation_start_index:
            prediction_start_index
        ].copy()
    
        if borrowed_test_rows.empty:
            raise ValueError(
                "No rows were selected from the long test "
                "recording to extend training."
            )
    
        train_filled = pd.concat(
            [
                train_filled,
                borrowed_test_rows,
            ],
            axis=0,
            ignore_index=True,
        )
    
        train_timestamps = pd.DatetimeIndex(
            np.concatenate(
                [
                    train_timestamps.to_numpy(),
                    test_timestamps[
                        continuation_start_index:
                        prediction_start_index
                    ].to_numpy(),
                ]
            )
        )
    
        # The returned test split now contains only rows that were not
        # included in training.
        test_filled = test_filled.iloc[
            prediction_start_index:
        ].reset_index(
            drop=True
        )
    
        test_timestamps = test_timestamps[
            prediction_start_index:
        ]
        
    if max_rows is not None:
        limit = int(max_rows)
    
        train_filled = train_filled.iloc[
            :limit
        ].copy()
    
        test_filled = test_filled.iloc[
            :limit
        ].copy()
    
        if train_timestamps is not None:
            train_timestamps = train_timestamps[
                :limit
            ]
    
        if test_timestamps is not None:
            test_timestamps = test_timestamps[
                :limit
            ]
    
    if train_filled.empty or test_filled.empty:
        raise ValueError(
            "POWDER train/test splits must both contain "
            "at least one row after chronological construction."
        )
    
    # Validate CSV interpolation
    if train_filled.isna().any().any():
        raise ValueError(
            "NaN values remain in the POWDER training CSV "
            "after interpolation."
        )
    
    if test_filled.isna().any().any():
        raise ValueError(
            "NaN values remain in the POWDER test CSV "
            "after interpolation."
        )
        

    train_array = train_filled.to_numpy(dtype=np.float32)
    test_array = test_filled.to_numpy(dtype=np.float32)
    train_model = train_array
    test_model = test_array
    normalization = None
    
    if normalize:
        fit_end = int(
            len(train_array) * (1.0 - val_fraction)
        )
    
        if fit_end <= 0:
            raise ValueError(
                "POWDER normalization training portion is empty."
            )
    
        mean, std = fit_per_frequency_normalization(
            train_array[:fit_end]
        )
    
        train_model = apply_per_frequency_normalization(
            train_array,
            mean,
            std,
        ).astype(np.float32)
    
        test_model = apply_per_frequency_normalization(
            test_array,
            mean,
            std,
        ).astype(np.float32)
    
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train_before_validation",
            "frequency_axis": -1,
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

# --- Powder Map Loader ---

def _load_powder_map(
    path: Path,
    map_key: str,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    max_rows: int | None,
    require_timestamps: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex | None]:
    
    with np.load(path) as archive:
        if map_key not in archive:
            raise KeyError(
                f"{path} does not contain map key {map_key!r}"
            )
    
        if "freqs_mhz" not in archive:
            raise KeyError(
                f"{path} does not contain 'freqs_mhz'"
            )
    
        map_data = np.asarray(
            archive[map_key],
            dtype=np.float32,
        )
    
        frequencies = np.asarray(
            archive["freqs_mhz"],
            dtype=np.float32,
        )
            
        timestamps: pd.DatetimeIndex | None = None
        
        if "timestamps" in archive:
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(
                    archive["timestamps"],
                    utc=True,
                    errors="raise",
                )
            )
        elif require_timestamps:
            raise KeyError(
                f"{path} does not contain 'timestamps'. "
                "Timestamp metadata is required when "
                "data.prediction_start_row is configured."
            )
    
    if timestamps is not None:
        if len(timestamps) != len(map_data):
            raise ValueError(
                f"{path} timestamp count does not match "
                "the map time dimension."
            )
    
        if timestamps.hasnans:
            raise ValueError(
                f"{path} contains missing timestamps."
            )
    
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"{path} timestamps are not chronologically ordered."
            )
    
        if timestamps.has_duplicates:
            raise ValueError(
                f"{path} contains duplicate timestamps."
            )
        
    if map_data.ndim != 4:
        raise ValueError(
            f"{path} map must be shaped (T, H, W, F), "
            f"got {map_data.shape}"
        )

    if map_data.shape[-1] != len(frequencies):
        raise ValueError(
            f"{path} frequency count does not match map channels."
        )

    selected = (
        (frequencies >= chunk_start_mhz)
        & (frequencies <= chunk_end_mhz)
    )

    if not selected.any():
        raise ValueError(
            f"No frequencies from {chunk_start_mhz} to "
            f"{chunk_end_mhz} MHz in {path}"
        )

    map_data = map_data[..., selected]
    frequencies = frequencies[selected]

    if max_rows is not None:
        limit = int(max_rows)
        map_data = map_data[:limit]
    
        if timestamps is not None:
            timestamps = timestamps[:limit]
            

    return map_data, frequencies, timestamps

def load_powder_map_data(
    train_path: Path,
    test_path: Path,
    map_key: str,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    normalize: bool = False,
    reference_site: str = "POWDER",
    max_rows: int | None = None,
    val_fraction: float = 0.1,
    prediction_start_row: int | None = None,
) -> LoadedSpectrumData:
    
    require_timestamps = (
        prediction_start_row is not None
    )
    
    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError(
            "data.max_rows must be positive when provided."
        )
    train_raw, train_freqs, train_timestamps = _load_powder_map(
        train_path,
        map_key,
        chunk_start_mhz,
        chunk_end_mhz,
        None,
        require_timestamps=require_timestamps,
    )

    test_raw, test_freqs, test_timestamps = _load_powder_map(
        test_path,
        map_key,
        chunk_start_mhz,
        chunk_end_mhz,
        None,
        require_timestamps=require_timestamps,
    )

    if not np.array_equal(train_freqs, test_freqs):
        raise ValueError(
            "POWDER train and test maps have different frequencies."
        )

    extension_indices = None
    
    if prediction_start_row is not None:
        if (
            train_timestamps is None
            or test_timestamps is None
        ):
            raise ValueError(
                "Map training extension requires timestamps "
                "for both training and test recordings."
            )
    
        extension_indices = (
            _prediction_extension_indices(
                train_timestamps=train_timestamps,
                test_timestamps=test_timestamps,
                prediction_start_row=prediction_start_row,
            )
        )
    
    if extension_indices is not None:
        (
            continuation_start_index,
            prediction_start_index,
        ) = extension_indices
    
        borrowed_test_rows = test_raw[
            continuation_start_index:
            prediction_start_index
        ]
    
        if len(borrowed_test_rows) == 0:
            raise ValueError(
                "No map rows were selected from the long test "
                "recording to extend training."
            )
    
        train_raw = np.concatenate(
            [
                train_raw,
                borrowed_test_rows,
            ],
            axis=0,
        )
    
        train_timestamps = pd.DatetimeIndex(
            np.concatenate(
                [
                    train_timestamps.to_numpy(),
                    test_timestamps[
                        continuation_start_index:
                        prediction_start_index
                    ].to_numpy(),
                ]
            )
        )
    
        # Final testing begins at prediction_start_index.
        test_raw = test_raw[
            prediction_start_index:
        ]
    
        test_timestamps = test_timestamps[
            prediction_start_index:
        ]
    
    
    # Apply smoke-test truncation only after the final train and test
    # sequences have been constructed.
    if max_rows is not None:
        limit = int(max_rows)
    
        train_raw = train_raw[
            :limit
        ]
    
        test_raw = test_raw[
            :limit
        ]
    
        if train_timestamps is not None:
            train_timestamps = train_timestamps[
                :limit
            ]
    
        if test_timestamps is not None:
            test_timestamps = test_timestamps[
                :limit
            ]

    if len(train_raw) == 0 or len(test_raw) == 0:
        raise ValueError(
            "POWDER map train/test splits must both contain "
            "at least one timestep after chronological construction."
        )
        
    # Remove unusable timesteps before calculating the train/validation
    # boundary so that cleaning cannot shift validation rows into the
    # normalization/imputation fitting portion.
    train_valid_rows = ~np.isnan(train_raw).all(axis=(1, 2, 3))
    test_valid_rows = ~np.isnan(test_raw).all(axis=(1, 2, 3))

    train_raw = train_raw[train_valid_rows]
    test_raw = test_raw[test_valid_rows]

    if train_timestamps is not None:
        train_timestamps = train_timestamps[
            train_valid_rows
        ]
    
    if test_timestamps is not None:
        test_timestamps = test_timestamps[
            test_valid_rows
        ]
        

    if len(train_raw) == 0:
        raise ValueError(
            "POWDER training map contains no usable timesteps."
        )

    if len(test_raw) == 0:
        raise ValueError(
            "POWDER test map contains no usable timesteps."
        )

    fit_end = int(
        len(train_raw) * (1.0 - val_fraction)
    )

    if fit_end <= 0 or fit_end >= len(train_raw):
        raise ValueError(
            "POWDER map train/validation split leaves an empty "
            "training or validation portion."
        )

    train_clean, fallback_values = clean_interpolated_map(
        train_raw,
        fit_rows=fit_end,
    )

    test_clean, _ = clean_interpolated_map(
        test_raw,
        fallback_frequency_values=fallback_values,
    )

    train_model = train_clean
    test_model = test_clean
    normalization = None

    if normalize:
        mean, std = fit_per_frequency_normalization(
            train_clean[:fit_end]
        )

        train_model = apply_per_frequency_normalization(
            train_clean,
            mean,
            std,
        ).astype(np.float32)

        test_model = apply_per_frequency_normalization(
            test_clean,
            mean,
            std,
        ).astype(np.float32)

        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train_before_validation",
            "frequency_axis": -1,
        }

    train_len = len(train_clean)
    total_len = train_len + len(test_clean)

    return LoadedSpectrumData(
        files={
            "train_map": train_path,
            "test_map": test_path,
        },
        raw_frames={},
        filled_frames={},
        reference_site=reference_site,
        frequencies=[
            float(value)
            for value in train_freqs
        ],
        splits={
            f"{reference_site}_train": SplitArrays(
                raw_dbm=train_clean,
                model_input=train_model,
                row_start=0,
                row_end=train_len - 1,
            ),
            f"{reference_site}_test": SplitArrays(
                raw_dbm=test_clean,
                model_input=test_model,
                row_start=train_len,
                row_end=total_len - 1,
            ),
        },
        normalization=normalization,
    )



# ---- Other Helper ---

def _prediction_extension_indices(
    *,
    train_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
    prediction_start_row: int | None,
) -> tuple[int, int] | None:
    """
    Resolve the portion of the original long test recording that should
    extend training.

    Returns:
        continuation_start_index
        prediction_start_index

    Both returned indices are zero-based indices into the original test
    recording.

    continuation_start_index:
        First test row strictly after the final short-training timestamp.

    prediction_start_index:
        First row reserved for final testing.
    """

    if prediction_start_row is None:
        return None

    if len(train_timestamps) == 0:
        raise ValueError(
            "Cannot extend training from an empty training recording."
        )

    if len(test_timestamps) == 0:
        raise ValueError(
            "Cannot extend training from an empty test recording."
        )

    if not train_timestamps.is_monotonic_increasing:
        raise ValueError(
            "Training timestamps must be chronologically ordered."
        )

    if not test_timestamps.is_monotonic_increasing:
        raise ValueError(
            "Test timestamps must be chronologically ordered."
        )

    if train_timestamps.has_duplicates:
        raise ValueError(
            "Training timestamps contain duplicate rows."
        )

    if test_timestamps.has_duplicates:
        raise ValueError(
            "Test timestamps contain duplicate rows."
        )

    # Config uses one-based row numbering.
    prediction_start_index = (
        int(prediction_start_row) - 1
    )

    if prediction_start_index <= 0:
        raise ValueError(
            "data.prediction_start_row must leave at least "
            "one original test row before the final test split."
        )

    if prediction_start_index >= len(
        test_timestamps
    ):
        raise ValueError(
            "data.prediction_start_row is beyond the "
            f"original test recording. Got row "
            f"{prediction_start_row}, but the test recording "
            f"has {len(test_timestamps)} rows."
        )

    final_train_timestamp = train_timestamps[-1]

    # The continuation starts at the first test timestamp strictly later
    # than the final short-training timestamp. This removes the overlap.
    continuation_start_index = int(
        test_timestamps.searchsorted(
            final_train_timestamp,
            side="right",
        )
    )

    if continuation_start_index == 0:
        raise ValueError(
            "The configured training-extension behavior expects "
            "the training and test recordings to overlap, but the "
            "test recording begins after the training recording ends."
        )

    if continuation_start_index >= len(
        test_timestamps
    ):
        raise ValueError(
            "The long test recording contains no timestamps "
            "after the short training recording ends."
        )

    if (
        continuation_start_index
        >= prediction_start_index
    ):
        continuation_timestamp = (
            test_timestamps[
                continuation_start_index
            ]
            if continuation_start_index
            < len(test_timestamps)
            else None
        )

        prediction_timestamp = test_timestamps[
            prediction_start_index
        ]

        raise ValueError(
            "data.prediction_start_row does not leave any "
            "non-overlapping test rows to extend training. "
            f"Continuation begins at test index "
            f"{continuation_start_index} "
            f"({continuation_timestamp}), while final testing "
            f"begins at test index {prediction_start_index} "
            f"({prediction_timestamp})."
        )

    return (
        continuation_start_index,
        prediction_start_index,
    )