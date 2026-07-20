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
- validating frequency equality between POWDER train and test data;
- validating timestamp availability when chronological extension is enabled;
- cleaning POWDER train and test recordings independently;
- preventing preprocessing across the train/test recording boundary;
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
import re

import numpy as np
import pandas as pd

from training.common.preprocessing import (
    LoadedSpectrumData,
    SplitArrays,
    apply_per_frequency_normalization,
    clean_spectrum_data,
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

def load_powder_data(
    train_path: Path,
    test_path: Path,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    input_format: str,
    map_key: str = "map_db",
    normalize: bool = False,
    reference_site: str = "POWDER",
    max_rows: int | None = None,
    val_fraction: float = 0.1,
    prediction_start_row: int | None = None,
    impute: bool = False,
) -> LoadedSpectrumData:
    """
        Loads and chronologically aligns POWDER 2D and 4D  data from train and test paths. 
        Handles time-sequence extensions, optional spatial-temporal imputation, and per-frequency channel standardization. 
        Returns a structured data container holding unscaled raw metrics and processed model inputs partitioned by split.

    """
    input_format = str(input_format).lower()

    if input_format not in {"csv", "map"}:
        raise ValueError(
            "Error! input_format must be either 'csv' or 'map'."
        )

    if chunk_start_mhz > chunk_end_mhz:
        raise ValueError(
            "Error! chunk_start_mhz cannot be greater than "
            "chunk_end_mhz."
        )

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "Error! val_fraction must be strictly between 0 and 1."
        )

    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError(
            "Error! data.max_rows must be positive when provided."
        )

    require_timestamps = (
        prediction_start_row is not None
    )
    
    
    # Load either 2D CSV spectrum data or 4D map spectrum data.
    if input_format == "csv":
        train_raw_frame, train_freqs, train_timestamps = _load_powder_csv(
            path=train_path,
            chunk_start_mhz=chunk_start_mhz,
            chunk_end_mhz=chunk_end_mhz,
            require_timestamps=require_timestamps,
        )

        test_raw_frame, test_freqs, test_timestamps = _load_powder_csv(
            path=test_path,
            chunk_start_mhz=chunk_start_mhz,
            chunk_end_mhz=chunk_end_mhz,
            require_timestamps=require_timestamps,
        )

        train_raw = train_raw_frame.to_numpy(dtype=np.float32)

        test_raw = test_raw_frame.to_numpy(dtype=np.float32)

    else:
        train_raw, train_freqs, train_timestamps = (
            _load_powder_map(
                path=train_path,
                map_key=map_key,
                chunk_start_mhz=chunk_start_mhz,
                chunk_end_mhz=chunk_end_mhz,
                require_timestamps=require_timestamps,
            )
        )

        test_raw, test_freqs, test_timestamps = (
            _load_powder_map(
                path=test_path,
                map_key=map_key,
                chunk_start_mhz=chunk_start_mhz,
                chunk_end_mhz=chunk_end_mhz,
                require_timestamps=require_timestamps,
            )
        )

        train_raw_frame = None
        test_raw_frame = None
    
    # Ensure that train and test sets' channels align
    if not np.array_equal(train_freqs, test_freqs):
        raise ValueError(
            " Error! POWDER train and test recordings have different frequencies."
        )
        
    #===============================================================================================================
    # The following code is ONLY for STS-PredNet where we are using a subset of testing data to extend training set
    #===============================================================================================================
    extension_indices = None
    
    # Throw an error if timesteps are missing: required for alignment
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
    # Check if indices were successfully extracted
    if extension_indices is not None:
        (
            continuation_start_index,
            prediction_start_index,
        ) = extension_indices
        # Check to ensure start point is before end
        if ( continuation_start_index < prediction_start_index ):
            
            # Extrat subset of test set
            borrowed_test_rows = test_raw[
                continuation_start_index:
                prediction_start_index
            ]
            # Concatenate extension to train set along time axis
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
        test_raw = test_raw[prediction_start_index:]

    
        test_timestamps = test_timestamps[prediction_start_index:]
    #===============================================================================================================
    #===============================================================================================================
    
    # Apply smoke-test truncation only after the final train and test
    # sequences have been constructed.
    # Max rows allows for taking a shorter duration of the dataset
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
            
    if input_format == "csv":
        train_raw_frame = pd.DataFrame(
            train_raw,
            columns=[
                str(float(value))
                for value in train_freqs
            ],
        )

        test_raw_frame = pd.DataFrame(
            test_raw,
            columns=[
                str(float(value))
                for value in test_freqs
            ],
        )
        
    if len(train_raw) == 0 or len(test_raw) == 0:
        raise ValueError(
            "Error! POWDER  train/test splits must both contain "
            "at least one timestep after chronological construction."
        )
        
    # Calculate the train/validation boundary
    fit_end = int(len(train_raw) * (1.0 - val_fraction))

    if fit_end <= 0 or fit_end >= len(train_raw):
        raise ValueError(
            "POWDER  train/validation split leaves an empty "
            "training or validation portion."
        )
    
    # Impute Missing Values if imputation is enabled
    # Store the data after optional imputation but before normalization.
    if impute:
        # Clean train and test independently so imputation never crosses
        # the boundary between the two recordings.
        train_filled = clean_spectrum_data(
            train_raw
        )

        test_filled = clean_spectrum_data(
            test_raw
        )
    else:
        # No filling was requested, so the filled stage is identical
        # to the raw stage.
        train_filled = train_raw
        test_filled = test_raw

    train_filled = np.asarray(
        train_filled,
        dtype=np.float32,
    )

    test_filled = np.asarray(
        test_filled,
        dtype=np.float32,
    )
    
    
    if not np.isfinite(train_filled).all():
        raise ValueError(
            "Error! Non-finite values remain in the POWDER "
            "training data after preprocessing."
        )

    if not np.isfinite(test_filled).all():
        raise ValueError(
            "Error! Non-finite values remain in the POWDER "
            "testing data after preprocessing."
        )

    # Model input begins as the filled, unnormalized representation.
    train_model = train_filled
    test_model = test_filled
        
    # Create dictionary to store values used for normalization
    normalization = None

    if normalize:

        # Calculate mean and std for training set only
        mean, std = fit_per_frequency_normalization(
            train_model[:fit_end]
        )
        # Debugging
        print("Debugging normalization: ", mean.mean(), std.mean())
        
        # Normalize both train and val set with training set mean/std
        train_model = apply_per_frequency_normalization(
            train_model,
            mean,
            std,
        ).astype(np.float32)
        
        # Normalize test set with training set mean/std
        test_model = apply_per_frequency_normalization(
            test_model,
            mean,
            std,
        ).astype(np.float32)
        
        # Store normalization info in dictionary
        normalization = {
            "mean_dbm": mean,
            "std_dbm": std,
            "site": reference_site,
            "source_split": "train_before_validation",
            "frequency_axis": -1,
        }

    train_len = len(train_model)
    test_len = len(test_model)

    # Build representation-specific inspection metadata.
    if input_format == "csv":
        raw_frames = {
            "train": train_raw_frame,
            "test": test_raw_frame,
        }

        filled_frames = {
            "train": pd.DataFrame(
                train_filled,
                columns=train_raw_frame.columns,
            ),
            "test": pd.DataFrame(
                test_filled,
                columns=test_raw_frame.columns,
            ),
        }

        files = {
            "train_csv": train_path,
            "test_csv": test_path,
        }

    else:
        raw_frames = {
            "train": train_raw,
            "test": test_raw,
        }

        filled_frames = {
            "train": train_filled,
            "test": test_filled,
        }

        files = {
            "train_map": train_path,
            "test_map": test_path,
        }

    return LoadedSpectrumData(
        files=files,
        raw_frames=raw_frames,
        filled_frames=filled_frames,
        reference_site=reference_site,
        frequencies=[
            float(value)
            for value in train_freqs
        ],
        splits={
            f"{reference_site}_train": SplitArrays(
                raw_dbm=train_raw,
                model_input=train_model,
                row_start=0,
                row_end=train_len - 1,
            ),
            f"{reference_site}_test": SplitArrays(
                raw_dbm=test_raw,
                model_input=test_model,
                row_start=0,
                row_end=test_len - 1,
            ),
        },
        normalization=normalization,
    )

def _load_powder_csv(
    path: Path,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    require_timestamps: bool,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DatetimeIndex | None,
]:
    """
    Load one POWDER CSV recording.

    The function:

    1. Reads the CSV file.
    2. Extracts and validates timestamp metadata when available.
    3. Converts frequency-column names into a consistent numeric format.
    4. Converts spectrum values to numeric values.
    5. Selects the requested frequency chunk.
    6. Returns the selected spectrum frame, frequency metadata,
       and timestamps.

    Expected CSV structure:

        timestamp_utc, 600.0, 600.8, 601.6, ...

    The returned spectrum frame has shape:

        (T, F)

    where:

        T = number of timesteps
        F = number of selected frequency bins
    """

    # Ensure that the requested frequency range is valid.
    if chunk_start_mhz > chunk_end_mhz:
        raise ValueError(
            "Error! chunk_start_mhz cannot be greater than "
            "chunk_end_mhz."
        )

    # Ensure that the source file exists before attempting to read it.
    if not path.exists():
        raise FileNotFoundError(
            f"Error! POWDER CSV file does not exist: {path}"
        )

    # Read the complete CSV file.
    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(
            f"Error! POWDER CSV file is empty: {path}"
        )

    # Timestamp metadata is optional unless the calling workflow
    # explicitly requires it, such as chronological training extension.
    timestamps: pd.DatetimeIndex | None = None

    if "timestamp_utc" in frame.columns:
        # Remove the timestamp column from the spectrum data and parse it
        # separately as timezone-aware UTC datetime values.
        timestamp_values = frame.pop(
            "timestamp_utc"
        )

        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                timestamp_values,
                utc=True,
                errors="raise",
            )
        )

    elif require_timestamps:
        raise ValueError(
            f"Error! {path} does not contain timestamp_utc. "
            "Timestamp metadata is required for this loading mode."
        )

    # Validate timestamp metadata whenever it is available.
    if timestamps is not None:
        # Every spectrum row must have exactly one corresponding timestamp.
        if len(timestamps) != len(frame):
            raise ValueError(
                f"Error! {path} timestamp count does not match "
                "the number of spectrum rows."
            )

        if timestamps.hasnans:
            raise ValueError(
                f"Error! {path} contains missing timestamps."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"Error! {path} timestamps are not "
                "chronologically ordered."
            )

        if timestamps.has_duplicates:
            raise ValueError(
                f"Error! {path} contains duplicate timestamps."
            )

    # A valid spectrum CSV must contain at least one frequency column
    # after removing timestamp metadata.
    if frame.shape[1] == 0:
        raise ValueError(
            f"Error! {path} does not contain frequency columns."
        )

    # Convert every frequency-column label into one consistent string
    # representation. frequency_value() is responsible for extracting
    # the numeric MHz value from the original column label.
    normalized_columns = [str(frequency_value(column))for column in frame.columns]

    # Multiple original labels must not resolve to the same normalized
    # frequency because that would create ambiguous duplicate columns.
    if len(set(normalized_columns)) != len( normalized_columns):
        raise ValueError(
            f"Error! {path} contains duplicate frequency columns "
            "after frequency-label normalization."
        )

    frame.columns = normalized_columns

    # Convert spectrum values into numeric values.
    #
    # Invalid entries become NaN so that the shared preprocessing
    # function can handle them later.
    numeric_frame = frame.apply( pd.to_numeric, errors="coerce",)

    # Sort all available frequency columns numerically and select only
    # the frequencies inside the requested inclusive chunk.
    selected_columns = [
        column
        for column in sorted(
            numeric_frame.columns,
            key=frequency_value,
        )
        if (
            chunk_start_mhz
            <= frequency_value(column)
            <= chunk_end_mhz
        )
    ]

    if not selected_columns:
        raise ValueError(
            f"Error! No frequencies from {chunk_start_mhz} to "
            f"{chunk_end_mhz} MHz were found in {path}."
        )

    # Preserve only the selected frequencies in ascending frequency order.
    selected_frame = numeric_frame.loc[:,selected_columns,].copy()

    # Reset the row index so that row positions directly correspond to
    # chronological timestep positions from 0 to T - 1.
    selected_frame = selected_frame.reset_index(drop=True)

    # Store the selected frequency metadata in the same order as the
    # columns in selected_frame.
    frequencies = np.asarray(
        [
            frequency_value(column)
            for column in selected_columns
        ], dtype=np.float32,
    )

    # Verify that the number of returned frequency values matches the
    # number of selected data columns.
    if selected_frame.shape[1] != len(
        frequencies
    ):
        raise ValueError(
            f"Error! {path} frequency metadata count does not "
            "match the selected CSV columns."
        )

    return (
        selected_frame,
        frequencies,
        timestamps,
    )


def _load_powder_map(
    path: Path,
    map_key: str,
    chunk_start_mhz: float,
    chunk_end_mhz: float,
    require_timestamps: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex | None]:
    """ Loads Freq, Timestamp and (T, H, W, F) data from npz file, applies chunking and returns map data, frequency and timestamps """
    
    # Load the npz file: ['map_db','timestamps', 'freqs_mhz', 'lon_grid', 'lat_grid', 
    #'site_data_db','raw_site_data_db','site_lons','site_lats','site_names','metadata']
    with np.load(path) as archive:
        
        # Check if map key : map_db is in archive, map_db is our data (T, H, W, F)
        if map_key not in archive:
            raise KeyError(
                f"{path} does not contain map key {map_key!r}"
            )
            
        # Check if freqs_mhz is in archive. freqs_mhz: stores all freq channels
        if "freqs_mhz" not in archive:
            raise KeyError(
                f"{path} does not contain 'freqs_mhz'"
            )
        # Load data (T, H, W, F)
        map_data = np.asarray(
            archive[map_key],
            dtype=np.float32,
        )
        # Load all freq channels
        frequencies = np.asarray(
            archive["freqs_mhz"],
            dtype=np.float32,
        )

        # Creates timestamp object and load timestamps if present in archive
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
                f"Error! {path} does not contain 'timestamps'. "
                "Timestamp metadata is required when "
                "data.prediction_start_row is configured."
            )

    # Check for inconsistency between timestamps stored and the amount of data we have
    if timestamps is not None:
        if len(timestamps) != len(map_data):
            raise ValueError(
                f"Error! {path} timestamp count does not match "
                "the map time dimension."
            )
    
        if timestamps.hasnans:
            raise ValueError(
                f"Error! {path} contains missing timestamps."
            )
    
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"Error! {path} timestamps are not chronologically ordered."
            )
    
        if timestamps.has_duplicates:
            raise ValueError(
                f"Error! {path} contains duplicate timestamps."
            )
    # Check for dimension consistency    
    if map_data.ndim != 4:
        raise ValueError(
            f"Error! {path} map must be shaped (T, H, W, F), "
            f"got {map_data.shape}"
        )

    if map_data.shape[-1] != len(frequencies):
        raise ValueError(
            f"Error! {path} frequency count does not match map channels."
        )
    
    # Select Chunks we are insteretsed in based and start/end freq criteria
    selected = (
        (frequencies >= chunk_start_mhz)
        & (frequencies <= chunk_end_mhz)
    )

    if not selected.any():
        raise ValueError(
            f"Error! No frequencies from {chunk_start_mhz} to "
            f"{chunk_end_mhz} MHz in {path}"
        )
    # Extract those specific freqencies
    selected_frequencies = frequencies[selected]

    if len(
        np.unique(selected_frequencies)
    ) != len(selected_frequencies):
        raise ValueError(
            f"Error! {path} contains duplicate frequency "
            "channels in the requested chunk."
        )

    frequency_order = np.argsort(
        selected_frequencies
    )

    map_data = map_data[
        ..., selected
    ][
        ..., frequency_order
    ]

    frequencies = selected_frequencies[
        frequency_order
    ]

    # Returns map data, frequency and timestamps
    return map_data, frequencies, timestamps

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
    dataset.

    continuation_start_index:
        First test row strictly after the final short-training timestamp.

    prediction_start_index:
        First row reserved for final testing.
    """

    if prediction_start_row is None:
        return None
        
    # Check for inconsistency between timestamps stored and the amount of data we have
    if len(train_timestamps) == 0:
        raise ValueError(
            "Error! Cannot extend training from an empty training recording."
        )

    if len(test_timestamps) == 0:
        raise ValueError(
            "Error! Cannot extend training from an empty test recording."
        )

    if not train_timestamps.is_monotonic_increasing:
        raise ValueError(
            "Error! Training timestamps must be chronologically ordered."
        )

    if not test_timestamps.is_monotonic_increasing:
        raise ValueError(
            "Error! Test timestamps must be chronologically ordered."
        )

    if train_timestamps.has_duplicates:
        raise ValueError(
            "Error! Training timestamps contain duplicate rows."
        )

    if test_timestamps.has_duplicates:
        raise ValueError(
            "Error! Test timestamps contain duplicate rows."
        )

    # Config uses one-based row numbering.
    prediction_start_index = int(prediction_start_row) - 1
    
    # Check for invalid index
    if prediction_start_index <= 0:
        raise ValueError(
            "Error! data.prediction_start_row must leave at least "
            "one original test row before the final test split."
        )

    if prediction_start_index >= len(
        test_timestamps
    ):
        raise ValueError(
            "Error! data.prediction_start_row is beyond the "
            f"original test recording. Got row "
            f"{prediction_start_row}, but the test recording "
            f"has {len(test_timestamps)} rows."
        )
    
    # Store final timestep in training data
    final_train_timestamp = train_timestamps[-1]

    # The continuation starts at the first test timestamp strictly later
    # than the final short-training timestamp. This removes the overlap.
    continuation_start_index = int(
        test_timestamps.searchsorted(
            final_train_timestamp,
            side="right",
        )
    )
    
    # Check for invalid index
    if continuation_start_index == 0:
        raise ValueError(
            "Error! The configured training-extension behavior expects "
            "the training and test recordings to overlap, but the "
            "test recording begins after the training recording ends."
        )

    if continuation_start_index >= len(
        test_timestamps
    ):
        raise ValueError(
            "Error! The long test recording contains no timestamps "
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
            "Error! data.prediction_start_row does not leave any "
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