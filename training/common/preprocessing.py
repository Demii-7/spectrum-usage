"""
Shared spectrum-data structures, cleaning, interpolation, and normalization
utilities.

This module defines the common data containers and preprocessing operations used
by all dataset loaders, training scripts, evaluation scripts, metric functions,
and checkpoint-validation utilities.

The preprocessing functions support both frequency-vector data and spatial
spectrum maps while preserving frequency as the final axis in loader layout.

Primary responsibilities include:

- defining SplitArrays for paired raw dBm values and model-input arrays;
- defining LoadedSpectrumData for source files, frames, frequencies, splits,
  normalization metadata, and reference-site information;
- identifying the canonical training and testing split names;
- parsing frequency values from source column labels;
- reading and standardizing site-level spectrum CSV files;
- interpolating missing temporal values in individual recordings;
- cleaning partially missing map tensors;
- removing or replacing unusable map values using training-derived fallback
  statistics;
- fitting per-frequency mean and standard-deviation values;
- applying fitted normalization values to training, validation, and test data;
- preserving broadcasting behavior across vector and spatial layouts; and
- exposing metadata required for later denormalization.

Leakage prevention is a core requirement of this module. Imputation fallback
values and normalization statistics must be fitted only from the designated
training portion before validation and then applied unchanged to later data.

Expected loader layouts:

    CSV:
        (T, F)

    Map:
        (T, H, W, F)
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import ndimage


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

def frequency_value(column: str) -> float:
    return float(column)

def fit_per_frequency_normalization(
    training_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate independent normalization metrics (Mean and Standard Deviation) 
    for every frequency bin across all time steps and spatial coordinates.

    Behavior:
    1. Validates that the input data tensor has a valid structure.
    2. Identifies all leading tracking axes (Time, Lat, Long) to compress them.
    3. Calculates raw variance statistics strictly along the final frequency channel axis.
    4. Performs safety checks to guarantee no division-by-zero or infinite values exist.
    """
    # Safety Check: Data must at least have a Time axis and a Frequency axis
    if training_data.ndim < 2:
        raise ValueError(
            "Spectrum data must include time and frequency dimensions."
        )

    # Automatically identify all axes EXCEPT the very last one (the Frequency dimension).
    # For a (9000, 10, 10, 200) matrix, this creates a tuple of tracking indices: (0, 1, 2)
    reduction_axes = tuple(
        range(training_data.ndim - 1)
    )

    # Compute the average value for each frequency bin across all times and locations
    mean = np.mean(
        training_data,
        axis=reduction_axes,
    ).astype(np.float32)

    # Compute the spread/variance value for each frequency bin across all times and locations
    std = np.std(
        training_data,
        axis=reduction_axes,
    ).astype(np.float32)

    # Safety Check: Guard against corrupt dataset values like NaNs or Infs
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError(
            "Non-finite per-frequency normalization statistics."
        )

    # Safety Check: If a frequency is completely dead (variance is 0), division will crash
    if np.any(std == 0.0):
        raise ValueError(
            "Cannot normalize a zero-variance frequency."
        )

    # Returns the calculated statistics arrays (each will have a length of 200)
    return mean, std


def apply_per_frequency_normalization(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Standardize a data tensor using pre-calculated per-frequency statistics.

    Behavior:
    1. Calculates a matching placeholder broadcast shape (e.g., [1, 1, 1, 200]).
    2. Reshapes the flat 1D metric arrays to match that layout footprint.
    3. Leverages NumPy broadcasting to subtract the mean and divide by the standard
       deviation for all spatial and temporal elements simultaneously.
    """
    # Create a layout shape filled with 1s for every leading dimension, 
    # appending the explicit frequency channel count to the very end.
    # For a 4D array, this turns a flat line of 200 numbers into a (1, 1, 1, 200) shape.
    broadcast_shape = (
        (1,) * (data.ndim - 1)
        + (len(mean),)
    )

    # Perform the standardization math: (Data - Mean) / Std
    # Reshaping allows NumPy to perfectly align the 200 frequency metrics with 
    # the matching 200 channels inside every single pixel coordinate automatically.
    return (
        data
        - mean.reshape(broadcast_shape)
    ) / std.reshape(broadcast_shape)



def read_site_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(frequency_value(col)) for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()


def _fill_nearest_neighbor_2d(
    array: np.ndarray,
) -> np.ndarray:
    mask = ~np.isnan(array)

    if mask.all() or not mask.any():
        return array

    indices = ndimage.distance_transform_edt(
        (~mask).astype(np.uint8),
        return_distances=False,
        return_indices=True,
    )

    return array[tuple(indices)]


def clean_interpolated_map(
    data: np.ndarray,
    *,
    fit_rows: int | None = None,
    fallback_frequency_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean map data shaped (T, H, W, F).

    Returns:
        cleaned data
        per-frequency fallback values shaped (F,)
    """

    if data.ndim != 4:
        raise ValueError(
            "Map data must be shaped (time, height, width, frequency), "
            f"got {data.shape}"
        )

    data = np.asarray(data, dtype=np.float32).copy()

    fully_missing_timesteps = np.isnan(data).all(
        axis=(1, 2, 3)
    )
    data = data[~fully_missing_timesteps]

    if len(data) == 0:
        raise ValueError(
            "All map timesteps are entirely NaN."
        )

    time_steps, _, _, frequencies = data.shape

    # Spatial nearest-neighbour filling for each time/frequency slice.
    for time_index in range(time_steps):
        for frequency_index in range(frequencies):
            spatial_slice = data[
                time_index, :, :, frequency_index
            ]

            if np.isnan(spatial_slice).any():
                data[
                    time_index, :, :, frequency_index
                ] = _fill_nearest_neighbor_2d(
                    spatial_slice
                )

    remaining = np.isnan(data)

    if fallback_frequency_values is None:
        if fit_rows is None:
            fit_rows = len(data)

        fit_rows = min(max(int(fit_rows), 1), len(data))

        fallback_frequency_values = np.nanmean(
            data[:fit_rows],
            axis=(0, 1, 2),
        ).astype(np.float32)
    else:
        fallback_frequency_values = np.asarray(
            fallback_frequency_values,
            dtype=np.float32,
        )

    if np.isnan(fallback_frequency_values).any():
        raise ValueError(
            "At least one frequency has no usable training values "
            "for map imputation."
        )

    if remaining.any():
        broadcast_values = fallback_frequency_values.reshape(
            1, 1, 1, -1
        )
        data = np.where(
            remaining,
            broadcast_values,
            data,
        )

    if np.isnan(data).any():
        raise ValueError(
            "NaN values remain after map cleaning."
        )

    return data.astype(np.float32), fallback_frequency_values