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
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class SequenceSegment:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class SplitArrays:
    raw_dbm: np.ndarray
    model_input: np.ndarray
    row_start: int
    row_end: int
    segments: tuple[SequenceSegment, ...] = ()


@dataclass(frozen=True)
class LoadedSpectrumData:
    files: dict[str, Path]
    raw_frames: dict[str, pd.DataFrame]
    filled_frames: dict[str, pd.DataFrame]
    reference_site: str
    frequencies: list[float]
    splits: dict[str, SplitArrays]
    normalization: dict[str, object] | None
    feature_labels: list[str] | None = None

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
    allow_zero_variance: bool = False,
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
            "Error! Non-finite per-frequency normalization statistics."
        )

    # Safety Check: If a frequency is completely dead (variance is 0), division will crash
    if np.any(std == 0.0) and not allow_zero_variance:
        raise ValueError(
            "Error! Cannot normalize a zero-variance frequency."
        )
    if allow_zero_variance:
        std = np.where(std == 0.0, 1.0, std)

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
    """Vectorized 2D spatial cleaner that replaces NaN values by mapping every missing cell to its physically closest valid neighbor using a Euclidean distance transform"""
    
    # Create a boolean mask where True represents valid numbers and False represents NaNs
    mask = ~np.isnan(array)

    # Edge case protection: If the grid is completely full or 100% empty, return it as-is
    if mask.all() or not mask.any():
        return array

    # Calculate the Exact Euclidean Distance Transform (EDT).
    # We pass (~mask), targeting the NaN spaces. 
    # return_indices=True forces SciPy to output a coordinate grid mapping every single 
    # NaN cell to the (X, Y) coordinates of its physically closest non-NaN neighbor pixel.
    indices = ndimage.distance_transform_edt(
        (~mask).astype(np.uint8),
        return_distances=False,
        return_indices=True,
    )

    # Multi-dimensional indexing: Unpack the (2, H, W) coordinate map into a tuple.
    # This instantly remaps and copies the valid neighbor values into all NaN cells simultaneously.
    return array[tuple(indices)]


def clean_spectrum_data(
    data: np.ndarray,
) -> np.ndarray:
    """
    Clean spectrum data shaped either:
    
        (T, F)
        (T, H, W, F)
    
    For 4D map data, partially missing spatial slices are filled
    using nearest-neighbour imputation first.
    
    Remaining missing values are interpolated across time.
    """
    # Type enforcement: Force data into standard 32-bit floats and make a local copy
    data = np.asarray(data, dtype=np.float32).copy()

    # Shape validation: Ensure incoming data adheres strictly to the expected 2D or 4D tensor structure
    if data.ndim not in (2, 4):
        raise ValueError(
            "Error! Spectrum data must be shaped either "
            f"(T, F) or (T, H, W, F), got {data.shape}"
        )
        

    time_steps = data.shape[0]

    # Tier 1: Reject NaN in 4D spatial data
    if data.ndim == 4:
        if np.isnan(data).any():
            raise ValueError(
                "Error! Spatial map data contains NaN values. "
                "Imputation is disabled for 4D data."
            )

    # Tier 2: Temporal interpolation for remaining missing values in each flattened feature.
    # Flatten spatial and frequency dimensions into a 2D matrix [Time, Features]
    flattened_data = data.reshape(time_steps, -1)
    
    # Generate index arrays to locate valid and missing time blocks
    all_timesteps = np.arange(time_steps)
    
    # Loop through each flattened feature column to interpolate along the time axis
    for col_idx in range(flattened_data.shape[1]):
        column = flattened_data[:, col_idx]
        nan_mask = np.isnan(column)
        
        if not nan_mask.any():
            continue
            
        # If the entire column is NaN, throw error
        if nan_mask.all():
            raise ValueError(
                "Error! A complete feature column is NaN and cannot "
                "be temporally interpolated."
            )
            
        # Split into known points and targets to interpolate
        known_x = all_timesteps[~nan_mask]
        known_y = column[~nan_mask]
        
        # Interpolate internal gaps linearly and use the nearest
        # valid boundary value for gaps at the beginning or end.
        f_time = interp1d(known_x, known_y, kind='linear', bounds_error=False,fill_value=(known_y[0],known_y[-1],),)
        flattened_data[nan_mask, col_idx] = f_time(all_timesteps[nan_mask])

    # Reshape the cleaned matrix back to its original 2D or 4D shape.
    cleaned = flattened_data.reshape(data.shape)

    if not np.isfinite(cleaned).all():
        raise ValueError(
            "Error! Non-finite values remain after cleaning."
        )

    return cleaned.astype(np.float32, copy=False,)
