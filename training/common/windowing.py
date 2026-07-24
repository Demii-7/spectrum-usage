"""
Shared temporal-window construction and model-layout conversion utilities.

This module creates aligned lookback and target windows for training,
validation, and evaluation. It supports both frequency-vector recordings and
spatial spectrum-map recordings through a common time-first dataset interface.

Loader arrays use frequency-last map layout, while map models use
frequency-first channel layout. The conversion functions in this module provide
the authoritative boundary between those representations.

Primary responsibilities include:

- calculating valid target-row ranges for aligned forecasting tasks;
- constructing lagged history matrices;
- constructing aligned auxiliary-history matrices;
- selecting zero-based indices for configured forecast horizons;
- generating all valid input-window starting positions;
- validating lookback, rollout-horizon, stride, and sequence-length values;
- implementing a PyTorch Dataset that returns paired lookback and future-target
  tensors;
- splitting training data chronologically into training and validation portions;
- constructing shuffled training and ordered validation DataLoaders;
- materializing evaluation lookback windows in batches;
- converting loader arrays into model-ready layouts; and
- selecting raw target rows and converting them into model/output layout.
- honoring sequence segments so windows do not cross files, sites, or frequency bins.

Layout conversions:

    CSV loader to model:
        (T, F) -> (T, F)

    Map loader to model:
        (T, H, W, F) -> (T, F, H, W)

    Selected map targets:
        (N, H, W, F) -> (N, F, H, W)

This module does not perform forecasting or normalization. It only constructs
and aligns temporal samples and array layouts.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.common.data_loader import data_loader_kwargs
from training.common.preprocessing import SequenceSegment

def target_rows_for(
    split_length: int,
    history_offset: int,
    horizon: int,
    lookback: int,
    min_history: int = 0,
) -> np.ndarray:
    start = max(history_offset, horizon + lookback - 1, min_history)
    end = history_offset + split_length
    if start >= end:
        raise ValueError(
            f"Split length {split_length} is too short for horizon {horizon}, "
            f"lookback {lookback}, and min_history {min_history}."
        )
    return np.arange(start, end)


def lagged_matrix(series: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    return np.stack([series[start : start + lookback] for start in starts], axis=0).astype(np.float32)


def aligned_history_matrix(x: np.ndarray, target_rows: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    starts = target_rows - horizon - lookback + 1
    return np.stack([x[start : start + lookback] for start in starts], axis=0).astype(np.float32)


def selected_horizon_index(horizon: int) -> int:
    return int(horizon) - 1


# ===========================================================================
# Window creation
# ===========================================================================

class WindowDataset(Dataset):
    """Create input and target windows from time-first data."""

    def __init__(
        self,
        data: np.ndarray,
        starts: np.ndarray,
        lookback: int,
        rollout_horizon: int,
        segments: tuple[SequenceSegment, ...] = (),
    ):
        # If data is a 4D map (T, Lat, Long, Freq), swap axes to (T, Freq, Lat, Long)
        # Complete dataset with time as dimension 0.
        self.data = torch.from_numpy(
            to_model_layout(data)
        ).float()
        
        # Starting position of each valid input window.
        starts = np.asarray(starts, dtype=np.int64)
        if segments:
            starts = starts[
                [
                    any(
                        start >= segment.start
                        and start + lookback + rollout_horizon <= segment.end
                        for segment in segments
                    )
                    for start in starts
                ]
            ]
        self.starts = starts

        # Number of past timesteps given to the model.
        self.lookback = lookback

        # Number of future timesteps used as the training target.
        self.rollout_horizon = rollout_horizon


    def __len__(self) -> int:
        # Number of available windows.
        return len(self.starts)

    def __getitem__(self, idx: int):
        # Beginning of this input window.
        start = int(self.starts[idx])

        # Past input window.
        x = self.data[
            start : start + self.lookback
        ]

        # Future target immediately after the input window.
        y = self.data[
            start + self.lookback : start + self.lookback + self.rollout_horizon
        ]

        return x, y


def make_window_starts(
    n_timesteps: int,
    lookback: int,
    rollout_horizon: int,
    stride: int,
    segments: tuple[SequenceSegment, ...] = (),
) -> np.ndarray:
    """Create every valid starting position for a window."""
    
    if stride <= 0:
        raise ValueError(
            "Error! Window stride must be positive."
        )
    
    if lookback <= 0:
        raise ValueError("Error! Lookback must be greater than 0!")

    if rollout_horizon <= 0:
        raise ValueError("Error! Rollout_horizon must be greater than 0")

    if segments:
        starts = []
        for segment in segments:
            if segment.end - segment.start < lookback + rollout_horizon:
                continue
            segment_starts = make_window_starts(
                segment.end - segment.start,
                lookback,
                rollout_horizon,
                stride,
            )
            starts.extend(segment.start + segment_starts)
        return np.asarray(starts, dtype=np.int64)

    number_of_windows = (n_timesteps - lookback - rollout_horizon + 1 )

    if number_of_windows <= 0:
        raise ValueError(
            f"Error! Not enough timesteps ({n_timesteps}) for "
            f"lookback={lookback} and "
            f"rollout_horizon={rollout_horizon}"
        )

    return np.arange(0, number_of_windows, stride, dtype=np.int64)


def filter_target_rows(
    target_rows: np.ndarray,
    history: int,
    segments: tuple[SequenceSegment, ...],
) -> np.ndarray:
    """Keep targets whose required history stays within one segment."""
    if not segments:
        return target_rows
    return target_rows[
        [
            any(
                target - history >= segment.start
                and target < segment.end
                for segment in segments
            )
            for target in target_rows
        ]
    ]

def build_window_loaders(
    data: np.ndarray,
    lookback: int,
    rollout_horizon: int,
    batch_size: int,
    val_fraction: float,
    train_stride: int,
    val_stride: int,
    segments: tuple[SequenceSegment, ...] = (),
    data_loader_config: dict | None = None,
) -> tuple[DataLoader, DataLoader]:

    """Create training and validation DataLoaders."""
    print(f"[DEBUG] build_window_loaders: data.shape={data.shape}, lookback={lookback}, rollout_horizon={rollout_horizon}, batch_size={batch_size}, val_fraction={val_fraction}")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"Error! val_fraction must be between 0 and 1, got {val_fraction}"
        )
    
    # Find Split index
    split_index = int(len(data) * (1.0 - val_fraction))
    print(f"[DEBUG] build_window_loaders: split_index={split_index}, train_rows={split_index}, val_rows={len(data) - split_index}")
    
    # Split data into train and Validation sets
    train_data = data[:split_index]
    val_data = data[split_index:]

    train_segments = tuple(
        SequenceSegment(segment.start, min(segment.end, split_index), segment.label)
        for segment in segments
        if segment.start < split_index and segment.start < min(segment.end, split_index)
    )
    val_segments = tuple(
        SequenceSegment(max(segment.start, split_index) - split_index, segment.end - split_index, segment.label)
        for segment in segments
        if segment.end > split_index and max(segment.start, split_index) < segment.end
    )
    
    # Establish valid start positions for train set
    print(f"[DEBUG] build_window_loaders: computing train window starts ...")
    train_starts = make_window_starts(
        n_timesteps=len(train_data),
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        stride=train_stride,
        segments=train_segments,
    )
    print(f"[DEBUG] build_window_loaders: training windows: {len(train_starts)}")
    # Establish valid start positions for validation set
    val_starts = make_window_starts(
        n_timesteps=len(val_data),
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        stride=val_stride,
        segments=val_segments,
    )
    print(f"[DEBUG] build_window_loaders: validation windows: {len(val_starts)}")
    
    # Create window datasets for training and validation
    print(f"[DEBUG] build_window_loaders: creating WindowDatasets ...")
    train_dataset = WindowDataset(
        data=train_data,
        starts=train_starts,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        segments=train_segments,
    )

    val_dataset = WindowDataset(
        data=val_data,
        starts=val_starts,
        lookback=lookback,
        rollout_horizon=rollout_horizon,
        segments=val_segments,
    )
    
    # Create loaders for training and validation with torch DataLoader
    dl_kwargs = data_loader_kwargs(data_loader_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **dl_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **dl_kwargs,
    )

    return train_loader, val_loader



def make_window_batch_array(
    *,
    full_x: np.ndarray,
    start_rows: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Materialize all initial lookback windows for the selected origins.
    """

    if len(start_rows) == 0:
        return np.empty(
            (0, lookback, *full_x.shape[1:]),
            dtype=np.float32,
        )

    return np.stack(
        [
            full_x[
                origin :
                origin + lookback
            ]
            for origin in start_rows
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

# Reshaping data into desired input tensors format


def to_model_layout(
    data: np.ndarray,
) -> np.ndarray:
    """
    Convert loader output into model-ready time-first layout.

    CSV:
        (T, F) -> (T, F)

    Map:
        (T, H, W, F) -> (T, F, H, W)
    """

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    if data.ndim == 2:
        return data

    if data.ndim == 4:
        return np.transpose(
            data,
            (0, 3, 1, 2),
        ).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        "Expected data shaped either "
        "(time, frequency) or "
        "(time, height, width, frequency), "
        f"got {data.shape}"
    )


def raw_targets_to_model_layout(
    raw_dbm: np.ndarray,
    target_rows: np.ndarray,
) -> np.ndarray:
    """
    Select raw target rows and convert them to model/output layout.
    """

    selected = np.asarray(
        raw_dbm[target_rows],
        dtype=np.float32,
    )

    if selected.ndim == 2:
        return selected

    if selected.ndim == 4:
        return np.transpose(
            selected,
            (0, 3, 1, 2),
        ).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        "Expected selected targets shaped either "
        "(samples, frequency) or "
        "(samples, height, width, frequency), "
        f"got {selected.shape}"
    )
