from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class NormalizationStats:
    mean_dbm: np.ndarray
    std_dbm: np.ndarray


class WindowedSpectrumDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32, copy=False))
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_single_site_csv(config: dict[str, Any], csv_path: str | Path | None = None) -> np.ndarray:
    data_config = config["data"]
    dataset_path = resolve_path(csv_path or data_config["dataset_path"])
    skip_rows = 1 if data_config.get("has_header", False) else 0
    raw = np.loadtxt(dataset_path, delimiter=",", skiprows=skip_rows, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, None]

    max_rows = data_config.get("max_rows")
    if max_rows:
        raw = raw[: int(max_rows)]

    expected_bins = int(data_config["n_frequency_bins"])
    if raw.shape[1] != expected_bins:
        raise ValueError(
            f"Expected CSV with {expected_bins} frequency bins for a single site, got shape {raw.shape}."
        )
    return raw.astype(np.float32, copy=False)


def chronological_split(series: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    split_config = config["split"]
    if not split_config.get("chronological_split", True):
        raise ValueError("Only chronological_split=true is supported for VanillaLSTM.")

    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.6f}.")

    n_rows = int(series.shape[0])
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return {
        "train": series[:train_end],
        "val": series[train_end:val_end],
        "test": series[val_end:],
    }


def fit_normalization(train_split: np.ndarray, config: dict[str, Any]) -> NormalizationStats:
    normalization = config["preprocessing"].get("normalization", "zscore")
    if normalization != "zscore":
        raise ValueError(f"Unsupported normalization {normalization!r}; only 'zscore' is implemented.")
    mean = np.mean(train_split, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_split, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1e-8, std).astype(np.float32)
    return NormalizationStats(mean_dbm=mean, std_dbm=std)


def normalize_split(split: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((split - stats.mean_dbm) / stats.std_dbm).astype(np.float32)


def denormalize_array(values: np.ndarray, stats: NormalizationStats | dict[str, Any]) -> np.ndarray:
    mean_dbm = np.asarray(stats.mean_dbm if isinstance(stats, NormalizationStats) else stats["mean_dbm"], dtype=np.float32)
    std_dbm = np.asarray(stats.std_dbm if isinstance(stats, NormalizationStats) else stats["std_dbm"], dtype=np.float32)
    return (values * std_dbm + mean_dbm).astype(np.float32)


def create_windows(series: np.ndarray, input_length: int, output_length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError("Stride must be positive.")

    total_window = input_length + output_length
    max_start = series.shape[0] - total_window
    if max_start < 0:
        empty_x = np.empty((0, input_length, series.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, output_length, series.shape[1]), dtype=np.float32)
        return empty_x, empty_y

    starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
    inputs = np.stack([series[start : start + input_length] for start in starts], axis=0).astype(np.float32)
    targets = np.stack(
        [series[start + input_length : start + total_window] for start in starts],
        axis=0,
    ).astype(np.float32)
    return inputs, targets


def create_datasets(config: dict[str, Any], csv_path: str | Path | None = None) -> dict[str, Any]:
    raw = load_single_site_csv(config, csv_path=csv_path)
    raw_splits = chronological_split(raw, config)

    fit_on_train_only = bool(config["preprocessing"].get("fit_on_train_only", True))
    if not fit_on_train_only:
        raise ValueError("Only fit_on_train_only=true is supported for VanillaLSTM.")

    stats = fit_normalization(raw_splits["train"], config)
    normalized_splits = {name: normalize_split(split, stats) for name, split in raw_splits.items()}

    input_length = int(config["windowing"]["input_sequence_length"])
    output_length = int(config["windowing"]["prediction_horizon"])
    strides = {
        "train": int(config["windowing"]["train_stride"]),
        "val": int(config["windowing"]["val_stride"]),
        "test": int(config["windowing"]["test_stride"]),
    }

    datasets: dict[str, WindowedSpectrumDataset] = {}
    windows: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        x_norm, y_norm = create_windows(normalized_splits[split_name], input_length, output_length, strides[split_name])
        x_raw, y_raw = create_windows(raw_splits[split_name], input_length, output_length, strides[split_name])
        datasets[split_name] = WindowedSpectrumDataset(x_norm, y_norm)
        windows[split_name] = {
            "inputs_normalized": x_norm,
            "targets_normalized": y_norm,
            "inputs_raw": x_raw,
            "targets_raw": y_raw,
        }
        counts[split_name] = int(x_norm.shape[0])

    return {
        "raw": raw,
        "raw_splits": raw_splits,
        "normalized_splits": normalized_splits,
        "datasets": datasets,
        "windows": windows,
        "normalization_stats": stats,
        "window_counts": counts,
    }
