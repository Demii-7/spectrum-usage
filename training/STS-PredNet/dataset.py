"""
Dataset loading, preprocessing, and splitting for STS-PredNet.

Provides utilities to load raw CSV spectrograms, reshape to 3D tensor
(nodes x frequency bins), normalize using minmax or z-score, split data
chronologically or randomly, and build PyTorch Datasets with closeness /
period / trend input branches.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_csv(csv_path):
    """Load a CSV file as a float32 numpy array (rows = time steps, cols = features)."""
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def reshape_to_3d(arr, n_nodes, bins_per_node):
    """Reshape flat 2D array into (time, n_nodes, bins_per_node)."""
    return arr.reshape(-1, n_nodes, bins_per_node)


def compute_minmax_stats(data_3d):
    """Compute per-location min and max for minmax normalization."""
    dmin = data_3d.min(axis=0, keepdims=True)
    dmax = data_3d.max(axis=0, keepdims=True)
    return dmin.astype(np.float32), dmax.astype(np.float32)


def minmax_neg1_pos1(data_3d, dmin, dmax):
    """Normalize data to [-1, 1] range using precomputed min/max."""
    eps = 1e-8
    return (2.0 * (data_3d - dmin) / (dmax - dmin + eps) - 1.0).astype(np.float32)


def denormalize(data, dmin, dmax):
    """Reverse [-1, 1] normalization back to original scale."""
    eps = 1e-8
    return 0.5 * (data + 1.0) * (dmax - dmin + eps) + dmin


def zscore(data_3d, mean, std):
    """Standardize data to zero mean and unit variance."""
    return ((data_3d - mean) / (std + 1e-8)).astype(np.float32)


def split_indices(n_total, train_ratio, val_ratio, chronological=True):
    """Partition sample indices into train/val/test splits.

    Args:
        n_total: Total number of samples.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        chronological: If True, use sequential split preserving time order.

    Returns:
        Tuple of (train_idx, val_idx, test_idx) lists.
    """
    if n_total <= 0:
        return [], [], []
    if chronological:
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, n_total))
    else:
        # Random permutation with fixed seed for reproducibility
        perm = np.random.RandomState(42).permutation(n_total)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:n_train + n_val].tolist()
        test_idx = perm[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


class STSPredNetDataset(Dataset):
    """PyTorch Dataset for spectrum prediction with temporal branches.

    For each target time step, constructs:
      - closeness:  contiguous past frames up to 'lc' steps before the prediction point
      - period:     frames sampled with 'period_interval' spacing (e.g., daily pattern)
      - trend:      frames sampled with 'trend_interval' spacing (e.g., weekly pattern)
    """

    def __init__(self, data_3d, target_indices,
                 use_closeness, use_period, use_trend,
                 lc, lp, lq, period_interval, trend_interval,
                 prediction_offset, add_channel_dim=True):
        self.data = torch.from_numpy(data_3d).float()
        self.target_indices = target_indices
        self.use_closeness = use_closeness
        self.use_period = use_period
        self.use_trend = use_trend
        self.lc = lc
        self.lp = lp
        self.lq = lq
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.prediction_offset = prediction_offset
        self.add_channel_dim = add_channel_dim

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        target_idx = self.target_indices[idx]
        t = target_idx - self.prediction_offset
        result = {"target_idx": target_idx, "t": t}

        # Contiguous closeness window: lc frames ending at time t
        if self.use_closeness:
            c_start = t - self.lc + 1
            c_seq = self.data[c_start:t + 1]
            if self.add_channel_dim:
                c_seq = c_seq.unsqueeze(1)
            result["closeness"] = c_seq

        # Period branch: frames spaced by period_interval (e.g. daily pattern)
        if self.use_period:
            p_indices = [target_idx - i * self.period_interval
                         for i in range(self.lp, 0, -1)]
            p_seq = torch.stack([self.data[i] for i in p_indices], dim=0)
            if self.add_channel_dim:
                p_seq = p_seq.unsqueeze(1)
            result["period"] = p_seq

        # Trend branch: frames spaced by trend_interval (e.g. weekly pattern)
        if self.use_trend:
            q_indices = [target_idx - i * self.trend_interval
                         for i in range(self.lq, 0, -1)]
            q_seq = torch.stack([self.data[i] for i in q_indices], dim=0)
            if self.add_channel_dim:
                q_seq = q_seq.unsqueeze(1)
            result["trend"] = q_seq

        target = self.data[target_idx]
        if self.add_channel_dim:
            target = target.unsqueeze(0)
        result["target"] = target

        return result


def generate_target_indices(total_len, prediction_offset,
                            use_closeness, use_period, use_trend,
                            lc, lp, lq, period_interval, trend_interval):
    """Return indices for which all requested temporal branches have enough history.

    Filters out target positions near the start of the time series where
    the required look-back windows would exceed the available data.
    """
    indices = []
    for target_idx in range(prediction_offset, total_len):
        t = target_idx - prediction_offset
        valid = True
        if use_closeness:
            if t - lc + 1 < 0:
                valid = False
        if use_period:
            if target_idx - lp * period_interval < 0:
                valid = False
        if use_trend:
            if target_idx - lq * trend_interval < 0:
                valid = False
        if valid:
            indices.append(target_idx)
    return indices


def collate_branch_samples(batch):
    """Custom collation for branch-structured samples.

    Stack sequence tensors along the batch dimension while keeping
    scalar indices as 1D tensors.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ("target_idx", "t"):
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def load_map_npz(npz_path: str | os.PathLike, map_key: str = "map_db") -> np.ndarray:
    """Load interpolated map from .npz and transpose to (T, F, H, W).

    Expected raw shape: (T, H, W, F). Returns (T, F, H, W).
    """
    data = np.load(npz_path)[map_key].astype(np.float32)
    return data.transpose(0, 3, 1, 2)


def compute_norm_stats_freq(data_4d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency mean/std, broadcastable as (1, F, 1, 1)."""
    mean = np.mean(data_4d, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.std(data_4d, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean, std


def impute_nan_local_time(data_4d: np.ndarray, window_steps: int = 2) -> np.ndarray:
    """Impute NaN values along the time axis using local neighbours.

    For each (F, H, W) cell, for each NaN at time t, compute the mean of
    valid values in the window [t-window_steps, t-1] ∪ [t+1, t+window_steps].
    All imputation values are computed from the *original* neighbors at each
    position and applied simultaneously, avoiding cascading propagation.
    If no valid neighbours exist within the window, the NaN is left as-is
    (no fallback imputation).

    Operates on (T, F, H, W) layout.  Does not drop any timesteps.
    """
    T, F, H, W = data_4d.shape
    total_nan = int(np.isnan(data_4d).sum())
    if total_nan == 0:
        return data_4d

    fill_coords = []
    fill_vals = []

    for f in range(F):
        for h in range(H):
            for w in range(W):
                series = data_4d[:, f, h, w]
                if not np.isnan(series).any():
                    continue
                for t in np.where(np.isnan(series))[0]:
                    left = slice(max(0, t - window_steps), t)
                    right = slice(t + 1, min(T, t + window_steps + 1))
                    vals = np.concatenate([series[left], series[right]])
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        fill_coords.append((t, f, h, w))
                        fill_vals.append(valid.mean())

    for (t, f, h, w), v in zip(fill_coords, fill_vals):
        data_4d[t, f, h, w] = v

    remaining = int(np.isnan(data_4d).sum())
    print(f"  Imputed {len(fill_coords)} NaN cell(s) via local time-axis (window={window_steps})."
          + (f"  Remaining: {remaining}" if remaining else ""))
    return data_4d


def create_interpolated_map_datasets(
    map_path: str | os.PathLike,
    config: dict,
) -> tuple:
    """Create train/val/test datasets from an interpolated-map .npz file.

    Returns (train_ds, val_ds, test_ds, stats) matching the CSV-mode interface.
    """
    dcfg = config["data"]
    pcfg = config["preprocessing"]
    scfg = config["splits"]
    bcfg = config["branches"]
    wcfg = config.get("windowing", {})

    temporal = dcfg.get("temporal_overrides", {})
    if temporal:
        bcfg["lc"] = int(temporal.get("lc", bcfg["lc"]))
        bcfg["lp"] = int(temporal.get("lp", bcfg["lp"]))
        bcfg["period_interval"] = int(temporal.get("period_interval", bcfg["period_interval"]))

    use_c = bcfg["use_closeness"]
    use_p = bcfg["use_period"]
    use_t = bcfg["use_trend"]
    lc = bcfg["lc"]
    lp = bcfg["lp"]
    lq = bcfg["lq"]
    period_interval = bcfg["period_interval"]
    trend_interval = bcfg["trend_interval"]
    prediction_offset = bcfg.get("prediction_offset", 1)

    train_stride = wcfg.get("train_stride", 1)
    val_stride = wcfg.get("val_stride", 1)
    test_stride = wcfg.get("test_stride", 1)

    data_4d = load_map_npz(map_path, dcfg.get("map_key", "map_db"))
    T, F, H, W = data_4d.shape

    # Impute NaNs along time axis (configurable window size)
    ipcfg = dcfg.get("imputation", {})
    if ipcfg.get("enabled", True):
        window = int(ipcfg.get("window_steps", 2))
        data_4d = impute_nan_local_time(data_4d, window)
    else:
        nan_count = int(np.isnan(data_4d).sum())
        if nan_count:
            print(f"  WARNING: imputation disabled, {nan_count} NaN(s) remain in data")

    # Filter to indices with enough history for all requested branches
    all_valid = generate_target_indices(
        len(data_4d), prediction_offset,
        use_c, use_p, use_t,
        lc, lp, lq, period_interval, trend_interval,
    )
    n_valid = len(all_valid)
    train_idx_list, val_idx_list, test_idx_list = split_indices(
        n_valid, scfg["train_ratio"], scfg["val_ratio"],
        scfg.get("chronological_split", True),
    )

    train_targets = [all_valid[i] for i in train_idx_list][::train_stride]
    val_targets = [all_valid[i] for i in val_idx_list][::val_stride]
    test_targets = [all_valid[i] for i in test_idx_list][::val_stride]

    if pcfg["normalization"] == "zscore":
        if pcfg["fit_on_train_only"] and train_targets:
            train_end = max(train_targets) + 1
            mean, std = compute_norm_stats_freq(data_4d[:train_end])
        else:
            mean, std = compute_norm_stats_freq(data_4d)
        data_norm = zscore(data_4d, mean, std)
        stats = {"mean": mean, "std": std, "method": "zscore"}
    else:
        data_norm = data_4d.astype(np.float32)
        stats = {"method": "none"}

    stats["n_freq"] = F
    stats["grid_h"] = H
    stats["grid_w"] = W

    train_ds = STSPredNetDataset(data_norm, train_targets, use_c, use_p, use_t,
                                  lc, lp, lq, period_interval, trend_interval,
                                  prediction_offset, add_channel_dim=False) if train_targets else None
    val_ds = STSPredNetDataset(data_norm, val_targets, use_c, use_p, use_t,
                                lc, lp, lq, period_interval, trend_interval,
                                prediction_offset, add_channel_dim=False) if val_targets else None
    test_ds = STSPredNetDataset(data_norm, test_targets, use_c, use_p, use_t,
                                 lc, lp, lq, period_interval, trend_interval,
                                 prediction_offset, add_channel_dim=False) if test_targets else None

    return train_ds, val_ds, test_ds, stats


def create_datasets(csv_path, config):
    """Load, normalize, split, and wrap data into train/val/test datasets.

    Args:
        csv_path: Path to raw CSV file.
        config: Full configuration dictionary.

    Returns:
        Tuple (train_ds, val_ds, test_ds, stats) where each dataset is either
        an STSPredNetDataset or None if the split has no samples, and stats
        holds the normalization parameters for later denormalization.
    """
    dcfg = config["data"]
    pcfg = config["preprocessing"]
    scfg = config["splits"]
    bcfg = config["branches"]
    wcfg = config.get("windowing", {})

    n_nodes = dcfg["n_nodes"]
    bins_per_node = dcfg["bins_per_node"]
    normalization = pcfg["normalization"]
    fit_on_train_only = pcfg["fit_on_train_only"]

    use_c = bcfg["use_closeness"]
    use_p = bcfg["use_period"]
    use_t = bcfg["use_trend"]
    lc = bcfg["lc"]
    lp = bcfg["lp"]
    lq = bcfg["lq"]
    period_interval = bcfg["period_interval"]
    trend_interval = bcfg["trend_interval"]
    prediction_offset = bcfg.get("prediction_offset", 1)

    # Optional sub-sampling stride within each split
    train_stride = wcfg.get("train_stride", 1)
    val_stride = wcfg.get("val_stride", 1)
    test_stride = wcfg.get("test_stride", 1)

    raw = load_csv(csv_path)
    data_3d = reshape_to_3d(raw, n_nodes, bins_per_node)

    # Determine which time indices are valid (enough history for all branches)
    all_targets = generate_target_indices(
        len(data_3d), prediction_offset,
        use_c, use_p, use_t,
        lc, lp, lq, period_interval, trend_interval,
    )

    n_total = len(all_targets)
    train_idx_list, val_idx_list, test_idx_list = split_indices(
        n_total, scfg["train_ratio"], scfg["val_ratio"],
        scfg.get("chronological_split", True),
    )

    # Apply stride to reduce temporal density of each split
    train_targets = [all_targets[i] for i in train_idx_list][::train_stride]
    val_targets = [all_targets[i] for i in val_idx_list][::val_stride]
    test_targets = [all_targets[i] for i in test_idx_list][::test_stride]

    # Normalize using either minmax [-1,1] or z-score
    if normalization == "minmax_neg1_pos1":
        if fit_on_train_only and train_targets:
            train_data_end = max(train_targets) + 1
            train_segment = data_3d[:train_data_end]
            dmin, dmax = compute_minmax_stats(train_segment)
        else:
            dmin, dmax = compute_minmax_stats(data_3d)
        data_norm = minmax_neg1_pos1(data_3d, dmin, dmax)
        stats = {"dmin": dmin, "dmax": dmax, "method": "minmax_neg1_pos1"}
    elif normalization == "zscore":
        if fit_on_train_only and train_targets:
            train_data_end = max(train_targets) + 1
            train_segment = data_3d[:train_data_end]
            mean, std = train_segment.mean(axis=0, keepdims=True), train_segment.std(axis=0, keepdims=True)
        else:
            mean, std = data_3d.mean(axis=0, keepdims=True), data_3d.std(axis=0, keepdims=True)
        data_norm = zscore(data_3d, mean, std)
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "method": "zscore"}
    else:
        data_norm = data_3d.astype(np.float32)
        stats = {"method": "none"}

    train_ds = STSPredNetDataset(data_norm, train_targets, use_c, use_p, use_t,
                                  lc, lp, lq, period_interval, trend_interval,
                                  prediction_offset) if train_targets else None
    val_ds = STSPredNetDataset(data_norm, val_targets, use_c, use_p, use_t,
                                lc, lp, lq, period_interval, trend_interval,
                                prediction_offset) if val_targets else None
    test_ds = STSPredNetDataset(data_norm, test_targets, use_c, use_p, use_t,
                                 lc, lp, lq, period_interval, trend_interval,
                                 prediction_offset) if test_targets else None

    return train_ds, val_ds, test_ds, stats
