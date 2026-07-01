import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_3d, seq_len, label_len, pred_len, indices):
        self.data = torch.from_numpy(data_3d).float()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start = self.indices[idx]
        series = self.data[sample_idx]
        s_begin = start
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = series[s_begin:s_end]
        seq_y = series[r_begin:r_end]
        return seq_x, seq_y


def load_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def load_map_npz(npz_path, map_key):
    data = np.load(npz_path)[map_key].astype(np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected interpolated map with 4 dims (T,H,W,F), got {data.shape}")
    return data


def impute_nan_along_time(data_3d):
    """Fill NaNs along time for each sample/frequency stream.

    Uses linear interpolation over time, with edge values extended at the
    boundaries. Streams that are entirely NaN are filled with zeros.
    """
    n_samples, series_len, n_features = data_3d.shape
    filled = data_3d.copy()
    x = np.arange(series_len, dtype=np.float32)
    total_nan = int(np.isnan(filled).sum())
    if total_nan == 0:
        return filled, 0

    filled_count = 0
    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            series = filled[sample_idx, :, feature_idx]
            mask = ~np.isnan(series)
            if mask.all():
                continue
            if not mask.any():
                filled[sample_idx, :, feature_idx] = 0.0
                filled_count += series_len
                continue
            missing = ~mask
            filled[sample_idx, missing, feature_idx] = np.interp(x[missing], x[mask], series[mask])
            filled_count += int(missing.sum())
    return filled.astype(np.float32), filled_count


def compute_norm_stats(data_3d):
    mean = np.mean(data_3d, axis=(0, 1), keepdims=True)
    std = np.std(data_3d, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(data_3d, mean, std):
    return ((data_3d - mean) / std).astype(np.float32)


def _make_windows(series_len, seq_len, pred_len, stride):
    return list(range(0, series_len - seq_len - pred_len + 1, max(stride, 1)))


def _split_time_ranges(total_len, train_ratio, val_ratio, chronological=True):
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    if chronological:
        train_range = list(range(0, n_train))
        val_range = list(range(n_train, n_train + n_val))
        test_range = list(range(n_train + n_val, total_len))
    else:
        perm = np.random.RandomState(42).permutation(total_len)
        train_range = perm[:n_train].tolist()
        val_range = perm[n_train:n_train + n_val].tolist()
        test_range = perm[n_train + n_val:].tolist()
    return train_range, val_range, test_range


def _select_csv_features(raw, data_cfg):
    selected_nodes = data_cfg.get("selected_nodes") or data_cfg.get("node_names")
    node_names = data_cfg.get("node_names", [])
    bins_per_node = int(data_cfg["bins_per_node"])

    if data_cfg.get("cc2_only_smoke_test", False):
        selected_nodes = ["CC2"]

    if not selected_nodes or not node_names:
        return raw, {"selected_nodes": node_names, "n_features": raw.shape[1], "n_nodes": data_cfg.get("n_nodes", 1)}

    column_blocks = []
    for node in selected_nodes:
        if node not in node_names:
            raise ValueError(f"Unknown selected node: {node}")
        idx = node_names.index(node)
        start = idx * bins_per_node
        end = start + bins_per_node
        column_blocks.append(raw[:, start:end])
    selected = np.concatenate(column_blocks, axis=1)
    return selected, {
        "selected_nodes": selected_nodes,
        "n_features": selected.shape[1],
        "n_nodes": len(selected_nodes),
    }


def prepare_csv_series(csv_path, data_cfg):
    raw = load_csv(csv_path)
    selected, meta = _select_csv_features(raw, data_cfg)
    data_3d = selected[None, :, :]
    meta["mode"] = "csv"
    meta["bins_per_node"] = int(data_cfg["bins_per_node"])
    return data_3d, meta


def prepare_map_series(npz_path, map_key):
    raw = load_map_npz(npz_path, map_key)
    # (T, H, W, F) -> (G, T, F), treat each grid cell as its own sample stream.
    T, H, W, F = raw.shape
    series = raw.reshape(T, H * W, F).transpose(1, 0, 2)
    series, filled_count = impute_nan_along_time(series)
    meta = {
        "mode": "interpolated_map",
        "grid_height": H,
        "grid_width": W,
        "n_grid": H * W,
        "n_features": F,
        "nan_filled_count": filled_count,
    }
    return series, meta


def create_datasets(
    *,
    data_format,
    dataset_path,
    seq_len,
    label_len,
    pred_len,
    train_stride=1,
    val_stride=None,
    test_stride=None,
    train_ratio=0.8,
    val_ratio=0.1,
    chronological=True,
    normalization="zscore",
    fit_on_train_only=True,
    data_cfg=None,
):
    val_stride = pred_len if val_stride is None else val_stride
    test_stride = pred_len if test_stride is None else test_stride
    data_cfg = data_cfg or {}

    if data_format == "interpolated_map":
        data_3d, meta = prepare_map_series(dataset_path, data_cfg.get("map_key", "map_db"))
        max_grid_points = data_cfg.get("max_grid_points")
        if max_grid_points:
            data_3d = data_3d[: int(max_grid_points)]
            meta["n_grid"] = int(data_3d.shape[0])
    else:
        data_3d, meta = prepare_csv_series(dataset_path, data_cfg)

    n_samples, series_len, _ = data_3d.shape

    if fit_on_train_only:
        n_train_raw = int(series_len * train_ratio)
        train_segment = data_3d[:, :n_train_raw, :] if n_train_raw > 0 else data_3d[:, :1, :]
        mean, std = compute_norm_stats(train_segment)
    else:
        mean, std = compute_norm_stats(data_3d)

    if normalization == "zscore":
        data_norm = zscore(data_3d, mean, std)
    elif normalization == "none":
        data_norm = data_3d.astype(np.float32)
        mean = np.zeros((1, 1, data_3d.shape[-1]), dtype=np.float32)
        std = np.ones((1, 1, data_3d.shape[-1]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    train_range, val_range, test_range = _split_time_ranges(series_len, train_ratio, val_ratio, chronological)
    train_starts = _make_windows(len(train_range), seq_len, pred_len, train_stride)
    val_starts = _make_windows(len(val_range), seq_len, pred_len, val_stride)
    test_starts = _make_windows(len(test_range), seq_len, pred_len, test_stride)

    def build_indices(time_range, starts):
        indices = []
        for sample_idx in range(n_samples):
            for start in starts:
                indices.append((sample_idx, time_range[start]))
        return indices

    train_ds = SequenceDataset(data_norm, seq_len, label_len, pred_len, build_indices(train_range, train_starts)) if train_starts else None
    val_ds = SequenceDataset(data_norm, seq_len, label_len, pred_len, build_indices(val_range, val_starts)) if val_starts else None
    test_ds = SequenceDataset(data_norm, seq_len, label_len, pred_len, build_indices(test_range, test_starts)) if test_starts else None

    stats = {"mean": mean, "std": std, "data_format": data_format}
    stats.update(meta)
    return train_ds, val_ds, test_ds, stats
