import numpy as np
import torch
from torch.utils.data import Dataset


def node_column_slice(node_names, bins_per_node=250):
    known = {"CC1": 0, "CC2": 1, "LW1": 2}
    cols = []
    for name in node_names:
        idx = known.get(name)
        if idx is None:
            raise ValueError(f"Unknown node {name}, known: {list(known.keys())}")
        cols.extend(range(idx * bins_per_node, (idx + 1) * bins_per_node))
    return cols


def random_mask(shape, missing_rate):
    return (torch.rand(shape) > missing_rate).float()


def block_mask(shape, missing_rate):
    mask = torch.ones(shape)
    T, H, W, F = shape
    block_len = max(1, int(W * missing_rate * 0.5))
    n_blocks = max(1, int(T * H * missing_rate))
    for _ in range(n_blocks):
        t = np.random.randint(0, T)
        h = np.random.randint(0, H)
        w_start = np.random.randint(0, W - block_len + 1)
        mask[t, h, w_start:w_start + block_len, :] = 0.0
    return mask


def frequency_mask(shape, missing_rate):
    T, H, W, F = shape
    n_masked = max(1, int(F * missing_rate)) if F > 1 else max(1, int(W * missing_rate))
    mask = torch.ones(shape)
    for t in range(T):
        if F > 1:
            freq_indices = np.random.choice(F, n_masked, replace=False)
            mask[t, :, :, freq_indices] = 0.0
        else:
            width_indices = np.random.choice(W, n_masked, replace=False)
            mask[t, :, width_indices, :] = 0.0
    return mask


def spatial_mask(shape, missing_rate):
    T, H, W, F = shape
    total_cells = H * W
    n_masked = max(1, int(total_cells * missing_rate))
    mask = torch.ones(shape)
    for t in range(T):
        indices = np.random.choice(total_cells, n_masked, replace=False)
        for idx in indices:
            h = idx // W
            w = idx % W
            mask[t, h, w, :] = 0.0
    return mask


def node_mask(shape, missing_rate, n_nodes):
    T, H, W, F = shape
    mask = torch.ones(shape)
    n_masked = max(1, int(n_nodes * missing_rate))
    for t in range(T):
        node_indices = np.random.choice(n_nodes, n_masked, replace=False)
        mask[t, node_indices, :, :] = 0.0
    return mask


def _trim_trailing_all_nan_timesteps(data):
    valid_mask = ~np.isnan(data).all(axis=tuple(range(1, data.ndim)))
    if valid_mask.all():
        return data, []
    last_valid = np.where(valid_mask)[0]
    if len(last_valid) == 0:
        return data[:0], list(range(data.shape[0]))
    trimmed = list(range(last_valid[-1] + 1, data.shape[0]))
    return data[: last_valid[-1] + 1], trimmed


def impute_full_nan_timesteps(data_4d, window_steps=2):
    full_nan_steps = np.where(np.isnan(data_4d).all(axis=(1, 2, 3)))[0]
    if len(full_nan_steps) == 0:
        return data_4d, 0
    T = data_4d.shape[0]
    filled = 0
    for t in full_nan_steps:
        prev_frame = None
        next_frame = None
        for delta in range(1, window_steps + 1):
            prev_idx = t - delta
            next_idx = t + delta
            if prev_frame is None and prev_idx >= 0 and not np.isnan(data_4d[prev_idx]).all():
                prev_frame = data_4d[prev_idx]
            if next_frame is None and next_idx < T and not np.isnan(data_4d[next_idx]).all():
                next_frame = data_4d[next_idx]
            if prev_frame is not None and next_frame is not None:
                break
        if prev_frame is not None and next_frame is not None:
            data_4d[t] = ((prev_frame + next_frame) / 2.0).astype(np.float32)
            filled += 1
        elif prev_frame is not None:
            data_4d[t] = prev_frame.copy()
            filled += 1
        elif next_frame is not None:
            data_4d[t] = next_frame.copy()
            filled += 1
    return data_4d, filled


def impute_nan_local_time(data_4d, window_steps=2):
    total_nan = int(np.isnan(data_4d).sum())
    if total_nan == 0:
        return data_4d, 0
    T, F, H, W = data_4d.transpose(0, 3, 1, 2).shape  # for consistent loops on channel-last input
    fill_coords = []
    fill_vals = []
    for h in range(data_4d.shape[1]):
        for w in range(data_4d.shape[2]):
            for f in range(data_4d.shape[3]):
                series = data_4d[:, h, w, f]
                if not np.isnan(series).any():
                    continue
                for t in np.where(np.isnan(series))[0]:
                    left = slice(max(0, t - window_steps), t)
                    right = slice(t + 1, min(data_4d.shape[0], t + window_steps + 1))
                    vals = np.concatenate([series[left], series[right]])
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        fill_coords.append((t, h, w, f))
                        fill_vals.append(valid.mean())
    for coord, value in zip(fill_coords, fill_vals):
        data_4d[coord] = value
    return data_4d, len(fill_coords)


def impute_nan_local_frequency(data_4d, window_steps=2):
    if data_4d.shape[3] == 1:
        return data_4d, 0
    total_nan = int(np.isnan(data_4d).sum())
    if total_nan == 0:
        return data_4d, 0
    fill_coords = []
    fill_vals = []
    T, H, W, F = data_4d.shape
    for t in range(T):
        for h in range(H):
            for w in range(W):
                series = data_4d[t, h, w, :]
                if not np.isnan(series).any():
                    continue
                for f in np.where(np.isnan(series))[0]:
                    left = slice(max(0, f - window_steps), f)
                    right = slice(f + 1, min(F, f + window_steps + 1))
                    vals = np.concatenate([series[left], series[right]])
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        fill_coords.append((t, h, w, f))
                        fill_vals.append(valid.mean())
    for coord, value in zip(fill_coords, fill_vals):
        data_4d[coord] = value
    return data_4d, len(fill_coords)


def clean_map_nans(data_4d, time_window_steps=2, frequency_window_steps=None):
    frequency_window_steps = time_window_steps if frequency_window_steps is None else frequency_window_steps
    data_4d, trimmed = _trim_trailing_all_nan_timesteps(data_4d)
    data_4d, full_frames = impute_full_nan_timesteps(data_4d, time_window_steps)
    data_4d, time_filled = impute_nan_local_time(data_4d, time_window_steps)
    data_4d, freq_filled = impute_nan_local_frequency(data_4d, frequency_window_steps)
    remaining = np.where(np.isnan(data_4d).any(axis=(1, 2, 3)))[0].tolist()
    if remaining:
        raise RuntimeError(f"Internal timesteps still contain unresolved NaNs: {remaining}")
    stats = {
        "trimmed_trailing_timesteps": trimmed,
        "full_nan_timesteps_imputed": full_frames,
        "time_axis_fills": time_filled,
        "frequency_axis_fills": freq_filled,
        "remaining_nan_count": int(np.isnan(data_4d).sum()),
    }
    return data_4d.astype(np.float32), stats


def load_csv_pseudo_map(config, cc2_only=False):
    csv_path = config["data"]["dataset_path"]
    nodes = config["data"]["selected_nodes"]
    bins = config["data"]["bins_per_node"]
    node_names = config["data"]["node_names"]
    max_rows = config["data"].get("max_rows")
    if cc2_only:
        nodes = ["CC2"]
    cols = node_column_slice(nodes, bins)
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(-1, len(cols))
    if max_rows is not None:
        raw = raw[:max_rows]
    data_2d = raw[:, cols]
    T, _ = data_2d.shape
    H = len(nodes)
    W = bins
    data_map = data_2d.reshape(T, H, W, 1).astype(np.float32)
    return data_map, {"node_names": nodes if cc2_only else nodes, "grid_height": H, "grid_width": W, "n_freq_bins": 1}


def load_interpolated_map(config):
    npz_path = config["data"]["map_path"]
    map_key = config["data"].get("map_key", "map_db")
    raw = np.load(npz_path)[map_key].astype(np.float32)
    if raw.ndim != 4:
        raise ValueError(f"Expected interpolated map shape (T,H,W,F), got {raw.shape}")
    max_rows = config["data"].get("max_rows")
    if max_rows is not None:
        raw = raw[:max_rows]
    preproc = config.get("preprocessing", {})
    time_window = int(preproc.get("nan_time_window_steps", 2))
    freq_window = int(preproc.get("nan_frequency_window_steps", time_window))
    cleaned, nan_stats = clean_map_nans(raw, time_window_steps=time_window, frequency_window_steps=freq_window)
    T, H, W, F = cleaned.shape
    return cleaned, {"node_names": ["grid"], "grid_height": H, "grid_width": W, "n_freq_bins": F, "nan_stats": nan_stats}


class SpectrumMapDataset(Dataset):
    def __init__(self, data, config, split="train"):
        self.config = config
        self.split = split
        self.T_in = config["windowing"]["input_sequence_length"]
        self.T_out = config["windowing"]["prediction_horizon"]
        self.missing_rate = config["preprocessing"]["missing_rate"]
        self.missing_strategy = config["preprocessing"].get("missing_strategy", "random")
        self.mask_targets = config["preprocessing"].get("mask_targets", False)

        stride_key = f"{split}_stride"
        stride = config["windowing"].get(stride_key)
        if stride is None:
            stride = config["windowing"].get("test_stride" if split == "test" else "val_stride")
        if stride is None:
            stride = 1 if split == "train" else self.T_out
        self.stride = stride
        self.maps = torch.from_numpy(self._build_windows(data)).float()

    def _build_windows(self, data):
        T_total = data.shape[0]
        total_len = self.T_in + self.T_out
        starts = list(range(0, T_total - total_len + 1, self.stride))
        self.window_starts = starts
        return np.stack([data[s:s + total_len] for s in starts]) if starts else np.empty((0, total_len, *data.shape[1:]), dtype=np.float32)

    def __len__(self):
        return len(self.window_starts)

    def _generate_mask(self, shape):
        if self.missing_strategy == "block":
            return block_mask(shape, self.missing_rate)
        if self.missing_strategy == "frequency":
            return frequency_mask(shape, self.missing_rate)
        if self.missing_strategy == "node":
            return node_mask(shape, self.missing_rate, shape[1])
        if self.missing_strategy == "spatial":
            return spatial_mask(shape, self.missing_rate)
        return random_mask(shape, self.missing_rate)

    def __getitem__(self, idx):
        window = self.maps[idx]
        X = window[:self.T_in].clone()
        Y = window[self.T_in:self.T_in + self.T_out].clone()
        mask = self._generate_mask((self.T_in,) + tuple(X.shape[1:]))
        X_masked = X * mask
        return X_masked, mask, Y


def normalize_splits(train_data, val_data, test_data, config, full_data=None):
    norm_cfg = config["preprocessing"]
    method = norm_cfg.get("normalization", "minmax")
    fit_on_train = norm_cfg.get("fit_on_train_only", True)
    if fit_on_train or full_data is None:
        fit_data = train_data
    else:
        fit_data = full_data

    if method == "minmax":
        dmin = float(np.min(fit_data))
        dmax = float(np.max(fit_data))
        lo, hi = norm_cfg.get("minmax_range", [-1, 1])
        eps = 1e-8
        def _norm(data):
            return (((data - dmin) / (dmax - dmin + eps)) * (hi - lo) + lo).astype(np.float32)
        stats = {"method": "minmax", "dmin": dmin, "dmax": dmax, "range": [lo, hi]}
    elif method == "zscore":
        mean = float(np.mean(fit_data))
        std = float(np.std(fit_data))
        std = 1.0 if std < 1e-8 else std
        def _norm(data):
            return ((data - mean) / std).astype(np.float32)
        stats = {"method": "zscore", "mean": mean, "std": std}
    else:
        def _norm(data):
            return data.astype(np.float32)
        stats = {"method": "none"}

    return _norm(train_data), _norm(val_data), _norm(test_data), stats


def load_and_split(config, cc2_only=False):
    data_format = config["data"].get("format", "csv")
    if data_format == "interpolated_map":
        full_data, meta = load_interpolated_map(config)
    else:
        full_data, meta = load_csv_pseudo_map(config, cc2_only=cc2_only)

    T = full_data.shape[0]
    ratios = config["split"]
    tr, vr, ter = ratios["train_ratio"], ratios["val_ratio"], ratios["test_ratio"]
    n_train = int(T * tr)
    n_val = int(T * vr)
    n_test = int(T * ter)

    train_data = full_data[:n_train]
    val_data = full_data[n_train:n_train + n_val]
    test_data = full_data[n_train + n_val:n_train + n_val + n_test]
    train_norm, val_norm, test_norm, stats = normalize_splits(train_data, val_data, test_data, config, full_data=full_data)
    stats.update(meta)
    stats["data_format"] = data_format
    stats["n_nodes"] = meta["grid_height"] if data_format == "csv" else meta["grid_height"]
    return train_norm, val_norm, test_norm, stats, meta["node_names"]
