"""
Dataset loading, normalisation, and window-splitting utilities for
spectrum-usage time-series forecasting.

Provides the ``AERPAWDataset`` (PyTorch ``Dataset``) that extracts
(encoder_input, decoder_target) pairs from a CSV of power-spectral-density
measurements, together with helpers for z-score normalisation, train/val/test
splitting, and sliding-window index generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class AERPAWDataset(Dataset):
    """PyTorch Dataset for spectrum-usage time-series forecasting.

    Each sample consists of:
        - ``seq_x`` (encoder input): the look-back window of length *seq_len*.
        - ``seq_y`` (decoder target): a segment of length
          ``label_len + pred_len`` starting *label_len* steps before the end
          of the encoder window.  During training the first ``label_len``
          positions are used as the decoder's initial token sequence.

    Indices are precomputed by :func:`create_datasets` so that sliding-window
    positions are mapped back to the original (non-shifted) time indices.
    """

    def __init__(self, data_2d, seq_len, label_len, pred_len, start_indices):
        self.data = torch.from_numpy(data_2d).float()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.indices = start_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # Encoder window: [i, i + seq_len)
        s_begin = i
        s_end = s_begin + self.seq_len
        # Decoder window: overlaps the end of the encoder window by label_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y


def load_csv(csv_path):
    """Load a CSV of spectrum measurements as a float32 numpy array.

    Expected shape: ``(time_steps, n_features)``, no header.
    """
    return np.loadtxt(csv_path, delimiter=",").astype(np.float32)


def compute_norm_stats(data_2d):
    """Compute per-feature mean and standard deviation for normalisation.

    Standard deviations smaller than ``1e-8`` are clipped to avoid division
    by zero.
    """
    mean = np.mean(data_2d, axis=0, keepdims=True)
    std = np.std(data_2d, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(data_2d, mean, std):
    """Apply z-score normalisation: ``(data - mean) / std``."""
    return ((data_2d - mean) / std).astype(np.float32)


def _make_windows(data_len, seq_len, pred_len, stride):
    """Return a list of starting indices for sliding windows of length seq_len + pred_len.

    Only indices where a full window fits within ``[0, data_len)`` are kept.
    """
    return list(range(0, data_len - seq_len - pred_len + 1, max(stride, 1)))


def _split_time_ranges(total_len, train_ratio, val_ratio, chronological=True):
    """Split the integer range ``[0, total_len)`` into train/val/test segments.

    If *chronological* is ``True`` (default), the split is contiguous and
    preserves temporal order (no data leakage from future to past).
    Otherwise a random permutation is used (e.g. for cross-validation).
    """
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


def create_datasets(csv_path, seq_len, label_len, pred_len,
                    train_stride=1, val_stride=None, test_stride=None,
                    train_ratio=0.8, val_ratio=0.1, chronological=True,
                    normalization="zscore", fit_on_train_only=True):
    """Load CSV, normalise, split temporally, and return three ``AERPAWDataset`` instances.

    Args:
        csv_path: Path to the CSV file (time x features, no header).
        seq_len: Number of time steps in the encoder look-back window.
        label_len: Number of initial time steps provided to the decoder.
        pred_len: Number of time steps to forecast.
        train_stride, val_stride, test_stride: Spacing between consecutive
            sliding windows in each split.  Default ``None`` sets val/test
            stride to *pred_len* (non-overlapping evaluation).
        train_ratio, val_ratio: Fraction of the total time series used for
            training and validation.  The remainder is used for testing.
        chronological: If ``True`` (default), splits are contiguous and
            time-ordered; otherwise random.
        normalization: ``"zscore"`` (default) or ``"minmax"``.  If any other
            value, data is returned as-is with identity stats.
        fit_on_train_only: If ``True``, normalisation statistics are computed
            only on the training portion to avoid test-set information leakage.

    Returns:
        Tuple of ``(train_ds, val_ds, test_ds, stats_dict)`` where each
        dataset is ``None`` if the corresponding split is empty, and
        *stats_dict* contains ``"mean"`` and ``"std"``.
    """
    val_stride = pred_len if val_stride is None else val_stride
    test_stride = pred_len if test_stride is None else test_stride

    raw = load_csv(csv_path)
    T = len(raw)

    # Compute statistics only on training data to prevent look-ahead bias
    if fit_on_train_only:
        n_train_raw = int(T * train_ratio)
        train_segment = raw[:n_train_raw] if n_train_raw > 0 else raw[:1]
        mean, std = compute_norm_stats(train_segment)
    else:
        mean, std = compute_norm_stats(raw)

    if normalization == "zscore":
        data_norm = zscore(raw, mean, std)
    elif normalization == "minmax":
        dmin = raw.min(axis=0, keepdims=True)
        dmax = raw.max(axis=0, keepdims=True)
        data_norm = ((raw - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)
        mean, std = dmin.astype(np.float32), (dmax - dmin + 1e-8).astype(np.float32)
    else:
        data_norm = raw.astype(np.float32)
        mean = np.zeros((1, raw.shape[1]), dtype=np.float32)
        std = np.ones((1, raw.shape[1]), dtype=np.float32)

    # Obtain time-range indices for each split
    train_range, val_range, test_range = _split_time_ranges(
        T, train_ratio, val_ratio, chronological,
    )

    # Compute sliding-window start positions within each split
    train_starts = _make_windows(len(train_range), seq_len, pred_len, train_stride)
    val_starts = _make_windows(len(val_range), seq_len, pred_len, val_stride)
    test_starts = _make_windows(len(test_range), seq_len, pred_len, test_stride)

    # Map split-local indices back to original time indices
    train_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [train_range[s] for s in train_starts],
    ) if train_starts else None

    val_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [val_range[s] for s in val_starts],
    ) if val_starts else None

    test_ds = AERPAWDataset(
        data_norm, seq_len, label_len, pred_len,
        [test_range[s] for s in test_starts],
    ) if test_starts else None

    stats = {"mean": mean, "std": std}
    return train_ds, val_ds, test_ds, stats
