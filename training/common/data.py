from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from training.common.aerpaw_loader import LoadedSpectrumData, load_aerpaw_data
from training.common.config import ROOT, resolve_path


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    start_mhz: float
    end_mhz: float


def chunk_specs(config: dict[str, Any]) -> list[ChunkSpec]:
    return [
        ChunkSpec(str(chunk["id"]), float(chunk["start_mhz"]), float(chunk["end_mhz"]))
        for chunk in config["data"]["chunks"]
    ]


def load_chunk(config: dict[str, Any], chunk: ChunkSpec) -> LoadedSpectrumData:
    data_dir = resolve_path(config["data"]["data_dir"])
    normalize = bool(config["preprocessing"].get("normalize", True))
    reference_site = str(config["data"].get("reference_site", "CC2"))
    return load_aerpaw_data(
        data_dir,
        chunk.start_mhz,
        chunk.end_mhz,
        normalize=normalize,
        reference_site=reference_site,
    )


def model_matrix_to_convlstm_frames(x: np.ndarray) -> np.ndarray:
    """Convert (T, F) model input to ConvLSTM frame layout (T, 1, 200)."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix (time, frequency), got shape {x.shape}")
    return x[:, None, :].astype(np.float32)


def _fill_nearest_neighbor_2d(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values in a 2D array using nearest-neighbor interpolation.

    Uses ``scipy.ndimage.distance_transform_edt`` to find the closest
    non-NaN cell for each NaN cell.
    """
    mask = ~np.isnan(arr)
    if mask.all() or not mask.any():
        return arr
    inverted = (~mask).astype(np.uint8)
    indices = ndimage.distance_transform_edt(
        inverted, return_distances=False, return_indices=True,
    )
    return arr[tuple(indices)]


def clean_interpolated_map(
    data_4d: np.ndarray,
    train_ratio: float = 0.8,
    fit_on_train_only: bool = True,
) -> np.ndarray:
    """Remove or impute all NaN values in a (T, F, H, W) interpolated map array.

    Pipeline:
      1. Drop timesteps that are entirely NaN (all F×H×W values).
      2. Per (time, frequency) spatial slice, fill partial-NaN cells using
         nearest-neighbor interpolation (``_fill_nearest_neighbor_2d``).
      3. Impute any remaining NaN (e.g. slices that were fully NaN in step 2)
         with the per-frequency mean computed from the training portion.
      4. Assert that zero NaN cells remain.

    Args:
        data_4d: (T, F, H, W) float32 NumPy array, possibly containing NaN.
        train_ratio: Fraction of timesteps used as the training set for
                     computing per-frequency means in step 3.
        fit_on_train_only: If True, compute frequency means only from the
                           first ``train_ratio`` fraction of timesteps.

    Returns:
        Cleaned array of shape (T', F, H, W) with no NaN values.  T' ≤ T.

    Raises:
        AssertionError: If all timesteps are fully NaN or if any NaN remains
                        after cleaning.

    Side effect:
        Prints a detailed log of the cleaning process.
    """
    T, F, H, W = data_4d.shape
    total_cells = T * F * H * W
    initial_nan = np.isnan(data_4d)
    initial_nan_count = int(initial_nan.sum())

    if initial_nan_count == 0:
        print("[clean_interpolated_map] No NaN values found; no cleaning needed.")
        return data_4d

    print(
        f"[clean_interpolated_map] Original shape ({T}, {F}, {H}, {W}), "
        f"NaN cells: {initial_nan_count}/{total_cells} "
        f"({100.0 * initial_nan_count / total_cells:.2f}%)."
    )

    # ---- Step 1: drop fully-NaN timesteps ----
    all_nan_t = initial_nan.all(axis=(1, 2, 3))
    drop_count = int(all_nan_t.sum())
    nan_in_dropped = int(initial_nan[all_nan_t].sum()) if drop_count else 0
    data_4d = data_4d[~all_nan_t]
    T = int(data_4d.shape[0])
    if drop_count:
        print(
            f"[clean_interpolated_map]  Step 1: dropped {drop_count} fully-NaN "
            f"timestep(s) ({nan_in_dropped} NaN cells).  New T={T}."
        )

    # ---- Step 2: nearest-neighbour fill per (t, f) slice ----
    before_nn = int(np.isnan(data_4d).sum())
    for t in range(T):
        for f in range(F):
            sl = data_4d[t, f]
            if np.isnan(sl).any():
                data_4d[t, f] = _fill_nearest_neighbor_2d(sl)
    after_nn = int(np.isnan(data_4d).sum())
    nn_filled = before_nn - after_nn
    if nn_filled:
        print(
            f"[clean_interpolated_map]  Step 2: filled {nn_filled} cell(s) via "
            f"nearest-neighbour per (t,f) slice."
        )

    # ---- Step 3: impute remaining NaNs with per-frequency mean ----
    remaining = np.isnan(data_4d)
    remaining_count = int(remaining.sum())
    if remaining_count:
        if fit_on_train_only:
            n_train = max(1, int(T * train_ratio))
            train_seg = data_4d[:n_train]
        else:
            train_seg = data_4d
        freq_means = np.nanmean(train_seg, axis=(0, 2, 3), keepdims=True)
        data_4d = np.where(remaining, freq_means, data_4d)
        print(
            f"[clean_interpolated_map]  Step 3: filled {remaining_count} cell(s) "
            f"with per-frequency training-set mean."
        )

    # ---- Step 4: assert no NaNs remain and data is non-empty ----
    if T == 0:
        raise AssertionError(
            "[clean_interpolated_map] All timesteps were fully NaN — "
            "no usable data remains!"
        )
    final_nan = np.isnan(data_4d).any()
    if final_nan:
        raise AssertionError(
            f"[clean_interpolated_map] {int(np.isnan(data_4d).sum())} NaN cell(s) "
            f"remain after cleaning!"
        )
    print("[clean_interpolated_map]  Step 4: PASS — zero NaN cells remain.")

    total_filled = initial_nan_count - nan_in_dropped
    print(
        f"[clean_interpolated_map] Done — original NaN cells: {initial_nan_count}, "
        f"removed by drop: {nan_in_dropped}, "
        f"imputed: {total_filled}."
    )
    return data_4d
