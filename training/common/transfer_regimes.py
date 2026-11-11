"""Pure helpers for the three frozen-transfer regimes (no model, no I/O).

A  frozen: weights and the checkpoint's own source normalization; nothing derived from target data.
B  per-window centering: the lookback mean of each window and bin is removed before prediction (in the source normalization) and added back
   to the forecast. Not applied to models that normalize every window themselves (RevIN family), where it would be a second normalization.
C  causal normalizer refit: weights frozen; the per-bin mean AND standard deviation that map dBm to model units are refitted on a calibration
   interval that ends before the first scored forecast origin (see ``calibration_mask``), and used in place of the checkpoint's statistics for
   the whole scored interval. The refit standard deviation is floored at ``STD_FLOOR_FRACTION`` x the source standard deviation (declared before
   scoring, not tuned) so that a quiet calibration interval cannot blow up the input scale.
Cm same as C with the mean refitted only (the source standard deviation is kept).

For models with full intrinsic window normalization (every window is standardized inside the model: RevIN-MLP, PatchTST, OFA, Time-LLM,
MOMENT/TimeRAN) regimes B and C change nothing by construction: a per-bin affine change of the input scale cancels in the model's own per-window
standardization and in the de-standardization of the forecast. They are computed once (A) and verified on a subset, not re-run everywhere.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

REGIMES = ("A", "B", "C", "Cm")
STD_FLOOR_FRACTION = 0.25
MIN_CALIBRATION_ROWS = 48

# Models that standardize every input window inside the model (RevIN or equivalent). ``patchtst_noinorm`` explicitly does not.
FULL_INTRINSIC = frozenset({"revinmlp", "patchtst", "patchtst_joint", "patchtst_nopatch", "ofa", "ofa_rand", "ofa_scratch", "timellm",
                            "timeran_lp", "moment_lp"})
# The residual heads scale their correction by the window's scale and keep the level (partial).
PARTIAL_INTRINSIC = frozenset({"residuallinearar1d", "residuallinearar2d", "residualvanillalstm", "residualnonlinear2d"})


def intrinsic_class(model: str) -> str:
    """``full``, ``partial`` or ``none`` (no per-window normalization inside the model)."""
    if model in FULL_INTRINSIC:
        return "full"
    if model in PARTIAL_INTRINSIC:
        return "partial"
    return "none"


def regime_flags(model: str, regime: str) -> tuple[bool, str | None]:
    """(apply per-window centering, normalizer refit mode) for a regime. Refit mode is None, ``mean`` (Cm) or ``mean_std`` (C)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if regime == "A":
        return False, None
    if regime == "B":
        return intrinsic_class(model) != "full", None
    return False, "mean_std" if regime == "C" else "mean"


def calibration_mask(timestamps, target_start) -> np.ndarray:
    """Rows observed before the scored interval: every loaded row with a timestamp before ``target_start``.

    The loaded raw window of every target starts ``lookback`` (96) blocks before the first scored target row, so this is exactly the history of
    the first forecast origin, 24 hours of 15-minute blocks (strictly causal: nothing at or after the first scored target is used).
    """
    return np.asarray(timestamps < target_start)


def refit_normalization(base: Mapping[str, Any], data: np.ndarray, mask: np.ndarray, mode: str,
                        floor_fraction: float = STD_FLOOR_FRACTION, min_rows: int = MIN_CALIBRATION_ROWS) -> dict[str, Any]:
    """Copy of the checkpoint's normalization dict with the per-bin mean (and, for ``mean_std``, the standard deviation) taken from the
    calibration rows of ``data`` (T, F) in dB. Array shapes follow the checkpoint's; the standard deviation is floored at
    ``floor_fraction`` x the source standard deviation. Statistics are never taken from rows outside ``mask``."""
    if mode not in ("mean", "mean_std"):
        raise ValueError(f"unknown refit mode {mode!r}")
    rows = np.asarray(data, dtype=np.float64)[np.asarray(mask)]
    if rows.ndim != 2 or len(rows) < min_rows:
        raise ValueError(f"calibration interval has {len(rows)} rows, need at least {min_rows}")
    if not np.isfinite(rows).all():
        raise ValueError("calibration rows contain non-finite values")
    mean_shape = np.asarray(base["mean_dbm"]).shape
    source_std = np.asarray(base["std_dbm"], dtype=np.float64).reshape(-1)
    if rows.shape[1] != source_std.size:
        raise ValueError("calibration bin count differs from the checkpoint's normalization")
    out = dict(base)
    out["mean_dbm"] = rows.mean(axis=0).astype(np.float32).reshape(mean_shape)
    if mode == "mean_std":
        std = np.maximum(rows.std(axis=0), floor_fraction * source_std)
        out["std_dbm"] = std.astype(np.float32).reshape(np.asarray(base["std_dbm"]).shape)
    return out


def calibration_facts(timestamps, mask: np.ndarray) -> dict[str, Any]:
    stamps = timestamps[np.asarray(mask)]
    return {"calibration_rows": int(len(stamps)), "calibration_start": stamps[0].isoformat() if len(stamps) else None,
            "calibration_end": stamps[-1].isoformat() if len(stamps) else None, "std_floor_fraction": STD_FLOOR_FRACTION}
