"""Shared pieces of the NCAR transfer batch (``transfer-ncar-2026-11``): per-window arrays, per-bin metrics, extra POWDER targets, model tags.

A *cell* is one (model, seed, condition, target). Besides the existing ``summary.json``/``per_origin.csv``, a cell scored with
``save_arrays`` writes
  ``forecasts.npz``  per horizon h: ``origin_h{h}`` (N,) int64 epoch seconds of the forecast origin (last observed step; the target time is
                     origin + 15 min x h), ``pred_h{h}`` and ``target_h{h}`` (N, F) float32 dBm, rows sorted by origin, columns = bin index;
  ``per_bin.csv``    per (horizon, bin): n_windows, mae_db, mse_db2, rmse_db, mape_linear_pct, mape_dbm_pct, bias_db,
                     mae_after_per_bin_offset, rmse_after_per_bin_offset (error minus its mean over windows; the definition of
                     ``evaluation/analysis/loo_level_decomposition.py``), share_removed = 1 - mae_after / mae.
Every derived table (receiver/group means, restricted windows, level split) is computed from these arrays, so a cell is scored once.

Metric definitions are the scoring code's (``training.ray.transfer15.score_horizon`` and ``evaluation_integrated._percentage_error_arrays``):
MAE/MSE/RMSE of the dB error; MAPE linear = |10^(e/10) - 1| x 100 (the primary MAPE), MAPE dBm = |e| / |target dBm| x 100.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

PREFIX = "transfer-ncar-2026-11"
STEP = pd.Timedelta(minutes=15)
WEEK2 = (pd.Timestamp("2026-11-03T20:00:00Z"), pd.Timestamp("2026-11-10T20:00:00Z"))
CORE4 = ("humanities", "madsen", "moran", "sagepoint")

# --------------------------------------------------------------------------------------------- tags
# TCN is NOT in this set: every tuned 15-minute TCN final (both bands, seeds 40-44) has model.feature_mode = "independent", which reshapes
# the bins into the batch (models/TemporalConvNet.py), so it never mixes bins. Autoformer-CSA embeds all bins jointly (TokenEmbedding over
# 200 channels): yes. TimeRAN/MOMENT reshape channels into the batch: no.
SPECTRAL_YES = {"linearar2d", "residuallinearar2d", "vanillalstm", "residualvanillalstm:2d", "lstmattn", "autoformer_csa",
                "patchtst_joint", "revinmlp_mix", "tsslcd",
                # not in the brief's lists: Residual nonlinear 2D (ResidualLinearAR 2D base), iTransformer (variates are the tokens),
                # DeepSPred (3D Swin over time x frequency frames) all mix bins
                "residualnonlinear2d", "itransformer", "deepspred2d"}
SPATIAL_YES = {"linearar4d", "residuallinearar4d", "convlstm", "residualconvlstm", "dswinlstm_i", "stsprednet", "tsslcd", "revinmlp_nbr",
               "lookbackmean4d"}
LEVEL_KEEPING = {"revinmlp", "patchtst", "ofa", "timellm", "moment_lp", "timeran_lp", "nlinear", "residuallinearar1d", "residuallinearar2d"}


def tags(model: str, rep: str = "2d") -> dict[str, str]:
    """``spectral_context`` / ``spatial_context`` columns (brief lists; TCN, Autoformer-CSA and TimeRAN checked against the code, see
    CAMPAIGN_STATUS 40). ``residualvanillalstm`` exists as 1D and 2D; only the 2D one mixes bins."""
    key = f"{model}:{rep}" if model == "residualvanillalstm" else model
    return {"spectral_context": "yes" if key in SPECTRAL_YES else "no", "spatial_context": "yes" if model in SPATIAL_YES else "no",
            "level_keeping": "yes" if model in LEVEL_KEEPING else "no"}


# ------------------------------------------------------------------------------------------ targets
def core_targets(bands: list[str], window: str = "oct", receivers: tuple[str, ...] = CORE4) -> list[dict[str, Any]]:
    """The core POWDER receivers as targets (in distribution) on the October test week (``oct``) or the second test week (``week2``)."""
    from training.ray.powder15 import TEST_RUN

    out = []
    for b in bands:
        for r in receivers:
            t = {"platform": "powder", "site": r, "run": TEST_RUN, "band": b}
            if window == "oct":
                t["n_origins"] = 625
            else:
                t["window"] = window
            out.append(t)
    return out


def week_task(task, window: str | None):
    """The same task moved to the second test week (lookback history from the 24 h before it)."""
    import dataclasses

    if not window or window == "oct":
        return task
    if window != "week2":
        raise ValueError(window)
    start, end = WEEK2
    return dataclasses.replace(task, task_id=f"{task.task_id}-week2", evaluation=task.evaluation.replace("October test week", "second test week"),
                               raw_start=start - 96 * STEP, raw_end=end, target_start=start, target_end=end)


# ------------------------------------------------------------------------------------------- arrays
def matrices(result: dict[str, Any], rep: str) -> dict[int, dict[str, np.ndarray]]:
    """(origin, bin) matrices per horizon from ``transfer15.evaluate_frozen`` output (2D rows are already (N, F); 1D rows are one bin each)."""
    out = {}
    for h in result["predictions"]:
        pred, tgt, times = (np.asarray(result["predictions"][h], dtype=np.float32), np.asarray(result["targets"][h], dtype=np.float32),
                            np.asarray(result["times"][h], dtype="datetime64[s]").astype(np.int64))
        if rep == "1d" or pred.shape[1] == 1:
            bins = np.asarray(result["bins"][h]).reshape(-1).astype(np.int64)
            origins = np.unique(times)
            n_bins = int(bins.max()) + 1
            P = np.full((len(origins), n_bins), np.nan, np.float32)
            T = np.full_like(P, np.nan)
            row = np.searchsorted(origins, times)
            P[row, bins] = pred.reshape(-1)
            T[row, bins] = tgt.reshape(-1)
            if np.isnan(P).any():
                raise RuntimeError("1D rows do not fill the (origin, bin) grid")
        else:
            order = np.argsort(times, kind="stable")
            origins, P, T = times[order], pred[order], tgt[order]
        out[int(h)] = {"origin": origins, "pred": P, "target": T}
    return out


def per_bin_table(mats: dict[int, dict[str, np.ndarray]], windows: dict[int, np.ndarray] | None = None) -> pd.DataFrame:
    """Per (horizon, bin) metrics; ``windows`` optionally restricts each horizon to a boolean mask over its origins."""
    frames = []
    for h, m in sorted(mats.items()):
        keep = slice(None) if windows is None else windows[h]
        e = (m["pred"][keep].astype(np.float64) - m["target"][keep].astype(np.float64))
        tgt = m["target"][keep].astype(np.float32)
        if not len(e):
            continue
        e32 = e.astype(np.float32)
        lin = np.abs(np.power(np.float32(10.0), e32 / np.float32(10.0)) - np.float32(1.0)) * np.float32(100.0)
        dbm = np.abs(e32) / np.maximum(np.abs(tgt), np.float32(1e-6)) * np.float32(100.0)
        bias = e.mean(axis=0)
        centred = e - bias[None, :]
        mae = np.abs(e).mean(axis=0)
        mae_c = np.abs(centred).mean(axis=0)
        frames.append(pd.DataFrame({"horizon": h, "bin": np.arange(e.shape[1]), "n_windows": e.shape[0], "mae_db": mae,
                                    "mse_db2": (e ** 2).mean(axis=0), "rmse_db": np.sqrt((e ** 2).mean(axis=0)),
                                    "mape_linear_pct": lin.mean(axis=0, dtype=np.float64), "mape_dbm_pct": dbm.mean(axis=0, dtype=np.float64),
                                    "bias_db": bias, "mae_after_per_bin_offset": mae_c, "rmse_after_per_bin_offset": np.sqrt((centred ** 2).mean(axis=0)),
                                    "share_removed": 1.0 - mae_c / np.where(mae > 0, mae, np.nan)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def npz_bytes(mats: dict[int, dict[str, np.ndarray]]) -> bytes:
    payload = {}
    for h, m in mats.items():
        payload[f"origin_h{h}"] = m["origin"].astype(np.int64)
        payload[f"pred_h{h}"] = m["pred"].astype(np.float32)
        payload[f"target_h{h}"] = m["target"].astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def load_npz(data: bytes) -> dict[int, dict[str, np.ndarray]]:
    z = np.load(io.BytesIO(data))
    hs = sorted({int(k.split("_h")[1]) for k in z.files if k.startswith("origin_h")})
    return {h: {"origin": z[f"origin_h{h}"], "pred": z[f"pred_h{h}"], "target": z[f"target_h{h}"]} for h in hs}


def save_cell(client, bucket: str, base: str, mats: dict[int, dict[str, np.ndarray]], frequencies_mhz: list[float] | None = None) -> pd.DataFrame:
    """Write ``forecasts.npz`` and ``per_bin.csv`` under ``base`` (never under an existing source prefix; the caller owns ``base``)."""
    if not base.startswith(PREFIX + "/"):
        raise ValueError(f"refusing to write arrays outside {PREFIX}/: {base}")
    table = per_bin_table(mats)
    if frequencies_mhz is not None and len(table):
        table.insert(2, "frequency_mhz", np.asarray(frequencies_mhz, dtype=float)[table["bin"].to_numpy()])
    client.put_object(Bucket=bucket, Key=f"{base}/forecasts.npz", Body=npz_bytes(mats))
    client.put_object(Bucket=bucket, Key=f"{base}/per_bin.csv", Body=table.to_csv(index=False).encode())
    return table


def band_frequencies(band: str) -> list[float]:
    """Bin centres (MHz) of a 200-bin band ``<low>_<high>``: low + 0.5, ..., high - 0.5."""
    low = float(band.split("_")[0])
    return [low + 0.5 + i for i in range(200)]
