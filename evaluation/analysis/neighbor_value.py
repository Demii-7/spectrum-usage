"""Do the neighbours help? Two paired analyses on the 15-minute core receivers (frozen models, test week 2026-10-27 to 11-03).

A. ``compare``: the base 4D models (map of the four core receivers; tuning finals, ``neighbor-ablation-oct2026`` 'real' mode) against the
   per-receiver models that see only the receiver's own history (1D and 2D linear and residual-linear tuning finals scored in
   distribution by ``training.ray.transfer15``) and against persistence, on the SAME forecast origins and horizons (4, 12, 24, 48),
   per receiver and band.
B. ``ablation``: the 4D models with the other receivers' dynamics broken (shift96, shift210, level; see ``training.ray.neighbor_ablation``)
   against the same models on the real series, on origins valid in every mode.

Paired differences use a moving-block bootstrap over origins (block 96 = one day, 2,000 resamples) on the mean over horizons of the
seed-averaged per-origin MAE. A pooled value averages the per-receiver bootstrap draws (receivers resampled independently).

    python evaluation/analysis/neighbor_value.py [--out evaluation/results]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET = "spectrum"
HORIZONS = (4, 12, 24, 48)
SEEDS = (40, 41, 42, 43, 44)
BANDS = ("600_800", "2400_2600")
RECEIVERS = {"humanities": "humanities", "madsen": "madsen", "moran": "moran", "sage point": "sagepoint"}   # 4D name -> transfer15 name
PER_RECEIVER_MODELS = (("2d", "residuallinearar2d"), ("2d", "linearar2d"), ("1d", "residuallinearar1d"), ("1d", "linearar1d"))
FOUR_D = ("residuallinearar4d", "linearar4d", "stsprednet", "residualconvlstm", "dswinlstm_i")      # the neural models appear only once their per_origin.npz exist
SINGLE_SEED = ("lookbackmean4d", "dswinlstm_i")     # one seed (42) exists; rows of dswinlstm_i are labelled "dswinlstm_i (seed 42)"
MODES = ("real", "shift96", "shift210", "level")
RNG = np.random.default_rng(0)
BLOCK, DRAWS = 96, 2000


def read(client, key: str) -> bytes:
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def utc(values) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values), utc=True)


def ablation_arrays(client, band: str, model: str, seed: int) -> dict[str, np.ndarray] | None:
    try:
        with np.load(io.BytesIO(read(client, f"neighbor-ablation-oct2026/{band}/{model}/seed-{seed}/per_origin.npz")), allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception:
        return None


def four_d_series(client, band: str, model: str, mode: str, receiver: str) -> pd.DataFrame | None:
    """Seed-averaged per-origin MAE (mean over the four horizons) of one 4D model, indexed by origin time."""
    frames = []
    for seed in (SEEDS if model not in SINGLE_SEED else (42,)):
        arrays = ablation_arrays(client, band, model, seed)
        if arrays is None:
            continue
        horizons = list(arrays["horizons"])
        pick = [horizons.index(h) for h in HORIZONS]
        times = utc(arrays[f"{receiver}|{mode}|times"])
        mae = arrays[f"{receiver}|{mode}|mae"][:, pick]
        frames.append(pd.Series(mae.mean(axis=1), index=times))
    if not frames:
        return None
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame()


def per_receiver_series(client, band: str, receiver: str, rep: str, model: str) -> pd.DataFrame | None:
    frames = []
    for seed in SEEDS:
        try:
            table = pd.read_csv(io.BytesIO(read(client, f"transfer15-oct2026/{band}/{receiver}/{rep}/{model}/seed-{seed}/per_origin.csv")))
        except Exception:
            continue
        table = table[table.horizon.isin(HORIZONS)]
        frames.append(table.groupby("time").mae_db.mean())
    if not frames:
        return None
    out = pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna()
    out.index = utc(out.index.to_numpy())
    return out.rename("mae").to_frame()


def persistence_series(client, band: str, receiver: str) -> pd.DataFrame | None:
    try:
        table = pd.read_csv(io.BytesIO(read(client, f"transfer15-oct2026/{band}/{receiver}/baseline/persistence/per_origin.csv")))
    except Exception:
        return None
    out = table[table.horizon.isin(HORIZONS)].groupby("time").mae_db.mean()
    out.index = utc(out.index.to_numpy())
    return out.rename("mae").to_frame()


def block_bootstrap_delta(delta: np.ndarray) -> np.ndarray:
    """Moving-block bootstrap draws of the mean of a per-origin difference series."""
    n = len(delta)
    if n < 2 * BLOCK:
        block = max(1, n // 4)
    else:
        block = BLOCK
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    return delta[index].mean(axis=1)


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, int, float, float]:
    common = a.index.intersection(b.index)
    x, y = a.loc[common, "mae"].to_numpy(), b.loc[common, "mae"].to_numpy()
    return x - y, len(common), float(x.mean()), float(y.mean())


def summarize(rows: list[dict], keys: list[str], draws_key: str = "draws") -> pd.DataFrame:
    """Add the pooled-over-receivers row (mean of the per-receiver deltas, bootstrap draws averaged) per group."""
    frame = pd.DataFrame(rows)
    out = [frame.drop(columns=draws_key)]
    pooled = []
    for group, sub in frame.groupby(keys, dropna=False):
        draws = np.mean(np.stack(sub[draws_key].to_list()), axis=0)
        record = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        record.update(receiver="pooled", n_origins=int(sub.n_origins.sum()), mean_a=sub.mean_a.mean(), mean_b=sub.mean_b.mean(),
                      delta=sub.delta.mean(), ci_low=np.percentile(draws, 2.5), ci_high=np.percentile(draws, 97.5))
        pooled.append(record)
    return pd.concat([out[0], pd.DataFrame(pooled)], ignore_index=True)


def compare(client) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        for receiver_4d, receiver in RECEIVERS.items():
            references = {f"{rep} {name}": per_receiver_series(client, band, receiver, rep, name) for rep, name in PER_RECEIVER_MODELS}
            references["persistence"] = persistence_series(client, band, receiver)
            references["4D lookback mean"] = four_d_series(client, band, "lookbackmean4d", "real", receiver_4d)
            for model in FOUR_D:
                ours = four_d_series(client, band, model, "real", receiver_4d)
                if ours is None:
                    continue
                for name, other in references.items():
                    if other is None:
                        continue
                    delta, n, mean_a, mean_b = paired(ours, other)
                    rows.append({"band": band, "receiver": receiver, "model_4d": model + (" (seed 42)" if model == "dswinlstm_i" else ""), "versus": name, "n_origins": n, "mean_a": mean_a,
                                 "mean_b": mean_b, "delta": float(delta.mean()), "ci_low": np.percentile(b := block_bootstrap_delta(delta), 2.5),
                                 "ci_high": np.percentile(b, 97.5), "draws": b})
    table = summarize(rows, ["band", "model_4d", "versus"])
    return table.rename(columns={"mean_a": "mae_4d", "mean_b": "mae_other", "delta": "delta_4d_minus_other"})


def ablation(client) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        for receiver_4d in RECEIVERS:
            for model in (*FOUR_D, "lookbackmean4d"):
                series = {m: four_d_series(client, band, model, m, receiver_4d) for m in MODES}
                if any(v is None for v in series.values()):
                    continue
                common = series["real"].index
                for m in MODES[1:]:
                    common = common.intersection(series[m].index)
                base = series["real"].loc[common]
                for mode in MODES[1:]:
                    delta, n, mean_a, mean_b = paired(series[mode].loc[common], base)
                    draws = block_bootstrap_delta(delta)
                    rows.append({"band": band, "receiver": receiver_4d, "model": model + (" (seed 42)" if model == "dswinlstm_i" else ""), "mode": mode, "n_origins": n, "mean_a": mean_a,
                                 "mean_b": mean_b, "delta": float(delta.mean()), "ci_low": np.percentile(draws, 2.5),
                                 "ci_high": np.percentile(draws, 97.5), "draws": draws})
    table = summarize(rows, ["band", "model", "mode"])
    return table.rename(columns={"mean_a": "mae_ablated", "mean_b": "mae_real", "delta": "delta_ablated_minus_real"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results")
    args = parser.parse_args()
    client = spectrum_s3_client()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    a = compare(client)
    a.to_csv(out / "neighbor_value_compare.csv", index=False)
    print("A. 4D vs per-receiver models, pooled over the four core receivers (MAE dB, mean over horizons 4/12/24/48; delta < 0 means 4D is better)")
    print(a[a.receiver == "pooled"].drop(columns="receiver").round(3).to_string(index=False))
    b = ablation(client)
    b.to_csv(out / "neighbor_value_ablation.csv", index=False)
    print("\nB. neighbour ablation, pooled over receivers (delta > 0 means breaking the neighbours hurts)")
    print(b[b.receiver == "pooled"].drop(columns="receiver").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
