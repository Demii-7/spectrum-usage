"""Does the 4D-versus-per-receiver result depend on how the map is built? (15-minute, October test week, four core receivers)

Earlier finding (``neighbor_value.py``, default IDW power-2 map): the 4D linear models are 0.10-0.29 dB worse than per-receiver 1D/2D
linear models that see only the receiver's own history. The S2 variants of ``training.ray.spatial15`` change the map construction (IDW
power 1, 2, 4, 16, linear-power interpolation, grids 5x5 and 20x20, and ordinary kriging, ``core4-krige``). This script compares the 4D
linear models under every construction with the per-receiver models, persistence and the 4D lookback mean under the same map, on
IDENTICAL forecast origins: the per-origin errors of the finished spatial cells (``training.ray.spatial15_origins``, prefix
``spatial15-origin-errors-oct2026``) against the per-receiver per-origin files of ``training.ray.transfer15``.

Per origin the error is the mean over horizons t+4/12/24/48 (15-minute steps) of the MAE over the 200 bins, averaged over the seeds
that exist for the cell (seeds 40-44 for the base map, 40-42 for the other S2 variants, 40-44 for kriging; the lookback mean has seed
42 only). Paired differences use a moving-block bootstrap over origins (block 96 = one day, 2,000 resamples); the pooled value averages
the per-receiver draws (receivers resampled independently), as in ``neighbor_value.py``. An unpaired fallback from the cells'
``result.json`` (physical-receiver MAE, mean and seed sd) is printed next to it as a cross-check.

    python evaluation/analysis/map_construction_vs_per_receiver.py [--out evaluation/results/map_construction_vs_per_receiver.csv]
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
SPATIAL_PREFIX = "spatial15-tuned-oct2026"
ORIGIN_PREFIX = "spatial15-origin-errors-oct2026"
BANDS = ("600_800", "2400_2600")
VARIANTS = ("core4-p1", "core4-p2", "core4-p4", "core4-p16", "core4-p2-linear", "core4-p2-g5", "core4-p2-g20", "core4-krige")
MODELS_4D = ("linearar4d", "residuallinearar4d")
RECEIVERS = {"humanities": "humanities", "madsen": "madsen", "moran": "moran", "sage point": "sagepoint"}   # 4D name -> transfer15 name
PER_RECEIVER = (("1d", "residuallinearar1d"), ("1d", "linearar1d"), ("2d", "residuallinearar2d"), ("2d", "linearar2d"))
HORIZONS = (4, 12, 24, 48)
SEEDS = (40, 41, 42, 43, 44)
BLOCK, DRAWS = 96, 2000
RNG = np.random.default_rng(0)


def read(client, key: str) -> bytes:
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def utc(values) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values), utc=True)


def four_d_series(client, band: str, variant: str, model: str, receiver: str) -> tuple[pd.DataFrame | None, int]:
    """Seed-averaged per-origin MAE (mean over the four horizons) of one 4D cell at one receiver, and the number of seeds."""
    frames = []
    for seed in (SEEDS if model != "lookbackmean4d" else (42,)):
        try:
            with np.load(io.BytesIO(read(client, f"{ORIGIN_PREFIX}/{band}/{variant}/{model}/seed-{seed}/per_origin.npz")), allow_pickle=True) as npz:
                horizons = list(npz["horizons"])
                pick = [horizons.index(h) for h in HORIZONS]
                frames.append(pd.Series(npz[f"{receiver}|mae"][:, pick].mean(axis=1), index=utc(npz[f"{receiver}|times"])))
        except Exception:
            continue
    if not frames:
        return None, 0
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna().rename("mae").to_frame(), len(frames)


def per_receiver_series(client, band: str, receiver: str, rep: str, model: str) -> pd.DataFrame | None:
    frames = []
    for seed in SEEDS:
        try:
            table = pd.read_csv(io.BytesIO(read(client, f"transfer15-oct2026/{band}/{receiver}/{rep}/{model}/seed-{seed}/per_origin.csv")))
        except Exception:
            continue
        frames.append(table[table.horizon.isin(HORIZONS)].groupby("time").mae_db.mean())
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


def block_bootstrap(delta: np.ndarray) -> np.ndarray:
    n = len(delta)
    block = BLOCK if n >= 2 * BLOCK else max(1, n // 4)
    starts = RNG.integers(0, n - block + 1, size=(DRAWS, int(np.ceil(n / block))))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    return delta[index].mean(axis=1)


def paired(a: pd.DataFrame, b: pd.DataFrame):
    common = a.index.intersection(b.index)
    x, y = a.loc[common, "mae"].to_numpy(), b.loc[common, "mae"].to_numpy()
    return x - y, len(common), float(x.mean()), float(y.mean())


def compare(client) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        references = {}
        for receiver_4d, receiver in RECEIVERS.items():
            for rep, name in PER_RECEIVER:
                references[(receiver_4d, f"{rep} {name}")] = per_receiver_series(client, band, receiver, rep, name)
            references[(receiver_4d, "persistence")] = persistence_series(client, band, receiver)
        for variant in VARIANTS:
            for model in MODELS_4D:
                per_receiver = {}
                for receiver_4d in RECEIVERS:
                    ours, n_seeds = four_d_series(client, band, variant, model, receiver_4d)
                    if ours is None:
                        continue
                    lbm, _ = four_d_series(client, band, variant, "lookbackmean4d", receiver_4d)
                    refs = {name: references[(receiver_4d, name)] for _, name in [(None, f"{r} {m}") for r, m in PER_RECEIVER]}
                    refs["persistence"] = references[(receiver_4d, "persistence")]
                    refs["4D lookback mean (same map)"] = lbm
                    for name, other in refs.items():
                        if other is None:
                            continue
                        delta, n, mean_a, mean_b = paired(ours, other)
                        draws = block_bootstrap(delta)
                        per_receiver.setdefault(name, []).append((n, mean_a, mean_b, float(delta.mean()), draws, n_seeds))
                for name, entries in per_receiver.items():
                    if len(entries) != len(RECEIVERS):
                        continue                                         # pool only when all four receivers have both series
                    draws = np.mean(np.stack([e[4] for e in entries]), axis=0)
                    rows.append({"band": band, "variant": variant, "model_4d": model, "versus": name, "receiver": "pooled",
                                 "n_origins": int(sum(e[0] for e in entries)), "n_seeds_4d": int(entries[0][5]),
                                 "mae_4d": float(np.mean([e[1] for e in entries])), "mae_other": float(np.mean([e[2] for e in entries])),
                                 "delta_4d_minus_other": float(np.mean([e[3] for e in entries])),
                                 "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def unpaired_cells(client) -> pd.DataFrame:
    """Fallback: physical-receiver MAE of each finished cell from its result.json (mean over the four receivers, seed mean and sd)."""
    rows = []
    for band in BANDS:
        for variant in VARIANTS:
            for model in (*MODELS_4D, "lookbackmean4d"):
                values = []
                for seed in (SEEDS if model != "lookbackmean4d" else (42,)):
                    try:
                        r = json.loads(read(client, f"{SPATIAL_PREFIX}/{band}/{variant}/{model}/seed-{seed}/result.json"))
                        v = (r.get("receivers") or {}).get("geometry_correct", {}).get("mean_horizon_mae_db")
                        if r.get("ok") and v is not None:
                            values.append(float(v))
                    except Exception:
                        continue
                if values:
                    rows.append({"band": band, "variant": variant, "model": model, "n_seeds": len(values), "mae_mean": float(np.mean(values)),
                                 "mae_sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="evaluation/results/map_construction_vs_per_receiver.csv")
    args = parser.parse_args()
    client = spectrum_s3_client()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    paired_table = compare(client)
    paired_table.to_csv(out, index=False)
    unpaired = unpaired_cells(client)
    unpaired.to_csv(out.with_name(out.stem + "_unpaired.csv"), index=False)
    for band in BANDS:
        d = paired_table[paired_table.band == band]
        print(f"\n{band}: paired comparison on identical origins, mean over horizons 4/12/24/48, pooled over the four receivers "
              f"(delta = 4D minus other, dB; negative = 4D better)")
        for model in MODELS_4D:
            m = d[d.model_4d == model]
            if m.empty:
                continue
            wide = m.pivot_table(index="variant", columns="versus", values="delta_4d_minus_other").reindex([v for v in VARIANTS if v in set(m.variant)])
            low = m.pivot_table(index="variant", columns="versus", values="ci_low").reindex(wide.index)
            high = m.pivot_table(index="variant", columns="versus", values="ci_high").reindex(wide.index)
            mae = m.drop_duplicates("variant").set_index("variant").mae_4d.reindex(wide.index).rename("mae_4d")
            show = wide.copy()
            for column in show.columns:
                show[column] = [f"{v:+.3f} [{lo:+.3f},{hi:+.3f}]" if pd.notna(v) else "" for v, lo, hi in
                                zip(wide[column], low[column], high[column])]
            print(f"\n{model}")
            print(pd.concat([mae.round(3), show], axis=1).to_string())
    print("\nunpaired cross-check (physical-receiver MAE of the cells, result.json; mean over the four receivers, seed mean and sd)")
    print(unpaired.pivot_table(index=["band", "variant"], columns="model", values="mae_mean").round(3).to_string())


if __name__ == "__main__":
    main()
