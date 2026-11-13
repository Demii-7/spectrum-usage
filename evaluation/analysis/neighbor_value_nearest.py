#!/usr/bin/env python3
"""Neighbour-shift test of the 4D models at 24 h and 36 h, read at the nearest grid cell (15 minutes, October test week, default map core4-p2).

Inputs: ``neighbor-ablation-oct2026-nearest`` (``training.ray.neighbor_ablation --readout nearest_cell --modes shift96 shift144``: per-origin errors of
the modes real / shift96 (24 h) / shift144 (36 h)) and, for the old-readout 24 h delta side by side, ``neighbor-ablation-oct2026`` (as_evaluated,
modes real / shift96 / ...). Per band, model and receiver (and pooled = mean of the four receivers): MAE real, MAE shifted and
delta = shifted minus real, as the mean over t+4/12/24 (``h3``: 1, 3, 6 hours) and over all four horizons (``h4``); paired on the origins valid in
every mode, seeds averaged per origin, moving-block bootstrap over origins (block 96, 2,000 draws).

Outputs (``--out`` stem): ``<stem>.csv`` long format, ``<stem>.txt`` one table per band plus the checks. With ``--upload`` also to analysis-outputs/.

    python evaluation/analysis/neighbor_value_nearest.py [--out evaluation/results/neighbor_value_ablation_nearest] [--upload]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import neighbor_value as nv  # noqa: E402
from training.ray.storage import spectrum_s3_client  # noqa: E402

BUCKET = "spectrum"
NEW, OLD, CHECK = "neighbor-ablation-oct2026-nearest", "neighbor-ablation-oct2026", "neighbor-ablation-oct2026-nearest-check"
BANDS = ("600_800", "2400_2600")
MODELS = ("linearar4d", "residuallinearar4d", "lookbackmean4d", "stsprednet", "residualconvlstm", "dswinlstm_i")
SEEDS = (40, 41, 42, 43, 44)
RECEIVERS = ("humanities", "madsen", "moran", "sage point")
HSETS = {"h3": (0, 1, 2), "h4": (0, 1, 2, 3)}            # columns of (t+4, t+12, t+24, t+48)
REFERENCE_H3 = {("linearar4d", "600_800"): 0.760, ("linearar4d", "2400_2600"): 0.624, ("stsprednet", "600_800"): 0.968, ("stsprednet", "2400_2600"): 0.908,
                ("residualconvlstm", "600_800"): 1.015, ("residualconvlstm", "2400_2600"): 0.800, ("dswinlstm_i", "600_800"): 1.161,
                ("dswinlstm_i", "2400_2600"): 1.301}


def read(client, prefix, band, model, seed):
    try:
        body = client.get_object(Bucket=BUCKET, Key=f"{prefix}/{band}/{model}/seed-{seed}/per_origin.npz")["Body"].read()
        with np.load(io.BytesIO(body), allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def series(cells: dict[int, dict], receiver: str, mode: str, cols: tuple, common: dict[int, set]) -> pd.Series | None:
    frames = []
    for seed, cell in cells.items():
        times = pd.to_datetime(pd.Series(cell[f"{receiver}|{mode}|times"]), utc=True)
        mae = cell[f"{receiver}|{mode}|mae"][:, list(cols)].mean(axis=1)
        s = pd.Series(mae, index=times)
        frames.append(s[s.index.isin(common[seed])])
    return pd.concat(frames, axis=1).mean(axis=1, skipna=False).dropna() if frames else None


def common_origins(cell: dict, receiver: str, modes: list[str]) -> set:
    sets = [set(pd.to_datetime(pd.Series(cell[f"{receiver}|{m}|times"]), utc=True)) for m in modes]
    return set.intersection(*sets)


def rows_for(band, model, readout, cells, modes_shifted) -> list[dict]:
    seeds = sorted(cells)
    out, pooled = [], {}
    for hs, cols in HSETS.items():
        for mode in modes_shifted:
            per_receiver = []
            for receiver in RECEIVERS:
                common = {s: common_origins(cells[s], receiver, ["real", mode]) for s in seeds}
                real, shifted = series(cells, receiver, "real", cols, common), series(cells, receiver, mode, cols, common)
                idx = real.index.intersection(shifted.index)
                delta = (shifted.loc[idx] - real.loc[idx]).to_numpy()
                draws = nv.block_bootstrap_delta(delta)
                row = {"band": band, "model": model, "readout": readout, "receiver": receiver, "mode": mode, "horizon_set": hs, "n_seeds": len(seeds),
                       "seeds": ",".join(map(str, seeds)), "n_origins": len(idx), "mae_real": float(real.loc[idx].mean()),
                       "mae_shifted": float(shifted.loc[idx].mean()), "delta": float(delta.mean()), "ci_low": float(np.percentile(draws, 2.5)),
                       "ci_high": float(np.percentile(draws, 97.5))}
                out.append(row)
                per_receiver.append((row, draws))
            d = np.mean([x[1] for x in per_receiver], axis=0)
            out.append({"band": band, "model": model, "readout": readout, "receiver": "pooled", "mode": mode, "horizon_set": hs, "n_seeds": len(seeds),
                        "seeds": ",".join(map(str, seeds)), "n_origins": int(np.mean([r["n_origins"] for r, _ in per_receiver])),
                        "mae_real": float(np.mean([r["mae_real"] for r, _ in per_receiver])),
                        "mae_shifted": float(np.mean([r["mae_shifted"] for r, _ in per_receiver])),
                        "delta": float(np.mean([r["delta"] for r, _ in per_receiver])), "ci_low": float(np.percentile(d, 2.5)),
                        "ci_high": float(np.percentile(d, 97.5))})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/results/neighbor_value_ablation_nearest")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    client = spectrum_s3_client()
    rows, lines, expected = [], [], len(BANDS) * (5 * 5 + 1)
    done = 0
    for band in BANDS:
        for model in MODELS:
            seeds = (42,) if model == "lookbackmean4d" else SEEDS
            new = {s: c for s in seeds if (c := read(client, NEW, band, model, s)) is not None and "sage point|shift144|mae" in c}
            old = {s: c for s in seeds if (c := read(client, OLD, band, model, s)) is not None}
            done += len(new)
            if new:
                rows += rows_for(band, model, "nearest_cell", new, ["shift96", "shift144"])
            if old and new:
                rows += rows_for(band, model, "as_evaluated", {s: old[s] for s in sorted(set(old) & set(new))}, ["shift96"])
    table = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out + ".csv", index=False)
    lines.append(f"Neighbour shift at 24 h and 36 h, nearest-cell readout (core4-p2; dB; pooled over the four receivers; paired; cells finished {done} of {expected})")
    lines.append("MAE real = nearest cell; delta = shifted minus real; old = as_evaluated readout, 24 h, same seeds; h3 = mean over t+4/12/24, h4 = over t+4/12/24/48")
    for band in BANDS:
        lines += ["", band + ":"]
        sub = table[(table.band == band) & (table.receiver == "pooled")]
        for hs in ("h3", "h4"):
            lines.append(f"  [{hs}]  model                  seeds   MAE real    d24h nearest [95% CI]       d36h nearest [95% CI]        d24h old readout")
            for model in MODELS:
                n = sub[(sub.model == model) & (sub.readout == "nearest_cell") & (sub.horizon_set == hs)]
                if n.empty:
                    lines.append(f"         {model:20s}  (no cells yet)")
                    continue
                g = lambda mode: n[n["mode"] == mode].iloc[0]
                o = sub[(sub.model == model) & (sub.readout == "as_evaluated") & (sub.horizon_set == hs)]
                old_txt = f"{o.iloc[0].delta:+.3f}" if len(o) else "-"
                lines.append(f"         {model:20s}  {int(g('shift96').n_seeds)}     {g('shift96').mae_real:.3f}    {g('shift96').delta:+.3f} [{g('shift96').ci_low:+.3f},{g('shift96').ci_high:+.3f}]   "
                             f"{g('shift144').delta:+.3f} [{g('shift144').ci_low:+.3f},{g('shift144').ci_high:+.3f}]    {old_txt}")
    lines += ["", "Checks: real at the nearest cell (h3, pooled) against the reference values of the brief:"]
    for (model, band), ref in REFERENCE_H3.items():
        n = table[(table.band == band) & (table.model == model) & (table.readout == "nearest_cell") & (table.receiver == "pooled") & (table.horizon_set == "h3") & (table["mode"] == "shift96")]
        if len(n):
            lines.append(f"  {band} {model:18s} seeds {int(n.iloc[0].n_seeds)}: {n.iloc[0].mae_real:.3f} (reference {ref:.3f}, difference {n.iloc[0].mae_real - ref:+.3f})")
    # exact reproduction of one old-readout cell
    chk = read(client, CHECK, "600_800", "linearar4d", 40)
    old = read(client, OLD, "600_800", "linearar4d", 40)
    if chk is not None and old is not None:
        worst = 0.0
        for receiver in RECEIVERS:
            for mode in ("real", "shift96"):
                t1 = pd.Series(chk[f"{receiver}|{mode}|times"].astype(str)); t0 = pd.Series(old[f"{receiver}|{mode}|times"].astype(str))
                m1 = pd.Series(chk[f"{receiver}|{mode}|mae"].mean(axis=1), index=t1); m0 = pd.Series(old[f"{receiver}|{mode}|mae"].mean(axis=1), index=t0)
                idx = m1.index.intersection(m0.index)
                worst = max(worst, float((m1.loc[idx] - m0.loc[idx]).abs().max()))
        lines.append(f"  old-readout check cell (linearar4d seed 40, 600_800, as_evaluated, real and shift96): largest per-origin difference {worst:.2e} dB")
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.upload:
        for suffix in (".csv", ".txt"):
            f = a.out + suffix
            client.put_object(Bucket=BUCKET, Key="analysis-outputs/" + Path(f).name, Body=open(f, "rb").read())
            print("uploaded", f)


if __name__ == "__main__":
    main()
