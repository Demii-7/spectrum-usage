#!/usr/bin/env python3
"""Model advantage by entropy third: each model's MAE minus the tuned lookback median's, for the low / middle / high third of bins.

Bin level (200 one-MHz bins per band), averaged over forecast horizons and seeds; terciles of the bins' 1-minute power-difference entropy
within each band (the same bins at both resolutions). Rows: 1-minute measurements (horizons 1, 15, 60 min) and 15-minute block averages (horizons 1, 3, 6, 12 h);
columns: 600-800 and 2400-2600 MHz. Within each third, bars are ordered from the largest improvement to the largest deterioration; colour: relative difference from the lookback median, as in the tables (blue: lower error, red: higher; saturates at 30%).
Input: ``evaluation/results/entropy_advantage_october.csv`` (``entropy_advantage_october.py``).

    python evaluation/plotting/plot_entropy_tercile_bars_october.py --output latex/figures/entropy_tercile_bars_october.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Okabe-Ito family colours; light/dark shades separate models inside a family.
STYLE = {
    "1d linearar1d": ("#0072B2", "LinearAR 1D"), "1d residuallinearar1d": ("#56B4E9", "Residual LinearAR 1D"),
    "2d linearar2d": ("#004C7A", "LinearAR 2D"), "2d residuallinearar2d": ("#9AD0F0", "Residual LinearAR 2D"),
    "1d vanillalstm1d": ("#D55E00", "LSTM 1D"), "1d residualvanillalstm": ("#F0A060", "Residual LSTM 1D"),
    "2d vanillalstm": ("#8A3B00", "LSTM 2D"), "2d residualvanillalstm": ("#E69F00", "Residual LSTM 2D"),
    "2d lstmattn": ("#B34A00", "LSTMAttn 2D"), "2d temporalconvnet": ("#009E73", "TCN"), "2d autoformer_csa": ("#CC79A7", "Autoformer-CSA"),
    # direct models (one model per horizon)
    "direct revinmlp": ("#6A3D9A", "RevIN-MLP"), "direct patchtst": ("#9E7BC2", "PatchTST"), "direct nlinear": ("#4D4D4D", "NLinear"),
    "direct dlinear": ("#8C8C8C", "DLinear"), "direct ofa": ("#7A0177", "OFA (GPT-2)"), "direct timellm": ("#C51B8A", "Time-LLM (GPT-2)"),
    "direct deepspred2d": ("#B8860B", "3D-SwinSTB"),
}
TIERS = ["Low", "Middle", "High"]
BANDS = {"600_800": "600–800 MHz", "2400_2600": "2400–2600 MHz"}
RES = {"1min": "1 minute", "15min": "15 minutes"}
MID_GRAY = np.array([0.72, 0.72, 0.72])
YLIM = {"1min": (-0.85, 0.55), "15min": (-0.85, 0.55)}  # common range, so that panels are comparable


def shade(rel: float) -> tuple[float, float, float]:
    """Gray (no difference) to blue (lower error) or red (higher error), saturating at +-30%.

    The saturated ends are the tables' 52% blue/red; the midpoint is gray instead of white so that short bars stay visible."""
    p = min(abs(rel) / 0.30, 1.0)
    pure = np.array([0.0, 0.0, 1.0]) if rel < 0 else np.array([1.0, 0.0, 0.0])
    end = 0.48 * np.ones(3) + 0.52 * pure
    return tuple((1 - p) * MID_GRAY + p * end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "evaluation/results/entropy_advantage_october.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/figures/entropy_tercile_bars_october.png")
    args = parser.parse_args()
    d = pd.read_csv(args.input)
    d = d[d.level == "bin"]
    fs = 7
    plt.rcParams.update({"font.size": fs})
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.45), sharey="all", constrained_layout=True)
    panels = [(res, band) for res in RES for band in BANDS]
    for k, (res, band) in enumerate(panels):
        ax = axes[k]
        x = d[(d.resolution == res) & (d.band == band)].set_index("model")
        models = [m for m in STYLE if m in x.index]
        bw = 0.86 / len(models)
        for t in range(3):
            # best improvement (most negative gap) first; colour = relative difference from the lookback median, as in the tables
            pairs = sorted((x.loc[m, f"gap_db_tier{t + 1}"], x.loc[m, f"adv_tier{t + 1}"]) for m in models)
            for i, (v, rel) in enumerate(pairs):
                ax.bar(t + (i - (len(models) - 1) / 2) * bw, v, bw, color=shade(rel), edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="#202124", linewidth=0.8)
        ax.set_xticks(range(3), TIERS, fontsize=fs - 1)
        ax.set_ylim(*YLIM[res])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=fs - 1)
        ax.set_title(f"{RES[res]}\n{BANDS[band]}", fontsize=fs - 1)
        if k == 0:
            ax.set_ylabel("MAE $-$ lookback\nmedian MAE (dB)", fontsize=fs - 1)
    fig.supxlabel("Entropy third of bins (by 1-minute entropy)", fontsize=fs - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and .pdf")


if __name__ == "__main__":
    main()
