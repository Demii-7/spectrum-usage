#!/usr/bin/env python3
"""October entropy figure: region entropy against error, and each model's gap to the lookback mean, at 1 and 15 minutes (600-800 MHz).

Top row: the lookback mean's MAE in each of the 13 non-noise regions against the region's power-difference entropy (the error rises
with entropy). Bottom row: model MAE minus lookback-mean MAE (dB) per region against entropy; below zero = the model beats the lookback
mean. Inputs: ``evaluation/results/entropy_advantage_october_regions.csv`` (``entropy_advantage_regions_october.py``) and, for the
Spearman correlations quoted in the panels, ``entropy_advantage_october.csv``.

    python evaluation/plotting/plot_entropy_advantage_october.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Okabe-Ito colours by model family; markers separate models inside a family.
STYLE = {
    "1d linearar1d": ("#0072B2", "o", "LinearAR 1D"), "1d residuallinearar1d": ("#56B4E9", "s", "ResidualLinearAR 1D"),
    "2d linearar2d": ("#0072B2", "D", "LinearAR 2D"), "2d residuallinearar2d": ("#56B4E9", "P", "ResidualLinearAR 2D"),
    "2d temporalconvnet": ("#009E73", "X", "TCN"), "2d autoformer_csa": ("#CC79A7", "v", "Autoformer-CSA"),
    "2d residualvanillalstm": ("#E69F00", "^", "ResidualVanillaLSTM"), "2d vanillalstm": ("#D55E00", "<", "VanillaLSTM"),
    "2d lstmattn": ("#D55E00", ">", "LSTMAttn"),
}
TITLES = {"1min": "1 minute (t+1, 15, 60 min)", "15min": "15 minutes (t+1, 3, 6, 12 h)"}
YLIM = {"1min": (-0.2, 0.5), "15min": (-1.45, 0.85)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", type=Path, default=ROOT / "evaluation/results/entropy_advantage_october_regions.csv")
    parser.add_argument("--advantage", type=Path, default=ROOT / "evaluation/results/entropy_advantage_october.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/figures/entropy_model_advantage_october.png")
    args = parser.parse_args()
    d = pd.read_csv(args.regions)
    a = pd.read_csv(args.advantage)
    fs = 7
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs - 0.5})
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.9), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1.35]})
    for col, res in enumerate(("1min", "15min")):
        x = d[d.resolution == res]
        base = x.drop_duplicates("region").sort_values("entropy")
        rho_bin = a[(a.resolution == res) & (a.band == "600_800") & (a.level == "bin")].rho_entropy_baseline_error.iloc[0]
        rho_reg = a[(a.resolution == res) & (a.band == "600_800") & (a.level == "region")].rho_entropy_baseline_error.iloc[0]
        top = axes[0, col]
        top.scatter(base.entropy, base.lookbackmean_mae_db, s=14, color="#202124", zorder=3)
        top.set_title(TITLES[res], fontsize=fs)
        top.set_ylabel("LookbackMean MAE (dB)" if col == 0 else "")
        top.text(0.04, 0.96, f"Spearman $\\rho$ = {rho_reg:.2f} (13 regions)\n$\\rho$ = {rho_bin:.2f} (200 bins)", transform=top.transAxes,
                 va="top", fontsize=fs - 0.5)
        top.grid(alpha=0.25)
        top.set_axisbelow(True)
        bot = axes[1, col]
        bot.axhline(0, color="#202124", linewidth=0.9, zorder=1)
        for model, (color, marker, _) in STYLE.items():
            m = x[x.model == model]
            if len(m):
                bot.scatter(m.entropy, m.gap_db, s=13, color=color, marker=marker, alpha=0.85, linewidths=0, zorder=3)
        bot.set_ylim(*YLIM[res])
        bot.set_xlabel("Region power-difference entropy")
        bot.set_ylabel("Model MAE $-$ LookbackMean MAE (dB)" if col == 0 else "")
        bot.grid(alpha=0.25)
        bot.set_axisbelow(True)
        off = x[(x.gap_db < YLIM[res][0]) | (x.gap_db > YLIM[res][1])]
        if len(off):
            bot.text(0.98, 0.97, f"{len(off)} point{"s" if len(off) > 1 else ""} off scale\n(max {off.gap_db.max():.2f} dB)", transform=bot.transAxes, ha="right", va="top", fontsize=fs - 1)
        bot.text(0.02, 0.03, "model better $\\downarrow$", transform=bot.transAxes, fontsize=fs - 0.5, color="#555555", va="bottom")
    handles = [plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=4.5, label=lab) for c, mk, lab in STYLE.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and .pdf")


if __name__ == "__main__":
    main()
