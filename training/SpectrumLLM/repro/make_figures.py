"""Figures for the SpectrumLLM reproduction.

Run in the container that carries matplotlib (the training image does not):

  docker run --rm -v "$PWD:$PWD" -w "$PWD" --entrypoint python tsslcd-ours:mpl \
    training/SpectrumLLM/repro/make_figures.py            # every figure
  ... make_figures.py --only channel_heatmaps             # one figure

Writes PNGs next to this script, in figures/. Each figure is a function
registered in FIGURES; add new ones by decorating with @figure("name").
Only the standard library, numpy and matplotlib are used.
"""
import argparse
import csv
import datetime as dt
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(RE, "data")
OUT = os.path.join(RE, "figures")

FIGURES = OrderedDict()


def figure(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


def load(name):
    """-> (dates as float days since start, freqs MHz, array (T, C) raw units)."""
    with open(os.path.join(DATA, name)) as f:
        rows = list(csv.reader(f))
    freqs = [float(c) for c in rows[0][1:]]
    dates, data = [], []
    for r in rows[1:]:
        dates.append(dt.datetime.strptime(r[0], "%Y/%m/%d %H:%M"))
        data.append([float(x) for x in r[1:]])
    t0 = dates[0]
    days = np.array([(d - t0).total_seconds() / 86400.0 for d in dates])
    return days, freqs, np.array(data, dtype=np.float64)


def time_ticks(ax, days, step=2):
    n = int(days[-1])
    ax.set_xticks(np.arange(0, n + 1, step))
    ax.set_xticklabels([f"+{d}d" for d in range(0, n + 1, step)], rotation=45,
                       ha="right", fontsize=7)


def heatmap(ax, days, freqs, z, mark=(), title=""):
    im = ax.imshow(z.T, aspect="auto", origin="lower", interpolation="nearest",
                   extent=[days[0], days[-1] + 1 / 96, -0.5, len(freqs) - 0.5],
                   cmap="viridis")
    ax.set_yticks(range(len(freqs)))
    labels = [f"{f:.0f}" for f in freqs]
    ax.set_yticklabels(labels, fontsize=7)
    for i, f in enumerate(freqs):
        if f in mark:
            ax.get_yticklabels()[i].set_color("crimson")
    ax.set_ylabel("channel (MHz)")
    ax.set_title(title, fontsize=9)
    time_ticks(ax, days)
    return im


@figure("channel_heatmaps")
def fig_channel_heatmaps():
    """Every channel's PSD over the month, both scenarios, with per-channel spread."""
    days25, f25, z25 = load("spec25_data.csv")
    days4, f4, z4 = load("spec4_data.csv")
    shared = set(f4)
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [25, 4], "height_ratios": [4, 1]})

    im = heatmap(axes[0][0], days25, f25, z25, mark=shared,
                 title="spec25 -- 25 channels (crimson ticks = the four spec4 frequencies)")
    heatmap(axes[0][1], days4, f4, z4, title="spec4 -- 4 channels")
    fig.colorbar(im, ax=axes[0].tolist(), label="PSD (raw CSV units)", shrink=0.9)

    for ax, freqs, z, lab in ((axes[1][0], f25, z25, "spec25"), (axes[1][1], f4, z4, "spec4")):
        ax.plot(range(len(freqs)), z.std(0), color="crimson", marker="o", ms=3)
        ax.set_xlim(-0.5, len(freqs) - 0.5)
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f"{f:.0f}" for f in freqs], fontsize=6)
        ax.set_ylabel("std", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{lab}: per-channel std (variability)", fontsize=9)
    axes[1][0].tick_params(labelsize=7)

    fig.suptitle("ElectroSense Madrid, 2872 rows @ 15 min (2021-05-28 to 06-26), raw PSD units",
                 fontsize=10)
    fig.savefig(os.path.join(OUT, "channel_heatmaps.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("channel std  spec25:", np.round(z25.std(0), 2).tolist())
    print("channel std  spec4 :", np.round(z4.std(0), 2).tolist())


@figure("crop_check")
def fig_crop_check():
    """Is spec4 just a crop of spec25? Same timestamps and centre frequencies,
    but the measured values differ -- except at 500 MHz."""
    days4, f4, z4 = load("spec4_data.csv")
    _, f25, z25 = load("spec25_data.csv")
    idx = [f25.index(f) for f in f4]
    zsub = z25[:, idx]
    diff = z4 - zsub

    fig, axes = plt.subplots(3, 1, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    im0 = heatmap(axes[0], days4, f4, z4, title="spec4, its own four channels")
    im1 = heatmap(axes[1], days4, f4, zsub,
                  title="spec25, the same four centre frequencies (column crop)")
    fig.colorbar(im1, ax=axes[:2].tolist(), label="PSD (raw CSV units)", shrink=0.9)

    v = np.nanmax(np.abs(diff))
    im2 = axes[2].imshow(diff.T, aspect="auto", origin="lower", interpolation="nearest",
                         extent=[days4[0], days4[-1] + 1 / 96, -0.5, len(f4) - 0.5],
                         cmap="coolwarm", vmin=-v, vmax=v)
    axes[2].set_yticks(range(len(f4)))
    axes[2].set_yticklabels([f"{f:.0f}" for f in f4], fontsize=7)
    axes[2].set_ylabel("channel (MHz)")
    time_ticks(axes[2], days4)
    per = [f"{f:.0f}: max|d|={np.abs(diff[:, i]).max():.1f}, "
           f"identical {int((diff[:, i] == 0).sum())}/{len(diff)}"
           for i, f in enumerate(f4)]
    axes[2].set_title("difference (spec4 - spec25 crop):  " + "; ".join(per), fontsize=8)
    fig.colorbar(im2, ax=axes[2], label="difference", shrink=0.9)

    fig.suptitle("spec4 is a column SUBSET of spec25 by centre frequency -- but not a value crop",
                 fontsize=10)
    fig.savefig(os.path.join(OUT, "crop_check.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for i, f in enumerate(f4):
        print(f"{f:.0f} MHz: identical rows {int((diff[:, i] == 0).sum())}/{len(diff)}, "
              f"max |diff| {np.abs(diff[:, i]).max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated figure names; default all")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(FIGURES)
    for name in wanted:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; have {list(FIGURES)}")
        FIGURES[name]()
        print(f"figure '{name}' -> figures/")


if __name__ == "__main__":
    main()
