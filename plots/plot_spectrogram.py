#!/usr/bin/env python3
"""Plot merged 250-bin per-site ground-truth spectrum heatmaps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


LABELS = ("CC1", "CC2", "LW1")
BINS_PER_SITE = 250


def smooth_like_tss_lcd(data, window_length=51, polyorder=2):
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    scale = col_max - col_min
    scale[scale == 0] = 1e-8

    norm = (data - col_min) / scale
    smooth = savgol_filter(
        norm,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    return smooth * scale + col_min


def plot(input_path, output_path, time_steps, from_end, smooth):
    data = pd.read_csv(input_path, header=0).to_numpy(dtype=float)
    if data.shape[1] != BINS_PER_SITE * len(LABELS):
        raise ValueError(f"expected 750 columns, found {data.shape[1]}")
    if smooth:
        data = smooth_like_tss_lcd(data)
    if time_steps is not None:
        data = data[-time_steps:] if from_end else data[:time_steps]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, axes = plt.subplots(1, len(LABELS), figsize=(14.4, 1.65), squeeze=False)

    for i, (ax, label) in enumerate(zip(axes[0], LABELS)):
        block = data[:, i * BINS_PER_SITE : (i + 1) * BINS_PER_SITE]
        finite = block[np.isfinite(block)]
        vmin, vmax = np.percentile(finite, [1, 99])

        im = ax.imshow(
            block.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=[0, block.shape[0], 0, BINS_PER_SITE],
        )

        ax.set_title(f"{label}\nGround-Truth", fontsize=8, fontweight="bold", pad=2)
        ax.set_xlabel("Time Step", fontsize=8, fontweight="bold", labelpad=1)
        ax.set_ylabel("Frequency Bin", fontsize=8, fontweight="bold", labelpad=1)
        ax.set_xlim(0, block.shape[0])
        ax.set_xticks(np.arange(500, block.shape[0] + 1, 500))
        ax.set_yticks(np.arange(50, BINS_PER_SITE + 1, 50))
        ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9, pad=1)

        colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
        colorbar.ax.tick_params(labelsize=7, direction="in", length=2, width=0.8, pad=1)

    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.32, top=0.78, wspace=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/merged_power_data_sub6GHz_avg_per_minute.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/merged_ground_truth.png"),
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=3000,
        help="Number of time steps to plot. Use 0 to plot the full CSV.",
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Plot the last --max-time rows instead of the first rows.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Plot raw values instead of applying TSS-LCD-style smoothing.",
    )
    args = parser.parse_args()

    time_steps = None if args.max_time == 0 else args.max_time
    plot(args.input, args.output, time_steps, args.from_end, not args.no_smooth)


if __name__ == "__main__":
    main()
