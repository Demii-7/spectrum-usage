#!/usr/bin/env python3
"""Plot per-band entropy and MAE comparisons as colored rectangles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--band-column", default="band")
    parser.add_argument("--entropy-column", default="entropy")
    parser.add_argument("--lookback-mae-column", default="lookback_mae_db")
    parser.add_argument("--model-mae-column", default="model_mae_db")
    parser.add_argument("--rectangle-width", type=float, default=None)
    parser.add_argument("--label-bands", action="store_true")
    return parser.parse_args()


def rectangle_width(entropy: np.ndarray) -> float:
    values = np.unique(np.sort(entropy))
    if len(values) > 1:
        return float(np.median(np.diff(values)) * 0.55)
    return 0.03


def plot(data: pd.DataFrame, args: argparse.Namespace) -> None:
    columns = [
        args.band_column,
        args.entropy_column,
        args.lookback_mae_column,
        args.model_mae_column,
    ]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    data = data[columns].dropna().copy()
    if data.empty:
        raise ValueError("No complete rows to plot")

    entropy = data[args.entropy_column].to_numpy(dtype=float)
    differences = (
        data[args.model_mae_column].to_numpy(dtype=float)
        - data[args.lookback_mae_column].to_numpy(dtype=float)
    )
    width = args.rectangle_width or rectangle_width(entropy)
    color_limit = max(float(np.max(np.abs(differences))), 1e-12)
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("PRGn")

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for (_, row), difference in zip(data.iterrows(), differences):
        lookback_mae = float(row[args.lookback_mae_column])
        model_mae = float(row[args.model_mae_column])
        bottom = min(lookback_mae, model_mae)
        height = abs(model_mae - lookback_mae)
        x = float(row[args.entropy_column]) - width / 2
        ax.add_patch(
            Rectangle(
                (x, bottom),
                width,
                height,
                facecolor=cmap(norm(difference)),
                edgecolor="#263238",
                linewidth=0.6,
            )
        )
        if height == 0:
            ax.hlines(bottom, x, x + width, color=cmap(norm(0)), linewidth=3)
        if args.label_bands:
            ax.annotate(
                str(row[args.band_column]),
                (float(row[args.entropy_column]), max(lookback_mae, model_mae)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    mae = data[[args.lookback_mae_column, args.model_mae_column]].to_numpy(dtype=float)
    padding = max(float(np.ptp(mae)) * 0.08, 0.05)
    ax.set_xlim(float(entropy.min()) - width, float(entropy.max()) + width)
    ax.set_ylim(max(0, float(mae.min()) - padding), float(mae.max()) + padding)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("MAE (dB)")
    ax.set_title(f"{args.model_name} vs LookbackMean")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label(
        f"{args.model_name} MAE - LookbackMean MAE (dB)\n"
        "purple: model lower, green: LookbackMean lower"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(pd.read_csv(args.input), args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
