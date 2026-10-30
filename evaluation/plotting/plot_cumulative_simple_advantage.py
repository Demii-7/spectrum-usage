#!/usr/bin/env python3
"""Plot how regional simple-model advantage accumulates with entropy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIMPLE_MODELS = ("lookbackmean2d", "linearar2d", "residuallinearar2d")
COMPLEX_MODELS = (
    "autoformer_csa",
    "lstmattn",
    "residualvanillalstm",
    "temporalconvnet",
    "vanillalstm",
)
COLORS = {
    "lookbackmean2d": "#e45756",
    "linearar2d": "#4c78a8",
    "residuallinearar2d": "#54a24b",
}
LABELS = {
    "lookbackmean2d": "LookbackMean",
    "linearar2d": "LinearAR2D",
    "residuallinearar2d": "ResidualLinearAR2D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "evaluation/results/tables/entropy_band_metrics_2d.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "evaluation/results/figures/cumulative_simple_advantage.png",
    )
    parser.add_argument("--table-output", type=Path, default=None)
    return parser.parse_args()


def prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model_name", "band_id", "mean_horizon_mae_db", "entropy", "bin_count"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")

    regional = (
        data.loc[data["model_name"].isin((*SIMPLE_MODELS, *COMPLEX_MODELS))]
        .groupby(["model_name", "band_id"], as_index=False)
        .agg(mae_db=("mean_horizon_mae_db", "mean"), entropy=("entropy", "mean"), bin_count=("bin_count", "mean"))
    )
    pivot = regional.pivot(index="band_id", columns="model_name", values="mae_db")
    metadata = regional.drop_duplicates("band_id").set_index("band_id")[["entropy", "bin_count"]]
    complex_mae = pivot[list(COMPLEX_MODELS)].mean(axis=1)

    rows = []
    for model in SIMPLE_MODELS:
        gap = complex_mae - pivot[model]
        total = float((gap * metadata["bin_count"]).sum())
        ordered = pd.DataFrame({"band_id": gap.index, "gap_db": gap.values}).join(metadata, on="band_id")
        ordered = ordered.sort_values(["entropy", "band_id"]).reset_index(drop=True)
        ordered["model_name"] = model
        ordered["cumulative_bins"] = ordered["bin_count"].cumsum() / ordered["bin_count"].sum()
        ordered["cumulative_advantage"] = (ordered["gap_db"] * ordered["bin_count"]).cumsum() / total
        ordered["region_order"] = np.arange(1, len(ordered) + 1)
        rows.append(ordered)
    cumulative = pd.concat(rows, ignore_index=True)

    ecdf = cumulative[["model_name", "band_id", "gap_db", "entropy"]].copy()
    ecdf["fraction_regions"] = ecdf.groupby("model_name")["gap_db"].rank(method="average", pct=True)
    return cumulative, ecdf


def plot(cumulative: pd.DataFrame, ecdf: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    for model, group in cumulative.groupby("model_name", sort=False):
        group = group.sort_values("region_order")
        axes[0].plot(
            group["cumulative_bins"], group["cumulative_advantage"],
            marker="o", markersize=4.5, linewidth=2, color=COLORS[model], label=LABELS[model],
        )
        for _, row in group.iloc[[0, 5, 10, 15, -1]].iterrows():
            axes[0].annotate(
                row["band_id"], (row["cumulative_bins"], row["cumulative_advantage"]),
                xytext=(3, 4), textcoords="offset points", fontsize=7, color=COLORS[model],
            )
    axes[0].plot([0, 1], [0, 1], "--", color="#616161", linewidth=1.2, label="Proportional contribution")
    axes[0].set_title("Cumulative advantage by entropy order")
    axes[0].set_xlabel("Cumulative fraction of frequency bins\n(lowest to highest entropy)")
    axes[0].set_ylabel("Cumulative fraction of total simple-model advantage")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.03)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    for model, group in ecdf.groupby("model_name", sort=False):
        values = np.sort(group["gap_db"].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        axes[1].step(values, y, where="post", linewidth=2, color=COLORS[model], label=LABELS[model])
    axes[1].axvline(0, color="#263238", linewidth=1)
    axes[1].set_title("Regional simple-model advantage")
    axes[1].set_xlabel("Complex-model MAE - simple-model MAE (dB)")
    axes[1].set_ylabel("Fraction of regions")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Simple-model advantage is distributed across entropy regions", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cumulative, ecdf = prepare(pd.read_csv(args.input))
    table_output = args.table_output or args.output.with_suffix(".csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    cumulative.to_csv(table_output, index=False)
    plot(cumulative, ecdf, args.output)
    print(f"Wrote {len(cumulative)} rows to {table_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
