#!/usr/bin/env python3
"""Plot entropy-ordered regional MAE with LookbackMean reference markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "evaluation/results/tables/entropy_band_metrics_2d.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/results/figures/regional_mae_vs_lookback.png",
    )
    parser.add_argument("--table-output", type=Path, default=None)
    parser.add_argument("--metric", default="mean_horizon_mae_db")
    parser.add_argument("--baseline", default="lookbackmean2d")
    parser.add_argument("--entropy-column", default="entropy")
    return parser.parse_args()


def aggregate(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {"model_name", "seed", "band_id", args.metric, args.entropy_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")

    selected = data[["model_name", "seed", "band_id", args.metric, args.entropy_column]].copy()
    selected["region_id"] = selected["band_id"].str.extract(r"R(\d+)").astype(int)
    selected = selected.dropna(subset=[args.metric, args.entropy_column])
    region_entropy = selected.groupby("region_id", as_index=False)[args.entropy_column].mean()
    region_entropy = region_entropy.sort_values(args.entropy_column).reset_index(drop=True)
    region_entropy["region_order"] = np.arange(len(region_entropy))

    averages = (
        selected.groupby(["model_name", "region_id"], as_index=False)[args.metric]
        .mean()
        .rename(columns={args.metric: "mae_db"})
    )
    result = averages.merge(region_entropy, on="region_id", validate="many_to_one")
    result["region_label"] = result.apply(
        lambda row: f"R{int(row['region_id'])}\n{row[args.entropy_column]:.3f}", axis=1
    )
    baseline = result.loc[result["model_name"].eq(args.baseline), ["region_id", "mae_db"]].rename(
        columns={"mae_db": "lookback_mae_db"}
    )
    result = result.merge(baseline, on="region_id", validate="many_to_one")
    result["gap_db"] = result["mae_db"] - result["lookback_mae_db"]
    return result.sort_values(["region_order", "model_name"])


def plot(data: pd.DataFrame, args: argparse.Namespace) -> None:
    models = [model for model in data["model_name"].unique() if model != args.baseline]
    if not models:
        raise ValueError("No non-baseline models to plot")
    labels = data.drop_duplicates("region_id").sort_values("region_order")["region_label"].tolist()
    x = np.arange(len(labels))
    columns = 2
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, model in zip(axes, models):
        group = data.loc[data["model_name"].eq(model)].sort_values("region_order")
        ax.bar(x, group["mae_db"], color="#4c78a8", alpha=0.82, label=model)
        ax.scatter(x, group["lookback_mae_db"], color="#e45756", marker="_", s=180, linewidths=2.5,
                   label="LookbackMean", zorder=3)
        ax.axhline(0, color="#263238", linewidth=0.8)
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    for ax in axes[len(models):]:
        ax.set_visible(False)
    for ax in axes[:len(models)]:
        ax.set_xticks(x, labels, rotation=0)
        ax.set_xlabel("Region ordered by entropy\n(region, mean entropy)")
        ax.set_ylabel("Mean MAE across horizons (dB)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Regional forecast error versus LookbackMean", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = aggregate(pd.read_csv(args.input), args)
    table_output = args.table_output or args.output.with_suffix(".csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(table_output, index=False)
    plot(data, args)
    print(f"Wrote {len(data)} rows to {table_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
