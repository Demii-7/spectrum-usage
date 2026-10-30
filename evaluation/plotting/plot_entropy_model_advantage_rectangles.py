#!/usr/bin/env python3
"""Compare regional model MAE with LookbackMean using entropy rectangles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "lookbackmean2d"
MODEL_LABELS = {
    "lookbackmean2d": "LookbackMean",
    "autoformer_csa": "Autoformer CSA",
    "linearar2d": "Linear AR2D",
    "lstmattn": "LSTM attention",
    "residuallinearar2d": "Residual Linear AR2D",
    "residualvanillalstm": "Residual Vanilla LSTM",
    "temporalconvnet": "Temporal ConvNet",
    "vanillalstm": "Vanilla LSTM",
}
MODEL_MARKERS = {
    "autoformer_csa": "o",
    "linearar2d": "s",
    "lstmattn": "^",
    "residuallinearar2d": "D",
    "residualvanillalstm": "P",
    "temporalconvnet": "X",
    "vanillalstm": "v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "evaluation/results/tables/entropy_band_metrics_2d.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "evaluation/results/figures/entropy_model_advantage_rectangles.png",
    )
    parser.add_argument("--table-output", type=Path, default=None)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--metric", default="mean_horizon_mae_db")
    parser.add_argument("--rectangle-width", type=float, default=None)
    return parser.parse_args()


def prepare(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {"model_name", "band_id", "entropy", args.metric}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    selected = data.loc[:, ["model_name", "seed", "band_id", "entropy", args.metric]].copy()
    selected = selected.dropna(subset=["entropy", args.metric])
    selected = (
        selected.groupby(["model_name", "band_id"], as_index=False)
        .agg(entropy=("entropy", "mean"), mae_db=(args.metric, "mean"))
    )
    baseline = selected.loc[selected["model_name"].eq(args.baseline), ["band_id", "mae_db"]].rename(
        columns={"mae_db": "baseline_mae_db"}
    )
    if baseline.empty:
        raise ValueError(f"Baseline {args.baseline!r} is not present")
    selected = selected.merge(baseline, on="band_id", validate="many_to_one")
    selected["gap_db"] = selected["mae_db"] - selected["baseline_mae_db"]
    selected["comparison"] = np.where(
        selected["model_name"].eq(args.baseline), "baseline",
        np.where(selected["gap_db"] < 0, "advantage", "disadvantage"),
    )
    return selected.sort_values(["entropy", "model_name", "band_id"])


def prepare_horizons(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for horizon, metric in (("1 minute", "t1_mae_db"), ("15 minutes", "t15_mae_db"), ("60 minutes", "t60_mae_db")):
        horizon_args = argparse.Namespace(**vars(args))
        horizon_args.metric = metric
        frame = prepare(data, horizon_args)
        frame["horizon"] = horizon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def width_for(entropy: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return requested
    values = np.unique(np.sort(entropy))
    return float(np.median(np.diff(values)) * 0.85) if len(values) > 1 else 0.03


def plot(data: pd.DataFrame, args: argparse.Namespace) -> None:
    font_size = 7
    models = [model for model in data["model_name"].unique() if model != args.baseline]
    comparison_colors = {"advantage": "#2ca25f", "disadvantage": "#de2d26"}
    width = width_for(data["entropy"].to_numpy(dtype=float), args.rectangle_width)
    jitter = {
        model: (index - (len(models) - 1) / 2) * width * 0.72
        for index, model in enumerate(models)
    }
    fig, axes = plt.subplots(1, 2, figsize=(4.0, 1.75), sharex=True, sharey=True, constrained_layout=True)
    ymax = float(data[["mae_db", "baseline_mae_db"]].max().max())
    baseline = data.loc[data["model_name"].eq(args.baseline)].drop_duplicates("band_id")
    all_points = data.loc[~data["model_name"].eq(args.baseline)]
    for axis, comparison in zip(
        axes,
        ("advantage", "disadvantage"),
    ):
        subset = all_points.loc[all_points["comparison"].eq(comparison)]
        for _, row in subset.iterrows():
            bottom = min(row["mae_db"], row["baseline_mae_db"])
            height = abs(row["gap_db"])
            axis.add_patch(Rectangle(
                (row["entropy"] + jitter[row["model_name"]] - width / 2, bottom), width, height,
                facecolor=comparison_colors[comparison], edgecolor="#263238",
                linewidth=0.6, alpha=0.72,
            ))
        axis.set_xlabel("Entropy", fontsize=font_size)
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        axis.set_ylim(0, ymax * 1.05)
        axis.tick_params(axis="both", labelsize=font_size)
        axis.scatter(
            baseline["entropy"], baseline["baseline_mae_db"], marker="_", s=90,
            linewidths=1.2, color="#202124", label="LookbackMean", zorder=4,
        )
    axes[0].set_ylabel("Mean MAE across horizons (dB)", fontsize=font_size)
    axes[0].legend(
        [plt.Line2D([], [], color="#2ca25f", linewidth=7, alpha=0.72),
         plt.Line2D([], [], color="#202124", marker="_", linewidth=0, markersize=10)],
        ["Model advantage", "LookbackMean"], frameon=False, fontsize=font_size, loc="upper left",
    )
    axes[1].legend(
        [plt.Line2D([], [], color="#de2d26", linewidth=7, alpha=0.72),
         plt.Line2D([], [], color="#202124", marker="_", linewidth=0, markersize=10)],
        ["Model disadvantage", "LookbackMean"], frameon=False, fontsize=font_size, loc="upper left",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = prepare(pd.read_csv(args.input), args)
    table_output = args.table_output or args.output.with_suffix(".csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(table_output, index=False)
    plot(data, args)
    print(f"Wrote {len(data)} rows to {table_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
