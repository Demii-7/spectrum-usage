#!/usr/bin/env python3
"""Plot the three all-region MAE evaluations as capped error rectangles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "evaluation/results/tables/spatial_grid_receiver_summary.csv"
DEFAULT_OUTPUT = ROOT / "evaluation/results/figures/spatial_error_caps.png"

HORIZONS = ("1", "15", "60")
HORIZON_LABELS = {"1": "t+1 minute", "15": "t+15 minutes", "60": "t+60 minutes"}
LOOKBACK_MODEL = "lookbackmean4d"
METRICS = (
    ("full_grid_mae_db", "Full grid / IDW", "#4C78A8"),
    ("receiver_grid_mae_db", "Receiver grid / IDW", "#F58518"),
    ("physical_receiver_mae_db", "Physical receiver", "#7A5195"),
)
MODEL_LABELS = {
    "convlstm": "ConvLSTM",
    "lookbackmean4d": "LookbackMean\n4D",
    "linearar4d": "LinearAR\n4D",
    "residuallinearar4d": "Residual\nLinearAR 4D",
    "residualconvlstm": "Residual\nConvLSTM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "model",
        "condition",
        "region_set",
        "horizon",
        *(metric[0] for metric in METRICS),
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")

    selected = data.loc[
        data["region_set"].eq("all_regions") & data["condition"].eq("geometry_correct")
    ].copy()
    selected["horizon"] = selected["horizon"].astype(str)
    selected = selected.loc[selected["horizon"].isin(HORIZONS)]
    if selected.empty:
        raise ValueError("No all-region geometry_correct rows were found")
    duplicate = selected.duplicated(["model", "horizon"])
    if duplicate.any():
        raise ValueError("Expected one all-region row per model and horizon")
    return selected


def plot(data: pd.DataFrame, output: Path) -> None:
    models = [model for model in dict.fromkeys(data["model"]) if model != LOOKBACK_MODEL]
    baseline = data.loc[data["model"].eq(LOOKBACK_MODEL)].set_index("horizon")
    if set(baseline.index) != set(HORIZONS):
        raise ValueError("LookbackMean4D must have one all-region row per horizon")

    figure, axes = plt.subplots(
        1,
        len(HORIZONS),
        figsize=(5.5, 2.0),
        sharey=True,
        constrained_layout=True,
    )
    maximum = float(data[list(column for column, _, _ in METRICS)].max().max())
    width = min(0.62, 0.78 / max(1, len(models)))

    for axis, horizon in zip(axes, HORIZONS):
        horizon_data = data.loc[
            data["horizon"].eq(horizon) & data["model"].isin(models)
        ].set_index("model")
        for model_index, model in enumerate(models):
            if model not in horizon_data.index:
                continue
            row = horizon_data.loc[model]
            values = {column: float(row[column]) for column, _, _ in METRICS}
            body = Rectangle(
                (model_index - width / 2, 0.0),
                width,
                max(values.values()),
                facecolor="#D9D9D9",
                edgecolor="#4D4D4D",
                linewidth=0.55,
                zorder=1,
            )
            axis.add_patch(body)
            for column, _, color in METRICS:
                axis.plot(
                    [model_index - width * 0.5, model_index + width * 0.5],
                    [values[column], values[column]],
                    color=color,
                    linewidth=1.5,
                    solid_capstyle="round",
                    zorder=3,
                )

        baseline_row = baseline.loc[horizon]
        for column, _, color in METRICS:
            axis.axhline(
                float(baseline_row[column]),
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
        axis.set_title(HORIZON_LABELS[horizon], fontsize=7.5, fontweight="bold")
        axis.set_xlim(-0.5, len(models) - 0.5)
        axis.set_ylim(0.0, maximum * 1.10)
        axis.set_xticks(range(len(models)))
        axis.set_xticklabels(
            [MODEL_LABELS.get(model, model.replace("_", " ").title()) for model in models],
            fontsize=6.5,
        )
        axis.tick_params(axis="y", labelsize=6.5)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("MAE (dB)", fontsize=8)
    handles = [
        Line2D([], [], color=color, linewidth=2.0, label=label)
        for _, label, color in METRICS
    ]
    handles.append(
        Line2D(
            [], [], color="#4D4D4D", linewidth=1.0, linestyle="--",
            label="LookbackMean4D",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=1.5,
        columnspacing=1.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot(load_data(args.table), args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
