#!/usr/bin/env python3
"""Plot spectral-context ablation results across forecast horizons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE = Path(__file__).resolve().parent.parent / "results" / "tables" / "spectrum_context.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "spectral_context_bars.png"

CONDITIONS = ("full_context", "region_only")
CONDITION_LABELS = {"full_context": "Full context", "region_only": "Region only"}
HORIZONS = (
    ("t1", "1 minute forecast"),
    ("t15", "15 minute forecast"),
    ("t60", "60 minute forecast"),
)
COLORS = {"full_context": "#2E7D8A", "region_only": "#D9822B"}

DISPLAY_NAMES = {
    "autoformer_csa": "Autoformer\nCSA",
    "lstmattn": "AttnLSTM",
    "residualvanillalstm": "Residual\nLSTM",
    "temporalconvnet": "TCN",
    "vanillalstm": "LSTM",
    "linearar2d": "LinearAR",
    "residuallinearar2d": "Residual\nLinearAR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def load_matched_data(table: Path) -> pd.DataFrame:
    data = pd.read_csv(table)
    data = data[data["run_status"] == "completed"].copy()
    keys = ["model_name", "seed", "band_id"]
    matched = data.groupby(keys)["condition"].nunique()
    matched = matched[matched == len(CONDITIONS)].index
    return data.set_index(keys).loc[matched].reset_index()


def compute_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, _ in HORIZONS:
        column = f"test_mae_db_{horizon}"
        grouped = data.groupby(["band_id", "model_name", "condition"])[column]
        for (band_id, model, condition), values in grouped:
            values = values.dropna()
            if values.empty:
                continue
            rows.append({
                "band_id": band_id,
                "model_name": model,
                "condition": condition,
                "horizon": horizon,
                "mean": values.mean(),
                "sem": values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                "n": len(values),
            })
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, output: Path) -> None:
    bands = sorted(stats["band_id"].unique(), key=lambda band: int(band.removeprefix("R")))
    fig, axes = plt.subplots(
        len(bands), 3, figsize=(10.5, max(7.0, len(bands) * 0.85)),
        constrained_layout=True, sharex=False, sharey="row",
    )
    axes = np.atleast_2d(axes)
    for row, band_id in enumerate(bands):
        for col, (horizon, horizon_label) in enumerate(HORIZONS):
            ax = axes[row, col]
            subset = stats[(stats["band_id"] == band_id) & (stats["horizon"] == horizon)]
            models = sorted(subset["model_name"].unique(), key=lambda m: subset[subset["model_name"] == m]["mean"].mean())
            x = np.arange(len(models))
            width = 0.36
            for condition_idx, condition in enumerate(CONDITIONS):
                values = []
                errors = []
                for model in models:
                    item = subset[(subset["model_name"] == model) & (subset["condition"] == condition)]
                    values.append(float(item["mean"].iloc[0]) if not item.empty else np.nan)
                    errors.append(float(item["sem"].iloc[0]) if not item.empty else 0.0)
                ax.bar(
                    x + (condition_idx - 0.5) * width,
                    values,
                    width=width,
                    yerr=errors,
                    capsize=2,
                    label=CONDITION_LABELS[condition],
                    color=COLORS[condition],
                    edgecolor="#263238",
                    linewidth=0.5,
                    ecolor="#263238",
                )
            ax.set_xticks(x)
            ax.set_xticklabels([DISPLAY_NAMES.get(model, model) for model in models], fontsize=6)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylabel(f"{band_id}\nMAE (dB)" if col == 0 else "", fontsize=7)
            ax.set_title(horizon_label, fontsize=8, fontweight="bold") if row == 0 else None
            ax.tick_params(axis="y", labelsize=7)
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Spectral context ablation: MAE by frequency region", fontsize=10, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stats = compute_stats(load_matched_data(args.table))
    plot(stats, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
