#!/usr/bin/env python3
"""Plot matched full-context and region-only spectral ablation errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "evaluation/results/tables/spectrum_context.csv"
OUTPUT = ROOT / "evaluation/results/figures/spectral_ablation_rectangles.png"
HORIZONS = (("t1", "1"), ("t15", "15"), ("t60", "60"))
HORIZON_ALPHA = {"1": 0.35, "15": 0.60, "60": 0.85}
MODEL_LABELS = {
    "autoformer_csa": "Autoformer\nCSA",
    "linearar2d": "LinearAR",
    "lstmattn": "AttnLSTM",
    "residuallinearar2d": "Residual\nLinearAR",
    "residualvanillalstm": "Residual\nLSTM",
    "temporalconvnet": "TCN",
    "vanillalstm": "LSTM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "seed", "condition", "band_id", "run_status"}
    required |= {f"test_mae_db_{horizon}" for horizon, _ in HORIZONS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")

    data = data.loc[data["run_status"].eq("completed")].copy()
    rows = []
    for horizon, horizon_label in HORIZONS:
        metric = f"test_mae_db_{horizon}"
        values = data.dropna(subset=[metric])
        grouped = values.groupby(["model_name", "band_id", "condition"], as_index=False)[metric].mean()
        pivot = grouped.pivot(index=["model_name", "band_id"], columns="condition", values=metric)
        if not {"full_context", "region_only"}.issubset(pivot.columns):
            raise ValueError("Each model, region, and horizon must have both ablation conditions")
        pivot = pivot.dropna(subset=["full_context", "region_only"]).reset_index()
        pivot["horizon"] = horizon_label
        pivot["full_context_mae"] = pivot["full_context"]
        pivot["region_only_mae"] = pivot["region_only"]
        difference = pivot["region_only_mae"] - pivot["full_context_mae"]
        pivot["winner"] = np.where(
            difference < -0.001, "region_only",
            np.where(difference > 0.001, "full_context", "tie"),
        )
        rows.append(pivot[["model_name", "band_id", "horizon", "full_context_mae", "region_only_mae", "winner"]])
    return pd.concat(rows, ignore_index=True)


def plot(data: pd.DataFrame, output: Path) -> None:
    model_error = data.assign(
        mean_mae=data[["full_context_mae", "region_only_mae"]].mean(axis=1)
    ).groupby("model_name")["mean_mae"].mean()
    models = model_error.sort_values(ascending=False).index.tolist()
    regions = sorted(data["band_id"].unique(), key=lambda value: int(str(value).removeprefix("R")))
    slots = [(region, horizon) for region in regions for _, horizon in HORIZONS]
    slot_offsets = {
        slot: (index - (len(slots) - 1) / 2) * 0.82 / len(slots)
        for index, slot in enumerate(slots)
    }
    width = 0.65 / len(slots)
    colors = {"region_only": "#d73027", "full_context": "#1a9850", "tie": "#777777"}

    figure, axis = plt.subplots(figsize=(max(5.5, len(models) * 0.85), 2), constrained_layout=True)
    for _, row in data.iterrows():
        x = models.index(row["model_name"]) + slot_offsets[(row["band_id"], row["horizon"])]
        bottom = min(row["full_context_mae"], row["region_only_mae"])
        height = abs(row["full_context_mae"] - row["region_only_mae"])
        color = colors[row["winner"]]
        alpha = HORIZON_ALPHA[row["horizon"]]
        axis.add_patch(Rectangle(
            (x - width / 2, bottom), width, height,
            facecolor=color, edgecolor=color, linewidth=0.45, alpha=alpha,
        ))
        # Endpoint caps keep small error differences visible without changing their scale.
        axis.plot(
            [x - width * 0.5, x + width * 0.5],
            [row["full_context_mae"], row["full_context_mae"]],
            color=color, alpha=alpha, linewidth=1.0, solid_capstyle="round", zorder=3,
        )
        axis.plot(
            [x - width * 0.5, x + width * 0.5],
            [row["region_only_mae"], row["region_only_mae"]],
            color=color, alpha=alpha, linewidth=1.0, solid_capstyle="round", zorder=3,
        )

    axis.set_xticks(range(len(models)))
    axis.set_xticklabels([MODEL_LABELS.get(model, model) for model in models], rotation=0, ha="center", fontsize=7)
    axis.set_ylabel("MAE (dB)", fontsize=8)
    maximum = float(data[["full_context_mae", "region_only_mae"]].max().max())
    axis.set_ylim(0, maximum * 1.08)
    axis.set_xlim(-0.5, len(models) - 0.5)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", labelsize=7)
    axis.legend(
        [plt.Line2D([], [], color="#d73027", linewidth=7), plt.Line2D([], [], color="#1a9850", linewidth=7),
         plt.Line2D([], [], linestyle="none"),
         plt.Line2D([], [], color="#263238", linewidth=7, alpha=0.35), plt.Line2D([], [], color="#263238", linewidth=7, alpha=0.60),
         plt.Line2D([], [], color="#263238", linewidth=7, alpha=0.85)],
        ["Region only advantage", "Full context advantage", " ", "1 min forecast", "15 min forecast", "60 min forecast"],
        frameon=False, fontsize=6.5, ncol=1, loc="upper right", bbox_to_anchor=(0.98, 0.98),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data = prepare(pd.read_csv(args.table))
    data.to_csv(args.output.with_suffix(".csv"), index=False)
    plot(data, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
