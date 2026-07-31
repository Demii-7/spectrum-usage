#!/usr/bin/env python3
"""Plot mean and SEM of MAE/RMSE across seeds for each model as bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

REPRESENTATIONS = ["1d", "2d", "4d"]
METRICS = [
    ("test_mean_horizon_mae_db", "MAE (dB)"),
    ("test_mean_horizon_rmse_db", "RMSE (dB)"),
]
HORIZON_METRICS = {
    "t1": [
        ("test_mae_db_t1", "MAE (dB)"),
        ("test_rmse_db_t1", "RMSE (dB)"),
    ],
    "t15": [
        ("test_mae_db_t15", "MAE (dB)"),
        ("test_rmse_db_t15", "RMSE (dB)"),
    ],
    "t60": [
        ("test_mae_db_t60", "MAE (dB)"),
        ("test_rmse_db_t60", "RMSE (dB)"),
    ],
}
HORIZON_LABELS = {
    "t1": "1 minute forecast",
    "t15": "15 minute forecast",
    "t60": "60 minute forecast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory containing seeds_*.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "seed_bars.png",
    )
    return parser.parse_args()


def load_seed_data(tables_dir: Path) -> pd.DataFrame:
    frames = []
    for rep in REPRESENTATIONS:
        path = tables_dir / f"seeds_{rep}.csv"
        df = pd.read_csv(path)
        df["representation"] = rep
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    horizon_rmse_cols = ["test_rmse_db_t1", "test_rmse_db_t15", "test_rmse_db_t60"]
    missing = data["test_mean_horizon_rmse_db"].isna()
    data.loc[missing, "test_mean_horizon_rmse_db"] = (
        data.loc[missing, horizon_rmse_cols].mean(axis=1)
    )
    return data


def compute_stats(
    data: pd.DataFrame, metrics: list[tuple[str, str]] = METRICS
) -> pd.DataFrame:
    grouped = data.groupby(["representation", "model"])
    stats = []
    for (rep, model), group in grouped:
        if model.startswith("lookbackmean") or pd.isna(group["seed"]).all():
            continue
        for col, label in metrics:
            values = group[col].dropna()
            n = len(values)
            if n == 0:
                continue
            stats.append(
                {
                    "representation": rep,
                    "model": model,
                    "metric": label,
                    "mean": values.mean(),
                    "sem": values.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                }
            )
    return pd.DataFrame(stats)


_DISPLAY_NAMES = {
    "linearar": "LinearAR",
    "residuallinearar": "Residual LinearAR",
    "autoformer_csa": "AutoformerCSA",
    "lstmattn": "AttnLSTM",
    "vanillalstm": "LSTM",
    "residualvanillalstm": "Residual LSTM",
    "temporalconvnet": "TCN",
    "convlstm": "ConvLSTM",
    "residualconvlstm": "Residual ConvLSTM",
    "dswinlstm_i": "DSWinLSTM",
}


def format_model_name(name: str) -> str:
    for key, display in _DISPLAY_NAMES.items():
        if name.startswith(key):
            return display
    return name.replace("_", " ").title()


def format_tick_label(name: str) -> str:
    if name == "AutoformerCSA":
        return "Autoformer\nCSA"
    if name == "Residual LSTM":
        return "Residual\nLSTM"
    if name == "DSWinLSTM":
        return "DSWin\nLSTM"
    if len(name) > 13 and " " in name:
        split_at = name.rfind(" ", 0, len(name) // 2 + 2)
        return name[:split_at] + "\n" + name[split_at + 1:]
    return name


def extract_lookback(
    data: pd.DataFrame, metrics: list[tuple[str, str]] = METRICS
) -> dict[tuple[str, str], float]:
    lb = data[data["model"].str.startswith("lookbackmean")]
    baselines = {}
    for rep in REPRESENTATIONS:
        rows = lb[lb["representation"] == rep]
        if rows.empty:
            continue
        for col, label in metrics:
            values = rows[col].dropna()
            if not values.empty:
                baselines[(rep, label)] = float(values.mean())
    return baselines


def plot(stats: pd.DataFrame, baselines: dict[tuple[str, str], float],
         output: Path) -> None:
    width_ratios = [
        max(1, stats[stats["representation"] == rep]["model"].nunique())
        for rep in REPRESENTATIONS
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(8.5, 3.75), constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios},
        sharey="row",
    )

    for row_idx, (_, metric_label) in enumerate(METRICS):
        row_stats = stats[stats["metric"] == metric_label]
        ymax = row_stats["mean"].max() + row_stats["sem"].max() * 2 + 0.05

        for col_idx, rep in enumerate(REPRESENTATIONS):
            ax = axes[row_idx, col_idx]
            subset = stats[
                (stats["representation"] == rep) & (stats["metric"] == metric_label)
            ].copy()
            if subset.empty:
                ax.set_visible(False)
                continue
            subset = subset.sort_values("mean", ascending=False)
            labels = [format_tick_label(format_model_name(m)) for m in subset["model"]]
            means = subset["mean"].to_numpy()
            sems = subset["sem"].to_numpy()
            x = np.arange(len(labels))
            bl = baselines.get((rep, metric_label))
            for i in range(len(labels)):
                if bl is None:
                    color = "#5C6BC0"
                elif means[i] < bl:
                    color = "#43A047"
                else:
                    color = "#E53935"
                model_name = subset.iloc[i]["model"]
                alpha = 0.35 if "linearar" in model_name else 1.0
                ax.bar(x[i], means[i], yerr=sems[i], width=0.6, edgecolor="#263238",
                       linewidth=0.6, capsize=3, color=color, alpha=alpha,
                       ecolor="#263238")
            ax.set_xticks(x)
            tick_labels = ax.set_xticklabels(
                labels, rotation=0, ha="center", fontsize=7
            )
            for tick_label in tick_labels:
                tick_label.set_linespacing(0.95)
                tick_label.set_multialignment("right")
            ax.set_ylabel(metric_label)
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(f"{rep.upper()} models", fontweight="bold")
            if bl is not None:
                ax.axhline(bl, color="black", linestyle="--", linewidth=1.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_horizons(
    horizon_stats: dict[str, pd.DataFrame],
    horizon_baselines: dict[str, dict[tuple[str, str], float]],
    output: Path,
) -> None:
    font_size = 7
    fig, axes = plt.subplots(
        2, 3, figsize=(10.5, 3.75), constrained_layout=True, sharey="row"
    )

    for col_idx, horizon in enumerate(HORIZON_METRICS):
        stats = horizon_stats[horizon]
        baselines = horizon_baselines[horizon]
        for row_idx, (_, metric_label) in enumerate(HORIZON_METRICS[horizon]):
            ax = axes[row_idx, col_idx]
            row_stats = stats[stats["metric"] == metric_label]
            ymax = row_stats["mean"].max() + row_stats["sem"].max() * 2 + 0.05
            positions = []
            labels = []
            group_bounds = []
            position = 0

            for rep in REPRESENTATIONS:
                subset = stats[
                    (stats["representation"] == rep)
                    & (stats["metric"] == metric_label)
                ].sort_values("mean", ascending=False)
                if subset.empty:
                    continue
                start = position
                bl = baselines.get((rep, metric_label))
                for _, item in subset.iterrows():
                    mean = float(item["mean"])
                    sem = float(item["sem"])
                    if bl is None:
                        color = "#5C6BC0"
                    elif mean < bl:
                        color = "#43A047"
                    else:
                        color = "#E53935"
                    alpha = 0.35 if "linearar" in item["model"] else 1.0
                    ax.bar(
                        position,
                        mean,
                        yerr=sem,
                        width=0.48,
                        edgecolor="#263238",
                        linewidth=0.6,
                        capsize=2,
                        color=color,
                        alpha=alpha,
                        ecolor="#263238",
                    )
                    positions.append(position)
                    labels.append(format_tick_label(format_model_name(item["model"])))
                    position += 1
                group_bounds.append((rep, start, position - 1, bl))
                position += 0.5

            ax.set_xticks(positions)
            tick_labels = ax.set_xticklabels(
                labels,
                rotation=90,
                ha="center",
                va="top",
                fontsize=4,
            )
            for tick_label in tick_labels:
                tick_label.set_linespacing(0.95)
                tick_label.set_multialignment("right")
            ax.set_ylim(0, ymax)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=font_size)
            else:
                ax.set_ylabel("")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(HORIZON_LABELS[horizon], fontsize=font_size, fontweight="bold")
            ax.tick_params(axis="both", labelsize=font_size)

            for rep, start, end, bl in group_bounds:
                if bl is not None:
                    ax.hlines(bl, start - 0.3, end + 0.3, color="black",
                              linestyle="--", linewidth=0.9)
                midpoint = (start + end) / 2
                ax.text(
                    midpoint,
                    0.96,
                    rep.upper(),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=font_size,
                    fontweight="bold",
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_seed_data(args.tables_dir)
    outputs = [(METRICS, args.output)]
    outputs.extend(
        (metrics, args.output.with_name(f"{args.output.stem}_{horizon}{args.output.suffix}"))
        for horizon, metrics in HORIZON_METRICS.items()
    )
    for metrics, output in outputs:
        stats = compute_stats(data, metrics)
        baselines = extract_lookback(data, metrics)
        plot(stats, baselines, output)
        print(f"Wrote {output}")

    horizon_stats = {
        horizon: compute_stats(data, metrics)
        for horizon, metrics in HORIZON_METRICS.items()
    }
    horizon_baselines = {
        horizon: extract_lookback(data, metrics)
        for horizon, metrics in HORIZON_METRICS.items()
    }
    combined_output = args.output.with_name(f"{args.output.stem}_horizons{args.output.suffix}")
    plot_combined_horizons(horizon_stats, horizon_baselines, combined_output)
    print(f"Wrote {combined_output}")


if __name__ == "__main__":
    main()
