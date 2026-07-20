"""
Analysis and plotting script for hyperparameter search results.

Reads grid_search_results.csv, produces diagnostic plots and summary tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.config import load_config, resolve_path

HORIZONS = [1, 5, 15, 60]
HPARAMS = ["hidden_size", "num_layers", "dropout", "learning_rate", "batch_size", "weight_decay"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def _load_data(config_path: Path | None = None) -> tuple[pd.DataFrame, Path, Path, str, pd.Series, pd.DataFrame | None]:
    config = load_config(config_path)
    hp_cfg = config.get("hyperparameter_search")
    if not hp_cfg:
        raise ValueError("Config has no hyperparameter_search section.")

    output_base = resolve_path(hp_cfg["output_dir"])
    results_csv = output_base / "grid_search_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    df = df[df["status"] == "completed"].copy()
    metric = "mean_horizon_rmse_db"
    df = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()

    for col in HPARAMS + ["epochs_completed", "total_seconds", "training_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metric_vals = pd.to_numeric(df[metric], errors="coerce")
    best_idx = metric_vals.idxmin()
    best_row = df.loc[best_idx]
    best_trial_id = str(best_row["trial_id"])

    plots_dir = output_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epoch_log: pd.DataFrame | None = None
    ep_path = output_base / "epoch_logs" / f"{best_trial_id}_epochs.csv"
    if ep_path.exists():
        epoch_log = pd.read_csv(ep_path)
        for h in HORIZONS:
            col = f"val_loss_t{h}"
            if col in epoch_log.columns:
                epoch_log[col] = pd.to_numeric(epoch_log[col], errors="coerce")

    print(f"  Completed trials analyzed: {len(df)}")
    n_failed = len(pd.read_csv(results_csv))
    n_failed -= len(df)
    print(f"  Failed trials excluded:    {max(0, n_failed)}")
    print(f"  Best trial:                {best_trial_id}")
    print(f"  Best {metric}:              {metric_vals.min():.4f} dB")

    return df, output_base, plots_dir, best_trial_id, best_row, epoch_log


def _label_best_points(ax, x_col: str, y_col: str, df: pd.DataFrame, best_ids: list[str], offset_x: float = 0, offset_y: float = 0.002):
    for _, row in df.iterrows():
        tid = str(row.get("trial_id", ""))
        if tid in best_ids:
            ax.annotate(
                tid,
                (float(row[x_col]), float(row[y_col])),
                xytext=(5 + offset_x, 5 + offset_y),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )


def _save_and_close(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Plot 1: Ranked trial performance ──────────────────────────────────
def plot_01_trial_ranking(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    sorted_df = df.sort_values(metric, ascending=True).head(20).copy()
    sorted_df = sorted_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.35 * len(sorted_df) + 2))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sorted_df))]
    bars = ax.barh(range(len(sorted_df)), sorted_df[metric].values, color=colors, height=0.7)

    best_val = sorted_df[metric].iloc[-1]
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        val = float(row[metric])
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8,
                color="#e74c3c" if i == len(sorted_df) - 1 else "black")

    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["trial_id"].values, fontsize=8)
    ax.set_xlabel("Mean horizon RMSE (dB)")
    ax.set_title("Top 20 trials by mean horizon RMSE (lower is better)")
    ax.invert_yaxis()
    ax.set_xlim(left=sorted_df[metric].min() - 0.02)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "01_trial_ranking_mean_rmse.png")


# ── Plot 2: Horizon RMSE comparison for top 10 ────────────────────────
def plot_02_top_horizons(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    top10 = df.sort_values(metric, ascending=True).head(10).copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top10))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    for j, h in enumerate(HORIZONS):
        col = f"rmse_t{h}_db"
        if col not in top10.columns:
            continue
        vals = pd.to_numeric(top10[col], errors="coerce").values
        bars = ax.bar(x + j * width, vals, width, label=f"t+{h}", color=colors[j % len(colors)])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top10["trial_id"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Per-horizon RMSE for the 10 best trials")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "02_top_trials_horizon_rmse.png")


# ── Plots 3–8: Hyperparameter main effects ────────────────────────────
def _main_effect_plot(df: pd.DataFrame, param: str, plots_dir: Path, use_log: bool = False):
    if param not in df.columns:
        print(f"  SKIP main effect: {param} column missing")
        return

    vals = pd.to_numeric(df[param], errors="coerce")
    metric = "mean_horizon_rmse_db"
    dfp = df.copy()
    dfp[param] = vals

    grouped = dfp.groupby(param, observed=True)[metric].agg(["mean", "std", "min", "count"]).reset_index()
    x_vals = grouped[param].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(x_vals)), grouped["mean"].values, yerr=grouped["std"].values,
           capsize=5, color="#3498db", width=0.6, error_kw={"linewidth": 1.5})

    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(i, float(row["mean"]) + 0.005, f"{row['mean']:.4f}\nn={int(row['count'])}",
                ha="center", va="bottom", fontsize=8)

    if use_log and all(x > 0 for x in x_vals):
        x_labels = [str(x) for x in x_vals]
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_labels, fontsize=9)
    else:
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(x) for x in x_vals], fontsize=9)

    ax.set_ylabel(f"Mean {metric} (dB)")
    ax.set_title(f"Effect of {param} on {metric}")
    fig.tight_layout()

    sanitized = param.replace("_", "")
    _save_and_close(fig, plots_dir / f"03_{sanitized}_effect.png".replace("__", "_"))


def plot_main_effects(df: pd.DataFrame, plots_dir: Path):
    log_params = {"learning_rate", "weight_decay"}
    for param in HPARAMS:
        _main_effect_plot(df, param, plots_dir, use_log=param in log_params)


# ── Plot 9: Box plots ─────────────────────────────────────────────────
def _box_plot_single(df: pd.DataFrame, param: str, plots_dir: Path):
    if param not in df.columns:
        print(f"  SKIP box plot: {param} column missing")
        return

    metric = "mean_horizon_rmse_db"
    dfp = df.copy()
    dfp[param] = pd.to_numeric(dfp[param], errors="coerce")
    dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
    dfp = dfp.dropna(subset=[param, metric])

    groups = dfp.groupby(param, observed=True)[metric]
    labels = sorted(groups.groups.keys(), key=lambda x: (isinstance(x, str), x))
    data = [groups.get_group(l).values for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True,
                    widths=0.5, showmeans=True, meanline=True,
                    medianprops={"color": "red", "linewidth": 2})
    ax.set_xticklabels([str(x) for x in labels])

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
        grp = groups.get_group(label).values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grp))
        x_pos = labels.index(label) + 1 + jitter
        ax.scatter(x_pos, grp, alpha=0.5, s=20, color="#2c3e50", zorder=5)

    ax.set_ylabel(f"{metric} (dB)")
    ax.set_title(f"Distribution of {metric} by {param}")
    fig.tight_layout()
    _save_and_close(fig, plots_dir / f"09_boxplot_{param}.png")


def plot_09_boxplots(df: pd.DataFrame, plots_dir: Path):
    for param in HPARAMS:
        _box_plot_single(df, param, plots_dir)


# ── Plot 10: Learning rate scatter ────────────────────────────────────
def plot_10_lr_scatter(df: pd.DataFrame, plots_dir: Path):
    if "learning_rate" not in df.columns:
        print("  SKIP lr scatter: learning_rate column missing")
        return

    metric = "mean_horizon_rmse_db"
    dfp = df.copy()
    dfp["learning_rate"] = pd.to_numeric(dfp["learning_rate"], errors="coerce")
    dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
    dfp = dfp.dropna(subset=["learning_rate", metric])

    top5_ids = set(dfp.nsmallest(5, metric)["trial_id"].values)

    fig, ax = plt.subplots(figsize=(9, 6))
    for hs in sorted(dfp["hidden_size"].dropna().unique()):
        mask = dfp["hidden_size"] == hs
        ax.scatter(dfp.loc[mask, "learning_rate"], dfp.loc[mask, metric],
                   label=f"hs={int(hs)}", alpha=0.7, s=40)

    _label_best_points(ax, "learning_rate", metric, dfp, list(top5_ids))

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(f"{metric} (dB)")
    ax.set_title("Learning rate vs mean horizon RMSE (colored by hidden size)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "10_learning_rate_scatter.png")


# ── Plot 11: Model capacity scatter ───────────────────────────────────
def plot_11_capacity_scatter(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    dfp = df.copy()

    if "parameter_count" in dfp.columns:
        x_col = "parameter_count"
        use_log = True
        xlabel = "Parameter count"
    elif "hidden_size" in dfp.columns:
        x_col = "hidden_size"
        use_log = False
        xlabel = "Hidden size"
    else:
        print("  SKIP capacity scatter: no parameter_count or hidden_size column")
        return

    dfp[x_col] = pd.to_numeric(dfp[x_col], errors="coerce")
    dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
    dfp = dfp.dropna(subset=[x_col, metric])

    top5_ids = set(dfp.nsmallest(5, metric)["trial_id"].values)

    fig, ax = plt.subplots(figsize=(9, 6))

    if "num_layers" in dfp.columns:
        for nl in sorted(dfp["num_layers"].dropna().unique()):
            mask = dfp["num_layers"] == nl
            ax.scatter(dfp.loc[mask, x_col], dfp.loc[mask, metric],
                       label=f"{int(nl)} layers", alpha=0.7, s=40)
        ax.legend(fontsize=8)
    else:
        ax.scatter(dfp[x_col], dfp[metric], alpha=0.7, s=40)

    _label_best_points(ax, x_col, metric, dfp, list(top5_ids))

    if use_log and dfp[x_col].min() > 0:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric} (dB)")
    ax.set_title("Model capacity vs mean horizon RMSE")
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "11_model_capacity_scatter.png")


# ── Plot 12: Accuracy vs training time ────────────────────────────────
def plot_12_accuracy_time(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    time_col = "total_seconds" if "total_seconds" in df.columns else "training_seconds"
    if time_col not in df.columns:
        print(f"  SKIP accuracy vs time: {time_col} column missing")
        return

    dfp = df.copy()
    dfp[time_col] = pd.to_numeric(dfp[time_col], errors="coerce")
    dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
    dfp = dfp.dropna(subset=[time_col, metric])

    top5_ids = set(dfp.nsmallest(5, metric)["trial_id"].values)
    best_rmse = dfp[metric].min()
    fast_within_2pct = dfp[dfp[metric] <= best_rmse * 1.02].nsmallest(1, time_col)
    annotate_ids = top5_ids | set(fast_within_2pct["trial_id"].values)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dfp[time_col], dfp[metric], alpha=0.6, s=40, color="#3498db")

    if not fast_within_2pct.empty:
        fr = fast_within_2pct.iloc[0]
        ax.scatter(fr[time_col], fr[metric], color="#e74c3c", s=80, marker="D", zorder=5, label="Fastest within 2% of best")

    _label_best_points(ax, time_col, metric, dfp, list(annotate_ids))

    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel(f"{metric} (dB)")
    ax.set_title("Accuracy vs training cost")
    if not fast_within_2pct.empty:
        ax.legend(fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "12_accuracy_vs_training_time.png")


# ── Plot 13: Epochs vs performance ────────────────────────────────────
def plot_13_epochs_vs_performance(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    if "epochs_completed" not in df.columns:
        print("  SKIP epochs vs perf: epochs_completed column missing")
        return

    dfp = df.copy()
    dfp["epochs_completed"] = pd.to_numeric(dfp["epochs_completed"], errors="coerce")
    dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
    dfp = dfp.dropna(subset=["epochs_completed", metric])

    top5_ids = set(dfp.nsmallest(5, metric)["trial_id"].values)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dfp["epochs_completed"], dfp[metric], alpha=0.6, s=40, color="#2ecc71")
    _label_best_points(ax, "epochs_completed", metric, dfp, list(top5_ids))

    ax.set_xlabel("Epochs completed")
    ax.set_ylabel(f"{metric} (dB)")
    ax.set_title("Epochs completed vs mean horizon RMSE")
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "13_epochs_vs_performance.png")


# ── Plot 14: Interaction heatmaps ─────────────────────────────────────
def _heatmap(ax, pivot: pd.DataFrame, title: str, cmap: str = "viridis_r"):
    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title(title)
        return

    data = pivot.values.astype(float)
    mask = np.isnan(data)
    data_plot = np.ma.masked_where(mask, data)

    im = ax.imshow(data_plot, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean RMSE (dB)", shrink=0.8)

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not mask[i, j]:
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if data[i, j] > data.mean() else "black")

    ax.set_title(title, fontsize=11)


def plot_14_heatmaps(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    pairs = [
        ("learning_rate", "hidden_size"),
        ("learning_rate", "num_layers"),
        ("hidden_size", "num_layers"),
        ("dropout", "num_layers"),
        ("weight_decay", "learning_rate"),
        ("batch_size", "learning_rate"),
    ]

    for p1, p2 in pairs:
        if p1 not in df.columns or p2 not in df.columns:
            print(f"  SKIP heatmap {p1}x{p2}: columns missing")
            continue

        dfp = df.copy()
        dfp[p1] = pd.to_numeric(dfp[p1], errors="coerce")
        dfp[p2] = pd.to_numeric(dfp[p2], errors="coerce")
        dfp[metric] = pd.to_numeric(dfp[metric], errors="coerce")
        sub = dfp.dropna(subset=[p1, p2, metric])

        if len(sub) == 0:
            print(f"  SKIP heatmap {p1}x{p2}: no data after filtering")
            continue

        pivot = sub.pivot_table(values=metric, index=p1, columns=p2, aggfunc="mean")

        fig, ax = plt.subplots(figsize=(7, 5))
        _heatmap(ax, pivot, f"Interaction: {p1} vs {p2} (mean {metric})")
        fig.tight_layout()
        _save_and_close(fig, plots_dir / f"14_interaction_{p1}_{p2}.png")


# ── Plot 15: Parallel coordinates ─────────────────────────────────────
def plot_15_parallel_coords(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    params = [p for p in HPARAMS if p in df.columns]

    if len(params) < 2:
        print("  SKIP parallel coords: fewer than 2 hyperparameters available")
        return

    top15 = df.sort_values(metric, ascending=True).head(15).copy()
    if len(top15) < 5:
        print("  SKIP parallel coords: fewer than 5 completed trials")
        return

    for p in params:
        top15[p] = pd.to_numeric(top15[p], errors="coerce")
    top15[metric] = pd.to_numeric(top15[metric], errors="coerce")
    top15 = top15.dropna(subset=params + [metric])

    if len(top15) < 2:
        print("  SKIP parallel coords: no valid data")
        return

    norm = top15[params].apply(lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-10))
    norm[metric] = (top15[metric] - top15[metric].min()) / max(top15[metric].max() - top15[metric].min(), 1e-10)

    all_cols = params + [metric]
    x_pos = list(range(len(all_cols)))
    cmap = plt.colormaps["viridis_r"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(norm.iterrows()):
        color = cmap(i / max(len(norm) - 1, 1))
        ax.plot(x_pos, [float(row[c]) for c in all_cols], marker="o", color=color,
                linewidth=1.2, alpha=0.8, label=str(row.get("trial_id", ""))[:12])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_cols, fontsize=9)
    ax.set_ylabel("Normalized value (0–1)")
    ax.set_title("Parallel coordinates: top 15 trials (best at top)")
    if len(norm) <= 15:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "15_top_trials_parallel_coordinates.png")


# ── Plot 16: Best-trial training curves ───────────────────────────────
def plot_16_best_curves(df: pd.DataFrame, plots_dir: Path, best_trial_id: str, epoch_log: pd.DataFrame | None):
    if epoch_log is None or len(epoch_log) == 0:
        print("  SKIP best-trial curves: epoch log not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = epoch_log["epoch"].values

    if "train_loss" in epoch_log.columns:
        ax.plot(epochs, epoch_log["train_loss"].values, label="Training loss", color="#3498db", linewidth=1.5)

    if "val_teacher_loss" in epoch_log.columns:
        tf_vals = epoch_log["val_teacher_loss"].values
        ax.plot(epochs, tf_vals, label="Teacher-forced val loss", color="#2ecc71", linewidth=1.5)

    if "val_loss" in epoch_log.columns:
        ar_vals = epoch_log["val_loss"].values
        ax.plot(epochs, ar_vals, label="Autoregressive val loss", color="#e74c3c", linewidth=1.5)

    best_epoch_row = df[df["trial_id"] == best_trial_id]
    if len(best_epoch_row) > 0:
        be = int(best_epoch_row.iloc[0]["best_epoch"])
        ax.axvline(x=be, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(be, ax.get_ylim()[1] * 0.95, f"Best epoch {be}", ha="center", fontsize=9, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalized MSE)")
    ax.set_title(f"Training curves for best trial: {best_trial_id}")

    has_all_positive = True
    for c in ["train_loss", "val_teacher_loss", "val_loss"]:
        if c in epoch_log.columns and (epoch_log[c] <= 0).any():
            has_all_positive = False
            break
    if has_all_positive:
        ax.set_yscale("log")

    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "16_best_trial_training_curves.png")


# ── Plot 17: Best-trial horizon curves ────────────────────────────────
def plot_17_best_horizon_curves(df: pd.DataFrame, plots_dir: Path, best_trial_id: str, epoch_log: pd.DataFrame | None):
    if epoch_log is None or len(epoch_log) == 0:
        print("  SKIP best-trial horizon curves: epoch log not available")
        return

    horizon_cols = [f"val_loss_t{h}" for h in HORIZONS]
    if not all(c in epoch_log.columns for c in horizon_cols):
        print("  SKIP best-trial horizon curves: val_loss_t{h} columns missing")
        return

    best_row = df[df["trial_id"] == best_trial_id]
    if len(best_row) == 0:
        return

    best_epoch = int(best_row.iloc[0]["best_epoch"])

    scale = {}
    best_ep_row = epoch_log[epoch_log["epoch"] == best_epoch]
    for h in HORIZONS:
        mse_norm = best_ep_row[f"val_loss_t{h}"].values[0]
        rmse_db = float(best_row.iloc[0].get(f"rmse_t{h}_db", 0))
        if mse_norm > 0 and rmse_db > 0:
            scale[h] = rmse_db / np.sqrt(mse_norm)
        else:
            scale[h] = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    for j, h in enumerate(HORIZONS):
        vals = epoch_log[f"val_loss_t{h}"].values
        db_vals = np.sqrt(np.maximum(vals, 1e-10)) * scale[h]
        ax.plot(epoch_log["epoch"].values, db_vals, label=f"t+{h}", color=colors[j % len(colors)], linewidth=1.5)

    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(best_epoch, ax.get_ylim()[1] * 0.95, f"Best epoch {best_epoch}", ha="center", fontsize=9, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title(f"Per-horizon RMSE for best trial: {best_trial_id}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, plots_dir / "17_best_trial_horizon_curves.png")


# ── Top 10 CSV ────────────────────────────────────────────────────────
def write_top10_csv(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    cols = [
        "trial_id", "hidden_size", "num_layers", "dropout",
        "learning_rate", "batch_size", "weight_decay", "optimizer",
        "best_epoch", "rmse_t1_db", "rmse_t5_db", "rmse_t15_db", "rmse_t60_db",
        "mean_horizon_rmse_db", "mean_horizon_mae_db",
        "total_seconds", "checkpoint_path",
    ]
    available = [c for c in cols if c in df.columns]
    top10 = df.sort_values(metric, ascending=True).head(10)[available].copy()
    top10.insert(0, "rank", range(1, 11))
    top10.to_csv(plots_dir / "top_10_trials.csv", index=False)
    print(f"  Top-10 trials  -> {plots_dir / 'top_10_trials.csv'}")


# ── Hyperparameter effects CSV ────────────────────────────────────────
def write_hyperparameter_effects(df: pd.DataFrame, plots_dir: Path):
    metric = "mean_horizon_rmse_db"
    rows: list[dict[str, Any]] = []

    for param in HPARAMS:
        if param not in df.columns:
            continue
        vals = pd.to_numeric(df[param], errors="coerce")
        metrics_v = pd.to_numeric(df[metric], errors="coerce")
        sub = pd.DataFrame({param: vals, metric: metrics_v}).dropna()

        for val, grp in sub.groupby(param, observed=True):
            m = grp[metric]
            rows.append({
                "hyperparameter": param,
                "value": val,
                "num_trials": len(grp),
                "mean_rmse": f"{m.mean():.4f}",
                "std_rmse": f"{m.std():.4f}",
                "median_rmse": f"{m.median():.4f}",
                "min_rmse": f"{m.min():.4f}",
                "max_rmse": f"{m.max():.4f}",
            })

    pd.DataFrame(rows).to_csv(plots_dir / "hyperparameter_effects.csv", index=False)
    print(f"  Hyperparameter effects -> {plots_dir / 'hyperparameter_effects.csv'}")


# ── Best-configuration JSON ───────────────────────────────────────────
def write_best_config_json(df: pd.DataFrame, plots_dir: Path, best_trial_id: str, epoch_log: pd.DataFrame | None):
    best = df[df["trial_id"] == best_trial_id]
    if len(best) == 0:
        return
    b = best.iloc[0]

    best_params = {}
    for p in HPARAMS + ["optimizer", "gradient_clip_norm", "output_strategy", "bidirectional"]:
        if p in b and str(b[p]) != "" and not (isinstance(b[p], float) and np.isnan(b[p])):
            val = b[p]
            if isinstance(val, np.generic):
                val = val.item()
            best_params[p] = val

    config_data = {
        "best_trial_id": best_trial_id,
        "selection_rule": "lowest finite mean_horizon_rmse_db",
        "model_parameters": best_params,
        "training_parameters": {
            "learning_rate": float(b.get("learning_rate", 0)) if str(b.get("learning_rate", "")) != "" else None,
            "batch_size": int(b.get("batch_size", 0)) if str(b.get("batch_size", "")) != "" else None,
            "weight_decay": float(b.get("weight_decay", 0)) if str(b.get("weight_decay", "")) != "" else None,
            "optimizer": str(b.get("optimizer", "")),
            "gradient_clip_norm": float(b.get("gradient_clip_norm", 0)) if str(b.get("gradient_clip_norm", "")) != "" else None,
            "epochs_completed": int(b.get("epochs_completed", 0)) if str(b.get("epochs_completed", "")) != "" else None,
            "best_epoch": int(b.get("best_epoch", 0)) if str(b.get("best_epoch", "")) != "" else None,
        },
        "best_epoch": int(b.get("best_epoch", 0)),
        "horizon_rmse_db": {f"t+{h}": float(b.get(f"rmse_t{h}_db", 0)) for h in HORIZONS if f"rmse_t{h}_db" in b},
        "horizon_mae_db": {f"t+{h}": float(b.get(f"mae_t{h}_db", 0)) for h in HORIZONS if f"mae_t{h}_db" in b},
        "mean_horizon_rmse_db": float(b.get("mean_horizon_rmse_db", 0)),
        "mean_horizon_mae_db": float(b.get("mean_horizon_mae_db", 0)),
        "total_training_time_sec": float(b.get("total_seconds", 0)),
        "checkpoint_path": str(b.get("checkpoint_path", "")),
        "epoch_log_path": str(b.get("epoch_log_path", "")),
        "results_file": "grid_search_results.csv",
    }

    json_path = plots_dir / "best_configuration.json"
    with open(json_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Best config JSON -> {json_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Hyperparameter Search Analysis")
    print("=" * 60)

    df, output_base, plots_dir, best_trial_id, best_row, epoch_log = _load_data()

    print(f"\nGenerating plots in: {plots_dir}")
    print("-" * 40)

    plot_01_trial_ranking(df, plots_dir)
    print("  1/17  Trial ranking")

    plot_02_top_horizons(df, plots_dir)
    print("  2/17  Top-trial horizon comparison")

    plot_main_effects(df, plots_dir)
    print("  3-8/17  Main-effect plots")

    plot_09_boxplots(df, plots_dir)
    print("  9/17  Box plots")

    plot_10_lr_scatter(df, plots_dir)
    print("  10/17  Learning-rate scatter")

    plot_11_capacity_scatter(df, plots_dir)
    print("  11/17  Model-capacity scatter")

    plot_12_accuracy_time(df, plots_dir)
    print("  12/17  Accuracy vs time")

    plot_13_epochs_vs_performance(df, plots_dir)
    print("  13/17  Epochs vs performance")

    plot_14_heatmaps(df, plots_dir)
    print("  14/17  Interaction heatmaps")

    plot_15_parallel_coords(df, plots_dir)
    print("  15/17  Parallel coordinates")

    plot_16_best_curves(df, plots_dir, best_trial_id, epoch_log)
    print("  16/17  Best-trial training curves")

    plot_17_best_horizon_curves(df, plots_dir, best_trial_id, epoch_log)
    print("  17/17  Best-trial horizon curves")

    write_top10_csv(df, plots_dir)
    write_hyperparameter_effects(df, plots_dir)
    write_best_config_json(df, plots_dir, best_trial_id, epoch_log)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Completed trials analyzed: {len(df)}")
    print(f"  Failed trials excluded:    {len(pd.read_csv(output_base / 'grid_search_results.csv')) - len(df)}")
    print(f"  Best trial:                {best_trial_id}")
    print(f"  Best mean horizon RMSE:    {best_row.get('mean_horizon_rmse_db', 'N/A'):.4f} dB")
    print(f"  Best config JSON:          {plots_dir / 'best_configuration.json'}")
    print(f"  Plots directory:           {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
