"""
Plot generation for exported spectrum forecasts and evaluation results.

This module reads forecast artifacts and metric tables produced by the
integrated evaluation pipeline and converts them into diagnostic visualizations
for model inspection and comparison.

It supports both vector-based frequency forecasts and spatial spectrum-map
forecasts. Plotting behavior is selected from the stored forecast layout and
metadata rather than from training-time model objects.

Generated visualizations may include:

- predicted and target spectrum traces over selected timesteps;
- absolute-error and squared-error summaries;
- per-frequency error profiles;
- frequency-band performance comparisons;
- error distributions and histograms;
- temporal forecast comparisons;
- spatial target, prediction, and error maps;
- selected frequency-channel map comparisons; and
- model- or chunk-level summary figures.

Primary responsibilities include:

- locating forecast and metric artifacts in a result directory;
- loading stored arrays and metadata;
- validating supported array layouts;
- selecting manageable subsets of timesteps, frequencies, or spatial channels;
- constructing readable figure titles, labels, legends, and axis values;
- saving figures with deterministic names;
- avoiding changes to forecast or metric data; and
- providing one top-level function that generates all supported plots for an
  evaluation run.

This module performs visualization only. It does not load checkpoints, run
models, calculate forecasts, or modify evaluation results.
"""


import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

HORIZONS = [1, 5, 15, 60]
HORIZON_COLORS = {1: "tab:blue", 5: "tab:orange", 15: "tab:green", 60: "tab:red"}
HORIZON_STYLES = {1: "-", 5: "--", 15: "-.", 60: ":"}
HORIZON_LABELS = {1: "1-min", 5: "5-min", 15: "15-min", 60: "60-min"}
SPECTROGRAM_HORIZONS = [1, 60]
SPECTROGRAM_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                       "tab:olive", "tab:cyan"]
SPECTROGRAM_LINESTYLES = ["-", "--", "-.", ":"]

BEHAVIOR_COLORS = {
    "noise_floor": "tab:blue",
    "diurnal_pattern": "tab:orange",
    "intermittent_occupancy": "tab:red",
    "bursty_short_timescale": "tab:purple",
    "constant_occupancy": "tab:green",
    "mixed_activity": "tab:gray",
}


def _update_horizons(horizons: list[int]) -> None:
    global HORIZONS, HORIZON_COLORS, HORIZON_STYLES, HORIZON_LABELS, SPECTROGRAM_HORIZONS
    HORIZONS = horizons
    HORIZON_COLORS = {
        h: SPECTROGRAM_PALETTE[i % len(SPECTROGRAM_PALETTE)]
        for i, h in enumerate(horizons)
    }
    HORIZON_STYLES = {
        h: SPECTROGRAM_LINESTYLES[i % len(SPECTROGRAM_LINESTYLES)]
        for i, h in enumerate(horizons)
    }
    HORIZON_LABELS = {h: f"{h}-min" for h in horizons}
    SPECTROGRAM_HORIZONS = (
        [horizons[0], horizons[-1]] if len(horizons) >= 2 else horizons
    )


def _band_id(chunk_id: str) -> str:
    return chunk_id.replace("powder_", "")


def _chunk_label(chunk_id: str) -> str:
    return chunk_id.replace("powder_", "").replace("_", "-") + " MHz"


def _squeeze_spatial(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        return np.mean(data, axis=(2, 3)).astype(data.dtype)
    return data


def _band_limits(band_id: str) -> tuple[float, float]:
    return (600.0, 800.0) if band_id == "600_800" else (2400.0, 2600.0)


def load_metadata(results_dir: Path):
    meta_files = sorted(results_dir.glob("forecasts/*_metadata.json"))
    if not meta_files:
        return None
    with open(meta_files[0]) as f:
        return json.load(f)

def _filter_test_split(
    frame: pd.DataFrame,
    split_filter: str | None,
) -> pd.DataFrame:
    if split_filter is not None:
        exact = frame[
            frame["split"] == split_filter
        ]

        if not exact.empty:
            return exact

    return frame[
        frame["split"]
        .astype(str)
        .str.endswith("_test")
    ]
    
def load_forecast(results_dir: Path):
    pred_files = sorted(results_dir.glob("forecasts/*_predictions.npz"))
    target_files = sorted(results_dir.glob("forecasts/*_targets.npz"))
    if not pred_files or not target_files:
        return None, None
    return np.load(pred_files[0]), np.load(target_files[0])


def load_norm_stats(results_dir: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = meta.get("mean_dbm")
    std = meta.get("std_dbm")
    if mean is not None and std is not None:
        return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

    band_id = _band_id(meta.get("chunk_id", ""))
    human = pd.read_csv(ROOT / "data" / f"powder_20260618T0036Z_humanities_{band_id}.csv").iloc[:, 1:].values.astype(np.float32)
    guest = pd.read_csv(ROOT / "data" / f"powder_20260618T0036Z_guesthouse_{band_id}.csv").iloc[:, 1:].values.astype(np.float32)
    train = np.vstack([human, guest])
    mn = np.nanmean(train, axis=0).astype(np.float32)
    sd = np.nanstd(train, axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    return mn, sd


def plot_mae_rmse_vs_horizon(results_dir: Path, out_dir: Path, split_filter: str = "test"):
    csv_path = results_dir / "aggregate_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = _filter_test_split(df,split_filter,)
    if df.empty:
        return
    horizons = df["horizon"].values
    mae = df["mae_db"].values
    rmse = df["rmse_db"].values

    fig, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(7, 3))
    ax_mae.plot(horizons, mae, "o-", color="tab:blue", linewidth=1.2, markersize=5)
    ax_rmse.plot(horizons, rmse, "o-", color="tab:blue", linewidth=1.2, markersize=5)
    for ax in (ax_mae, ax_rmse):
        ax.set_xlabel("Horizon (minutes)", fontsize=7.5, fontweight="bold")
        ax.set_xticks(HORIZONS)
        ax.tick_params(labelsize=6.5, top=True, right=True, length=3, width=0.9)
    ax_mae.set_ylabel("MAE (dB)", fontsize=7.5, fontweight="bold")
    ax_rmse.set_ylabel("RMSE (dB)", fontsize=7.5, fontweight="bold")
    ax_mae.set_title("Mean Absolute Error", fontsize=7.5, fontweight="bold", pad=2)
    ax_rmse.set_title("Root Mean Squared Error", fontsize=7.5, fontweight="bold", pad=2)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.25, top=0.85, wspace=0.3)
    fig.savefig(out_dir / "mae_rmse_vs_horizon.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / 'mae_rmse_vs_horizon.png'}")
    plt.close(fig)


def plot_spectrograms(results_dir: Path, out_dir: Path, meta: dict,
                      pred_npz, target_npz, max_steps: int,
                      model_name: str = "VanillaLSTM"):
    freqs_mhz = meta.get("frequencies_mhz", [])
    if not freqs_mhz:
        return
    freqs_arr = np.array(freqs_mhz)
    freq_start = freqs_arr[0] - 0.5
    freq_end = freqs_arr[-1] + 0.5
    band_id = _band_id(meta.get("chunk_id", ""))
    chunk_label = _chunk_label(meta.get("chunk_id", ""))

    fig, axes = plt.subplots(2, 2, figsize=(5.2 * 2, 1.8 * 2), sharex="col", sharey="row")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    for col, horizon in enumerate(SPECTROGRAM_HORIZONS):
        key = f"t_plus_{horizon}"
        pred = _squeeze_spatial(pred_npz[key][:max_steps]).astype(np.float64)
        target = _squeeze_spatial(target_npz[key][:max_steps]).astype(np.float64)

        all_valid = np.concatenate([pred[np.isfinite(pred)], target[np.isfinite(target)]])
        if all_valid.size == 0:
            continue
        vmin, vmax = np.percentile(all_valid, [1, 99])
        extent = [0, pred.shape[0], freq_start, freq_end]

        for row, (data, label) in enumerate([
            (target, "Ground-Truth"),
            (pred, "Prediction"),
        ]):
            ax = axes[row, col]
            masked = np.ma.masked_invalid(data)
            im = ax.imshow(masked.T, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent)
            ax.set_title(f"{model_name} / {chunk_label} / {HORIZON_LABELS[horizon]} / {label}",
                         fontsize=7.5, fontweight="bold", pad=2)
            ax.set_ylabel("Frequency (MHz)", fontsize=7.5, fontweight="bold", labelpad=1)
            ax.set_xlabel("Time Step", fontsize=7.5, fontweight="bold", labelpad=1)

            start_tick = int(np.ceil(freq_start / 25.0) * 25)
            stop_tick = int(np.floor(freq_end / 25.0) * 25)
            yticks = np.arange(start_tick, stop_tick + 1, 25)
            ax.set_yticks(yticks)
            ax.tick_params(labelsize=6.5, top=True, right=True, length=3, width=0.9, pad=1)
            ax.label_outer()

    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.1, top=0.88, hspace=0.3, wspace=0.2)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.012, 0.78])
    colorbar = fig.colorbar(im, cax=cbar_ax)
    colorbar.ax.tick_params(labelsize=6.5, direction="in", length=2, width=0.8, pad=1)
    colorbar.set_label("Power (dBm)", fontsize=7)

    out_path = out_dir / f"spectrograms_{band_id}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_mae_vs_frequency(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                           model_name: str = "VanillaLSTM",
                           split_filter: str = "test"):
    csv_path = results_dir / "per_frequency_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = _filter_test_split(df,split_filter,)
    if df.empty:
        return
    lo, hi = _band_limits(band_id)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        sub = df[df["horizon"] == h].sort_values("frequency_mhz")
        ax.plot(sub["frequency_mhz"].values, sub["mae_db"].values,
                linestyle=HORIZON_STYLES[h], color=HORIZON_COLORS[h],
                label=f"h={h}", linewidth=1.2)

    ax.set_xlabel("Frequency (MHz)", fontsize=9, fontweight="bold")
    ax.set_ylabel("MAE (dB)", fontsize=9, fontweight="bold")
    ax.set_title(f"{model_name} / {chunk_label} / MAE per Frequency", fontsize=9, fontweight="bold", pad=2)
    ax.legend(fontsize=7, prop={"family": "serif"})
    ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(out_dir / f"mae_vs_frequency_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'mae_vs_frequency_{band_id}.png'}")
    plt.close(fig)


def plot_r2_vs_frequency(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                           pred_npz, target_npz, freqs_mhz,
                           model_name: str = "VanillaLSTM"):
    freqs = np.array(freqs_mhz)
    lo, hi = _band_limits(band_id)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        key = f"t_plus_{h}"
        pred = _squeeze_spatial(pred_npz[key]).astype(np.float32)
        tgt = _squeeze_spatial(target_npz[key]).astype(np.float32)

        ss_res = np.nansum((pred - tgt) ** 2, axis=0)
        ss_tot = np.nansum((tgt - np.nanmean(tgt, axis=0)) ** 2, axis=0)
        r2 = 1 - ss_res / np.where(ss_tot == 0, 1, ss_tot)

        ax.plot(freqs, r2, linestyle=HORIZON_STYLES[h], color=HORIZON_COLORS[h],
                label=f"h={h}", linewidth=1.2)

    ax.set_xlabel("Frequency (MHz)", fontsize=9, fontweight="bold")
    ax.set_ylabel("R²", fontsize=9, fontweight="bold")
    ax.set_title(f"{model_name} / {chunk_label} / R² per Frequency", fontsize=9, fontweight="bold", pad=2)
    ax.legend(fontsize=7, prop={"family": "serif"})
    ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(lo, hi)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / f"r2_vs_frequency_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'r2_vs_frequency_{band_id}.png'}")
    plt.close(fig)


def plot_mean_power(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                     pred_npz, target_npz, freqs_mhz,
                     model_name: str = "VanillaLSTM"):
    freqs = np.array(freqs_mhz)
    lo, hi = _band_limits(band_id)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx // 2][idx % 2]
        key = f"t_plus_{h}"
        pred = _squeeze_spatial(pred_npz[key]).astype(np.float32)
        tgt = _squeeze_spatial(target_npz[key]).astype(np.float32)

        n = pred.shape[0]
        mean_pred = np.nanmean(pred, axis=0)
        sem_pred = np.nanstd(pred, axis=0) / np.sqrt(n)
        mean_tgt = np.nanmean(tgt, axis=0)
        sem_tgt = np.nanstd(tgt, axis=0) / np.sqrt(n)

        ax.plot(freqs, mean_tgt, color="black", label="Actual", linewidth=1.4)
        ax.fill_between(freqs, mean_tgt - sem_tgt, mean_tgt + sem_tgt, color="black", alpha=0.15)
        ax.plot(freqs, mean_pred, color=HORIZON_COLORS[h], label="Predicted", linewidth=1.4)
        ax.fill_between(freqs, mean_pred - sem_pred, mean_pred + sem_pred, color=HORIZON_COLORS[h], alpha=0.15)

        ax.set_title(f"h={h}", fontsize=9, fontweight="bold", pad=2)
        ax.legend(fontsize=7, prop={"family": "serif"})
        ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
        ax.set_xlim(lo, hi)

    for ax in axes[1, :]:
        ax.set_xlabel("Frequency (MHz)", fontsize=9, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Power (dBm)", fontsize=9, fontweight="bold")
    fig.suptitle(f"{model_name} / {chunk_label} / Mean Power per Bin", fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"mean_power_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'mean_power_{band_id}.png'}")
    plt.close(fig)


def plot_time_series(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                     pred_npz, target_npz, freqs_mhz,
                     bins=(30, 50), max_steps=500,
                     model_name: str = "VanillaLSTM"):
    freqs = np.array(freqs_mhz)
    n_bins = len(bins)

    fig, axes = plt.subplots(1, n_bins, figsize=(6 * n_bins, 4), sharey=True)
    if n_bins == 1:
        axes = [axes]

    gt_key = "t_plus_1"
    gt_rows = target_npz[f"rows_{gt_key}"][:max_steps]
    gt_tgt = _squeeze_spatial(target_npz[gt_key][:max_steps]).astype(np.float32)

    for idx, bin_idx in enumerate(bins):
        ax = axes[idx]
        freq_label = f"{freqs[bin_idx]:.1f} MHz"

        ax.plot(gt_rows, gt_tgt[:, bin_idx], color="black", label="Actual", linewidth=1.0, alpha=0.8)
        for h in HORIZONS:
            key = f"t_plus_{h}"
            rows = pred_npz[f"rows_{key}"][:max_steps]
            pred_arr = _squeeze_spatial(pred_npz[key][:max_steps])
            pred = pred_arr[:, bin_idx].astype(np.float32)
            ax.plot(rows, pred, color=HORIZON_COLORS[h], linestyle=HORIZON_STYLES[h],
                    label=HORIZON_LABELS[h], linewidth=0.9, alpha=0.8)

        ax.set_title(f"Bin {bin_idx} ({freq_label})", fontsize=9, fontweight="bold", pad=2)
        ax.set_xlabel("Source Row Index", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
        ax.legend(fontsize=6.5, prop={"family": "serif"}, ncol=2)

    axes[0].set_ylabel("Power (dBm)", fontsize=9, fontweight="bold")
    fig.suptitle(f"{model_name} / {chunk_label} / Time Series", fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    bins_str = "_".join(str(b) for b in bins)
    fig.savefig(out_dir / f"time_series_bins_{bins_str}_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'time_series_bins_{bins_str}_{band_id}.png'}")
    plt.close(fig)


def plot_per_site_mae_rmse_vs_horizon(
    results_dir: Path, out_dir: Path, site_name: str,
):
    csv_path = results_dir / "aggregate_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(7, 3))
    for approach_tag, style, label in [
        ("npz", "-o", f"{site_name} (interpolated npz)"),
        ("raw", "-s", f"{site_name} (raw CSV)"),
    ]:
        split = f"test_site_{approach_tag}_{site_name}"
        sub = df[df["split"] == split].sort_values("horizon")
        if sub.empty:
            continue
        ax_mae.plot(sub["horizon"].values, sub["mae_db"].values, style,
                    color="tab:blue" if approach_tag == "npz" else "tab:orange",
                    linewidth=1.2, markersize=5, label=label)
        ax_rmse.plot(sub["horizon"].values, sub["rmse_db"].values, style,
                     color="tab:blue" if approach_tag == "npz" else "tab:orange",
                     linewidth=1.2, markersize=5, label=label)
    if not ax_mae.lines:
        plt.close(fig)
        return
    for ax in (ax_mae, ax_rmse):
        ax.set_xlabel("Horizon (minutes)", fontsize=7.5, fontweight="bold")
        ax.set_xticks(HORIZONS)
        ax.tick_params(labelsize=6.5, top=True, right=True, length=3, width=0.9)
        ax.legend(fontsize=6.5, prop={"family": "serif"})
    ax_mae.set_ylabel("MAE (dB)", fontsize=7.5, fontweight="bold")
    ax_rmse.set_ylabel("RMSE (dB)", fontsize=7.5, fontweight="bold")
    ax_mae.set_title(f"{site_name} — MAE", fontsize=7.5, fontweight="bold", pad=2)
    ax_rmse.set_title(f"{site_name} — RMSE", fontsize=7.5, fontweight="bold", pad=2)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.25, top=0.85, wspace=0.3)
    fig.savefig(out_dir / f"per_site_mae_rmse_{site_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'per_site_mae_rmse_{site_name}.png'}")
    plt.close(fig)


def plot_per_site_mae_vs_frequency(
    results_dir: Path, out_dir: Path, band_id: str, chunk_label: str, site_name: str,
):
    csv_path = results_dir / "per_frequency_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lo, hi = _band_limits(band_id)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for col, (approach_tag, approach_label) in enumerate([("npz", "Interpolated npz"), ("raw", "Raw CSV")]):
        ax = axes[col]
        split = f"test_site_{approach_tag}_{site_name}"
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        for h in HORIZONS:
            hsub = sub[sub["horizon"] == h].sort_values("frequency_mhz")
            if hsub.empty:
                continue
            ax.plot(hsub["frequency_mhz"].values, hsub["mae_db"].values,
                    linestyle=HORIZON_STYLES[h], color=HORIZON_COLORS[h],
                    label=f"h={h}", linewidth=1.2)
        ax.set_title(f"{site_name} / {approach_label}", fontsize=8, fontweight="bold", pad=2)
        ax.set_xlabel("Frequency (MHz)", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=6.5, prop={"family": "serif"})
    axes[0].set_ylabel("MAE (dB)", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_site_mae_vs_frequency_{site_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'per_site_mae_vs_frequency_{site_name}.png'}")
    plt.close(fig)


def plot_interpolation_error(
    results_dir: Path, out_dir: Path, band_id: str, chunk_label: str, site_name: str,
):
    """Per-site MAE vs frequency: npz ground truth vs raw CSV ground truth."""
    csv_path = results_dir / "per_frequency_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lo, hi = _band_limits(band_id)
    fig, ax = plt.subplots(figsize=(8, 4))
    for h in HORIZONS:
        npz = df[(df["split"] == f"test_site_npz_{site_name}") & (df["horizon"] == h)].sort_values("frequency_mhz")
        raw = df[(df["split"] == f"test_site_raw_{site_name}") & (df["horizon"] == h)].sort_values("frequency_mhz")
        if npz.empty or raw.empty:
            continue
        color = HORIZON_COLORS[h]
        ax.plot(npz["frequency_mhz"].values, npz["mae_db"].values,
                linestyle="-", marker="o", color=color,
                label=f"h={h} npz", linewidth=1.2, markersize=4)
        ax.plot(raw["frequency_mhz"].values, raw["mae_db"].values,
                linestyle="--", marker="s", color=color,
                label=f"h={h} raw", linewidth=1.2, markersize=4)
    ax.set_xlabel("Frequency (MHz)", fontsize=8, fontweight="bold")
    ax.set_ylabel("MAE (dB)", fontsize=8, fontweight="bold")
    ax.set_title(f"{site_name} — Per-Site MAE: npz vs raw", fontsize=8, fontweight="bold", pad=2)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
    ax.legend(fontsize=6.5, prop={"family": "serif"})
    fig.tight_layout()
    fig.savefig(out_dir / f"interpolation_error_{site_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'interpolation_error_{site_name}.png'}")
    plt.close(fig)


def write_summary(results_dir: Path, out_dir: Path, extra_lines: list[str] | None = None,
                  model_name: str = "VanillaLSTM"):
    csv_path = results_dir / "aggregate_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lines = [f"Model: {model_name}"]
    if not df.empty:
        lines.append(f"Aggregate rows: {len(df)}")
        best = df.sort_values("mae_db").iloc[0]
        lines.append(
            f"Best aggregate MAE: {best['mae_db']:.4f} dB on {best['chunk_id']} {best['split']} h={int(best['horizon'])}"
        )
    if extra_lines:
        lines.extend(extra_lines)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {out_dir / 'summary.txt'}")


def plot_spatial_error_maps(
    out_dir: Path,
    band_id: str,
    chunk_label: str,
    pred_npz,
    target_npz,
    model_name: str = "ConvLSTM",
) -> None:
    """
    Plot RMSE and MAE per spatial grid cell for each horizon (map mode only).

    Expects 4D prediction/target arrays ``(num_origins, F, H, W)``.
    Produces two 2x2 figures (RMSE, MAE) with one heatmap per horizon.
    """
    horizons = [1, 5, 15, 60]

    fig_rmse, axes_rmse = plt.subplots(2, 2, figsize=(7, 6))
    fig_mae, axes_mae = plt.subplots(2, 2, figsize=(7, 6))

    for idx, h in enumerate(horizons):
        key = f"t_plus_{h}"
        if key not in pred_npz:
            continue
        pred = pred_npz[key].astype(np.float64)
        tgt = target_npz[key].astype(np.float64)

        se = (pred - tgt) ** 2
        ae = np.abs(pred - tgt)
        rmse = np.sqrt(np.nanmean(se, axis=(0, 1)))
        mae = np.nanmean(ae, axis=(0, 1))

        row, col = idx // 2, idx % 2
        for ax, data, label in [
            (axes_rmse[row, col], rmse, "RMSE"),
            (axes_mae[row, col], mae, "MAE"),
        ]:
            im = ax.imshow(data, cmap="hot", aspect="equal", interpolation="nearest")
            ax.set_title(f"h={h} {label}", fontsize=8, fontweight="bold", pad=2)
            ax.set_xlabel("Grid X", fontsize=7)
            ax.set_ylabel("Grid Y", fontsize=7)
            ax.tick_params(labelsize=6.5)

    for fig, suffix in [(fig_rmse, "rmse"), (fig_mae, "mae")]:
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.92, hspace=0.35, wspace=0.3)
        cbar_ax = fig.add_axes([0.90, 0.08, 0.015, 0.84])
        fig.colorbar(axes_rmse[0, 0].images[0], cax=cbar_ax).set_label("dB", fontsize=7)
        fig.suptitle(
            f"{model_name} / {chunk_label} / Spatial {suffix.upper()}",
            fontsize=9, fontweight="bold",
        )
        path = out_dir / f"spatial_{suffix}_map_{band_id}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {path}")
        plt.close(fig)


def plot_site_time_series(
    out_dir: Path,
    meta: dict,
    pred_npz,
    target_npz,
    band_id: str,
    chunk_label: str,
    max_steps: int = 500,
    model_name: str = "ConvLSTM",
) -> None:
    """
    Per-site time series: actual vs predicted power at the centre frequency.

    For each real sensor site, plots the actual sensor reading (from the
    original NPZ) alongside the model's prediction at the closest grid cell
    for horizons 1 and 60.
    """
    site_names = meta.get("site_names", [])
    site_positions = meta.get("site_grid_positions", [])
    test_map_path = meta.get("test_map_path")
    if not site_names or not site_positions or not test_map_path:
        return

    freqs = meta.get("frequencies_mhz", [])
    if not freqs:
        return
    f_mid = len(freqs) // 2

    loaded = np.load(test_map_path, allow_pickle=True)
    site_data_db = loaded["site_data_db"].astype(np.float32)

    for site_idx, site_name in enumerate(site_names):
        if site_idx >= len(site_positions):
            continue
        h_pos, w_pos = site_positions[site_idx]

        fig, ax = plt.subplots(figsize=(8, 3.5))

        for h in (1, 60):
            key = f"t_plus_{h}"
            rows_key = f"rows_{key}"
            if key not in pred_npz:
                continue
            rows = pred_npz[rows_key][:max_steps]
            pred_site = pred_npz[key][:max_steps, f_mid, h_pos, w_pos]
            ax.plot(rows, pred_site, color="tab:blue" if h == 1 else "tab:red",
                    linestyle="--" if h == 1 else ":", linewidth=1.0, label=f"Pred h={h}")
            if h == 1:
                ax.plot(rows, site_data_db[site_idx, rows, f_mid],
                        color="black", linewidth=1.2, label="Actual")

        ax.set_xlabel("Time Step", fontsize=8, fontweight="bold")
        ax.set_ylabel("Power (dBm)", fontsize=8, fontweight="bold")
        ax.set_title(
            f"{model_name} / {chunk_label} / {site_name} @ {freqs[f_mid]:.1f} MHz",
            fontsize=8, fontweight="bold", pad=2,
        )
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, prop={"family": "serif"})
        fig.tight_layout()
        path = out_dir / f"site_time_series_{site_name}_{band_id}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {path}")
        plt.close(fig)

    loaded.close()


def plot_site_scatter(
    out_dir: Path,
    meta: dict,
    pred_npz,
    target_npz,
    band_id: str,
    chunk_label: str,
    max_steps: int = 500,
    model_name: str = "ConvLSTM",
) -> None:
    """
    Per-site scatter: predicted vs actual power across all frequencies and
    time steps.  One 2x2 figure per site (horizons 1, 5, 15, 60).
    Each panel shows the y=x line and annotates R² and MAE.
    """
    site_names = meta.get("site_names", [])
    site_positions = meta.get("site_grid_positions", [])
    test_map_path = meta.get("test_map_path")
    if not site_names or not site_positions or not test_map_path:
        return

    horizons = [1, 5, 15, 60]

    loaded = np.load(test_map_path, allow_pickle=True)
    site_data_db = loaded["site_data_db"].astype(np.float32)

    for site_idx, site_name in enumerate(site_names):
        if site_idx >= len(site_positions):
            continue
        h_pos, w_pos = site_positions[site_idx]

        fig, axes = plt.subplots(2, 2, figsize=(7, 6))

        for idx, h in enumerate(horizons):
            key = f"t_plus_{h}"
            rows_key = f"rows_{key}"
            if key not in pred_npz:
                continue
            rows = pred_npz[rows_key][:max_steps]
            pred_site = pred_npz[key][:max_steps, :, h_pos, w_pos]
            actual_site = site_data_db[site_idx, rows, :]

            pred_flat = pred_site.ravel()
            actual_flat = actual_site.ravel()
            valid = np.isfinite(pred_flat) & np.isfinite(actual_flat)
            pred_flat = pred_flat[valid]
            actual_flat = actual_flat[valid]
            if len(pred_flat) == 0:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            ax.scatter(actual_flat, pred_flat, s=1, alpha=0.3,
                       c="tab:blue", edgecolors="none", rasterized=True)

            vmin = min(pred_flat.min(), actual_flat.min())
            vmax = max(pred_flat.max(), actual_flat.max())
            margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
            ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                    "--", color="gray", linewidth=0.8, zorder=0)
            ax.set_xlim(vmin - margin, vmax + margin)
            ax.set_ylim(vmin - margin, vmax + margin)
            ax.set_aspect("equal")

            ss_res = np.nansum((pred_flat - actual_flat) ** 2)
            ss_tot = np.nansum((actual_flat - np.nanmean(actual_flat)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            mae = np.nanmean(np.abs(pred_flat - actual_flat))
            ax.set_title(f"h={h}  R²={r2:.3f}  MAE={mae:.2f} dB",
                         fontsize=7.5, fontweight="bold", pad=2)
            ax.set_xlabel("Actual (dBm)", fontsize=7)
            ax.set_ylabel("Predicted (dBm)", fontsize=7)
            ax.tick_params(labelsize=6.5)

        fig.suptitle(
            f"{model_name} / {chunk_label} / {site_name}",
            fontsize=9, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = out_dir / f"site_scatter_{site_name}_{band_id}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {path}")
        plt.close(fig)

    loaded.close()


def plot_per_band_mae(results_dir: Path, out_dir: Path, model_name: str = "VanillaLSTM") -> None:
    band_csv = results_dir / "per_band_metrics.csv"
    agg_csv = results_dir / "aggregate_metrics.csv"

    if not band_csv.exists() or not agg_csv.exists():
        return

    try:
        band_df = pd.read_csv(band_csv)
        agg_df = pd.read_csv(agg_csv)
    except pd.errors.EmptyDataError:
        return

    if band_df.empty or agg_df.empty:
        return

    band_df = _filter_test_split(band_df, "test")
    agg_df = _filter_test_split(agg_df, "test")

    if band_df.empty or agg_df.empty:
        return

    band_df = band_df.sort_values("start_mhz").reset_index(drop=True)
    regions = band_df["band_id"].unique()
    horizons = sorted(band_df["horizon"].unique())

    agg_mae = dict(zip(agg_df["horizon"], agg_df["mae_db"]))

    n_regions = len(regions)
    n_horizons = len(horizons)

    fig, ax = plt.subplots(figsize=(max(10, n_regions * 0.55), 4.5))

    group_width = 0.8
    bar_width = group_width / n_horizons

    for h_idx, horizon in enumerate(horizons):
        h_data = band_df[band_df["horizon"] == horizon].set_index("band_id")
        positions = np.arange(n_regions) + (h_idx - n_horizons / 2 + 0.5) * bar_width
        mae_values = [h_data.loc[region, "mae_db"] if region in h_data.index else 0 for region in regions]
        ax.bar(positions, mae_values, bar_width,
               label=HORIZON_LABELS.get(horizon, f"h={horizon}"),
               color=HORIZON_COLORS.get(horizon, "gray"),
               linewidth=0.3, edgecolor="white")

    for horizon in horizons:
        if horizon in agg_mae:
            ax.axhline(y=agg_mae[horizon], color=HORIZON_COLORS.get(horizon, "gray"),
                       linestyle=HORIZON_STYLES.get(horizon, "--"),
                       linewidth=0.8, alpha=0.6)

    first_h = band_df[band_df["horizon"] == horizons[0]].set_index("band_id")
    tick_labels = []
    tick_colors = []
    for region in regions:
        if region in first_h.index:
            row = first_h.loc[region]
            start = int(round(float(row["start_mhz"])))
            end = int(round(float(row["end_mhz"])))
            tick_labels.append(f"{start}-{end}")
            tick_colors.append(BEHAVIOR_COLORS.get(row["behavior_category"], "black"))
        else:
            tick_labels.append(region)
            tick_colors.append("black")

    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=45, ha="right")
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)

    ax.set_xlabel("Frequency region (MHz) — labels colored by behavior category", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("MAE (dB)", fontsize=7.5, fontweight="bold")
    ax.set_title(f"Per-Region MAE — {model_name}", fontsize=8, fontweight="bold", pad=3)
    ax.legend(fontsize=6.5, loc="upper left", title="Horizon", title_fontsize=6.5)
    ax.tick_params(labelsize=6.5, top=True, right=True, length=3, width=0.9)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.9)
    fig.savefig(out_dir / "mae_per_band.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / 'mae_per_band.png'}")
    plt.close(fig)


def generate_all_plots(
    results_dir: str | Path,
    model_name: str = "VanillaLSTM",
    out_dir: str | Path | None = None,
    bins: tuple[int, ...] = (30, 150),
    max_steps: int = 500,
    horizons: list[int] | None = None,
) -> None:
    
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(results_dir)
    if meta is None:
        print(f"No metadata found in {results_dir / 'forecasts'}, skipping plots")
        return

    if horizons is not None:
        meta["stored_horizons"] = sorted(horizons)

    _update_horizons(
        sorted([
            int(value)
            for value in meta.get(
                "stored_horizons",
                HORIZONS,
            )
        ])
    )

    band_id = _band_id(meta.get("chunk_id", ""))
    chunk_label = _chunk_label(meta.get("chunk_id", ""))
    freqs_mhz = meta.get("frequencies_mhz", [])
    if not freqs_mhz:
        print("No frequencies in metadata, skipping plots")
        return

    pred_npz, target_npz = load_forecast(results_dir)
    if pred_npz is None:
        print(f"No forecasts in {results_dir / 'forecasts'}, skipping per-frequency plots")
        pred_npz, target_npz = None, None

    site_names = (
        meta.get("site_names")
        or (meta.get("test_map_metadata") or {}).get("site_names", [])
    )

    plot_mae_rmse_vs_horizon(results_dir, out_dir, split_filter="test")

    if pred_npz is not None:
        plot_spectrograms(results_dir, out_dir, meta, pred_npz, target_npz, max_steps, model_name=model_name)
        plot_r2_vs_frequency(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, freqs_mhz, model_name=model_name)
        plot_mean_power(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, freqs_mhz, model_name=model_name)
        plot_time_series(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, freqs_mhz, bins=bins, max_steps=max_steps, model_name=model_name)

        # Map-mode-only plots (4D prediction/target arrays)
        if pred_npz["t_plus_1"].ndim == 4:
            plot_spatial_error_maps(out_dir, band_id, chunk_label, pred_npz, target_npz, model_name=model_name)
            plot_site_time_series(out_dir, meta, pred_npz, target_npz, band_id, chunk_label, max_steps, model_name=model_name)
            plot_site_scatter(out_dir, meta, pred_npz, target_npz, band_id, chunk_label, max_steps, model_name=model_name)

    plot_mae_vs_frequency(results_dir, out_dir, band_id, chunk_label, model_name=model_name, split_filter="test")

    for site_name in site_names:
        plot_per_site_mae_rmse_vs_horizon(results_dir, out_dir, site_name)
        plot_per_site_mae_vs_frequency(results_dir, out_dir, band_id, chunk_label, site_name)
        plot_interpolation_error(results_dir, out_dir, band_id, chunk_label, site_name)

    write_summary(results_dir, out_dir, model_name=model_name)

    plot_per_band_mae(results_dir, out_dir, model_name=model_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate forecast plots and summary")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", default="VanillaLSTM")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, nargs="+", default=[30, 50])
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    generate_all_plots(
        results_dir=args.results_dir,
        model_name=args.model,
        out_dir=args.out_dir,
        bins=tuple(args.bins),
        max_steps=args.max_steps,
    )
