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
    meta_files = list(results_dir.glob("forecasts/*_metadata.json"))
    if not meta_files:
        return None
    with open(meta_files[0]) as f:
        return json.load(f)


def load_forecast(results_dir: Path):
    pred_files = list(results_dir.glob("forecasts/*_predictions.npz"))
    target_files = list(results_dir.glob("forecasts/*_targets.npz"))
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
    df = df[df["split"] == split_filter]
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
                      pred_npz, target_npz, train_mean, train_std, max_steps: int):
    freqs_mhz = meta.get("frequencies_mhz", [])
    if not freqs_mhz:
        return
    freqs_arr = np.array(freqs_mhz)
    freq_start = freqs_arr[0] - 0.5
    freq_end = freqs_arr[-1] + 0.5
    band_id = _band_id(meta.get("chunk_id", ""))
    chunk_label = _chunk_label(meta.get("chunk_id", ""))

    train_mean_2d = train_mean[np.newaxis, :]
    train_std_2d = train_std[np.newaxis, :]

    fig, axes = plt.subplots(2, 2, figsize=(5.2 * 2, 1.8 * 2), sharex="col", sharey="row")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    for col, horizon in enumerate(SPECTROGRAM_HORIZONS):
        key = f"t_plus_{horizon}"
        pred_norm = _squeeze_spatial(pred_npz[key][:max_steps]) * train_std_2d + train_mean_2d
        target = _squeeze_spatial(target_npz[key][:max_steps]).astype(np.float64)

        all_valid = np.concatenate([pred_norm[np.isfinite(pred_norm)], target[np.isfinite(target)]])
        if all_valid.size == 0:
            continue
        vmin, vmax = np.percentile(all_valid, [1, 99])
        extent = [0, pred_norm.shape[0], freq_start, freq_end]

        for row, (data, label) in enumerate([
            (target, "Ground-Truth"),
            (pred_norm, "Prediction"),
        ]):
            ax = axes[row, col]
            masked = np.ma.masked_invalid(data)
            im = ax.imshow(masked.T, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent)
            ax.set_title(f"VanillaLSTM / {chunk_label} / {HORIZON_LABELS[horizon]} / {label}",
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
                           split_filter: str = "test"):
    csv_path = results_dir / "per_frequency_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split_filter]
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
    ax.set_title(f"VanillaLSTM / {chunk_label} / MAE per Frequency", fontsize=9, fontweight="bold", pad=2)
    ax.legend(fontsize=7, prop={"family": "serif"})
    ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(out_dir / f"mae_vs_frequency_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'mae_vs_frequency_{band_id}.png'}")
    plt.close(fig)


def plot_r2_vs_frequency(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                          pred_npz, target_npz, train_mean, train_std, freqs_mhz):
    freqs = np.array(freqs_mhz)
    lo, hi = _band_limits(band_id)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        key = f"t_plus_{h}"
        pred = (_squeeze_spatial(pred_npz[key]) * train_std + train_mean).astype(np.float32)
        tgt = _squeeze_spatial(target_npz[key]).astype(np.float32)

        ss_res = np.nansum((pred - tgt) ** 2, axis=0)
        ss_tot = np.nansum((tgt - np.nanmean(tgt, axis=0)) ** 2, axis=0)
        r2 = 1 - ss_res / np.where(ss_tot == 0, 1, ss_tot)

        ax.plot(freqs, r2, linestyle=HORIZON_STYLES[h], color=HORIZON_COLORS[h],
                label=f"h={h}", linewidth=1.2)

    ax.set_xlabel("Frequency (MHz)", fontsize=9, fontweight="bold")
    ax.set_ylabel("R²", fontsize=9, fontweight="bold")
    ax.set_title(f"VanillaLSTM / {chunk_label} / R² per Frequency", fontsize=9, fontweight="bold", pad=2)
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
                     pred_npz, target_npz, train_mean, train_std, freqs_mhz):
    freqs = np.array(freqs_mhz)
    lo, hi = _band_limits(band_id)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for idx, h in enumerate(HORIZONS):
        ax = axes[idx // 2][idx % 2]
        key = f"t_plus_{h}"
        pred = (_squeeze_spatial(pred_npz[key]) * train_std + train_mean).astype(np.float32)
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
    fig.suptitle(f"VanillaLSTM / {chunk_label} / Mean Power per Bin", fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"mean_power_{band_id}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'mean_power_{band_id}.png'}")
    plt.close(fig)


def plot_time_series(results_dir: Path, out_dir: Path, band_id: str, chunk_label: str,
                     pred_npz, target_npz, train_mean, train_std, freqs_mhz,
                     bins=(30, 150), max_steps=500):
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
            pred = (pred_arr[:, bin_idx] * train_std[bin_idx] + train_mean[bin_idx]).astype(np.float32)
            ax.plot(rows, pred, color=HORIZON_COLORS[h], linestyle=HORIZON_STYLES[h],
                    label=HORIZON_LABELS[h], linewidth=0.9, alpha=0.8)

        ax.set_title(f"Bin {bin_idx} ({freq_label})", fontsize=9, fontweight="bold", pad=2)
        ax.set_xlabel("Source Row Index", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
        ax.legend(fontsize=6.5, prop={"family": "serif"}, ncol=2)

    axes[0].set_ylabel("Power (dBm)", fontsize=9, fontweight="bold")
    fig.suptitle(f"VanillaLSTM / {chunk_label} / Time Series", fontsize=10, fontweight="bold", y=0.98)
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
    """Difference npz_mae - raw_mae per frequency; positive means interpolation underestimates error."""
    csv_path = results_dir / "per_frequency_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lo, hi = _band_limits(band_id)
    fig, ax = plt.subplots(figsize=(7, 3))
    for h in HORIZONS:
        npz = df[(df["split"] == f"test_site_npz_{site_name}") & (df["horizon"] == h)].sort_values("frequency_mhz")
        raw = df[(df["split"] == f"test_site_raw_{site_name}") & (df["horizon"] == h)].sort_values("frequency_mhz")
        if npz.empty or raw.empty:
            continue
        diff = npz["mae_db"].values - raw["mae_db"].values
        ax.plot(npz["frequency_mhz"].values, diff,
                linestyle=HORIZON_STYLES[h], color=HORIZON_COLORS[h],
                label=f"h={h}", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Frequency (MHz)", fontsize=8, fontweight="bold")
    ax.set_ylabel("ΔMAE (npz − raw) dB", fontsize=8, fontweight="bold")
    ax.set_title(f"{site_name} — Interpolation Error (npz − raw MAE)", fontsize=8, fontweight="bold", pad=2)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=7, top=True, right=True, length=3, width=0.9)
    ax.legend(fontsize=6.5, prop={"family": "serif"})
    fig.tight_layout()
    fig.savefig(out_dir / f"interpolation_error_{site_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_dir / f'interpolation_error_{site_name}.png'}")
    plt.close(fig)


def write_summary(results_dir: Path, out_dir: Path, extra_lines: list[str] | None = None):
    csv_path = results_dir / "aggregate_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    lines = ["Model: VanillaLSTM"]
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


def generate_all_plots(
    results_dir: str | Path,
    model_name: str = "VanillaLSTM",
    out_dir: str | Path | None = None,
    bins: tuple[int, ...] = (30, 150),
    max_steps: int = 500,
) -> None:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(results_dir)
    if meta is None:
        print(f"No metadata found in {results_dir / 'forecasts'}, skipping plots")
        return

    band_id = _band_id(meta.get("chunk_id", ""))
    chunk_label = _chunk_label(meta.get("chunk_id", ""))
    freqs_mhz = meta.get("frequencies_mhz", [])
    if not freqs_mhz:
        print("No frequencies in metadata, skipping plots")
        return

    train_mean, train_std = load_norm_stats(results_dir, meta)

    pred_npz, target_npz = load_forecast(results_dir)
    if pred_npz is None:
        print(f"No forecasts in {results_dir / 'forecasts'}, skipping per-frequency plots")
        pred_npz, target_npz = None, None

    site_names = (meta.get("test_map_metadata") or {}).get("site_names", [])

    plot_mae_rmse_vs_horizon(results_dir, out_dir, split_filter="test")

    if pred_npz is not None:
        plot_spectrograms(results_dir, out_dir, meta, pred_npz, target_npz, train_mean, train_std, max_steps)
        plot_r2_vs_frequency(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, train_mean, train_std, freqs_mhz)
        plot_mean_power(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, train_mean, train_std, freqs_mhz)
        plot_time_series(results_dir, out_dir, band_id, chunk_label, pred_npz, target_npz, train_mean, train_std, freqs_mhz, bins=bins, max_steps=max_steps)

    plot_mae_vs_frequency(results_dir, out_dir, band_id, chunk_label, split_filter="test")

    for site_name in site_names:
        plot_per_site_mae_rmse_vs_horizon(results_dir, out_dir, site_name)
        plot_per_site_mae_vs_frequency(results_dir, out_dir, band_id, chunk_label, site_name)
        plot_interpolation_error(results_dir, out_dir, band_id, chunk_label, site_name)

    write_summary(results_dir, out_dir)


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
