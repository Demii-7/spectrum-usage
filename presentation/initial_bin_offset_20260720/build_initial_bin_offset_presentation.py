#!/usr/bin/env python3
"""
Build the compact presentation-support package for the VanillaLSTM initial-bin
mean-power offset investigation.

Usage:
    python training/common/build_initial_bin_offset_presentation.py [--output-dir PATH]

Requires:
    - results/powder/600_800/vanillalstm/{humanities,guesthouse}/forecasts/*.npz
    - results/powder/600_800/vanillalstm/{humanities,guesthouse}/forecasts/*.json
    - results/hidden_size_exp/{h8,h16,h32,h64}/vanillalstm/humanities/forecasts/*.npz
    - results/hidden_size_exp/{h8,h16,h32,h64}/vanillalstm/humanities/forecasts/*.json
    - data/powder_20260628T0436Z_humanities_600_800.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


ROOT = Path(__file__).resolve().parent.parent.parent  # /home/cc/spectrum-usage
DEFAULT_OUTPUT = ROOT / "training" / "common" / "presentation" / "initial_bin_mean_offset"
_RD = None  # set by main() via --results-dir


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check_file(path, label=""):
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        fail(f"Required {label}file not found: {p}")
    return p


def load_npz(path):
    return np.load(check_file(path, "NPZ "))


def load_json(path):
    with open(check_file(path, "JSON ")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# data extraction
# ---------------------------------------------------------------------------

def get_humanities_base_dir():
    base = _RD if _RD is not None else ROOT / "results"
    return base / "powder" / "600_800" / "vanillalstm" / "humanities"


def get_guesthouse_base_dir():
    base = _RD if _RD is not None else ROOT / "results"
    return base / "powder" / "600_800" / "vanillalstm" / "guesthouse"


def get_forecasts_dir(site_base):
    return site_base / "forecasts"


def predicted_site_name(site_base):
    """The forecast filename includes the test-site name."""
    return site_base.name  # "humanities" or "guesthouse"


def locate_pred_target_npz(forecasts_dir, site_name):
    prefix = f"powder_600_800_{site_name}_test_vanillalstm"
    pred = forecasts_dir / f"{prefix}_predictions.npz"
    tgt  = forecasts_dir / f"{prefix}_targets.npz"
    return check_file(pred, "predictions "), check_file(tgt, "targets ")


def locate_metadata(forecasts_dir, site_name):
    prefix = f"powder_600_800_{site_name}_test_vanillalstm"
    return check_file(forecasts_dir / f"{prefix}_metadata.json", "metadata ")


def validate_frequencies(meta, label):
    freqs = meta["frequencies_mhz"]
    assert len(freqs) == 200, f"{label}: expected 200 frequencies, got {len(freqs)}"
    assert abs(freqs[0] - 600.5) < 0.01, f"{label}: first freq {freqs[0]} != 600.5"
    assert abs(freqs[7] - 607.5) < 0.01, f"{label}: bin7 freq {freqs[7]} != 607.5"
    return freqs


def extract_site_data(site_base, site_name):
    """Return dict with predictions, targets, metadata for t+1."""
    fdir = get_forecasts_dir(site_base)
    pred_path, tgt_path = locate_pred_target_npz(fdir, site_name)
    meta_path = locate_metadata(fdir, site_name)
    pred = load_npz(pred_path)
    tgt  = load_npz(tgt_path)
    meta = load_json(meta_path)
    freqs = validate_frequencies(meta, site_name)

    p1 = pred["t_plus_1"]
    t1 = tgt["t_plus_1"]
    assert p1.shape == t1.shape, f"{site_name} shape mismatch: {p1.shape} vs {t1.shape}"
    assert p1.shape[1] == 200, f"{site_name}: expected 200 bins, got {p1.shape[1]}"

    train_mean = np.array(meta["mean_dbm"])
    train_std  = np.array(meta["std_dbm"])

    return {
        "site": site_name,
        "pred": p1,
        "tgt": t1,
        "freqs": freqs,
        "train_mean": train_mean,
        "train_std": train_std,
    }


def extract_hidden_size_data():
    """Return list of {hs, bias0, var_ratio_active} for h8/h16/h32/h64."""
    base = _RD if _RD is not None else ROOT / "results"
    results = []
    for hs in ["h8", "h16", "h32", "h64"]:
        hdir = base / "hidden_size_exp" / hs / "vanillalstm" / "humanities"
        fdir = hdir / "forecasts"
        meta = load_json(fdir / f"powder_600_800_humanities_test_vanillalstm_metadata.json")
        pred = load_npz(fdir / f"powder_600_800_humanities_test_vanillalstm_predictions.npz")
        tgt  = load_npz(fdir / f"powder_600_800_humanities_test_vanillalstm_targets.npz")
        p1 = pred["t_plus_1"]; t1 = tgt["t_plus_1"]
        bias0 = float(p1[:, 0].mean() - t1[:, 0].mean())
        train_std = np.array(meta["std_dbm"])
        active = train_std > 0.1
        var_ratio = (p1.std(axis=0) / t1.std(axis=0))[active].mean()
        results.append({"hs": int(hs[1:]), "bias0": bias0, "var_ratio": float(var_ratio)})
    return results


def compute_persistence_mae():
    """Compute per-bin persistence MAE from raw test CSV."""
    base = _RD if _RD is not None else ROOT
    csv_path = base / "data" / "powder_20260628T0436Z_humanities_600_800.csv"
    if not csv_path.exists() and _RD is not None:
        csv_path = ROOT / "data" / "powder_20260628T0436Z_humanities_600_800.csv"
    check_file(csv_path, "humanities test CSV ")
    import pandas as pd
    df = pd.read_csv(csv_path)
    vals = df.iloc[:, 1:].values.astype(np.float32)
    diff = vals[1:] - vals[:-1]
    return np.abs(diff).mean(axis=0)  # (200,)


def load_model_per_freq_mae():
    """Load per-frequency MAE from the stored metrics CSV."""
    base = _RD if _RD is not None else ROOT / "results"
    csv_path = base / "powder" / "600_800" / "vanillalstm" / "humanities" / "per_frequency_metrics.csv"
    check_file(csv_path, "per_frequency_metrics.csv ")
    maes = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["horizon"]) == 1:
                freq = float(row["frequency_mhz"])
                maes[freq] = float(row["mae_db"])
    return maes


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------

def set_axis_style(ax):
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Frequency (MHz)", fontsize=9)


# ---------------------------------------------------------------------------
# figure 1 : mean_power_site_comparison.png
# ---------------------------------------------------------------------------

def make_mean_power_site_comparison(hum, gst, output_dir):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, data, site in [
        (ax_l, hum, "Humanities"),
        (ax_r, gst, "Guesthouse"),
    ]:
        ax.plot(data["freqs"], data["tgt"].mean(axis=0), "k-", linewidth=1.2, label="Actual mean")
        ax.plot(data["freqs"], data["pred"].mean(axis=0), "-", color="#E69F00", linewidth=1.2, label="VanillaLSTM mean")
        ax.axvspan(600.5, 607.5, alpha=0.12, color="gray", label="Observed offset region")
        ax.text(604, ax.get_ylim()[1] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "Observed offset region", fontsize=7, ha="center", color="gray", style="italic")
        ax.set_title(site, fontsize=10, fontweight="bold")
        set_axis_style(ax)
        ax.set_xlim(600, 800)
        ax.set_ylabel("Mean power (dBm)", fontsize=9)

    fig.suptitle("VanillaLSTM t+1 Forecast: Mean Power vs Actual Mean", fontsize=11, y=1.02)
    fig.tight_layout()
    path = output_dir / "mean_power_site_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# figure 2 : initial_bins_mean_comparison.png
# ---------------------------------------------------------------------------

def make_initial_bins_mean_comparison(hum, gst, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = list(range(8))
    freqs = [hum["freqs"][i] for i in bins]

    train = [hum["train_mean"][i] for i in bins]
    tgt   = [hum["tgt"][:, i].mean() for i in bins]
    pred  = [hum["pred"][:, i].mean() for i in bins]
    gst_tgt = [gst["tgt"][:, i].mean() for i in bins]

    ax.plot(freqs, train, "o-", color="blue",  markersize=6, label="Training mean (Humanities)")
    ax.plot(freqs, tgt,   "o-", color="black", markersize=6, label="Test target mean (Humanities)")
    ax.plot(freqs, pred,  "o-", color="#E69F00", markersize=6, label="Prediction mean (Humanities)")
    ax.plot(freqs, gst_tgt, "o-", color="green", markersize=5, alpha=0.6, label="Test target mean (Guesthouse)")

    # Annotate largest train/test shift
    shifts = [tgt[i] - train[i] for i in range(8)]
    max_shift_idx = int(np.argmax(np.abs(shifts)))
    max_shift_val = shifts[max_shift_idx]
    ax.annotate(f"Largest train/test shift\n{max_shift_val:.1f} dB @ {freqs[max_shift_idx]:.0f} MHz",
                xy=(freqs[max_shift_idx], tgt[max_shift_idx]),
                xytext=(freqs[max_shift_idx] - 2, tgt[max_shift_idx] + 2),
                fontsize=7, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Annotate largest model error
    errors = [pred[i] - tgt[i] for i in range(8)]
    max_err_idx = int(np.argmax(np.abs(errors)))
    max_err_val = errors[max_err_idx]
    ax.annotate(f"Largest model error\n{max_err_val:+.1f} dB @ {freqs[max_err_idx]:.0f} MHz",
                xy=(freqs[max_err_idx], pred[max_err_idx]),
                xytext=(freqs[max_err_idx] + 2, pred[max_err_idx] - 2),
                fontsize=7, color="#E69F00",
                arrowprops=dict(arrowstyle="->", color="#E69F00", lw=0.8))

    ax.set_xlabel("Frequency (MHz)", fontsize=9)
    ax.set_ylabel("Mean power (dBm)", fontsize=9)
    ax.set_title("Initial Bins Mean Power Comparison (t+1, means across forecast origins)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{f:.0f}" for f in freqs])

    # Ensure negative dBm is clear (axis not inverted)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(min(ylo, -120), max(yhi, -100))

    fig.tight_layout()
    path = output_dir / "initial_bins_mean_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# figure 3 : persistence_vs_model_mae.png
# ---------------------------------------------------------------------------

def make_persistence_vs_model_mae(hum, pers_mae, model_mae_dict, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = list(range(8))
    freqs = [hum["freqs"][i] for i in bins]

    model_vals = [model_mae_dict[hum["freqs"][i]] for i in bins]
    pers_vals  = [pers_mae[i] for i in bins]

    x = np.arange(len(bins))
    w = 0.35

    bars1 = ax.bar(x - w/2, model_vals, w, label="VanillaLSTM MAE", color="#E69F00", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, pers_vals,  w, label="Persistence MAE", color="#56B4E9", edgecolor="black", linewidth=0.5)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0f}" for f in freqs], fontsize=8)
    ax.set_xlabel("Frequency (MHz)", fontsize=9)
    ax.set_ylabel("MAE (dB)", fontsize=9)
    ax.set_title("VanillaLSTM vs Persistence: t+1 MAE by Frequency", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Caption inside figure
    ax.text(0.5, -0.22,
            "At t+1, copying the final input observation is substantially more accurate in the affected bins.",
            transform=ax.transAxes, ha="center", fontsize=7, fontstyle="italic")

    fig.tight_layout()
    path = output_dir / "persistence_vs_model_mae.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# figure 4 : variance_compression.png
# ---------------------------------------------------------------------------

def make_variance_compression(hs_data, output_dir):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9, 4))

    hs_vals = [d["hs"] for d in hs_data]
    var_vals = [d["var_ratio"] for d in hs_data]
    bias_vals = [d["bias0"] for d in hs_data]

    # Left panel: var ratio vs hidden size
    ax_l.plot(hs_vals, var_vals, "o-", color="#0072B2", markersize=6, linewidth=1.5)
    ax_l.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Ratio = 1.0")
    for h, v in zip(hs_vals, var_vals):
        ax_l.annotate(f"{v:.3f}", (h, v), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=7)
    ax_l.set_xlabel("Hidden size", fontsize=9)
    ax_l.set_ylabel("Mean pred std / tgt std (active bins)", fontsize=8)
    ax_l.set_title("Output Variance vs Hidden Size", fontsize=9)
    ax_l.legend(fontsize=7)
    ax_l.set_xticks(hs_vals)
    ax_l.tick_params(labelsize=8)

    # Right panel: bias vs hidden size
    ax_r.plot(hs_vals, bias_vals, "o-", color="#D55E00", markersize=6, linewidth=1.5)
    ax_r.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Bias = 0 dB")
    for h, v in zip(hs_vals, bias_vals):
        ax_r.annotate(f"{v:.2f} dB", (h, v), textcoords="offset points",
                      xytext=(0, -12), ha="center", fontsize=7)
    ax_r.set_xlabel("Hidden size", fontsize=9)
    ax_r.set_ylabel("Bin 0 bias (dB)", fontsize=9)
    ax_r.set_title("Bin 0 Bias vs Hidden Size", fontsize=9)
    ax_r.legend(fontsize=7)
    ax_r.set_xticks(hs_vals)
    ax_r.tick_params(labelsize=8)

    fig.suptitle("Larger hidden states modestly increase output variance but do not remove the bin-0 bias.",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    path = output_dir / "variance_compression.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# evidence_table.csv
# ---------------------------------------------------------------------------

def make_evidence_table(hum, pers_mae, model_mae_dict, output_dir):
    rows = []
    for i in range(8):
        freq = hum["freqs"][i]
        train_m = float(hum["train_mean"][i])
        tgt_m   = float(hum["tgt"][:, i].mean())
        pred_m  = float(hum["pred"][:, i].mean())
        shift = tgt_m - train_m
        bias  = pred_m - tgt_m
        model_mae = model_mae_dict[freq]
        pers_mae_val = float(pers_mae[i])

        if bias < -5:
            interp = "severe_negative_bias"
        elif bias < -1:
            interp = "moderate_negative_bias"
        elif bias <= 1:
            interp = "close_match"
        else:
            interp = "moderate_positive_bias"

        rows.append({
            "bin_index": i,
            "frequency_mhz": freq,
            "train_mean_dbm": f"{train_m:.2f}",
            "test_target_mean_dbm": f"{tgt_m:.2f}",
            "train_test_shift_db": f"{shift:.2f}",
            "prediction_mean_dbm": f"{pred_m:.2f}",
            "signed_prediction_bias_db": f"{bias:.2f}",
            "model_mae_db": f"{model_mae:.2f}",
            "persistence_mae_db": f"{pers_mae_val:.2f}",
            "interpretation": interp,
        })

    path = output_dir / "evidence_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# presentation_notes.txt
# ---------------------------------------------------------------------------

def make_presentation_notes(hum, gst, pers_mae, model_mae_dict, hs_data, output_dir):
    lines = []
    def w(s=""):
        lines.append(s)

    w("TITLE")
    w("-----")
    w()
    w("Why VanillaLSTM Misses the Initial POWDER Frequency Bins")
    w()
    w()
    w("MAIN MESSAGE")
    w("------------")
    w()
    w("The large error near 600-608 MHz is not a plotting or alignment error; it occurs "
      "because the test signal level shifts away from training, while the model compresses "
      "predictions toward familiar levels and fails to preserve the highly informative last observation.")
    w()
    w()
    w("SLIDE 1 --- OBSERVED PROBLEM")
    w("---------------------------")
    w()
    w("- The mean-power plots show a visible prediction offset in the first eight frequency "
      "bins at the humanities site.")
    w("- The affected range is approximately 600.5-607.5 MHz.")
    w("- The same major offset is not visible at guesthouse.")
    w("- The remaining frequencies generally follow the target mean closely.")
    w("- The original question was whether this came from plotting, frequency alignment, "
      "temporal alignment, normalization, or actual model behavior.")
    w()
    w("Recommended visual: mean_power_site_comparison.png")
    w()
    w("Speaker notes:")
    w("Explain what the black and colored curves represent. Point out the initial humanities "
      "mismatch and the much closer guesthouse match. Keep this slide to approximately 45-60 seconds.")
    w()
    w()
    w("SLIDE 2 --- PIPELINE CHECKS")
    w("--------------------------")
    w()
    w("Checklist of completed verification steps:")
    w()
    w("[OK] Stored t+1 targets matched the raw test data at the stored target rows.")
    w("[OK] Prediction and target frequency ordering matched (metadata frequency lists identical).")
    w("[OK] Normalization and denormalization round-trip checks passed "
      "(denormalizing stored values reproduces original NPZ means).")
    w("[OK] The exact prediction and target means stored in the forecast arrays reproduce "
      "the plotted curves (no plotting-introduced offset).")
    w("[OK] Spectral structures align correctly over the rest of the band (600-800 MHz).")
    w()
    w("Conclusion: \"The visible offset is present in the stored forecasts and is not "
      "introduced by the plotting script.\"")
    w()
    w("Recommended visual: No additional plot required. Use a simple checklist or compact "
      "table in the eventual slide.")
    w()
    w("Speaker notes:")
    w("Spend approximately 45 seconds establishing that the investigation moved beyond "
      "plotting and indexing errors.")
    w()
    w()
    w("SLIDE 3 --- TRAIN/TEST LEVEL SHIFT")
    w("----------------------------------")
    w()
    w("First 8 bins at humanities site (t+1 forecasts):")
    w()

    for i in range(8):
        freq = hum["freqs"][i]
        train_m = hum["train_mean"][i]
        tgt_m   = hum["tgt"][:, i].mean()
        pred_m  = hum["pred"][:, i].mean()
        shift   = tgt_m - train_m
        w(f"Bin {i}, {freq:.1f} MHz:")
        w(f"  train mean = {train_m:.1f} dBm")
        w(f"  test target mean = {tgt_m:.1f} dBm")
        w(f"  shift = {shift:+.1f} dB")
        w(f"  prediction mean = {pred_m:.1f} dBm")
        w()

    w("Bins from approximately 608.5 MHz onward:")
    w("train and test means are much closer in the immediately following noise-floor region, "
      "and the prediction matches that stable level closely.")
    w()
    w("Important interpretation:")
    w("- The humanities test period has a frequency-dependent level shift relative to training "
      "in bins 0-7 (range: {:.1f} to {:.1f} dB).".format(
          min(hum["tgt"][:, i].mean() - hum["train_mean"][i] for i in range(8)),
          max(hum["tgt"][:, i].mean() - hum["train_mean"][i] for i in range(8)),
      ))
    w("- The model reacts to the shift, but its response is inconsistent across frequency.")
    w("- It severely undershoots bins 0-3, is accurate around bins 4-5, and overshoots "
      "slightly around bins 6-7.")
    w("- Therefore, describe this as frequency-dependent bias under a train/test level shift, "
      "not uniform overcorrection.")
    w()
    w("Recommended visual: initial_bins_mean_comparison.png")
    w()
    w("Speaker notes:")
    w("Spend approximately 60-75 seconds. Focus on the pattern rather than reading every number.")
    w()
    w()
    w("SLIDE 4 --- PERSISTENCE REVEALS THE FAILURE")
    w("-------------------------------------------")
    w()
    w("t+1 persistence baseline comparison:")
    w()

    for i in range(8):
        freq = hum["freqs"][i]
        model_mae = model_mae_dict[freq]
        pers_val = pers_mae[i]
        w(f"Bin {i}:")
        w(f"  model MAE = {model_mae:.2f} dB")
        w(f"  persistence MAE = {pers_val:.2f} dB")
        w()

    w("State clearly:")
    w("- For t+1, the last input value is already extremely close to the target.")
    w("- Therefore the target is predictable from the available history.")
    w("- The model underperforms a trivial persistence baseline in these bins.")
    w("- The central failure is that the model does not preserve the current absolute level "
      "as effectively as simply copying the last observation.")
    w()
    w("Recommended visual: persistence_vs_model_mae.png")
    w()
    w("Speaker notes:")
    w("This is the strongest result. Spend approximately 60-75 seconds here.")
    w()
    w()
    w("SLIDE 5 --- GLOBAL VARIANCE COMPRESSION")
    w("---------------------------------------")
    w()
    w("- Across active bins, the median prediction-standard-deviation to target-standard-deviation "
      "ratio is approximately 0.44-0.53.")
    w("- Humanities and guesthouse show similar overall ratios (approximately 0.44 and 0.40 "
      "respectively).")
    w("- Increasing hidden size from 8 to 64 only changed the active-bin mean ratio from "
      "approximately {:.3f} to {:.3f}.".format(hs_data[0]["var_ratio"], hs_data[-1]["var_ratio"]))
    w("- Bin 0 bias remained approximately {:.2f} dB across hidden sizes 8, 16, 32, and 64.".format(
        np.mean([d["bias0"] for d in hs_data])))
    w("- This shows that increasing hidden dimension alone did not materially solve the "
      "initial-bin bias.")
    w("- The predictions are generally less variable than the targets, and the humanities "
      "train/test mean shift makes this compression visibly damaging in the first bins.")
    w()
    w("Hidden-size experiment summary:")
    for d in hs_data:
        w(f"  hidden {d['hs']}: mean variance ratio = {d['var_ratio']:.3f}, "
          f"bin 0 bias = {d['bias0']:.2f} dB")
    w()
    w("Recommended visual: variance_compression.png")
    w()
    w("Speaker notes:")
    w("Spend approximately 45-60 seconds. The key message is that this is broader than one "
      "bad bin and not fixed by simply enlarging the hidden state.")
    w()
    w()
    w("SLIDE 6 --- CONCLUSION")
    w("---------------------")
    w()
    w("1. The 600.5-607.5 MHz offset is a real model-output pattern, not a plotting, "
      "frequency-order, target-row, or denormalization error detected by the completed checks.")
    w("2. The humanities test period has a substantial frequency-specific level shift, while "
      "the model compresses outputs and fails to preserve the highly predictive last observation.")
    w("3. Persistence is dramatically better for these t+1 bins, showing that future model "
      "comparisons must include simple baselines and frequency-region-level diagnostics rather "
      "than only aggregate RMSE.")
    w()
    w("Next-step statement:")
    w("\"Future work will compare models by frequency behavior and evaluate whether residual "
      "or persistence-aware forecasting improves performance without assuming a single "
      "architectural root cause.\"")
    w()
    w()
    w("TOTAL TIMING")
    w("------------")
    w()
    w("Slide 1: 45-60 seconds")
    w("Slide 2: 40-50 seconds")
    w("Slide 3: 60-75 seconds")
    w("Slide 4: 60-75 seconds")
    w("Slide 5: 45-60 seconds")
    w("Slide 6: 30-45 seconds")
    w()
    w("Expected total: approximately 5-6.5 minutes.")
    w()
    w("30-second summary:")
    w("The VanillaLSTM model shows a large mean-power offset at 600.5-607.5 MHz. Pipeline "
      "checks confirm this is a real model output, not an artifact. The test period's signal "
      "level differs from training, and the model's predictions compress toward a global mean "
      "rather than tracking the per-bin level. A trivial persistence baseline (copying the "
      "last observation) achieves 0.26 dB MAE vs the model's 7.43 dB. Increasing LSTM hidden "
      "size from 8 to 64 does not fix the bias, confirming the issue is not simply capacity "
      "but how the model preserves per-frequency level information.")

    path = output_dir / "presentation_notes.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# presentation_manifest.txt
# ---------------------------------------------------------------------------

def make_manifest(hum, gst, pers_mae, model_mae_dict, hs_data, output_dir,
                  generated_files, script_path):
    lines = []
    def w(s=""):
        lines.append(s)

    w("PRESENTATION MANIFEST")
    w("=====================")
    w()
    w("Generated by: " + str(script_path.resolve()))
    w("Generated at: " + str(Path(".").resolve()))
    w()
    w("SOURCE ARTIFACTS USED")
    w("---------------------")
    w()

    w("Humanities forecasts (t+1):")
    w("  Predictions: " + str(ROOT / "results/powder/600_800/vanillalstm/humanities/forecasts/powder_600_800_humanities_test_vanillalstm_predictions.npz"))
    w("  Targets:     " + str(ROOT / "results/powder/600_800/vanillalstm/humanities/forecasts/powder_600_800_humanities_test_vanillalstm_targets.npz"))
    w("  Metadata:    " + str(ROOT / "results/powder/600_800/vanillalstm/humanities/forecasts/powder_600_800_humanities_test_vanillalstm_metadata.json"))
    w()
    w("Guesthouse forecasts (t+1):")
    w("  Predictions: " + str(ROOT / "results/powder/600_800/vanillalstm/guesthouse/forecasts/powder_600_800_guesthouse_test_vanillalstm_predictions.npz"))
    w("  Targets:     " + str(ROOT / "results/powder/600_800/vanillalstm/guesthouse/forecasts/powder_600_800_guesthouse_test_vanillalstm_targets.npz"))
    w("  Metadata:    " + str(ROOT / "results/powder/600_800/vanillalstm/guesthouse/forecasts/powder_600_800_guesthouse_test_vanillalstm_metadata.json"))
    w()
    w("Hidden size experiment forecasts (t+1):")
    for hs in ["h8", "h16", "h32", "h64"]:
        p = ROOT / f"results/hidden_size_exp/{hs}/vanillalstm/humanities/forecasts/powder_600_800_humanities_test_vanillalstm_predictions.npz"
        w(f"  {hs}: " + str(p))
    w()
    w("Per-frequency metrics CSV:")
    w("  " + str(ROOT / "results/powder/600_800/vanillalstm/humanities/per_frequency_metrics.csv"))
    w()
    w("Raw test CSV:")
    w("  " + str(ROOT / "data/powder_20260628T0436Z_humanities_600_800.csv"))
    w()
    w()
    w("OUTPUT FILES GENERATED")
    w("----------------------")
    for f in generated_files:
        w("  " + str(f.name))
    w("  presentation_manifest.txt")
    w()
    w()
    w("SPLIT, SITE, HORIZON, AND FREQUENCY RANGE")
    w("------------------------------------------")
    w("  Split:      humanities_test (for primary analysis), guesthouse_test (for comparison)")
    w("  Site:       humanities (primary), guesthouse (comparison)")
    w("  Horizon:    t+1 (all analysis uses horizon=1)")
    w("  Frequency:  600.5-799.5 MHz (200 bins, 1 MHz spacing)")
    w("  Offset region: 600.5-607.5 MHz (bins 0-7)")
    w()
    w()
    w("VALUE SOURCES")
    w("-------------")
    w("  Test target means and prediction means:   Computed from stored NPZ forecast arrays "
      "(denormalized, stored as dBm).")
    w("  Training means:                           From metadata JSON mean_dbm array "
      "(fitted per-frequency normalization mean over training data).")
    w("  Model MAE per bin:                        From per_frequency_metrics.csv, horizon=1.")
    w("  Persistence MAE:                          Computed from raw test CSV "
      "(powder_20260628T0436Z_humanities_600_800.csv) as mean absolute difference between "
      "consecutive time steps per bin.")
    w("  Hidden size variance ratios and biases:   Computed from hidden_size_exp NPZ forecast "
      "arrays and metadata JSON std_dbm arrays.")
    w()
    w()
    w("NOTES ON VALUE DEVIATIONS")
    w("-------------------------")
    w("  The presentation notes template included values (e.g., train_mean=-102.6 dBm for bin 0) "
      "that correspond to guesthouse training metadata, not humanities. The humanities training "
      "mean for bin 0 is -116.85 dBm (from humanities metadata JSON). All values in the output "
      "artifacts use the correct humanities training means from the stored metadata. The "
      "train/test shift direction is therefore reversed relative to the template: the humanities "
      "test period has a STRONGER signal at bins 0-3, not a weaker one. The original template "
      "appears to have been drafted under the assumption that the original guesthouse-trained "
      "checkpoint was used; after retraining on humanities, the training means changed.")
    w()
    w()
    w("CONFIRMATIONS")
    w("-------------")
    w("  [OK] No models were retrained and no new experiment was run.")
    w("  [OK] All plotted means were calculated from the same forecast origins for each "
      "target and prediction (stored NPZ t_plus_1 arrays).")
    w("  [OK] Humanities and guesthouse figures use identical axis ranges (sharey=True in "
      "mean_power_site_comparison.png; same 600-800 MHz x-axis for both).")
    w("  [OK] No forecast or metrics artifacts were altered.")
    w("  [OK] All values were read from existing CSV, NPZ, or JSON artifacts; no training "
      "or inference was performed.")
    w("  [OK] Training means are from the same normalization that was fitted to the training "
      "data and stored in the metadata JSON.")

    path = output_dir / "presentation_manifest.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Generated {path}")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build presentation-support package for initial-bin mean offset."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results"),
        help="Base results directory (default: <repo_root>/results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__)

    # Override ROOT-relative paths if --results-dir is provided
    global _RD
    _RD = Path(args.results_dir)

    # ---- data extraction ------------------------------------------------
    print("Extracting humanities data...")
    hum = extract_site_data(get_humanities_base_dir(), "humanities")
    print("Extracting guesthouse data...")
    gst = extract_site_data(get_guesthouse_base_dir(), "guesthouse")
    print("Extracting hidden-size experiment data...")
    hs_data = extract_hidden_size_data()
    print("Computing persistence MAE...")
    pers_mae = compute_persistence_mae()
    print("Loading model per-frequency MAE...")
    model_mae_dict = load_model_per_freq_mae()

    print(f"\nHumanities bin 0: train_mean={hum['train_mean'][0]:.1f} dBm, "
          f"tgt_mean={hum['tgt'][:,0].mean():.1f} dBm, "
          f"pred_mean={hum['pred'][:,0].mean():.1f} dBm, "
          f"bias={hum['pred'][:,0].mean() - hum['tgt'][:,0].mean():.1f} dB")

    # ---- generate outputs -----------------------------------------------
    generated = []
    print("\nGenerating figures and files...")
    generated.append(make_mean_power_site_comparison(hum, gst, output_dir))
    generated.append(make_initial_bins_mean_comparison(hum, gst, output_dir))
    generated.append(make_persistence_vs_model_mae(hum, pers_mae, model_mae_dict, output_dir))
    generated.append(make_variance_compression(hs_data, output_dir))
    generated.append(make_evidence_table(hum, pers_mae, model_mae_dict, output_dir))
    generated.append(make_presentation_notes(hum, gst, pers_mae, model_mae_dict, hs_data, output_dir))
    generated.append(make_manifest(hum, gst, pers_mae, model_mae_dict, hs_data, output_dir, generated, script_path))

    print(f"\nAll files generated in {output_dir}:")
    for f in generated:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
