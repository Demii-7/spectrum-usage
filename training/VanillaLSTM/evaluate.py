from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import create_datasets, denormalize_array
from model import VanillaLSTMForecaster
from utils import (
    compute_per_frequency_rmse,
    compute_per_horizon_rmse,
    compute_regression_metrics,
    flatten_forecasts,
    get_device,
    load_config,
    load_normalization_stats,
    plot_error_analysis,
    plot_spectrogram,
    resolve_path,
    save_array_csv,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the VanillaLSTM spectrum forecaster.")
    parser.add_argument("--checkpoint", default="training/VanillaLSTM/checkpoints/best_model.pt")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--csv", default=None, help="Optional CSV path override.")
    parser.add_argument("--output-dir", default=None, help="Optional evaluation output directory override.")
    return parser.parse_args()


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for inputs, labels in loader:
        outputs = model(inputs.to(device)).cpu().numpy()
        predictions.append(outputs)
        targets.append(labels.numpy())
    if not predictions:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = load_config(args.config) if args.config else checkpoint["config"]
    device = get_device(str(config["device"]["device"]))
    bundle = create_datasets(config, csv_path=args.csv)
    test_dataset = bundle["datasets"]["test"]
    if len(test_dataset) == 0:
        raise RuntimeError("Test split produced zero windows. Check CSV size and split settings.")

    batch_size = int(config["training"]["batch_size"])
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VanillaLSTMForecaster(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_norm, tgt_norm = run_inference(model, loader, device)
    stats = load_normalization_stats(Path(config["paths"]["checkpoints_dir"]) / "normalization_stats.pt")
    pred_dbm = denormalize_array(pred_norm, stats)
    tgt_dbm = denormalize_array(tgt_norm, stats)

    overall = compute_regression_metrics(pred_dbm, tgt_dbm)
    per_horizon_rmse = compute_per_horizon_rmse(pred_dbm, tgt_dbm)
    per_frequency_rmse = compute_per_frequency_rmse(pred_dbm, tgt_dbm)

    metrics = {
        "overall_rmse": overall["rmse"],
        "overall_mae": overall["mae"],
        "overall_r2": overall["r2"],
        "per_horizon_rmse": per_horizon_rmse,
        "per_frequency_rmse": per_frequency_rmse,
        "n_test_windows": bundle["window_counts"]["test"],
    }

    output_dir = Path(args.output_dir or config["paths"]["evaluation_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)

    if bool(config["evaluation"].get("export_predictions", True)):
        save_array_csv(output_dir / "predictions.csv", flatten_forecasts(pred_dbm))
        save_array_csv(output_dir / "ground_truth.csv", flatten_forecasts(tgt_dbm))

    if bool(config["evaluation"].get("plot_denormalized_dbm", True)):
        site_name = str(config["data"].get("site_name", "site"))
        plot_spectrogram(pred_dbm[0], tgt_dbm[0], site_name, output_dir / f"spectrogram_{site_name}.png")
        plot_error_analysis(pred_dbm, tgt_dbm, per_frequency_rmse, output_dir / "error_analysis.png")

    print(f"Test windows: {bundle['window_counts']['test']}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.6f}")
    print(f"Overall MAE: {metrics['overall_mae']:.6f}")
    print(f"Overall R2: {metrics['overall_r2']:.6f}")
    print(f"Per-horizon RMSE: {metrics['per_horizon_rmse']}")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
