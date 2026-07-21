"""Evaluate trained LookbackMean baselines on test set (20260628) and report MAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.LookbackMean import LookbackMeanForecaster
from training.common.forecasting import forecast
from training.common.preprocessing import apply_per_frequency_normalization
from training.common.windowing import WindowDataset, make_window_starts
from training.common.runtime import device_for


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def load_and_concat_csvs(file_entries: list[dict], start_mhz: float, end_mhz: float) -> tuple[np.ndarray, list[float]]:
    arrays = []
    freqs = None
    for entry in file_entries:
        path = resolve_path(entry["path"])
        df = pd.read_csv(path)
        freq_cols = [c for c in df.columns[1:] if start_mhz <= float(c) <= end_mhz]
        freq_cols.sort(key=float)
        if freqs is None:
            freqs = [float(c) for c in freq_cols]
        arrays.append(df[freq_cols].to_numpy(dtype=np.float32))
    return np.concatenate(arrays, axis=0), freqs


def make_full_test_loader(data_norm: np.ndarray, lookback: int, rollout_horizon: int, batch_size: int = 32) -> DataLoader:
    starts = make_window_starts(len(data_norm), lookback, rollout_horizon, stride=1)
    ds = WindowDataset(data_norm, starts, lookback, rollout_horizon)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def mae_denorm(pred_norm: np.ndarray, target_norm: np.ndarray, mean, std) -> float:
    pred_dbm = pred_norm * std + mean
    target_dbm = target_norm * std + mean
    return float(np.mean(np.abs(pred_dbm - target_dbm)))


def evaluate_csv(ckpt_path: Path, train_files: list[dict], test_files: list[dict], freq_index: int | None = None, lookback: int = 60) -> float:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    norm = ckpt["normalization"]
    mean = np.asarray(norm["mean_dbm"], dtype=np.float32)
    std = np.asarray(norm["std_dbm"], dtype=np.float32)

    start_mhz, end_mhz = 600.0, 800.0
    train_arr, _ = load_and_concat_csvs(train_files, start_mhz, end_mhz)
    test_arr, _ = load_and_concat_csvs(test_files, start_mhz, end_mhz)

    if freq_index is not None:
        train_arr = train_arr[:, freq_index:freq_index+1]
        test_arr = test_arr[:, freq_index:freq_index+1]

    train_norm = apply_per_frequency_normalization(train_arr, mean, std).astype(np.float32)
    test_norm = apply_per_frequency_normalization(test_arr, mean, std).astype(np.float32)

    test_loader = make_full_test_loader(test_norm, lookback, rollout_horizon=1)

    input_size = int(np.prod(train_arr.shape[1:]))
    model_cfg = {"model": {"input_sequence_length": lookback, "prediction_horizon": 1, "input_size": input_size}}
    model = LookbackMeanForecaster(model_cfg).to(device_for({"training": {"device": "auto"}}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device_for({"training": {"device": "auto"}})), y.to(device_for({"training": {"device": "auto"}}))
            pred = forecast(model, x, prediction_horizon=1, rollout_horizon=1, targets=None)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    targ_np = np.concatenate(all_targets, axis=0)
    return mae_denorm(pred_np, targ_np, mean, std)


def evaluate_map(ckpt_path: Path, lookback: int = 60, map_dir: Path | None = None, map_name: str = "powder_600_800") -> float:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    norm = ckpt["normalization"]
    mean = norm["mean_dbm"].astype(np.float32)
    std = norm["std_dbm"].astype(np.float32)

    start_mhz, end_mhz = 600.0, 800.0
    if map_dir is None:
        map_dir = ROOT / "evaluation" / "results" / "idw"
    train_map = map_dir / f"{map_name}_train.npz"
    test_map_path = map_dir / f"{map_name}_test.npz"

    train_raw = np.load(str(train_map))["map_db"].astype(np.float32)
    test_raw = np.load(str(test_map_path))["map_db"].astype(np.float32)
    freqs_arr = np.load(str(train_map))["freqs_mhz"]
    freq_mask = np.array([(start_mhz <= f <= end_mhz) for f in freqs_arr])
    train_raw = train_raw[:, :, :, freq_mask]
    test_raw = test_raw[:, :, :, freq_mask]

    train_norm = apply_per_frequency_normalization(train_raw, mean, std).astype(np.float32)
    test_norm = apply_per_frequency_normalization(test_raw, mean, std).astype(np.float32)

    test_loader = make_full_test_loader(test_norm, lookback, rollout_horizon=1)

    input_size = int(np.prod(train_raw.shape[1:]))
    model_cfg = {"model": {"input_sequence_length": lookback, "prediction_horizon": 1, "input_size": input_size}}
    model = LookbackMeanForecaster(model_cfg).to(device_for({"training": {"device": "auto"}}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device_for({"training": {"device": "auto"}})), y.to(device_for({"training": {"device": "auto"}}))
            pred = forecast(model, x, prediction_horizon=1, rollout_horizon=1, targets=None)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    targ_np = np.concatenate(all_targets, axis=0)

    mean_br = mean.reshape(1, 1, -1, 1, 1)
    std_br = std.reshape(1, 1, -1, 1, 1)
    return mae_denorm(pred_np, targ_np, mean_br, std_br)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["1d", "2d", "4d", "all"], default="all")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("--freq-index", type=int, default=100)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROOT / "training" / "LookbackMean" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_files = config.get("data", {}).get("files", [])
    train_files = [f for f in data_files if f.get("partition") == "train"]
    test_files = [f for f in data_files if f.get("partition") == "test"]

    ckpt_dir = ROOT / "training" / "results" / "baselines"
    lookback = 60

    results = {}
    if args.mode in ("1d", "all"):
        mae = evaluate_csv(ckpt_dir / "lookbackmean1d.pt", train_files, test_files, freq_index=args.freq_index, lookback=lookback)
        results["lookbackmean1d"] = mae
        print(f"  lookbackmean1d  MAE: {mae:.4f} dB")

    if args.mode in ("2d", "all"):
        mae = evaluate_csv(ckpt_dir / "lookbackmean2d.pt", train_files, test_files, freq_index=None, lookback=lookback)
        results["lookbackmean2d"] = mae
        print(f"  lookbackmean2d  MAE: {mae:.4f} dB")

    if args.mode in ("4d", "all"):
        map_cfg = config.get("data", {}).get("map", {})
        map_dir = resolve_path(map_cfg.get("output_dir", "evaluation/results/idw")) if map_cfg.get("output_dir") else None
        map_name = map_cfg.get("name", "powder_600_800")
        mae = evaluate_map(ckpt_dir / "lookbackmean4d.pt", lookback=lookback, map_dir=map_dir, map_name=map_name)
        results["lookbackmean4d"] = mae
        print(f"  lookbackmean4d  MAE: {mae:.4f} dB")

    print("\nTest MAE (20260628):")
    for name, m in results.items():
        print(f"  {name:18s}  {m:.4f} dB")


if __name__ == "__main__":
    main()
