"""Evaluate the trained daily-history STS model and matched Linear AR."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset import resolve_branch_config
from linear_ar_baseline import LagMatchedLinearAR, lag_sets, train_variant
from stsprednet import STSPredNet
from train_integrated import to_sts_layout
from training.common.config import load_config
from training.common.data import chunk_specs, load_chunk
def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    timestamps = np.asarray(archive["timestamps"]).astype("datetime64[m]")
    return archive["map_db"].astype(np.float32), timestamps


def norm_map(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (data - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    return np.moveaxis(normalized, -1, 1).astype(np.float32)


def score(pred: np.ndarray, target: np.ndarray, mean, std) -> tuple[float, float]:
    pred_dbm = pred * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    err = pred_dbm - target
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err * err)))


def history_or_generated(generated: dict, history: dict, timestamp: np.datetime64) -> np.ndarray:
    """Use an origin's recursive prediction only when it has already been generated."""
    if timestamp in generated:
        return generated[timestamp]
    return history[timestamp]


def sts_predict(model, device, history: dict, origins: list[np.datetime64], horizon: int, branches):
    origin = np.asarray(origins) - np.timedelta64(horizon, "m")
    generated = [{} for _ in origins]
    for step in range(1, horizon + 1):
        current = origin + np.timedelta64(step, "m")
        close = np.stack([
            np.stack([
                history_or_generated(generated[index], history, current[index] - np.timedelta64(i, "m"))
                for i in range(branches["lc"], 0, -1)
            ])
            for index in range(len(origins))
        ], axis=0)
        period = np.stack([
            np.stack([history[t] for t in current - np.timedelta64(i * branches["period_interval"], "m")])
            for i in range(branches["lp"], 0, -1)
        ], axis=1)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(close).to(device),
                torch.from_numpy(period).to(device), None,
            ).cpu().numpy()
        for index, (timestamp, value) in enumerate(zip(current, pred)):
            generated[index][timestamp] = value
    return np.stack([generated[index][timestamp] for index, timestamp in enumerate(origins)])


def ar_predict(model, device, history: dict, origins: list[np.datetime64], horizon: int, lags: list[int]):
    origin = np.asarray(origins) - np.timedelta64(horizon, "m")
    generated = [{} for _ in origins]
    for step in range(1, horizon + 1):
        current = origin + np.timedelta64(step, "m")
        frames = np.stack([
            np.stack([
                history_or_generated(generated[index], history, current[index] - np.timedelta64(lag, "m"))
                for lag in lags
            ])
            for index in range(len(origins))
        ], axis=0)
        with torch.no_grad():
            pred = model(torch.from_numpy(frames).to(device)).cpu().numpy()
        for index, (timestamp, value) in enumerate(zip(current, pred)):
            generated[index][timestamp] = value
    return np.stack([generated[index][timestamp] for index, timestamp in enumerate(origins)])


def evaluate_model(model, predictor, device, data, timestamps, target_times, horizons, branches, mean, std, label, out):
    rows = []
    lookup = {timestamp: frame for timestamp, frame in zip(timestamps, data)}
    for horizon in horizons:
        valid = []
        targets = []
        for t in target_times:
            try:
                target = lookup[t]
                origin = t - np.timedelta64(horizon, "m")
                if all(origin - np.timedelta64(i, "m") in lookup for i in range(60)) and all(t - np.timedelta64(i * branches["period_interval"], "m") in lookup for i in range(1, branches["lp"] + 1)):
                    valid.append(t)
                    targets.append(target)
            except KeyError:
                pass
        if not valid:
            continue
        pred = predictor(lookup, valid, horizon, branches)
        mae, rmse = score(pred, np.stack(targets), mean, std)
        rows.append({"model": label, "horizon": horizon, "n_targets": len(valid), "mae_db": mae, "rmse_db": rmse})
    out.extend(rows)


def valid_origins(target_times, lookup, horizon, lags, recent_lags, recent_times=None):
    """Select targets whose observed inputs cover every recursive rollout step."""
    valid = []
    for target in target_times:
        ok = True
        origin = target - np.timedelta64(horizon, "m")
        for step in range(1, horizon + 1):
            current = origin + np.timedelta64(step, "m")
            for lag in lags:
                # Short lags produced earlier in this same rollout are predictions,
                # rather than required observed frames.
                if lag < step:
                    continue
                timestamp = current - np.timedelta64(lag, "m")
                if timestamp not in lookup or (recent_times is not None and lag in recent_lags and timestamp not in recent_times):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(target)
    return valid


def report_preflight(label: str, split: str, horizon: int, targets, valid) -> None:
    print(f"{label} {split} horizon={horizon}: {len(valid)}/{len(targets)} targets have complete recursive history")


def write_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def shared_valid(target_times, lookup, max_horizon, lags, recent_lags, recent_times=None):
    """Compute valid origins for the largest horizon, reuse for all."""
    return valid_origins(target_times, lookup, max_horizon, lags, recent_lags, recent_times)


def main() -> None:
    config_path = Path(sys.argv[1])
    checkpoint_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    config = load_config(config_path)
    chunk = chunk_specs(config)[0]
    data = load_chunk(config, chunk, val_fraction=0.1)
    train_split = data.splits[data.train_split]
    validation_split = data.splits[data.validation_split]
    test_split = data.splits[data.test_split]
    train_map, train_times = load_npz("data/maps/plan_powder_600_800_train.npz")
    validation_map, validation_times = load_npz("data/maps/plan_powder_600_800_validation.npz")
    test_map, test_times = load_npz("data/maps/plan_powder_600_800_test.npz")
    train_map_cwh = np.moveaxis(train_map, -1, 1)
    validation_map_cwh = np.moveaxis(validation_map, -1, 1)
    test_map_cwh = np.moveaxis(test_map, -1, 1)
    mean = data.normalization["mean_dbm"]
    std = data.normalization["std_dbm"]
    branches = resolve_branch_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = STSPredNet(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_norm = norm_map(train_map, mean, std)
    validation_norm = norm_map(validation_map, mean, std)
    test_norm = norm_map(test_map, mean, std)

    results = []
    sts_lags = list(range(1, branches["lc"] + 1)) + [
        i * branches["period_interval"] for i in range(1, branches["lp"] + 1)
    ]
    sts_recent_lags = set(range(1, branches["lc"] + 1))
    horizons = (1, 15, 60)

    # --- T4 validation ---
    validation_times_full = np.concatenate((train_times, validation_times))
    validation_data_full = np.concatenate((train_norm, validation_norm))
    validation_history = {t: x for t, x in zip(validation_times_full, validation_data_full)}
    validation_history_raw = {t: x for t, x in zip(validation_times_full, np.concatenate((train_map_cwh, validation_map_cwh)))}
    valid = shared_valid(validation_times, validation_history, 60, sts_lags, sts_recent_lags)
    report_preflight("STS-PredNet", "T4_validation", 60, validation_times, valid)
    for horizon in horizons:
        pred = sts_predict(model, device, validation_history, valid, horizon, branches)
        target = np.stack([validation_history_raw[t] for t in valid])
        mae, rmse = score(pred, target, mean, std)
        results.append({"model": "STS-PredNet", "split": "T4_validation", "horizon": horizon, "n_targets": len(valid), "mae_db": mae, "rmse_db": rmse})

    # --- T4 context → T6 ---
    train_times_full = np.concatenate((train_times, validation_times))
    train_data_full = np.concatenate((train_norm, validation_norm))
    train_data_raw_full = np.concatenate((train_map_cwh, validation_map_cwh))
    t4t6_lookup = {t: x for t, x in zip(np.concatenate((train_times_full, test_times)), np.concatenate((train_data_full, test_norm)))}
    t4t6_lookup_raw = {t: x for t, x in zip(np.concatenate((train_times_full, test_times)), np.concatenate((train_data_raw_full, test_map_cwh)))}
    valid = shared_valid(list(test_times), t4t6_lookup, 60, sts_lags, sts_recent_lags, set(test_times))
    report_preflight("STS-PredNet", "T4_context_T6", 60, test_times, valid)
    for horizon in horizons:
        pred = sts_predict(model, device, t4t6_lookup, valid, horizon, branches)
        target = np.stack([t4t6_lookup_raw[t] for t in valid])
        mae, rmse = score(pred, target, mean, std)
        results.append({"model": "STS-PredNet", "split": "T4_context_T6", "horizon": horizon, "n_targets": len(valid), "mae_db": mae, "rmse_db": rmse})
    write_results(output_path, results)

    # --- T6 only (no T4 context) ---
    t6_lookup = {t: x for t, x in zip(test_times, test_norm)}
    t6_lookup_raw = {t: x for t, x in zip(test_times, test_map_cwh)}
    valid = shared_valid(list(test_times), t6_lookup, 60, sts_lags, sts_recent_lags)
    report_preflight("STS-PredNet", "T6_only", 60, test_times, valid)
    for horizon in horizons:
        pred = sts_predict(model, device, t6_lookup, valid, horizon, branches)
        target = np.stack([t6_lookup_raw[t] for t in valid])
        mae, rmse = score(pred, target, mean, std)
        results.append({"model": "STS-PredNet", "split": "T6_only", "horizon": horizon, "n_targets": len(valid), "mae_db": mae, "rmse_db": rmse})
    write_results(output_path, results)

    # --- Linear AR ---
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ar_lags = lag_sets(config)["recent_daily"]
    ar_checkpoint = train_variant(config, data, chunk, "recent_daily", ar_lags, checkpoint_dir)
    saved = torch.load(ar_checkpoint, map_location="cpu", weights_only=False)
    ar = LagMatchedLinearAR(tuple(saved["feature_shape"]), len(saved["lags"])).to(device)
    ar.load_state_dict(saved["model_state_dict"])
    ar.eval()
    ar_splits = [
        ("T4_validation", validation_times, validation_history, validation_history_raw, None),
        ("T4_context_T6", test_times, t4t6_lookup, t4t6_lookup_raw, set(test_times)),
        ("T6_only", test_times, t6_lookup, t6_lookup_raw, None),
    ]
    for split_name, times, lookup, lookup_raw, recent_times in ar_splits:
        valid = shared_valid(list(times), lookup, 60, saved["lags"], sts_recent_lags, recent_times)
        report_preflight("Daily-history Linear AR", split_name, 60, times, valid)
        for horizon in horizons:
            pred = ar_predict(ar, device, lookup, valid, horizon, saved["lags"])
            target = np.stack([lookup_raw[t] for t in valid])
            mae, rmse = score(pred, target, mean, std)
            results.append({"model": "Daily-history Linear AR", "split": split_name, "horizon": horizon, "n_targets": len(valid), "mae_db": mae, "rmse_db": rmse})
            write_results(output_path, results)


if __name__ == "__main__":
    main()
