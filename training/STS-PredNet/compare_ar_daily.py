"""Ablation: evaluate lp0, lp1, lp2 Linear AR on lp2's target samples.

For each split (T4_validation, T4_context_T6, T6_only):
1. Compute valid origins using lp2 lags (most restrictive: needs t-2880).
2. Train lp0, lp1, lp2 AR variants.
3. Evaluate each on the same lp2-eligible targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from linear_ar_baseline import LagMatchedLinearAR, train_variant, _targets
from training.common.config import load_config
from training.common.data import chunk_specs, load_chunk


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    timestamps = np.asarray(archive["timestamps"]).astype("datetime64[m]")
    return archive["map_db"].astype(np.float32), timestamps


def norm_map(data, mean, std):
    normalized = (data - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    return np.moveaxis(normalized, -1, 1).astype(np.float32)


def score(pred, target, mean, std):
    pred_dbm = pred * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    err = pred_dbm - target
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err * err)))


def history_or_generated(generated, history, timestamp):
    if timestamp in generated:
        return generated[timestamp]
    return history[timestamp]


def ar_predict(model, device, history, origins, horizon, lags):
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


def valid_origins(target_times, lookup, horizon, lags, recent_lags, recent_times=None):
    valid = []
    for target in target_times:
        ok = True
        origin = target - np.timedelta64(horizon, "m")
        for step in range(1, horizon + 1):
            current = origin + np.timedelta64(step, "m")
            for lag in lags:
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


def main() -> None:
    config_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    config = load_config(config_path)
    chunk = chunk_specs(config)[0]
    data = load_chunk(config, chunk, val_fraction=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_map, train_times = load_npz("data/maps/plan_powder_600_800_train.npz")
    validation_map, validation_times = load_npz("data/maps/plan_powder_600_800_validation.npz")
    test_map, test_times = load_npz("data/maps/plan_powder_600_800_test.npz")
    train_map_cwh = np.moveaxis(train_map, -1, 1)
    validation_map_cwh = np.moveaxis(validation_map, -1, 1)
    test_map_cwh = np.moveaxis(test_map, -1, 1)
    mean = data.normalization["mean_dbm"]
    std = data.normalization["std_dbm"]

    train_norm = norm_map(train_map, mean, std)
    validation_norm = norm_map(validation_map, mean, std)
    test_norm = norm_map(test_map, mean, std)

    # All three lag sets
    recent_lags = list(range(60, 0, -1))
    lp0_lags = recent_lags                    # [60..1]
    lp1_lags = recent_lags + [1440]           # [60..1, 1440]
    lp2_lags = recent_lags + [1440, 2880]     # [60..1, 1440, 2880]
    recent_set = set(range(1, 61))
    horizons = (1, 15, 60)

    # Build history dicts
    validation_times_full = np.concatenate((train_times, validation_times))
    validation_data_full = np.concatenate((train_norm, validation_norm))
    validation_history = {t: x for t, x in zip(validation_times_full, validation_data_full)}
    validation_history_raw = {t: x for t, x in zip(validation_times_full, np.concatenate((train_map_cwh, validation_map_cwh)))}

    train_times_full = np.concatenate((train_times, validation_times))
    train_data_full = np.concatenate((train_norm, validation_norm))
    train_data_raw_full = np.concatenate((train_map_cwh, validation_map_cwh))
    t4t6_lookup = {t: x for t, x in zip(np.concatenate((train_times_full, test_times)), np.concatenate((train_data_full, test_norm)))}
    t4t6_lookup_raw = {t: x for t, x in zip(np.concatenate((train_times_full, test_times)), np.concatenate((train_data_raw_full, test_map_cwh)))}

    t6_lookup = {t: x for t, x in zip(test_times, test_norm)}
    t6_lookup_raw = {t: x for t, x in zip(test_times, test_map_cwh)}

    splits = [
        ("T4_validation", validation_times, validation_history, validation_history_raw, None),
        ("T4_context_T6", list(test_times), t4t6_lookup, t4t6_lookup_raw, set(test_times)),
        ("T6_only", list(test_times), t6_lookup, t6_lookup_raw, None),
    ]

    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Train all three variants
    ar_configs = [
        ("lp0", lp0_lags),
        ("lp1", lp1_lags),
        ("lp2", lp2_lags),
    ]

    models = {}
    for variant, lags in ar_configs:
        print(f"\n=== Training {variant} AR (n_lags={len(lags)}, max_lag={max(lags)}) ===")
        ckpt = train_variant(config, data, chunk, f"recent_{variant}", lags, checkpoint_dir)
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = LagMatchedLinearAR(tuple(saved["feature_shape"]), len(saved["lags"])).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        models[variant] = (model, saved["lags"])

    # Evaluate each variant on lp2-eligible targets
    for split_name, times, lookup, lookup_raw, recent_times in splits:
        lp2_valid = valid_origins(list(times), lookup, 60, lp2_lags, recent_set, recent_times)
        if not lp2_valid:
            print(f"\n  {split_name}: no lp2-eligible targets")
            continue
        print(f"\n  {split_name}: {len(lp2_valid)} lp2-eligible targets")

        for variant, (model, lags) in models.items():
            for horizon in horizons:
                pred = ar_predict(model, device, lookup, lp2_valid, horizon, lags)
                target = np.stack([lookup_raw[t] for t in lp2_valid])
                mae, rmse = score(pred, target, mean, std)
                row = {
                    "model": f"LinearAR-{variant}",
                    "split": split_name,
                    "horizon": horizon,
                    "n_targets": len(lp2_valid),
                    "mae_db": mae,
                    "rmse_db": rmse,
                }
                results.append(row)
                print(f"    {variant} h={horizon}: MAE={mae:.4f} RMSE={rmse:.4f}")

    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
