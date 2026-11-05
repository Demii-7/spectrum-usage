"""Does training the shared-weight 1D ResidualLinearAR on a raw-unit loss close the gap on ElectroSense?

The SpectrumLLM baselines train on raw PSD with a raw-unit loss; our pipeline trains on per-channel standardized data,
so every channel counts equally in the loss, while the reported metric (raw MSE) weights noisy channels more. The
pipeline flattens 1D samples and loses the channel id, so this is a controlled A/B outside it: identical model, data,
optimizer and rollout evaluation, only the per-sample loss weight differs.

    arm "standardized": weight 1 for every (window, channel) sample (what our pipeline does)
    arm "raw":          weight std_c**2 / mean(std**2), i.e. the loss equals raw-unit MSE up to one constant

    python -m training.ray.loss_units --datasets spec4 spec25 spec151 --seeds 40 41 42 43 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

HORIZONS = (12, 24, 48, 96, 192)
SEQ_LEN = 192
RIDGE = 0.01
CLIP = 1.0
VAL_STRIDE = 5


def channel_weights(std: np.ndarray, arm: str) -> np.ndarray:
    """Per-channel loss weights, normalized to mean 1 so the learning-rate scale is unchanged."""
    if arm == "standardized":
        return np.ones_like(std, dtype=np.float64)
    if arm == "raw":
        var = np.asarray(std, dtype=np.float64) ** 2
        return var / var.mean()
    raise ValueError(f"unknown arm {arm!r}")


def rollout_raw_mse(model, values: np.ndarray, start: int, end: int, mean: np.ndarray, std: np.ndarray,
                    horizons=HORIZONS, seq_len: int = SEQ_LEN, batch: int = 256) -> dict[str, float]:
    """Raw-unit MSE per horizon (mean over the first H rollout steps of every window) on rows [start, end).

    Windows follow Time-Series-Library: a window needs seq_len rows of context plus H target rows inside the
    partition, so smaller horizons have more windows. One rollout of the longest horizon serves all of them.
    """
    import torch

    part = (values[start:end] - mean) / std                      # standardized, (rows, channels)
    rows, channels = part.shape
    h_max = max(horizons)
    n_windows = rows - seq_len - min(horizons) + 1
    errors = np.full((n_windows, h_max, channels), np.nan, dtype=np.float64)   # raw-unit squared error
    model.eval()
    with torch.no_grad():
        for lo in range(0, n_windows, batch):
            idx = np.arange(lo, min(lo + batch, n_windows))
            x = torch.as_tensor(np.stack([part[i:i + seq_len] for i in idx]), dtype=torch.float32)  # (B, L, C)
            x = x.permute(0, 2, 1).reshape(-1, seq_len, 1)       # every channel is its own sample (1D model)
            preds = []
            for _ in range(h_max):
                step = model(x)                                   # (B*C, 1, 1)
                preds.append(step)
                x = torch.cat([x[:, 1:], step], dim=1)
            pred = torch.cat(preds, dim=1).reshape(len(idx), channels, h_max).permute(0, 2, 1).numpy()
            for j, i in enumerate(idx):
                avail = min(h_max, rows - seq_len - i)
                target = part[i + seq_len:i + seq_len + avail]
                errors[i, :avail] = ((pred[j, :avail] - target) * std) ** 2
    out: dict[str, float] = {}
    for h in horizons:
        count = rows - seq_len - h + 1
        out[f"mse_raw_h{h}"] = float(np.mean(errors[:count, :h]))
    out["mse_raw_mean"] = float(np.mean([out[f"mse_raw_h{h}"] for h in horizons]))
    return out


def train_arm(values: np.ndarray, stats: dict[str, Any], arm: str, seed: int, epochs: int = 10, lr: float = 1e-3,
              batch_size: int = 64, device: str = "cpu") -> dict[str, float]:
    import torch

    sys.path.insert(0, os.getcwd())
    from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = np.asarray(stats["train_mean"]), np.asarray(stats["train_std"])
    t0, t1 = stats["borders"]["train"]
    standardized = ((values[t0:t1] - mean) / std).astype(np.float32)           # (rows, channels)
    rows, channels = standardized.shape
    n_windows = rows - SEQ_LEN                                                 # one-step targets
    weights = torch.as_tensor(channel_weights(std, arm), dtype=torch.float32, device=device)

    model = ResidualLinearAutoregressiveForecaster({"model": {
        "input_sequence_length": SEQ_LEN, "prediction_horizon": 1, "input_size": 1, "ridge_alpha": RIDGE}}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.as_tensor(standardized, device=device)
    v0, v1 = stats["borders"]["validation"]
    val = torch.as_tensor(((values[v0:v1] - mean) / std).astype(np.float32), device=device)
    val_starts = np.arange(0, len(val) - SEQ_LEN, VAL_STRIDE)

    def validation_loss() -> float:
        """One-step loss on the validation partition with the same per-channel weights as training."""
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for lo in range(0, len(val_starts), 256):
                idx = torch.as_tensor(val_starts[lo:lo + 256], device=device)
                x = val[idx[:, None] + torch.arange(SEQ_LEN, device=device)[None, :]]        # (B, L, C)
                y = val[idx + SEQ_LEN]                                                          # (B, C)
                pred = model(x.permute(0, 2, 1).reshape(-1, SEQ_LEN, 1)).reshape(len(idx), channels)
                total += float((weights[None, :] * (pred - y) ** 2).sum())
                count += pred.numel()
        return total / count

    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(n_windows * channels)                          # (window, channel) pairs
        for lo in range(0, len(order), batch_size):
            pick = order[lo:lo + batch_size]
            w_idx, c_idx = pick // channels, pick % channels
            gather = torch.as_tensor(w_idx[:, None] + np.arange(SEQ_LEN)[None, :], device=device)
            c = torch.as_tensor(c_idx, device=device)
            x = data[gather, c[:, None]].unsqueeze(-1)                         # (B, L, 1)
            y = data[torch.as_tensor(w_idx + SEQ_LEN, device=device), c].reshape(-1, 1, 1)
            loss = (weights[c] * (model(x) - y).squeeze() ** 2).mean() + model.ridge_penalty()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        current = validation_loss()                                            # keep the best epoch, as the pipeline
        if current < best_loss:
            best_loss, best_state = current, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return rollout_raw_mse(model, values, *stats["borders"]["test"], mean, std)


def job(spec: dict[str, Any]) -> dict[str, Any]:
    os.chdir(spec["repo_root"])
    sys.path.insert(0, spec["repo_root"])
    import pandas as pd

    from training.ray.electrosense import ensure_dataset
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    started = time.time()
    record = {"dataset": spec["dataset"], "arm": spec["arm"], "seed": spec["seed"], "epochs": spec["epochs"],
              "kind": "loss_units"}
    try:
        path, stats = ensure_dataset(spec["dataset"], spec["repo_root"], spec["bucket"], client)
        values = pd.read_csv(path).drop(columns=["timestamp_utc"]).to_numpy(dtype=np.float64)
        record.update(train_arm(values, stats, spec["arm"], spec["seed"], epochs=spec["epochs"]))
        record["ok"] = True
    except Exception as exc:  # recorded, not raised, so one failure doesn't hide the other runs
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    record["wall_seconds"] = round(time.time() - started, 1)
    key = f"{spec['output_prefix']}/{spec['dataset']}/{spec['arm']}/seed-{spec['seed']}.json"
    client.put_object(Bucket=spec["bucket"], Key=key, Body=json.dumps(record).encode())
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["spec4", "spec25", "spec151"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--arms", nargs="+", default=["standardized", "raw"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-prefix", default="electrosense-results/loss-units")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args(argv)
    import ray

    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client  # noqa: F401  (fail early if storage is unavailable)

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address="auto")
    specs = [{"dataset": d, "arm": a, "seed": s, "epochs": args.epochs, "repo_root": args.repo_root,
              "bucket": bucket, "output_prefix": args.output_prefix}
             for d in args.datasets for s in args.seeds for a in args.arms]
    print(f"[loss-units] {len(specs)} runs", flush=True)
    remote = ray.remote(num_cpus=2)(job)
    pending = [remote.remote(s) for s in specs]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        print(f"[loss-units] {r['dataset']} {r['arm']} seed {r['seed']} ok={r['ok']} "
              f"mean={r.get('mse_raw_mean', r.get('error'))} ({r['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
