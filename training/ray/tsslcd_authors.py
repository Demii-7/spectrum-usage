"""The TSS-LCD authors' released model on POWDER, with the two smoothing settings as explicit switches.

The authors' released pipeline has two undocumented smoothing steps that carry its headline result (see
``training/TSS-LCD/MODEL_CARD.md``): a centred Savitzky-Golay filter (window 51, order 2) applied to the whole series
before windowing, and a centred moving average applied to the predictions and the ground truth at evaluation time (their
test script hard-codes both windows to 1 = off; their reported numbers come from larger windows). Neither can be switched
off in their code. Here both are settings, and everything else follows the benchmark protocol:

* data: 4 core receivers x 200 bins (600-800 MHz) = 800 columns, receiver-major, aligned on common minutes; short gaps
  (<= 9 min) interpolated; train / validation / test ranges as ``training/configs/oct2026/ray_tuning_2d.yaml`` with the
  final test range; windows never cross a gap or a partition boundary;
* the authors' min-max scaler, **fit on the training partition only** (their code fits it on whichever file it loads);
* the authors' model (``models/tsslcd_authors``), their three training stages, losses, learning rate and sampler; lookback 60
  and one 60-step diffusion sample per window (as for the other benchmark models: t+1, t+15, t+60 are slices of it);
* the authors' 10% zero-masking of the context during training; **off at test time** (their dataset also masks the test
  inputs) unless ``--eval-mask-ratio`` is set.

Settings
    ``--savgol``        off | centered (authors', window 51 order 2) | causal (trailing window, no future samples)
    ``--eval-smooth``   ``PRED:GT:MODE`` (repeatable), e.g. ``1:1:centered`` = off, ``6:3:centered`` = a large centred
                        window on predictions (6) and ground truth (3), ``6:3:causal`` = trailing windows.

The savgol setting changes the data the model is trained on, so each value is its own training run. Eval-time smoothing only
post-processes predictions, so one trained model is scored under every ``--eval-smooth`` setting. Every setting is scored
against two ground truths: the series the model was trained to predict (``model_target``: smoothed when savgol is on, and
with the eval-time GT smoothing applied; the authors' protocol) and the raw unfiltered measurement (``raw_target``: the
only one that is comparable across settings).

    python -m training.ray.tsslcd_authors --savgol off centered --seeds 40 41 42 43 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SITES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
RUNS = {"train": "20261015T1434Z", "validation": "20261025T1355Z", "test": "20261025T1355Z"}
RANGES = {"train": ("2026-10-15T14:34:00Z", "2026-10-22T13:13:00Z"),
          "validation": ("2026-10-25T13:55:00Z", "2026-10-27T13:55:00Z"),
          "test": ("2026-10-27T20:00:00Z", "2026-11-03T20:00:00Z")}
RAW_NAME = "power_1mhz_avg_per_minute.csv"
CONTEXT, FUTURE, HORIZONS = 60, 60, (1, 15, 60)
MAX_GAP = 9
SAVGOL_WINDOW, SAVGOL_ORDER = 51, 2
LATENT_DIM, N_TIMESTEPS, BATCH, LEARNING_RATE = 32, 1000, 256, 1e-4        # the authors' config.py defaults
EPOCHS = {"ctx": 300, "fut": 300, "diff": 2000}                              # the authors' config.py defaults
TRAIN_MASK = 0.1
DEFAULT_EVAL_SMOOTH = ("1:1:centered", "6:3:centered", "6:3:causal")


# ----------------------------------------------------------------------------------------------------------- data
def load_partition(root: str, partition: str, band: str = "600_800", sites=SITES, max_gap: int = MAX_GAP):
    """(values (T, sites*bins) dBm, minute index, valid mask). Receivers joined on common minutes; short gaps filled."""
    start, end = (pd.Timestamp(t) for t in RANGES[partition])
    frames = []
    for site in sites:
        raw = pd.read_csv(Path(root) / "data" / "powder" / site / RUNS[partition] / band / RAW_NAME)
        stamps = pd.to_datetime(raw.pop("timestamp_utc"), utc=True).dt.floor("min")
        raw.index = stamps
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        raw.columns = [f"{site}:{c}" for c in raw.columns]
        frames.append(raw)
    bins = {f.shape[1] for f in frames}
    if len(bins) != 1:
        raise ValueError(f"receivers differ in bin count: {sorted(bins)}")
    grid = pd.date_range(start, end, freq="min")
    joined = pd.concat([f.reindex(grid) for f in frames], axis=1)
    both = joined.interpolate(method="linear", limit=max_gap, limit_area="inside")
    valid = np.isfinite(both.to_numpy()).all(axis=1)
    return both.to_numpy(dtype=np.float64), grid, valid


def contiguous_segments(valid: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """[start, end) runs of valid rows that are at least ``minimum`` long."""
    padded = np.concatenate([[False], valid, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]


class MinMax:
    """The authors' per-column min-max scaler to [0, 1] (``MinMaxScalerCustom``), here fit on training rows only."""

    def fit(self, data: np.ndarray) -> "MinMax":
        self.min = data.min(axis=0)
        self.scale = data.max(axis=0) - self.min
        self.scale[self.scale == 0] = 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / self.scale

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale + self.min


def smooth_series(x: np.ndarray, kind: str, window: int = SAVGOL_WINDOW, order: int = SAVGOL_ORDER) -> np.ndarray:
    """Savitzky-Golay filter along axis 0 of one contiguous segment.

    off      : unchanged.
    centered : the authors' filter (``savgol_filter(..., mode='interp')``); every output uses samples up to window//2 ahead.
    causal   : the same polynomial fit over the trailing window only (the filter evaluated at its last sample), so no
               output uses a later sample. The first window-1 samples have no full window and are left unfiltered.
    """
    from scipy.signal import savgol_coeffs, savgol_filter

    if kind == "off":
        return x
    if len(x) < window:
        raise ValueError(f"segment of {len(x)} rows is shorter than the filter window {window}")
    if kind == "centered":
        return savgol_filter(x, window_length=window, polyorder=order, axis=0, mode="interp")
    if kind == "causal":
        coeffs = savgol_coeffs(window, order, pos=window - 1, use="dot")
        out = x.copy()
        windows = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)       # (T-w+1, D, w), oldest..newest
        out[window - 1:] = windows @ coeffs
        return out
    raise ValueError(f"unknown savgol setting {kind!r}")


def window_starts(segments: list[tuple[int, int]], length: int, stride: int = 1) -> np.ndarray:
    """Start rows of every window of ``length`` rows that lies inside one segment."""
    starts = [np.arange(a, b - length + 1, stride) for a, b in segments]
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def build_partition(root: str, partition: str, scaler: MinMax | None, savgol: str, band: str = "600_800"):
    """Series for one partition: min-max normalized raw (``raw``) and filtered (``model``), concatenated over valid
    segments, plus the segment boundaries in that concatenated coordinate. The scaler is fit here on the training rows."""
    values, _, valid = load_partition(root, partition, band)
    segments = contiguous_segments(valid, CONTEXT + FUTURE)
    if not segments:
        raise ValueError(f"no gap-free run of {CONTEXT + FUTURE} minutes in the {partition} partition")
    if scaler is None:
        scaler = MinMax().fit(np.concatenate([values[a:b] for a, b in segments]))
    raw_parts, model_parts, bounds, offset = [], [], [], 0
    for a, b in segments:
        normalized = scaler.transform(values[a:b])
        raw_parts.append(normalized)
        model_parts.append(smooth_series(normalized, savgol))
        bounds.append((offset, offset + b - a))
        offset += b - a
    return np.concatenate(raw_parts), np.concatenate(model_parts), bounds, scaler


# ------------------------------------------------------------------------------------------------ eval smoothing
def parse_eval_smooth(spec: str) -> tuple[int, int, str]:
    pred, gt, mode = spec.split(":")
    if mode not in ("centered", "causal"):
        raise ValueError(f"eval-smooth mode must be centered or causal, got {mode!r}")
    if int(pred) < 1 or int(gt) < 1:
        raise ValueError("eval-smooth windows must be >= 1")
    return int(pred), int(gt), mode


def moving_average(x: np.ndarray, window: int, mode: str) -> np.ndarray:
    """Moving average along axis 0 (consecutive windows of one segment). window=1 is the identity. ``centered`` uses
    samples on both sides (hindsight); ``causal`` only the current and earlier ones. Edges average what is available."""
    if window == 1:
        return x
    frame = pd.DataFrame(x.reshape(len(x), -1))                     # any trailing shape, e.g. (horizons, bins)
    return frame.rolling(window, center=(mode == "centered"), min_periods=1).mean().to_numpy().reshape(x.shape)


def smooth_windows(series: np.ndarray, segment_of_window: np.ndarray, window: int, mode: str) -> np.ndarray:
    """Apply ``moving_average`` separately inside each segment (series is (N windows, ...) in origin order)."""
    if window == 1:
        return series
    out = series.copy()
    for seg in np.unique(segment_of_window):
        idx = np.flatnonzero(segment_of_window == seg)
        out[idx] = moving_average(series[idx], window, mode)
    return out


def score(pred_h: np.ndarray, gt_h: np.ndarray) -> dict[str, float]:
    """MAE and MSE in dB per horizon for (N, len(HORIZONS), D) arrays, plus the mean over horizons."""
    out: dict[str, float] = {}
    for j, h in enumerate(HORIZONS):
        err = pred_h[:, j] - gt_h[:, j]
        out[f"mae_db_t{h}"] = float(np.abs(err).mean())
        out[f"mse_db2_t{h}"] = float((err ** 2).mean())
    out["mean_horizon_mae_db"] = float(np.mean([out[f"mae_db_t{h}"] for h in HORIZONS]))
    out["mean_horizon_mse_db2"] = float(np.mean([out[f"mse_db2_t{h}"] for h in HORIZONS]))
    return out


def score_grid(pred_h: np.ndarray, gt_model_h: np.ndarray, gt_raw_h: np.ndarray, segment_of_window: np.ndarray,
               eval_smooth: list[tuple[int, int, str]], consecutive: bool) -> dict[str, float]:
    """Metrics for every eval-smoothing setting against both ground truths (all arrays in dB, (N, horizons, D)).

    ``model_target``: the series the model was trained on, with the setting's GT smoothing applied (authors' protocol).
    ``raw_target``: the raw measurement, never smoothed; only the predictions are smoothed.
    Smoothing needs consecutive windows (test stride 1); otherwise only the identity setting is scored.
    """
    out: dict[str, float] = {}
    for pred_w, gt_w, mode in eval_smooth:
        name = f"p{pred_w}g{gt_w}{mode[:3]}" if (pred_w, gt_w) != (1, 1) else "off"
        if name != "off" and not consecutive:
            continue
        if name in {k.split("__")[1] for k in out}:
            continue
        pred = smooth_windows(pred_h, segment_of_window, pred_w, mode)
        gt_model = smooth_windows(gt_model_h, segment_of_window, gt_w, mode)
        for target, gt in (("model_target", gt_model), ("raw_target", gt_raw_h)):
            for key, value in score(pred, gt).items():
                out[f"{target}__{name}__{key}"] = value
    return out


# ------------------------------------------------------------------------------------------------------ training
def run_one(args: dict[str, Any]) -> dict[str, Any]:
    """Train the authors' model with one savgol setting and score it under every eval-smoothing setting."""
    import torch
    import torch.nn.functional as F

    root = args["repo_root"]
    sys.path.insert(0, root)
    from models.tsslcd_authors.Context2CondNew import ContextTransformerAE
    from models.tsslcd_authors.F2Cnet import FutureAutoencoder
    from models.tsslcd_authors.NoiseNet import DiffusionModel, inference

    device = torch.device("cuda" if torch.cuda.is_available() and args.get("device") != "cpu" else "cpu")
    seed, savgol = int(args["seed"]), args["savgol"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    result: dict[str, Any] = {"savgol": savgol, "seed": seed, "epochs": args["epochs"], "device": str(device),
                              "context": CONTEXT, "future": FUTURE, "latent_dim": LATENT_DIM, "n_timesteps": N_TIMESTEPS,
                              "batch_size": BATCH, "train_mask_ratio": TRAIN_MASK,
                              "eval_mask_ratio": args["eval_mask_ratio"], "sites": list(SITES)}

    train_raw, train_model, train_bounds, scaler = build_partition(root, "train", None, savgol)
    test_raw, test_model, test_bounds, _ = build_partition(root, "test", scaler, savgol)
    n_cols = train_model.shape[1]
    train_starts = window_starts(train_bounds, CONTEXT + FUTURE)
    test_starts = window_starts(test_bounds, CONTEXT + FUTURE, args["test_stride"])
    result.update(columns=n_cols, train_windows=len(train_starts), test_windows=len(test_starts),
                  test_stride=args["test_stride"])

    train_series = torch.as_tensor(train_model, dtype=torch.float32, device=device)
    ctx_offsets = torch.arange(CONTEXT, device=device)
    fut_offsets = torch.arange(CONTEXT, CONTEXT + FUTURE, device=device)

    def batch(series, starts_t):
        return series[starts_t[:, None] + ctx_offsets], series[starts_t[:, None] + fut_offsets]

    def masked(ctx, ratio):
        return ctx.masked_fill(torch.rand_like(ctx) < ratio, 0.0) if ratio > 0 else ctx

    ae_ctx = ContextTransformerAE(seq_len=CONTEXT, input_dim=n_cols, latent_dim=LATENT_DIM,
                                  n_locations=len(SITES)).to(device)
    ae_fut = FutureAutoencoder(FUTURE, n_cols, LATENT_DIM).to(device)
    diffusion = DiffusionModel(cond_dim=LATENT_DIM, latent_dim=LATENT_DIM, n_timestep=N_TIMESTEPS, device=device).to(device)
    result["parameters"] = {"ae_ctx": sum(p.numel() for p in ae_ctx.parameters()),
                            "ae_fut": sum(p.numel() for p in ae_fut.parameters()),
                            "diffusion": sum(p.numel() for p in diffusion.parameters())}
    starts_all = torch.as_tensor(train_starts, device=device)

    def epochs_of(stage, model, step_fn, epochs, lr):
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        t0, last = time.time(), float("nan")
        for _ in range(epochs):
            order = starts_all[torch.randperm(len(starts_all), device=device)]
            total, count = 0.0, 0
            for lo in range(0, len(order), BATCH):
                ctx, fut = batch(train_series, order[lo:lo + BATCH])
                loss = step_fn(masked(ctx, TRAIN_MASK), fut)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total, count = total + float(loss), count + 1
            last = total / max(count, 1)
        result[f"stage_{stage}"] = {"epochs": epochs, "seconds": round(time.time() - t0, 1),
                                    "seconds_per_epoch": round((time.time() - t0) / max(epochs, 1), 2), "final_loss": last}

    ae_ctx.train()
    epochs_of("ctx", ae_ctx, lambda ctx, fut: F.mse_loss(ae_ctx(ctx)[0], ctx), args["epochs"]["ctx"], LEARNING_RATE)
    ae_fut.train()
    epochs_of("fut", ae_fut, lambda ctx, fut: F.mse_loss(ae_fut(fut)[0], fut), args["epochs"]["fut"], LEARNING_RATE)
    ae_ctx.eval()
    ae_fut.eval()

    def diffusion_loss(ctx, fut):
        with torch.no_grad():
            z_ctx, z_fut = ae_ctx(ctx)[1], ae_fut(fut)[1]
        t = torch.randint(0, N_TIMESTEPS, (z_fut.size(0),), device=device)
        noise = torch.randn_like(z_fut)
        return F.mse_loss(diffusion(diffusion.q_sample(z_fut, t, noise), z_ctx, t), noise)

    diffusion.train()
    epochs_of("diff", diffusion, diffusion_loss, args["epochs"]["diff"], LEARNING_RATE)
    diffusion.eval()

    import tempfile

    from training.ray.artifacts import save_artifact
    from training.ray.storage import spectrum_s3_client

    with tempfile.TemporaryDirectory(prefix="tsslcd-authors-save-") as scratch:
        for part, module in (("ae_ctx", ae_ctx), ("ae_fut", ae_fut), ("diffusion", diffusion)):
            path = Path(scratch) / f"{part}.pth"
            torch.save(module.state_dict(), path)
            saved = save_artifact(spectrum_s3_client(), args["bucket"],
                                  f"{args['output_prefix'].strip('/')}/weights/savgol-{savgol}/seed-{seed}", path,
                                  {"savgol": savgol, "seed": seed, "epochs": args["epochs"], "columns": n_cols,
                                   "timing": {k: v for k, v in result.items() if k.startswith("stage_")}}, repo_root=root)
            result.setdefault("artifact_keys", []).append(saved.get("weights_key"))
            if "artifact_error" in saved:
                result["artifact_error"] = saved["artifact_error"]

    # inference: one diffusion sample per test window; keep the three scored horizons
    torch.manual_seed(seed + 1000)
    test_series = torch.as_tensor(test_model, dtype=torch.float32, device=device)
    t0, preds = time.time(), []
    with torch.no_grad():
        for lo in range(0, len(test_starts), BATCH):
            idx = torch.as_tensor(test_starts[lo:lo + BATCH], device=device)
            ctx, _ = batch(test_series, idx)
            ctx = masked(ctx, args["eval_mask_ratio"])
            preds.append(inference(ae_ctx, ae_fut, diffusion, ctx, device)[:, [h - 1 for h in HORIZONS]])
    result["inference_seconds"] = round(time.time() - t0, 1)
    pred_h = scaler.inverse(np.concatenate(preds))                                    # (N, horizons, D) dB

    horizon_rows = test_starts[:, None] + CONTEXT + np.array([h - 1 for h in HORIZONS])[None, :]
    gt_model_h = scaler.inverse(test_model[horizon_rows])
    gt_raw_h = scaler.inverse(test_raw[horizon_rows])
    segment_of_window = np.searchsorted([b for _, b in test_bounds], test_starts, side="right")
    result["metrics"] = score_grid(pred_h, gt_model_h, gt_raw_h, segment_of_window, args["eval_smooth"],
                                   consecutive=args["test_stride"] == 1)
    result["wall_seconds"] = round(time.time() - started, 1)
    result["ok"] = True
    return result


def job(spec: dict[str, Any]) -> dict[str, Any]:
    os.chdir(spec["repo_root"])
    sys.path.insert(0, spec["repo_root"])
    from training.ray.storage import spectrum_s3_client

    record: dict[str, Any] = {"savgol": spec["savgol"], "seed": spec["seed"], "kind": "tsslcd_authors"}
    try:
        record = run_one(spec)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    key = f"{spec['output_prefix'].strip('/')}/savgol-{spec['savgol']}/seed-{spec['seed']}.json"
    try:
        spectrum_s3_client().put_object(Bucket=spec["bucket"], Key=key, Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--savgol", nargs="+", default=["off", "centered"], choices=["off", "centered", "causal"])
    p.add_argument("--eval-smooth", action="append", default=None, metavar="PRED:GT:MODE",
                   help=f"repeatable; default {list(DEFAULT_EVAL_SMOOTH)}")
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    p.add_argument("--epochs-ctx", type=int, default=EPOCHS["ctx"])
    p.add_argument("--epochs-fut", type=int, default=EPOCHS["fut"])
    p.add_argument("--epochs-diff", type=int, default=EPOCHS["diff"])
    p.add_argument("--test-stride", type=int, default=1, help="1 for real results; larger only for timing runs")
    p.add_argument("--eval-mask-ratio", type=float, default=0.0)
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="tsslcd-authors-results")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import ray

    args = parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    eval_smooth = [parse_eval_smooth(s) for s in (args.eval_smooth or DEFAULT_EVAL_SMOOTH)]
    ray.init(address=args.address, ignore_reinit_error=True)
    client = spectrum_s3_client()
    specs = []
    for savgol in args.savgol:
        for seed in args.seeds:
            key = f"{args.output_prefix.strip('/')}/savgol-{savgol}/seed-{seed}.json"
            try:
                if json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"):
                    continue
            except Exception:
                pass
            specs.append({"savgol": savgol, "seed": seed, "repo_root": args.repo_root, "bucket": bucket,
                          "output_prefix": args.output_prefix, "device": args.device,
                          "epochs": {"ctx": args.epochs_ctx, "fut": args.epochs_fut, "diff": args.epochs_diff},
                          "test_stride": args.test_stride, "eval_mask_ratio": args.eval_mask_ratio,
                          "eval_smooth": eval_smooth})
    print(f"[tsslcd-authors] {len(specs)} runs", flush=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpu_fraction, max_retries=1)(job)
    pending = [remote.remote(s) for s in specs]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        shown = {k: round(v, 3) for k, v in r.get("metrics", {}).items() if k.startswith("raw_target__off__mean_horizon_mae")}
        print(f"[tsslcd-authors] savgol={r.get('savgol')} seed={r.get('seed')} ok={r.get('ok')} {shown or r.get('error')} "
              f"({r.get('wall_seconds')}s)", flush=True)


if __name__ == "__main__":
    main()
