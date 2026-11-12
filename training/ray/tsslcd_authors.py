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

Eval-only mode (``--eval-only --weights-prefix <source output prefix> --test-start T0 --test-end T1 --output-prefix <new prefix>``): no training;
the three saved networks of each (savgol, seed) run of the source prefix (``<prefix>/weights/savgol-X/seed-N/{ae_ctx,ae_fut,diffusion}.pth``,
hyperparameters from the source record) are loaded and scored on another test range with the same data pipeline (scaler fit on the same
training partition), sampler (``manual_seed(seed + 1000)``) and scoring code, under a new output prefix.
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
# Resolution profiles. "1min" is the original setting. "15min": lookback 100 (the model needs a multiple of its 5 attention heads; 96 is not), future 48,
# horizons 1/4/12/24/48) on the converted ``data/powder15`` traces; the test partition is loaded ``context - 1`` blocks early so the
# forecast origins are exactly the benchmark's (first origin = first block of the test week); everything else (model, stages, epochs,
# masking, savgol switch) is unchanged.
PROFILES = {
    "1min": {"root": "data/powder", "raw": RAW_NAME, "freq": "min", "context": CONTEXT, "future": FUTURE, "horizons": HORIZONS, "pad_test": False},
    "15min": {"root": "data/powder15", "raw": "power_15min.csv", "freq": "15min", "context": 100, "future": 48,
              "horizons": (1, 4, 12, 24, 48), "pad_test": True},
}


# ----------------------------------------------------------------------------------------------------------- data
def load_partition(root: str, partition: str, band: str = "600_800", sites=SITES, max_gap: int = MAX_GAP, profile: str = "1min"):
    """(values (T, sites*bins) dBm, minute index, valid mask). Receivers joined on common minutes; short gaps filled."""
    prof = PROFILES[profile]
    start, end = (pd.Timestamp(t) for t in RANGES[partition])
    start, end = start.ceil(prof["freq"]), end.floor(prof["freq"])          # block-aligned (a no-op at 1 minute)
    if prof["pad_test"] and partition == "test":
        start = start - (prof["context"] - 1) * pd.Timedelta(prof["freq"])
    frames = []
    for site in sites:
        raw = pd.read_csv(Path(root) / prof["root"] / site / RUNS[partition] / band / prof["raw"])
        stamps = pd.to_datetime(raw.pop("timestamp_utc"), utc=True).dt.floor(prof["freq"])
        raw.index = stamps
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        raw.columns = [f"{site}:{c}" for c in raw.columns]
        frames.append(raw)
    bins = {f.shape[1] for f in frames}
    if len(bins) != 1:
        raise ValueError(f"receivers differ in bin count: {sorted(bins)}")
    grid = pd.date_range(start, end, freq=prof["freq"])
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


def apply_input_offset(values: np.ndarray, offset_db: float) -> np.ndarray:
    """``values`` (dBm) plus a constant offset in dB; the identity (the same array) for 0."""
    return values if not offset_db else np.asarray(values, dtype=np.float64) + float(offset_db)


def build_partition(root: str, partition: str, scaler: MinMax | None, savgol: str, band: str = "600_800", profile: str = "1min",
                    input_offset_db: float = 0.0):
    """Series for one partition: min-max normalized raw (``raw``) and filtered (``model``), concatenated over valid
    segments, plus the segment boundaries in that concatenated coordinate. The scaler is fit here on the training rows.

    ``input_offset_db`` (default 0 = unchanged) adds a constant to every raw value (dBm) BEFORE the scaler, for the offset-response test of a
    frozen model; the scaler must then be the one fit on the unshifted training partition (pass it in), never fit on shifted data."""
    prof = PROFILES[profile]
    values, _, valid = load_partition(root, partition, band, profile=profile)
    values = apply_input_offset(values, input_offset_db)
    segments = contiguous_segments(valid, prof["context"] + prof["future"])
    if not segments:
        raise ValueError(f"no gap-free run of {prof['context'] + prof['future']} steps in the {partition} partition")
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


def score(pred_h: np.ndarray, gt_h: np.ndarray, horizons=HORIZONS) -> dict[str, float]:
    """MAE and MSE in dB per horizon for (N, len(HORIZONS), D) arrays, plus the mean over horizons."""
    out: dict[str, float] = {}
    for j, h in enumerate(horizons):
        err = pred_h[:, j] - gt_h[:, j]
        out[f"mae_db_t{h}"] = float(np.abs(err).mean())
        out[f"mse_db2_t{h}"] = float((err ** 2).mean())
    out["mean_horizon_mae_db"] = float(np.mean([out[f"mae_db_t{h}"] for h in horizons]))
    out["mean_horizon_mse_db2"] = float(np.mean([out[f"mse_db2_t{h}"] for h in horizons]))
    return out


def score_grid(pred_h: np.ndarray, gt_model_h: np.ndarray, gt_raw_h: np.ndarray, segment_of_window: np.ndarray,
               eval_smooth: list[tuple[int, int, str]], consecutive: bool, horizons=HORIZONS) -> dict[str, float]:
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
            for key, value in score(pred, gt, horizons).items():
                out[f"{target}__{name}__{key}"] = value
    return out


# ------------------------------------------------------------------------------------------------------ training
def offset_sweep(args, build, scaler, savgol, band, profile, partition, test_starts, batch, masked, nets, device, seed, base_pred, horizons, context):
    """Offset response of a frozen model: add c dB to every raw input value (all four receivers, before the scaler), sample the forecasts with the
    SAME sampler seed and origins as the unshifted run, and return {receiver: rows} (rows as ``training.ray.offset_response.response_rows``;
    shift = forecast(c) - forecast(0) in dB per (window, bin), gain = shift / c) for the horizons 4/12/24/48."""
    import torch

    from training.ray.offset_response import response_rows

    ae_ctx, ae_fut, diffusion, inference = nets
    columns = base_pred.shape[2]
    n_receivers = len(SITES)
    width = columns // n_receivers
    names = ["humanities", "madsen", "moran", "sage point"]
    out = {name: [] for name in names}
    root = args["repo_root"]
    for offset in args["input_offsets"]:
        _, shifted_model, _, _ = build(root, partition, scaler, savgol, band=band, profile=profile, input_offset_db=float(offset))
        series = torch.as_tensor(shifted_model, dtype=torch.float32, device=device)
        torch.manual_seed(seed + 1000)
        preds = []
        with torch.no_grad():
            for lo in range(0, len(test_starts), BATCH):
                idx = torch.as_tensor(test_starts[lo:lo + BATCH], device=device)
                ctx, _ = batch(series, idx)
                preds.append(inference(ae_ctx, ae_fut, diffusion, masked(ctx, args["eval_mask_ratio"]), device)[:, [h - 1 for h in horizons]])
        shifted_pred = scaler.inverse(np.concatenate(preds))
        for r, name in enumerate(names):
            cols = slice(r * width, (r + 1) * width)
            out[name].extend(response_rows({h: base_pred[:, horizons.index(h), cols] for h in (4, 12, 24, 48)},
                                           {h: shifted_pred[:, horizons.index(h), cols] for h in (4, 12, 24, 48)}, float(offset), (4, 12, 24, 48)))
    return out


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
    profile = args.get("resolution", "1min")
    band = args.get("band", "600_800")
    CONTEXT, FUTURE, HORIZONS = (PROFILES[profile][k] for k in ("context", "future", "horizons"))      # shadow the 1-minute module constants
    hp = {"latent_dim": LATENT_DIM, "learning_rate": LEARNING_RATE, "batch_size": BATCH, "train_mask_ratio": TRAIN_MASK,
          **(args.get("hparams") or {})}          # the authors' defaults unless a tuning trial overrides them (constructor / optimiser settings only)
    latent_dim, learning_rate, batch_size, train_mask = (int(hp["latent_dim"]), float(hp["learning_rate"]), int(hp["batch_size"]),
                                                         float(hp["train_mask_ratio"]))
    score_partition = args.get("score_partition", "test")      # "validation" for tuning trials
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    result: dict[str, Any] = {"savgol": savgol, "seed": seed, "epochs": args["epochs"], "device": str(device),
                              "context": CONTEXT, "future": FUTURE, "latent_dim": latent_dim, "n_timesteps": N_TIMESTEPS,
                              "batch_size": batch_size, "train_mask_ratio": train_mask, "learning_rate": learning_rate,
                              "score_partition": score_partition, "resolution": profile, "band": band, "horizons": list(HORIZONS),
                              "eval_mask_ratio": args["eval_mask_ratio"], "sites": list(SITES)}

    if profile != "1min":
        from training.ray.powder15 import ensure_powder15
        ensure_powder15(band, root, tuple(SITES))
    train_raw, train_model, train_bounds, scaler = build_partition(root, "train", None, savgol, band=band, profile=profile)
    test_raw, test_model, test_bounds, _ = build_partition(root, score_partition, scaler, savgol, band=band, profile=profile)
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

    ae_ctx = ContextTransformerAE(seq_len=CONTEXT, input_dim=n_cols, latent_dim=latent_dim,
                                  n_locations=len(SITES)).to(device)
    ae_fut = FutureAutoencoder(FUTURE, n_cols, latent_dim).to(device)
    diffusion = DiffusionModel(cond_dim=latent_dim, latent_dim=latent_dim, n_timestep=N_TIMESTEPS, device=device).to(device)
    result["parameters"] = {"ae_ctx": sum(p.numel() for p in ae_ctx.parameters()),
                            "ae_fut": sum(p.numel() for p in ae_fut.parameters()),
                            "diffusion": sum(p.numel() for p in diffusion.parameters())}
    if args.get("eval_only"):
        import tempfile as _tf

        from training.ray.storage import spectrum_s3_client as _client
        wp = f"{args['weights_prefix'].strip('/')}/weights/savgol-{savgol}/seed-{seed}"
        with _tf.TemporaryDirectory(prefix="tsslcd-eval-only-") as scratch:
            for part, module in (("ae_ctx", ae_ctx), ("ae_fut", ae_fut), ("diffusion", diffusion)):
                local = Path(scratch) / f"{part}.pth"
                _client().download_file(args["bucket"], f"{wp}/{part}.pth", str(local))
                module.load_state_dict(torch.load(local, map_location=device))
        result.update(eval_only=True, weights_prefix=wp)
        ae_ctx.eval()
        ae_fut.eval()
        diffusion.eval()
    else:
        starts_all = torch.as_tensor(train_starts, device=device)

        def epochs_of(stage, model, step_fn, epochs, lr):
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            t0, last = time.time(), float("nan")
            for _ in range(epochs):
                order = starts_all[torch.randperm(len(starts_all), device=device)]
                total, count = 0.0, 0
                for lo in range(0, len(order), batch_size):
                    ctx, fut = batch(train_series, order[lo:lo + batch_size])
                    loss = step_fn(masked(ctx, train_mask), fut)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total, count = total + float(loss), count + 1
                last = total / max(count, 1)
            result[f"stage_{stage}"] = {"epochs": epochs, "seconds": round(time.time() - t0, 1),
                                        "seconds_per_epoch": round((time.time() - t0) / max(epochs, 1), 2), "final_loss": last}

        ae_ctx.train()
        epochs_of("ctx", ae_ctx, lambda ctx, fut: F.mse_loss(ae_ctx(ctx)[0], ctx), args["epochs"]["ctx"], learning_rate)
        ae_fut.train()
        epochs_of("fut", ae_fut, lambda ctx, fut: F.mse_loss(ae_fut(fut)[0], fut), args["epochs"]["fut"], learning_rate)
        ae_ctx.eval()
        ae_fut.eval()

        def diffusion_loss(ctx, fut):
            with torch.no_grad():
                z_ctx, z_fut = ae_ctx(ctx)[1], ae_fut(fut)[1]
            t = torch.randint(0, N_TIMESTEPS, (z_fut.size(0),), device=device)
            noise = torch.randn_like(z_fut)
            return F.mse_loss(diffusion(diffusion.q_sample(z_fut, t, noise), z_ctx, t), noise)

        diffusion.train()
        epochs_of("diff", diffusion, diffusion_loss, args["epochs"]["diff"], learning_rate)
        diffusion.eval()

        import tempfile

        from training.ray.artifacts import save_artifact
        from training.ray.storage import spectrum_s3_client

        with tempfile.TemporaryDirectory(prefix="tsslcd-authors-save-") as scratch:
            for part, module in ((("ae_ctx", ae_ctx), ("ae_fut", ae_fut), ("diffusion", diffusion)) if args.get("save_weights", True) else ()):
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
                                   consecutive=args["test_stride"] == 1, horizons=HORIZONS)
    if args.get("source_metrics"):               # eval-only: the same data and sampler must give the source run's stored scores back
        shared = [k for k in result["metrics"] if k in args["source_metrics"]]
        result["zero_offset_max_abs_diff_db"] = float(max(abs(result["metrics"][k] - args["source_metrics"][k]) for k in shared)) if shared else None
    if args.get("input_offsets"):
        result["offset_rows"] = offset_sweep(args, build_partition, scaler, savgol, band, profile, score_partition, test_starts, batch, masked,
                                             (ae_ctx, ae_fut, diffusion, inference), device, seed, pred_h, HORIZONS, CONTEXT)
    result["hparams"] = {"latent_dim": latent_dim, "learning_rate": learning_rate, "batch_size": batch_size,
                         "train_mask_ratio": train_mask}
    result["objective"] = result["metrics"].get("raw_target__off__mean_horizon_mae_db")    # tuning objective: raw-target MAE, no smoothing
    result["wall_seconds"] = round(time.time() - started, 1)
    result["ok"] = True
    return result


def job(spec: dict[str, Any]) -> dict[str, Any]:
    os.chdir(spec["repo_root"])
    sys.path.insert(0, spec["repo_root"])
    from training.ray.storage import spectrum_s3_client

    record: dict[str, Any] = {"savgol": spec["savgol"], "seed": spec["seed"], "kind": "tsslcd_authors"}
    part = spec.get("score_partition", "test")
    original_range = RANGES[part]
    try:
        if spec.get("test_range"):                       # eval-only on another week: the same loader, a different range
            RANGES[part] = tuple(spec["test_range"])
        record = run_one(spec)
        if spec.get("test_range"):
            record["test_range"] = list(spec["test_range"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    finally:
        RANGES[part] = original_range
    offset_rows = record.pop("offset_rows", None)       # per-receiver offset summaries go to their own keys, not into the run record
    if offset_rows:
        for receiver, rows in offset_rows.items():
            summary = {"kind": "offset_response", "band": record.get("band"), "receiver": receiver, "rep": "4d", "model": "tsslcd", "seed": spec["seed"],
                       "intrinsic_window_normalization": "none (min-max scaler fit on the training week; frozen)",
                       "offsets_db": list(spec["input_offsets"]), "horizons": [4, 12, 24, 48], "origin_stride": spec["test_stride"], "rows": rows,
                       "n_origins": record.get("test_windows"), "weights_prefix": spec.get("weights_prefix"), "savgol": spec["savgol"],
                       "zero_offset_max_abs_diff_db": record.get("zero_offset_max_abs_diff_db"), "ok": bool(record.get("ok"))}
            try:
                spectrum_s3_client().put_object(
                    Bucket=spec["bucket"], Key=f"{spec['offset_prefix'].strip('/')}/tsslcd/{record.get('band')}/{receiver}/seed-{spec['seed']}/summary.json",
                    Body=json.dumps(summary, default=str).encode())
            except Exception:
                pass
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
    p.add_argument("--band", default="600_800", choices=("600_800", "2400_2600"), help="the 4 core receivers, 200 one-MHz bins of this band")
    p.add_argument("--resolution", default="1min", choices=sorted(PROFILES), help="15min: benchmark windowing on data/powder15")
    p.add_argument("--hparams", default=None, help="JSON of constructor/optimiser overrides (latent_dim, learning_rate, batch_size, train_mask_ratio)")
    p.add_argument("--test-stride", type=int, default=1, help="1 for real results; larger only for timing runs")
    p.add_argument("--eval-mask-ratio", type=float, default=0.0)
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--eval-only", action="store_true", help="no training: score the saved weights of --weights-prefix (see the module docstring)")
    p.add_argument("--weights-prefix", default=None, help="eval-only: output prefix of the run whose saved weights and record (hyperparameters) are used")
    p.add_argument("--test-start", default=None, help="eval-only: start of the scored range (UTC), default the original test range")
    p.add_argument("--test-end", default=None, help="eval-only: end of the scored range (UTC)")
    p.add_argument("--input-offsets", nargs="+", type=float, default=None,
                   help="eval-only: offset-response test: add each constant (dB) to every raw input value before the scaler and record the forecast shift "
                        "per receiver (needs --offset-prefix); default off")
    p.add_argument("--offset-prefix", default=None,
                   help="with --input-offsets: S3 prefix of the per-receiver offset summaries (<prefix>/tsslcd/<band>/<receiver>/seed-N/summary.json); "
                        "must differ from --weights-prefix; existing ok summaries are kept (skip-done)")
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
    if args.eval_only:
        if not args.weights_prefix or args.weights_prefix.strip("/") == args.output_prefix.strip("/"):
            raise SystemExit("--eval-only needs --weights-prefix and a different --output-prefix (October results are never overwritten)")
        if bool(args.test_start) != bool(args.test_end):
            raise SystemExit("--test-start and --test-end go together")
        if args.input_offsets and (not args.offset_prefix or args.offset_prefix.strip("/") in (args.weights_prefix.strip("/"), args.output_prefix.strip("/"))):
            raise SystemExit("--input-offsets needs --offset-prefix, different from --weights-prefix and --output-prefix")
    elif args.input_offsets:
        raise SystemExit("--input-offsets is an eval-only option")
    specs = []
    for savgol in args.savgol:
        for seed in args.seeds:
            key = f"{args.output_prefix.strip('/')}/savgol-{savgol}/seed-{seed}.json"
            try:
                if json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"):
                    continue
            except Exception:
                pass
            extra = {}
            if args.eval_only:
                try:
                    source = json.loads(client.get_object(Bucket=bucket, Key=f"{args.weights_prefix.strip('/')}/savgol-{savgol}/seed-{seed}.json")["Body"].read())
                except Exception as error:
                    print(f"[tsslcd-authors] no source record for savgol={savgol} seed={seed}: {error}", flush=True)
                    continue
                if not source.get("ok") or not source.get("artifact_keys"):
                    print(f"[tsslcd-authors] source run savgol={savgol} seed={seed} not ok or without saved weights, skipped", flush=True)
                    continue
                extra = {"eval_only": True, "weights_prefix": args.weights_prefix, "save_weights": False, "hparams": source.get("hparams"),
                         "resolution": source.get("resolution", args.resolution), "band": source.get("band", args.band)}
                if args.test_start:
                    extra["test_range"] = [args.test_start, args.test_end]
                if args.input_offsets:
                    extra.update(input_offsets=list(args.input_offsets), offset_prefix=args.offset_prefix, source_metrics=source.get("metrics"))
            specs.append({"savgol": savgol, "seed": seed, "repo_root": args.repo_root, "bucket": bucket,
                          "output_prefix": args.output_prefix, "device": args.device,
                          "epochs": {"ctx": args.epochs_ctx, "fut": args.epochs_fut, "diff": args.epochs_diff},
                          "test_stride": args.test_stride, "eval_mask_ratio": args.eval_mask_ratio, "resolution": args.resolution, "band": args.band,
                          "hparams": json.loads(args.hparams) if args.hparams else None,
                          "eval_smooth": eval_smooth, **extra})
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
