"""Zero-shot MOMENT forecasting as masked reconstruction (no training, no fitted head), on the 15-minute POWDER and external targets.

MOMENT (Goswami et al. 2024) is pretrained by masked reconstruction of 512-step series split into patches of 8 steps; it has no forecasting head.
This module forecasts with the PRETRAINED reconstruction head: the context is placed at the start of the 512-step input, the next
``m`` steps are marked as unobserved, and the model reconstructs them; the forecast at horizon h is the reconstruction at step ``context + h - 1``.

Exact construction (``build_masked_input``), per window and per channel (every one of the 200 frequency bins is an independent univariate series;
MOMENT folds channels into the batch):
  * positions ``0 .. context-1``                 the observed context, raw dB values (MOMENT normalizes each window itself: its RevIN uses the
                                                  observed values only and de-normalizes the output, so no source statistics are involved);
  * positions ``context .. context+m-1``         the masked future: value 0, ``mask = 0`` (the patch embedding replaces masked patches by its learned
                                                  mask token), ``input_mask = 1`` (they take part in attention, like masked patches in pretraining);
  * positions ``context+m .. 511``               padding: value 0, ``mask = 0`` and ``input_mask = 0`` (excluded from attention and from the statistics).
``m`` is always a multiple of the patch length 8 (a patch counts as observed only if all its 8 steps are, so masking whole patches is the only
clean choice); the requested step is read from the reconstruction. Nothing is fitted; the weights are the published MOMENT-1 weights.

Design choices that are free (chosen on the VALIDATION range only and frozen in ``evaluation/analysis/moment_zeroshot_design.json`` before the
test week is scored): how many future steps are masked and whether the horizons are read from one pass or from several:
  single     one pass with ``future_patches`` masked patches (default 6 = 48 steps); every horizon is read from that pass;
  per_horizon one pass per horizon with exactly ceil(h / 8) masked patches;
  chunked    autoregressive: mask ``chunk_patches`` patches, append the reconstruction to the context as if observed, repeat until 48 steps.
The context is 96 steps (24 h) like every other model; ``--context`` runs a longer context as a separately labelled result.

Evaluation reuses the frozen-transfer machinery (``training.ray.transfer15`` / ``transfer_all``): the same 625 origins per receiver and band, the same
horizons 1/4/12/24/48, the same scoring (five metrics, per-origin errors). Outputs (S3): ``<prefix>/A/<platform>/<site>/<run>/<band>/2d/moment_zeroshot_<size>/
seed-0/{summary.json, per_origin.csv}`` (deterministic; the seed folder is a placeholder so that the transfer_all layout is kept).

    python -m training.ray.moment_zeroshot validate            # design choice on the validation range (writes the validation table)
    python -m training.ray.moment_zeroshot run --skip-done     # the October test week (core receivers, held-out receivers, external targets)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEQ_LEN = 512
PATCH = 8
LOOKBACK = 96
HORIZONS = (1, 4, 12, 24, 48)
MAX_HORIZON = max(HORIZONS)
DESIGN_FILE = "evaluation/analysis/moment_zeroshot_design.json"
PREFIX = "moment-zeroshot-oct2026"
VALIDATION = ("2026-10-25T14:00:00Z", "2026-10-27T13:55:00Z")      # the range every other model is tuned on
CORE_RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")
CANDIDATES = [
    {"name": "single6", "mode": "single", "future_patches": 6},
    {"name": "per_horizon", "mode": "per_horizon"},
    {"name": "chunk1", "mode": "chunked", "chunk_patches": 1},
    {"name": "chunk3", "mode": "chunked", "chunk_patches": 3},
]


# ------------------------------------------------------------------------------------------------ pure helpers
def build_masked_input(context: np.ndarray, masked_steps: int, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Context ``(B, C, L)`` -> MOMENT inputs ``x (B, C, seq_len)``, ``input_mask (B, seq_len)``, ``mask (B, seq_len)`` (see the module docstring)."""
    batch, channels, length = context.shape
    if masked_steps % PATCH or masked_steps <= 0:
        raise ValueError(f"masked_steps must be a positive multiple of {PATCH}, got {masked_steps}")
    if length + masked_steps > seq_len:
        raise ValueError(f"context {length} + masked {masked_steps} exceeds the model's {seq_len} steps")
    x = np.zeros((batch, channels, seq_len), dtype=np.float32)
    x[:, :, :length] = context
    input_mask = np.zeros((batch, seq_len), dtype=np.float32)
    input_mask[:, :length + masked_steps] = 1.0
    mask = np.zeros((batch, seq_len), dtype=np.float32)
    mask[:, :length] = 1.0
    return x, input_mask, mask


def patches_for(horizon: int) -> int:
    return int(math.ceil(horizon / PATCH))


def read_out(reconstruction: np.ndarray, length: int, horizons=HORIZONS) -> dict[int, np.ndarray]:
    """Reconstruction ``(B, C, seq_len)`` -> {horizon: (B, C)} = the value at step ``length + h - 1``."""
    return {h: reconstruction[:, :, length + h - 1] for h in horizons}


# ------------------------------------------------------------------------------------------------ the forecaster
class MomentZeroShot:
    """Frozen MOMENT-1 with its pretrained reconstruction head; ``predict`` maps windows ``(N, L, F)`` in dB to {horizon: (N, F)} in dB."""

    def __init__(self, size: str = "small", device: str | None = None):
        import torch
        from training.ray.llm_powder import use_isolated_libs

        use_isolated_libs()
        from momentfm import MOMENTPipeline

        self.torch = torch
        self.size = size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = MOMENTPipeline.from_pretrained(f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "reconstruction"})
        model.init()                                                    # no-op for the reconstruction task: the pretrained head is kept
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _reconstruct(self, context: np.ndarray, masked_steps: int) -> np.ndarray:
        x, input_mask, mask = build_masked_input(context, masked_steps)
        torch = self.torch
        with torch.no_grad():
            out = self.model(x_enc=torch.from_numpy(x).to(self.device), input_mask=torch.from_numpy(input_mask).to(self.device),
                             mask=torch.from_numpy(mask).to(self.device))
        return out.reconstruction.float().cpu().numpy()

    def predict(self, windows: np.ndarray, design: dict[str, Any], batch_windows: int = 4, horizons=HORIZONS) -> dict[int, np.ndarray]:
        """``windows`` (N, L, F) -> {h: (N, F)}."""
        n = len(windows)
        out = {h: np.empty((n, windows.shape[2]), dtype=np.float32) for h in horizons}
        for lo in range(0, n, batch_windows):
            context = np.ascontiguousarray(windows[lo:lo + batch_windows].transpose(0, 2, 1), dtype=np.float32)       # (B, F, L)
            length = context.shape[2]
            mode = design["mode"]
            if mode == "single":
                recon = self._reconstruct(context, int(design.get("future_patches", 6)) * PATCH)
                got = read_out(recon, length, horizons)
            elif mode == "per_horizon":
                got = {}
                for h in horizons:
                    got[h] = read_out(self._reconstruct(context, patches_for(h) * PATCH), length, (h,))[h]
            elif mode == "chunked":
                chunk = int(design["chunk_patches"]) * PATCH
                grown, produced = context, 0
                while produced < MAX_HORIZON:
                    recon = self._reconstruct(grown, chunk)
                    step = recon[:, :, grown.shape[2]:grown.shape[2] + chunk]
                    grown = np.concatenate([grown, step], axis=2)
                    produced += chunk
                got = read_out(np.concatenate([grown, np.zeros((*grown.shape[:2], SEQ_LEN - grown.shape[2]), dtype=np.float32)], axis=2), length, horizons)
            else:
                raise ValueError(f"unknown design mode {mode!r}")
            for h in horizons:
                out[h][lo:lo + len(context)] = got[h]
        return out


# ------------------------------------------------------------------------------------------------ scoring on a target
def score_target(forecaster: MomentZeroShot, task, file, design: dict[str, Any], lookback: int = LOOKBACK, horizons=HORIZONS) -> dict[str, Any]:
    """The frozen-transfer origins and scoring of ``transfer15.evaluate_frozen`` for a forecaster that needs no checkpoint."""
    from training.common import transfer_2d
    from training.common.evaluation_integrated import calculate_errors_and_export_arrays
    from training.common.windowing import make_window_batch_array
    from training.ray import transfer15 as t15

    source = transfer_2d.load_target(task, [file], impute=False, max_missing_gap=0)
    starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=lookback, max_horizon=MAX_HORIZON, stride=1,
                                              target_start=task.target_start, target_end=task.target_end)
    if not len(starts):
        raise ValueError("no origins")
    windows = make_window_batch_array(full_x=np.asarray(source.data, dtype=np.float32), start_rows=starts, lookback=lookback)
    t0 = time.time()
    predicted = forecaster.predict(windows, design, horizons=horizons)
    predictions, targets, bins, times = {}, {}, {}, {}
    for h in horizons:
        rows = starts + lookback + h - 1
        predictions[h], targets[h], _, _ = calculate_errors_and_export_arrays(
            prediction_normalized=predicted[h], target_raw_model_layout=np.asarray(source.data, dtype=np.float32)[rows], normalization=None)
        bins[h], times[h] = np.empty(0), t15.origin_times(source, rows, h)
    scored = t15.score_all(predictions, targets, bins, times)
    scored["n_origins"] = int(len(starts))
    scored["forecast_seconds"] = round(time.time() - t0, 1)
    scored["last_value_mae"] = {h: float(np.abs(np.asarray(source.data, dtype=np.float32)[starts + lookback - 1] -
                                                np.asarray(source.data, dtype=np.float32)[starts + lookback + h - 1]).mean()) for h in horizons}
    return scored


def validation_task(band: str, receiver: str):
    """The 2-day validation range as a transfer task on a core receiver (history before the first origin is the file's own start)."""
    import pandas as pd

    from training.ray import transfer15 as t15

    task = t15.target_task(band, receiver)
    start, end = (pd.Timestamp(t) for t in VALIDATION)
    import dataclasses

    return dataclasses.replace(task, task_id=f"validation-{band}-{receiver}", raw_start=start, raw_end=end, target_start=start, target_end=end)


# ------------------------------------------------------------------------------------------------ jobs
def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one target (a core receiver, held-out receiver or external site), the frozen design, or every candidate on the validation range."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    t = spec["target"]
    record: dict[str, Any] = {"target": t, "size": args["size"], "kind": spec["kind"]}
    started = time.time()
    try:
        forecaster = MomentZeroShot(args["size"])
        if spec["kind"] == "validation":
            from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

            ensure_powder15(t["band"], root, SPATIAL_NODES)
            from training.ray import transfer15 as t15

            task, file = validation_task(t["band"], t["site"]), t15.target_file(root, t["band"], t["site"])
            record["designs"] = {}
            for design in CANDIDATES:
                scored = score_target(forecaster, task, file, design)
                record["designs"][design["name"]] = {"n_origins": scored["n_origins"], "forecast_seconds": scored["forecast_seconds"],
                                                     "metrics": scored["metrics"], "last_value_mae": scored["last_value_mae"]}
            record["ok"] = True
            client.put_object(Bucket=bucket, Key=f"{args['prefix']}/validation/{t['band']}/{t['site']}.json", Body=json.dumps(record, default=str).encode())
        else:
            from training.ray import transfer_all as TA

            design = json.loads(Path(root, DESIGN_FILE).read_text())["design"]
            if args.get("context") and int(args["context"]) != LOOKBACK:
                design = {**design, "context": int(args["context"])}
            task, file = TA.prepare_target(root, t)
            scored = score_target(forecaster, task, file, design, lookback=int(design.get("context", LOOKBACK)))
            label = f"moment_zeroshot_{args['size']}" + (f"_ctx{design['context']}" if design.get("context", LOOKBACK) != LOOKBACK else "")
            summary = {"kind": "moment_zeroshot", "model": label, "design": design, "target": TA.target_key(t), "platform": t["platform"], "site": t["site"],
                       "run": t["run"], "target_band": t["band"], "n_origins": scored["n_origins"], "forecast_seconds": scored["forecast_seconds"],
                       "last_value_mae": scored["last_value_mae"], "code_revision": args["code_revision"], "metrics": scored["metrics"]}
            base = TA.output_base(args["prefix"], "A", t, "2d", label, 0, None)
            client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(summary, default=str, indent=1).encode())
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.csv", Body=scored["per_origin"].to_csv(index=False).encode())
            record["mae"] = {h: scored["metrics"][str(h)]["mae_db"] for h in HORIZONS}
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("validate", "run"))
    p.add_argument("--size", default="small", choices=("small", "base", "large"))
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    p.add_argument("--groups", nargs="+", default=["powder", "cosmos", "aerpaw", "ara"], help="powder = guesthouse and law73; the core receivers are always run")
    p.add_argument("--no-core", action="store_true")
    p.add_argument("--context", type=int, default=None, help="longer context (separately labelled result); default 96")
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--gpu-fraction", type=float, default=1.0, help="1.0 on 24 GB cards (fractional shares ran out of memory there); a small share is fine on an 80 GB H100")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    from training.ray import transfer_all as TA
    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    try:
        revision = subprocess.run(["git", "-C", a.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        revision = "unknown"
    from training.ray.powder15 import TEST_RUN

    if a.command == "validate":
        jobs = [{"kind": "validation", "target": {"platform": "powder", "site": r, "run": TEST_RUN, "band": b}} for b in a.bands for r in CORE_RECEIVERS]
    else:
        tgts = [] if a.no_core else [{"platform": "powder", "site": r, "run": TEST_RUN, "band": b, "n_origins": TA.POWDER_ORIGINS}
                                     for b in a.bands for r in CORE_RECEIVERS]
        tgts += TA.targets(a.repo_root, a.groups, a.bands)
        jobs = [{"kind": "test", "target": t} for t in tgts]
        label = f"moment_zeroshot_{a.size}" + (f"_ctx{a.context}" if a.context and a.context != LOOKBACK else "")
        if a.skip_done:
            def done(j):
                key = TA.output_base(a.prefix, "A", j["target"], "2d", label, 0, None)
                try:
                    client.head_object(Bucket=bucket, Key=f"{key}/summary.json")
                    return True
                except Exception:
                    return False
            jobs = [j for j in jobs if not done(j)]
    print(f"[moment-zeroshot] {a.command}: {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "size": a.size, "code_revision": revision, "context": a.context}
    ray.init(address=a.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, num_gpus=a.gpu_fraction, memory=24 * 2 ** 30, max_retries=1)(job)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            pending.append(remote.remote(queue.pop(0), shared))
        done_refs, pending = ray.wait(pending, num_returns=1)
        row = ray.get(done_refs[0])
        ok += bool(row.get("ok"))
        print(f"[moment-zeroshot] {row['target'].get('platform')}/{row['target'].get('site')}/{row['target'].get('band')} ok={row.get('ok')} "
              f"{row.get('error', '')} {row.get('mae', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[moment-zeroshot] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
