"""The TSS-LCD authors' first commit (d2d4f29) on the authors' own data: savgol and eval-time averaging as switches.

The collaborator's reproduction (``training/TSS-LCD/repro/``) reproduced the first commit's result with both smoothing steps on
(MSE 0.164-0.195 over three seeds, their overlapping split). This runner answers what it comes to with each step off, using the
same protocol, which ``training/SpectrumLLM``-style harness files in this repo document (``run_firstcommit_eval.py``) and the
authors' first-commit ``train.py`` defines:

* data: the authors' recovered series (``datasets/tsslcd/trainDataset.csv`` + ``testDataset.csv``, 6838 rows x 750 columns =
  3 AERPAW nodes x 250 bins, dBm, 1 minute), min-max scaled on the whole series (as they do);
* windows: context 50, future 10, over the whole series; train windows ``range(int(N*0.8))``, test windows
  ``range(int(N*0.555), N)`` (the two ranges overlap: 55% of the test windows are also training windows);
* model and training: the authors' model (``models/tsslcd_authors``; identical to d2d4f29), batch 256, lr 1e-4, 300 + 300 + 2000
  epochs, latent 32, 1000 diffusion steps, 10% context masking (also at test time, as their dataset class does);
* scoring: first future step only, inverse min-max to dBm.

Switches (everything else identical)
    ``--savgol on|off``   on = the authors' centred Savitzky-Golay(51, 2) over the whole series before windowing; off = none.
    eval-time averaging   the authors' w6/w3: a centred moving average (window 6) on the predictions and (window 3) on the ground
                          truth, ``np.convolve(..., mode="same")``. Always scored both ways.

Every model is scored in a 2 x 2 grid: averaging on/off x ground truth = the series the model was trained on ("model target", the
authors' protocol: smoothed when savgol is on) or the raw unfiltered measurement ("raw target"). Persistence is reported on the
raw series.

    python -m training.ray.tsslcd_firstcommit score --arms paper_s0 paper_s1 paper_s2   # existing collaborator checkpoints, CPU
    python -m training.ray.tsslcd_firstcommit train --savgol off on --seeds 0 1 2       # retrain with the switches, GPU
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

C, FH, D, LATENT, STEPS = 50, 10, 750, 32, 1000
BATCH, LR, MASK = 256, 1e-4, 0.1
EPOCHS = {"ctx": 300, "fut": 300, "diff": 2000}
TRAIN_FRACTION, TEST_START = 0.8, 0.555
DATA = "datasets/tsslcd"
CKPT = "checkpoints/repro/tsslcd_firstcommit"
MASK_SEED = 1234                       # the collaborator's evaluation seed for the test-time masking


# ------------------------------------------------------------------------------------------------------ data
def load_source(client, bucket: str) -> np.ndarray:
    """(6838, 750) dBm: both recovered files, each read the authors' way (row 0 is consumed as a header)."""
    parts = []
    for name in ("trainDataset.csv", "testDataset.csv"):
        body = client.get_object(Bucket=bucket, Key=f"{DATA}/{name}")["Body"].read()
        parts.append(pd.read_csv(io.BytesIO(body)).values.astype(np.float32)[:, :D])
    return np.concatenate(parts)


def prepare(source: np.ndarray, savgol: str):
    """(scaler, norm, y): min-max over the whole series (the authors' choice), then the optional centred savgol filter."""
    from training.ray.tsslcd_authors import MinMax, smooth_series

    scaler = MinMax().fit(source)
    norm = scaler.transform(source).astype(np.float32)
    y = smooth_series(norm, "centered" if savgol == "on" else "off").astype(np.float32)
    return scaler, norm, y


def window_ranges(rows: int) -> tuple[range, range]:
    n = rows - C - FH
    return range(int(n * TRAIN_FRACTION)), range(int(n * TEST_START), n)


def authors_average(matrix: np.ndarray, window: int) -> np.ndarray:
    """Their moving average: ``np.convolve(..., mode='same')`` per column (zero-padded at both ends, centred)."""
    if window == 1:
        return matrix
    kernel = np.ones(window) / window
    out = matrix.copy()
    for column in range(out.shape[1]):
        out[:, column] = np.convolve(out[:, column], kernel, mode="same")
    return out


def metrics(pred_dbm: np.ndarray, truth_dbm: np.ndarray) -> dict[str, float]:
    err = pred_dbm - truth_dbm
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}


def score_grid(pred: np.ndarray, gt_model: np.ndarray, gt_raw: np.ndarray, scaler, persistence_inputs=None) -> dict[str, Any]:
    """Scores for (n_test, 750) first-step arrays in normalized units.

    model_target: ground truth is the series the model was trained on; ``averaging`` applies the authors' w3 to it.
    raw_target: ground truth is the raw measurement, never averaged; only the predictions are averaged (w6).
    """
    inv = scaler.inverse
    out: dict[str, Any] = {}
    out["authors_protocol__model_target__averaging_on"] = metrics(inv(authors_average(pred, 6)), inv(authors_average(gt_model, 3)))
    out["model_target__averaging_off"] = metrics(inv(pred), inv(gt_model))
    out["raw_target__averaging_on"] = metrics(inv(authors_average(pred, 6)), inv(gt_raw))
    out["raw_target__averaging_off"] = metrics(inv(pred), inv(gt_raw))
    if persistence_inputs is not None:
        last, nxt = persistence_inputs
        out["persistence__raw_target"] = metrics(inv(last), inv(nxt))
    return out


# ------------------------------------------------------------------------------------------- model and runs
def build_models(device):
    from models.tsslcd_authors.Context2CondNew import ContextTransformerAE
    from models.tsslcd_authors.F2Cnet import FutureAutoencoder
    from models.tsslcd_authors.NoiseNet import DiffusionModel

    ae_ctx = ContextTransformerAE(seq_len=C, input_dim=D, latent_dim=LATENT).to(device)
    ae_fut = FutureAutoencoder(FH, D, LATENT).to(device)
    diffusion = DiffusionModel(cond_dim=LATENT, latent_dim=LATENT, n_timestep=STEPS, device=device).to(device)
    return ae_ctx, ae_fut, diffusion


def predict_first_step(models, y: np.ndarray, test_idx: range, device, batch: int = BATCH) -> np.ndarray:
    """First future step for every test window (10% context masking at test time, as the authors' dataset class does)."""
    import torch
    from models.tsslcd_authors.NoiseNet import inference

    ae_ctx, ae_fut, diffusion = models
    for m in models:
        m.eval()
    torch.manual_seed(MASK_SEED)
    series = torch.as_tensor(y, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for lo in range(0, len(test_idx), batch):
            idx = torch.as_tensor(list(test_idx[lo:lo + batch]))
            ctx = series[idx[:, None] + torch.arange(C)]
            ctx = ctx.masked_fill(torch.rand_like(ctx) < MASK, 0.0)
            preds.append(inference(ae_ctx, ae_fut, diffusion, ctx.to(device), device)[:, 0, :])
    return np.concatenate(preds)


def evaluate(models, source, savgol: str, device) -> dict[str, Any]:
    scaler, norm, y = prepare(source, savgol)
    _, test_idx = window_ranges(len(source))
    starts = np.fromiter(test_idx, dtype=np.int64)
    pred = predict_first_step(models, y, test_idx, device)
    gt_model, gt_raw = y[starts + C], norm[starts + C]
    persistence = (norm[starts + C - 1], norm[starts + C])
    return {"n_test": len(starts), "grid": score_grid(pred, gt_model, gt_raw, scaler, persistence)}


def train_models(source, savgol: str, seed: int, epochs: dict[str, int], device, log) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, y = prepare(source, savgol)
    train_idx, _ = window_ranges(len(source))
    series = torch.as_tensor(y, dtype=torch.float32, device=device)
    starts_all = torch.as_tensor(list(train_idx), device=device)
    ctx_off, fut_off = torch.arange(C, device=device), torch.arange(C, C + FH, device=device)
    ae_ctx, ae_fut, diffusion = build_models(device)
    timing: dict[str, Any] = {"train_windows": len(train_idx)}

    def run_stage(name, model, step_fn, n_epochs):
        optimiser = torch.optim.Adam(model.parameters(), lr=LR)
        t0, last = time.time(), float("nan")
        for epoch in range(n_epochs):
            order = starts_all[torch.randperm(len(starts_all), device=device)]
            total, count = 0.0, 0
            for lo in range(0, len(order), BATCH):
                idx = order[lo:lo + BATCH]
                ctx, fut = series[idx[:, None] + ctx_off], series[idx[:, None] + fut_off]
                ctx = ctx.masked_fill(torch.rand_like(ctx) < MASK, 0.0)
                loss = step_fn(ctx, fut)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total, count = total + float(loss), count + 1
            last = total / max(count, 1)
            if (epoch + 1) % max(1, n_epochs // 4) == 0:
                log(f"{name} epoch {epoch + 1}/{n_epochs} loss {last:.5f}")
        timing[name] = {"epochs": n_epochs, "seconds": round(time.time() - t0, 1), "final_loss": last}

    ae_ctx.train()
    run_stage("ctx", ae_ctx, lambda ctx, fut: F.mse_loss(ae_ctx(ctx)[0], ctx), epochs["ctx"])
    ae_fut.train()
    run_stage("fut", ae_fut, lambda ctx, fut: F.mse_loss(ae_fut(fut)[0], fut), epochs["fut"])
    ae_ctx.eval()
    ae_fut.eval()

    def diffusion_loss(ctx, fut):
        with torch.no_grad():
            z_ctx, z_fut = ae_ctx(ctx)[1], ae_fut(fut)[1]
        t = torch.randint(0, STEPS, (z_fut.size(0),), device=device)
        noise = torch.randn_like(z_fut)
        return F.mse_loss(diffusion(diffusion.q_sample(z_fut, t, noise), z_ctx, t), noise)

    diffusion.train()
    run_stage("diff", diffusion, diffusion_loss, epochs["diff"])
    return (ae_ctx, ae_fut, diffusion), timing


def job(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), spec["bucket"]
    device = torch.device("cuda" if torch.cuda.is_available() and spec["device"] != "cpu" else "cpu")
    torch.set_num_threads(int(spec.get("threads", 8)))
    name = spec["name"]
    record: dict[str, Any] = {"kind": f"tsslcd_firstcommit_{spec['mode']}", "name": name, "device": str(device)}
    started = time.time()
    try:
        source = load_source(client, bucket)
        record["rows"] = len(source)
        if spec["mode"] == "score":
            savgol = "on"                                     # the collaborator's arms were trained on the filtered series
            models = build_models(device)
            with tempfile.TemporaryDirectory(prefix="tsslcd-ckpt-") as scratch:
                for part, module in zip(("ae_ctx", "ae_fut", "diffusion"), models):
                    path = Path(scratch) / f"{part}.pth"
                    client.download_file(bucket, f"{CKPT}/{spec['arm']}/{part}.pth", str(path))
                    module.load_state_dict(torch.load(path, map_location=device))
            record.update(arm=spec["arm"], savgol=savgol)
        else:
            savgol, seed = spec["savgol"], int(spec["seed"])
            record.update(savgol=savgol, seed=seed, epochs=spec["epochs"])
            models, timing = train_models(source, savgol, seed, spec["epochs"], device, lambda m: print(f"[{name}] {m}", flush=True))
            record["timing"] = timing
        t0 = time.time()
        record.update(evaluate(models, source, savgol, device))
        record["inference_seconds"] = round(time.time() - t0, 1)
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=f"{spec['output_prefix']}/{spec['mode']}/{name}.json",
                          Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("score", help="score the collaborator's first-commit checkpoints (CPU)")
    s.add_argument("--arms", nargs="+", default=["paper_s0", "paper_s1", "paper_s2"])
    t = sub.add_parser("train", help="retrain with the switches (GPU)")
    t.add_argument("--savgol", nargs="+", default=["off", "on"], choices=["on", "off"])
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    t.add_argument("--epochs-ctx", type=int, default=EPOCHS["ctx"])
    t.add_argument("--epochs-fut", type=int, default=EPOCHS["fut"])
    t.add_argument("--epochs-diff", type=int, default=EPOCHS["diff"])
    for q in (s, t):
        q.add_argument("--cpus", type=int, default=8)
        q.add_argument("--gpu-fraction", type=float, default=None, help="default 0.5 for train, 0 for score")
        q.add_argument("--device", default=None)
        q.add_argument("--bucket", default=None)
        q.add_argument("--output-prefix", default="tsslcd-firstcommit-results")
        q.add_argument("--address", default="auto")
        q.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    base = {"mode": args.mode, "repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
            "device": args.device or ("cpu" if args.mode == "score" else "cuda"), "threads": args.cpus}
    if args.mode == "score":
        specs = [{**base, "name": arm, "arm": arm} for arm in args.arms]
        gpus = 0 if args.gpu_fraction is None else args.gpu_fraction
    else:
        specs = [{**base, "name": f"savgol-{sv}-seed-{seed}", "savgol": sv, "seed": seed,
                  "epochs": {"ctx": args.epochs_ctx, "fut": args.epochs_fut, "diff": args.epochs_diff}}
                 for sv in args.savgol for seed in args.seeds]
        gpus = 0.5 if args.gpu_fraction is None else args.gpu_fraction
    print(f"[tsslcd-firstcommit] {len(specs)} {args.mode} runs", flush=True)
    remote = ray.remote(num_cpus=args.cpus, num_gpus=gpus, max_retries=1)(job)
    pending = [remote.remote(sp) for sp in specs]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        g = r.get("grid", {})
        print(f"[tsslcd-firstcommit] {r['name']} ok={r.get('ok')} "
              + " ".join(f"{k.split('__')[0][:12]}/{k.split('__')[-1]}={v['mse']:.4f}" for k, v in g.items() if 'persistence' not in k)
              + (f" error={r.get('error')}" if not r.get("ok") else "") + f" ({r.get('wall_seconds')}s)", flush=True)


if __name__ == "__main__":
    main()
