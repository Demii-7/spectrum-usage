"""Frozen-input frequency ablation: do the trained 15-minute models use the frequency bins outside the region they forecast? (evaluation only)

Same idea as ``training.ray.neighbor_ablation`` (neighbouring receivers), for frequency. The trained model is FROZEN, nothing is retrained or
fitted; only the test inputs change. For each non-noise region of the band's plan (13 at 600-800, 5 at 2400-2600) the input window of every
bin OUTSIDE the region is replaced by the same bin shifted in time

    real      the measured series (the normal frozen evaluation)
    shift96   24 hours (96 steps, wrapping): keeps each bin's level, statistics and daily phase, loses its co-movement with the region (the control)
    shift210  52.5 hours (210 steps, wrapping): also breaks the daily phase

while the region's own bins stay real, and ONLY the region's bins are scored against the real targets (horizons t+4/12/24/48), on the same forecast
origins in every mode. If the error on the region does not change the model is not using the other bins; if it rises it is. Caveat: a frozen
model fed inconsistent inputs can do worse for reasons other than lost information; the 24 h shift is the control for that, as in the spatial section.

For the rolled-out models the shift applies to the initial input window only (the autoregressive rollout then feeds the model's own outputs back as usual).
Windows whose history starts before row ``shift`` of the series contain wrapped rows (the first 96 / 210 origins); the analysis also reports the
delta without them. Models: the tuned 15-minute 2D rolled-out finals (``tuning/powder15``) and the direct finals (``direct-tuning/15min``: one checkpoint
per trained horizon); the four core receivers' October test week, stride 1.

Output per (model, band, seed[, horizon]): ``<prefix>/<model>/<band>/seed-N[/h<h>]/per_origin.npz`` (per receiver, region and mode the per-origin, per-horizon
mean absolute error over the region's bins; origin times and start rows; the real-mode per-bin errors over the origins whose lookback lies inside the test week, for the identity check) and ``summary.json``.

    python -m training.ray.frequency_ablation --models revinmlp residuallinearar2d vanillalstm --seeds 42 --skip-done --gpu-fraction 0.1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HORIZONS = (4, 12, 24, 48)
RECEIVERS = ("humanities", "madsen", "moran", "sagepoint")
MODES = (("real", 0), ("shift96", 96), ("shift210", 210))
ROLLED = ("linearar2d", "residuallinearar2d", "vanillalstm", "residualvanillalstm", "lstmattn", "temporalconvnet", "autoformer_csa")
DIRECT = ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "itransformer", "nlinear", "dlinear")
PLANS = {"600_800": "evaluation/analysis/plan_regions_600_800.csv", "2400_2600": "evaluation/analysis/plan_regions_2400_2600.csv"}
CONTROLS = ("revinmlp", "linearar2d", "residuallinearar2d")        # per-bin / separate-weights models: no cross-bin path, must be unaffected


def regions_of(root: str, band: str) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(Path(root) / PLANS[band])
    noise = frame["is_noise_floor"].astype(str).str.lower().isin({"true", "1"})
    return frame.loc[~noise].to_dict(orient="records")


def out_base(prefix: str, entry: dict[str, Any]) -> str:
    base = f"{prefix.strip('/')}/{entry['model']}/{entry['band']}/seed-{entry['seed']}"
    return base + (f"/h{entry['horizon']}" if entry.get("source") == "direct" else "")


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one checkpoint, four receivers, every region and mode."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry = spec["entry"]
    band, label, seed = entry["band"], entry["model"], int(entry["seed"])
    name = entry.get("pipeline", label)
    direct = entry.get("source") == "direct"
    horizons = [int(entry["horizon"])] if direct else list(HORIZONS)
    base = out_base(args["prefix"], entry)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record: dict[str, Any] = {"kind": "frequency_ablation", "band": band, "model": label, "seed": seed, "rep": "2d", "source": entry.get("source"),
                              "trained_horizon": horizons[0] if direct else None, "horizons": horizons, "checkpoint_key": entry["pt_key"],
                              "modes": [m[0] for m in MODES], "code_revision": args.get("code_revision")}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(max(1, int(args.get("threads", 2))))
        from training.common import transfer_2d as module
        from training.common.evaluation_integrated import calculate_errors_and_export_arrays, evaluation_parameters
        from training.common.forecasting import forecast
        from training.common.frequency_shift import band_frequencies, region_errors, region_mask, shift_other_bins
        from training.common.model_factory import build_model
        from training.common.windowing import make_window_batch_array
        from training.ray import transfer15 as t15
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        ensure_powder15(band, root, SPATIAL_NODES)
        meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
        config = meta["config"]
        config["windowing"] = {"lookback": t15.LOOKBACK, "horizons": horizons}
        config[name].setdefault("train", {})["test_stride"] = int(args.get("stride", 1))
        regions = regions_of(root, band)
        if args.get("max_regions"):
            regions = regions[:int(args["max_regions"])]
        receivers = list(args.get("receivers") or RECEIVERS)
        frequencies = band_frequencies(band)
        masks = {r["band_id"]: region_mask(frequencies, r["start_mhz"], r["end_mhz"]) for r in regions}
        with tempfile.TemporaryDirectory(prefix="freq-abl-") as scratch_dir:
            path = Path(scratch_dir) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(path))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            normalization = checkpoint["normalization"]
            parameters = evaluation_parameters(config, name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = None
            collected: dict[str, np.ndarray] = {}

            def predict(source, data, starts):
                """Frozen forecasts (dB) of ``data`` (the series the windows are cut from) at the shared origins, one array per horizon."""
                normalized = module.apply_source_normalization(data, normalization)
                nonlocal model
                if model is None:
                    model = build_model(name, config, normalized)
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    model.to(device).eval()
                    torch.manual_seed(seed)
                windows = make_window_batch_array(full_x=normalized, start_rows=starts, lookback=parameters["lookback"])
                parts = {h: [] for h in parameters["horizons"]}
                with torch.no_grad():
                    for offset in range(0, len(windows), parameters["batch_size"]):
                        values = torch.from_numpy(windows[offset:offset + parameters["batch_size"]]).float().to(device)
                        predicted = forecast(model=model, x=values, prediction_horizon=parameters["prediction_horizon"],
                                             rollout_horizon=parameters["max_horizon"], targets=None)
                        for h in parts:
                            parts[h].append(predicted[:, h - 1].cpu().numpy())
                out = {}
                for h, chunk in parts.items():
                    rows = starts + parameters["lookback"] + h - 1
                    pred_db, target_db, _, _ = calculate_errors_and_export_arrays(
                        prediction_normalized=np.concatenate(chunk), target_raw_model_layout=source.data[rows], normalization=dict(normalization))
                    out[h] = (np.asarray(pred_db, dtype=np.float64), np.asarray(target_db, dtype=np.float64))
                return out

            for receiver in receivers:
                task = t15.target_task(band, receiver)
                source = module.load_target(task, [t15.target_file(root, band, receiver)], impute=False, max_missing_gap=0)
                starts = module.target_window_starts(source.timestamps, source.segments, lookback=parameters["lookback"],
                                                     max_horizon=max(parameters["max_horizon"], 48), stride=parameters["test_stride"],
                                                     target_start=task.target_start, target_end=task.target_end)
                if not len(starts):
                    raise ValueError("no origins")
                data = np.asarray(source.data, dtype=np.float32)
                times = np.asarray((source.timestamps[starts + parameters["lookback"] - 1]).values, dtype="datetime64[ns]").astype(str)
                collected[f"times|{receiver}"] = times
                real = predict(source, data, starts)
                targets = {h: real[h][1] for h in real}
                collected[f"starts|{receiver}"] = np.asarray(starts, dtype=np.int64)
                main = np.asarray(starts) >= int(parameters["lookback"])        # the stored per-bin tables score windows whose whole lookback lies inside the test week
                collected[f"binmae|{receiver}"] = np.stack([np.abs(real[h][0][main] - real[h][1][main]).mean(axis=0) for h in horizons])      # (H, bins), stored-comparable origins
                for region_id, mask in masks.items():
                    collected[f"mae|{receiver}|{region_id}|real"] = np.stack(
                        [region_errors(real[h][0], targets[h], mask) for h in horizons], axis=1).astype(np.float32)
                    for mode, shift in MODES[1:]:
                        shifted = predict(source, shift_other_bins(data, mask, shift), starts)
                        collected[f"mae|{receiver}|{region_id}|{mode}"] = np.stack(
                            [region_errors(shifted[h][0], targets[h], mask) for h in horizons], axis=1).astype(np.float32)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, horizons=np.asarray(horizons), regions=np.asarray(list(masks)), receivers=np.asarray(receivers), **collected)
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.npz", Body=buffer.getvalue())
        summary: dict[str, Any] = {}
        for region_id in masks:
            row = {}
            for mode, _ in MODES:
                row[mode] = float(np.mean([collected[f"mae|{r}|{region_id}|{mode}"].mean() for r in receivers]))
            summary[region_id] = row
        record.update(n_origins=int(len(collected[f"times|{receivers[0]}"])), regions=list(masks), mean_region_mae=summary, ok=True)
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=list(PLANS))
    p.add_argument("--models", nargs="+", default=list(ROLLED[:2]) + list(DIRECT[:1]))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="trained horizons (direct source: one checkpoint each)")
    p.add_argument("--receivers", nargs="+", default=list(RECEIVERS), choices=list(RECEIVERS))
    p.add_argument("--max-regions", type=int, default=0, help="smoke tests only (use another --prefix)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--gpu-fraction", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="frequency-ablation-2026-11")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    try:
        revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        revision = "unknown"
    rolled = [m for m in args.models if m in ROLLED]
    direct = [m for m in args.models if m in DIRECT]
    entries = []
    if rolled:
        entries += [e for e in discover_sources(client, bucket, ["p15tuned"], rolled)
                    if e["band"] in args.bands and e["rep"] == "2d" and int(e["seed"]) in args.seeds]
    if direct:
        entries += [e for e in discover_sources(client, bucket, ["direct"], direct)
                    if e["resolution"] == "15min" and e["band"] in args.bands and int(e["seed"]) in args.seeds and int(e["horizon"]) in args.horizons]
    unknown = [m for m in args.models if m not in ROLLED and m not in DIRECT]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    jobs = [{"entry": e} for e in entries]
    if args.skip_done:
        def done(j):
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=out_base(args.prefix, j["entry"]) + "/summary.json")["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    print(f"[freq-ablation] {len(jobs)} jobs (of {len(entries)} checkpoints), revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision, "threads": args.threads,
              "stride": args.stride, "max_regions": args.max_regions, "receivers": args.receivers}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, max_retries=1)(job)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.options(num_gpus=args.gpu_fraction, memory=int(args.memory_gb * 2 ** 30)).remote(queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[freq-ablation] {row.get('band')} {row.get('model')} seed{row.get('seed')} h{row.get('trained_horizon')} ok={row.get('ok')} "
              f"{row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[freq-ablation] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
