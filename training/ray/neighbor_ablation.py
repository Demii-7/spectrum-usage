"""Neighbour ablation: do a receiver's neighbours' dynamics help its forecast? (15-minute 4D, frozen base models, evaluation only)

For each core receiver r (the target) the four-receiver input map is rebuilt with r's own series untouched and the other three
receivers' series changed:

    real      the measured series (the normal geometry_correct evaluation)
    shift96   each neighbour shifted in time by 24 hours (96 steps, wrapping): keeps its level and its daily phase, breaks its
              short-term co-movement with r
    shift210  shifted by 52.5 hours (210 steps): breaks the daily phase too
    level     each neighbour replaced by its per-bin mean level (no dynamics at all)

The model is scored only at r. If the 4D models use neighbours' dynamics, the error at r rises when they are broken; if they forecast r
from r's own history (r dominates the interpolated map around its own location), it does not. The parameter-free 4D lookback mean is
the control (it has no learned use of neighbour dynamics). Every mode of one (model, seed, receiver) is scored on the origins that are
valid in all four modes, so the comparison is paired.

Models: the tuned 15-minute 4D winners' final seeds (``tuning/powder15_4d/<band>/4d/<model>/weights/seed-N``), the same models as
``core4-p2``. Linear models and the lookback-mean control run on CPU; the neural models (STS-PredNet, ResidualConvLSTM, ConvLSTM) run on a
GPU share (``--gpu-fraction``, default 0.5; ``--gpu-fraction 0`` forces CPU, which is slow for them). STS-PredNet loads through
``training.common.spatial_checkpoint_evaluation.load_stsprednet_for_forecast`` inside ``evaluate_condition``; ResidualConvLSTM and DSwinLSTM-I load
like the other neural models from their tuning finals (DSwinLSTM-I: seed 42 only). Output per (band, model, seed): ``<prefix>/<band>/<model>/seed-N/per_origin.npz`` (origin times and
per-origin, per-horizon mean absolute error, signed bias in dB and mean squared error in dB^2 (``mse``, added 2026-11-12) for every receiver x mode) and ``summary.json``.

    python -m training.ray.neighbor_ablation --models linearar4d residuallinearar4d lookbackmean4d --seeds 40 41 42 43 44
    python -m training.ray.neighbor_ablation --models stsprednet residualconvlstm --seeds 40 41 42 43 44 --skip-done --gpu-fraction 0.5
    python -m training.ray.neighbor_ablation --models stsprednet --bands 600_800 --seeds 40 --origin-stride 20 --prefix neighbor-ablation-smoke   # smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MODES = (("real", None, 0), ("shift96", "shift", 96), ("shift210", "shift", 210), ("level", "level", 0))
NEURAL = ("stsprednet", "residualconvlstm", "convlstm", "dswinlstm_i")        # GPU models; everything else (linear 4D, lookback mean) runs on CPU
SINGLE_SEED = {"lookbackmean4d": [42], "dswinlstm_i": [42]}      # the control has no weights; dswinlstm_i is one seed (inherited configuration)
KNOWN_MODELS = ("linearar4d", "residuallinearar4d", "lookbackmean4d", *NEURAL)


def device_for(model: str, gpu_fraction: float) -> str:
    """"cuda" for the neural models when a GPU share is requested, "cpu" otherwise."""
    return "cuda" if model in NEURAL and gpu_fraction > 0 else "cpu"


def make_transform(keep: int, mode: str | None, shift: int):
    """A series transform that leaves receiver ``keep`` untouched and changes the others as described."""
    def transform(series):
        if mode is None:
            return series
        values = np.array(series.values_dbm, copy=True)
        for index in range(values.shape[1]):
            if index == keep:
                continue
            if mode == "shift":
                values[:, index] = np.roll(values[:, index], shift, axis=0)
            elif mode == "level":
                values[:, index] = np.nanmean(values[:, index], axis=0, keepdims=True)
            else:
                raise ValueError(mode)
        return dataclasses.replace(series, values_dbm=values)
    return transform


def per_origin_errors(npz, receiver_index: int, normalization) -> dict[str, np.ndarray]:
    """Per-origin, per-horizon mean absolute error and signed bias (dB, mean over bins) at one receiver, with the origin times."""
    prediction = npz["predictions_normalized"][:, :, receiver_index, :].astype(np.float64)       # (N, H, F)
    target = npz["targets_dbm"][:, :, receiver_index, :].astype(np.float64)
    if normalization is not None:
        mean = np.asarray(normalization["mean_dbm"], dtype=np.float64).squeeze()
        std = np.asarray(normalization["std_dbm"], dtype=np.float64).squeeze()
        prediction = prediction * std + mean
    error = prediction - target
    return {"times": np.asarray(npz["origin_timestamps"]).astype(str), "mae": np.abs(error).mean(axis=2), "bias": error.mean(axis=2),
            "mse": (error ** 2).mean(axis=2)}


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: all receivers x all modes for one (band, model, seed)."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray.powder15 import ensure_powder15
    from training.ray.spatial15 import CORE, HORIZONS, SIX, experiment_yaml, grid_for, model_config_for, six_receiver_frame
    from training.ray.storage import spectrum_s3_client

    client, bucket, band = spectrum_s3_client(), args["bucket"], spec["band"]
    model, seed = spec["model"], int(spec["seed"])
    base = f"{args['prefix'].strip('/')}/{band}/{model}/seed-{seed}"
    record: dict[str, Any] = {"band": band, "model": model, "seed": seed, "modes": [m[0] for m in MODES],
                              "origin_stride": args["origin_stride"], "device": device_for(model, args["gpu_fraction"])}
    started = time.time()
    try:
        import torch

        ensure_powder15(band, root, tuple(SIX))
        names = [SIX[n] for n in CORE]
        with tempfile.TemporaryDirectory(prefix="neighbor-abl-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint, normalization = None, None
            if model == "lookbackmean4d":
                config = {"windowing": {"lookback": 96, "horizons": list(HORIZONS)}}
                model_yaml = config
            else:
                prefix = f"{args['tuning_prefix']}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/"
                keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                        for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
                if not keys:
                    raise FileNotFoundError(f"no tuning final weights under {prefix}")
                checkpoint = scratch / Path(keys[0]).name
                client.download_file(bucket, keys[0], str(checkpoint))
                meta = json.loads(client.get_object(Bucket=bucket, Key=keys[0] + ".meta.json")["Body"].read())
                _, normalization = _checkpoint_metadata(checkpoint)
                model_yaml = model_config_for(meta["config"], model)
                record.update(weights_key=keys[0], weights_sha256=meta.get("sha256"))
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_yaml))
            frame = six_receiver_frame(root)
            variant = {"name": "core4-p2", "nodes": list(CORE), "grid": {}}
            exp_path = Path(root) / "training/configs" / f".neighbor-{os.getpid()}-{model}-{seed}.yaml"
            exp_path.write_text(yaml.safe_dump(experiment_yaml(band, variant["nodes"], None, grid_for(variant, frame))))
            collected: dict[str, np.ndarray] = {}
            try:
                for receiver_index, receiver in enumerate(names):
                    for mode_name, mode, shift in MODES:
                        out = scratch / f"out-{receiver_index}-{mode_name}"
                        evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint,
                                           model_name=model, seed=seed, condition="geometry_correct", output_directory=out,
                                           batch_size=args["batch_size"], device=device_for(model, args["gpu_fraction"]),
                                           origin_stride=args["origin_stride"],
                                           series_transform=make_transform(receiver_index, mode, shift))
                        with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                            errors = per_origin_errors(npz, receiver_index, normalization)
                        for name, array in errors.items():
                            collected[f"{receiver}|{mode_name}|{name}"] = array
                        for name in ("receiver_forecasts.npz", "map_metrics.csv"):          # large; the per-origin tables are what we keep
                            (out / name).unlink(missing_ok=True)
            finally:
                exp_path.unlink(missing_ok=True)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, horizons=np.asarray(HORIZONS), **collected)
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.npz", Body=buffer.getvalue())
            summary = {}
            for receiver in names:
                common = None
                for mode_name, _, _ in MODES:
                    t = set(collected[f"{receiver}|{mode_name}|times"].tolist())
                    common = t if common is None else common & t
                row = {}
                for mode_name, _, _ in MODES:
                    times = collected[f"{receiver}|{mode_name}|times"].tolist()
                    keep = np.asarray([t in common for t in times])
                    row[mode_name] = {"n_common_origins": int(keep.sum()),
                                      "mae_by_horizon": collected[f"{receiver}|{mode_name}|mae"][keep].mean(axis=0).round(4).tolist(),
                                      "mean_horizon_mae": float(collected[f"{receiver}|{mode_name}|mae"][keep].mean())}
                summary[receiver] = row
            record["summary"] = summary
            client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps({**record, "ok": True}, default=str).encode())
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
        client.put_object(Bucket=bucket, Key=f"{base}/error.json", Body=json.dumps(record, default=str).encode())
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=("600_800", "2400_2600"))
    p.add_argument("--models", nargs="+", default=["linearar4d", "residuallinearar4d", "lookbackmean4d"], choices=KNOWN_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=24.0)
    p.add_argument("--gpu-fraction", type=float, default=0.5, help="GPU share of each neural-model job (0 = CPU); the other models never use a GPU")
    p.add_argument("--origin-stride", type=int, default=1, help="1 for real results; larger only for smoke tests (use another --prefix)")
    p.add_argument("--max-parallel", type=int, default=10)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="neighbor-ablation-oct2026")
    p.add_argument("--tuning-prefix", default="tuning")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = [{"band": b, "model": m, "seed": s} for b in args.bands for m in args.models
             for s in (SINGLE_SEED.get(m) or args.seeds)]
    if args.skip_done:
        def done(spec):
            try:
                client.head_object(Bucket=bucket, Key=f"{args.prefix.strip('/')}/{spec['band']}/{spec['model']}/seed-{spec['seed']}/summary.json")
                return True
            except Exception:
                return False
        specs = [sp for sp in specs if not done(sp)]
    print(f"[neighbor-ablation] {len(specs)} jobs", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "tuning_prefix": args.tuning_prefix,
              "batch_size": args.batch_size, "gpu_fraction": args.gpu_fraction, "origin_stride": args.origin_stride}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, num_gpus=0, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    queue, pending, ok = list(specs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            spec = queue.pop(0)
            gpus = args.gpu_fraction if device_for(spec["model"], args.gpu_fraction) == "cuda" else 0
            pending.append(remote.options(num_gpus=gpus).remote(spec, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[neighbor-ablation] {row.get('band')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[neighbor-ablation] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
