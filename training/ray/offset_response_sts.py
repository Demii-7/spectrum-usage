"""Constant-offset response of 4D models (default STS-PredNet; ``--models residualconvlstm dswinlstm_i`` for the extra test, S3 prefix
``offset-extra-2026-11/<model>/<band>/<receiver>/seed-N/summary.json`` with ``--layout model-first``; the zero-offset forecast is checked against the
stored ``neighbor-ablation-oct2026`` 'real' per-origin errors and recorded as ``zero_offset_max_abs_diff_db``; seeds are discovered from the weights
in S3 with ``--discover-seeds``, so a rerun picks up newly finished seeds).

Constant-offset response of STS-PredNet (4D): does the frozen spatial model carry a change of signal level through to its forecast?

Same definition as ``training.ray.offset_response`` (gain = mean forecast shift / input offset, offsets -20..+20 dB, identical origins across
offsets, nothing trained or fitted), for the 15-minute 4D tuning finals (``tuning/powder15_4d/<band>/4d/stsprednet/weights/seed-N``, the
base variant ``core4-p2``: the four core receivers on the default map). A constant of c dB is added to the series of all four receivers
(through the ``series_transform`` hook of ``training.common.spatial_ablation_eval``) before the inverse-distance-weighted maps are built,
so the input map shifts by exactly c (the interpolation weights sum to 1); the checkpoint's own normalization is applied afterwards. The
response is measured at the four physical receivers (forecast at the receiver's grid cell, as in the spatial evaluation), in dB.

Output: ``<prefix>/<band>/<receiver>/4d/stsprednet/seed-N/summary.json`` (same record layout as the 1D/2D and direct tests, so
``evaluation/analysis/offset_response_table.py --prefix offset-response-sts-oct2026/`` tabulates it).

    python -m training.ray.offset_response_sts --bands 600_800 2400_2600 --seeds 40 41 42 --skip-done
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.ray.offset_response import OFFSETS, response_rows

MODEL = "stsprednet"
STRIDE = 4                      # default origin thinning (the response does not need every origin); disclosed in every record


def make_shift(offset: float):
    """A series transform adding ``offset`` dB to every receiver."""
    def transform(series):
        return dataclasses.replace(series, values_dbm=np.asarray(series.values_dbm, dtype=np.float64) + float(offset))
    return transform


def to_dbm(predictions_normalized: np.ndarray, normalization) -> np.ndarray:
    """Un-normalize forecasts ((N, H, R, F), checkpoint units) to dBm with the checkpoint's per-bin statistics."""
    prediction = np.asarray(predictions_normalized, dtype=np.float64)
    if normalization is None:
        return prediction
    mean = np.asarray(normalization["mean_dbm"], dtype=np.float64).squeeze()
    std = np.asarray(normalization["std_dbm"], dtype=np.float64).squeeze()
    return prediction * std + mean


def receiver_arrays(prediction_dbm: np.ndarray, horizons: list[int], receiver_index: int) -> dict[int, np.ndarray]:
    """Forecasts of one receiver per horizon: {h: (N, F)}."""
    return {h: prediction_dbm[:, i, receiver_index, :] for i, h in enumerate(horizons)}


def summary_key(args: dict[str, Any], model: str, band: str, receiver: str, seed: int) -> str:
    prefix = args["prefix"].strip("/")
    if args.get("layout") == "model-first":
        return f"{prefix}/{model}/{band}/{receiver}/seed-{seed}/summary.json"
    return f"{prefix}/{band}/{receiver}/4d/{model}/seed-{seed}/summary.json"


def zero_offset_check(client, bucket: str, model: str, band: str, seed: int, times, prediction_dbm: np.ndarray, target_dbm: np.ndarray,
                      names: list[str], horizons: list[int]) -> dict[str, Any]:
    """Max |difference| (dB) between the unshifted forecasts' per-origin MAE and the stored per-origin errors of the same model and seed
    (``neighbor-ablation-oct2026/<band>/<model>/seed-N/per_origin.npz``, 'real' mode), matched on origin time. None if no stored file."""
    import io

    import pandas as pd

    try:
        with np.load(io.BytesIO(client.get_object(Bucket=bucket, Key=f"neighbor-ablation-oct2026/{band}/{model}/seed-{seed}/per_origin.npz")["Body"].read()),
                     allow_pickle=True) as stored:
            stored_h = [int(h) for h in stored["horizons"]]
            worst, matched = 0.0, 0
            ours = pd.DatetimeIndex(pd.to_datetime(np.asarray(times).astype(str), utc=True))
            for r, name in enumerate(names):
                theirs = pd.DatetimeIndex(pd.to_datetime(np.asarray(stored[f"{name}|real|times"]).astype(str), utc=True))
                position = pd.Series(np.arange(len(theirs)), index=theirs)
                keep = [i for i, t in enumerate(ours) if t in position.index]
                if not keep:
                    continue
                at = position.loc[ours[keep]].to_numpy()
                for j, h in enumerate(horizons):
                    if h not in stored_h:
                        continue
                    mine = np.abs(prediction_dbm[keep, j, r, :] - target_dbm[keep, j, r, :]).mean(axis=1)
                    worst = max(worst, float(np.abs(mine - stored[f"{name}|real|mae"][at, stored_h.index(h)]).max()))
                    matched += len(keep)
            return {"zero_offset_max_abs_diff_db": worst if matched else None, "zero_offset_values_compared": matched}
    except Exception as error:
        return {"zero_offset_max_abs_diff_db": None, "zero_offset_note": f"{type(error).__name__}: {str(error)[:120]}"}


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one (model, band, seed) of the base variant, every offset, all four receivers."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    band, seed = spec["band"], int(spec["seed"])
    MODEL = spec.get("model", "stsprednet")
    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray.powder15 import ensure_powder15
    from training.ray.spatial15 import CORE, HORIZONS, SIX, experiment_yaml, grid_for, model_config_for, six_receiver_frame
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    names = [SIX[n] for n in CORE]
    offsets = tuple(float(o) for o in (args.get("offsets") or OFFSETS))
    stride = int(args.get("stride") or STRIDE)
    record: dict[str, Any] = {"kind": "offset_response", "band": band, "rep": "4d", "model": MODEL, "seed": seed,
                              "intrinsic_window_normalization": "none (checkpoint normalization only)", "offsets_db": list(offsets),
                              "horizons": list(HORIZONS), "origin_stride": stride}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(max(1, int(args.get("threads", 2))))
        ensure_powder15(band, root, tuple(SIX))
        prefix = f"{args['tuning_prefix']}/powder15_4d/{band}/4d/{MODEL}/weights/seed-{seed}/"
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
        if not keys:
            raise FileNotFoundError(f"no tuning final weights under {prefix}")
        with tempfile.TemporaryDirectory(prefix="offset-sts-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint = scratch / Path(keys[0]).name
            client.download_file(bucket, keys[0], str(checkpoint))
            meta = json.loads(client.get_object(Bucket=bucket, Key=keys[0] + ".meta.json")["Body"].read())
            _, normalization = _checkpoint_metadata(checkpoint)
            record.update(checkpoint_key=keys[0], checkpoint_sha256=meta.get("sha256"))
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_config_for(meta["config"], MODEL)))
            variant = {"name": "core4-p2", "nodes": list(CORE), "grid": {}}
            exp_path = Path(root) / "training/configs" / f".offset-sts-{os.getpid()}-{band}-{seed}.yaml"
            exp_path.write_text(yaml.safe_dump(experiment_yaml(band, variant["nodes"], None, grid_for(variant, six_receiver_frame(root)))))
            predictions: dict[float, np.ndarray] = {}
            times = None
            try:
                for offset in (0.0, *offsets):
                    out = scratch / f"out-{offset:+g}"
                    evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint, model_name=MODEL,
                                       seed=seed, condition="geometry_correct", output_directory=out, origin_stride=stride,
                                       batch_size=args["batch_size"], device=args["device"], series_transform=make_shift(offset))
                    with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                        stamp = np.asarray(npz["origin_timestamps"]).astype(str)
                        if times is None:
                            times = stamp
                        elif not np.array_equal(times, stamp):
                            raise RuntimeError("origins changed with the offset")
                        predictions[offset] = to_dbm(npz["predictions_normalized"], normalization)
                        if offset == 0.0:
                            targets0 = np.asarray(npz["targets_dbm"], dtype=np.float64)
                    for name in ("receiver_forecasts.npz", "map_metrics.csv"):
                        (out / name).unlink(missing_ok=True)
            finally:
                exp_path.unlink(missing_ok=True)
        base = predictions[0.0]
        check = zero_offset_check(client, bucket, MODEL, band, seed, times, base, targets0, [SIX[n] for n in CORE], list(HORIZONS))
        record.update(check)
        for index, receiver in enumerate(names):
            rows = []
            for offset in offsets:
                rows.extend(response_rows(receiver_arrays(base, list(HORIZONS), index), receiver_arrays(predictions[offset], list(HORIZONS), index),
                                          offset, HORIZONS))
            summary = {**record, "receiver": receiver, "rows": rows, "n_origins": int(base.shape[0]), "code_revision": args["code_revision"], "ok": True}
            client.put_object(Bucket=bucket, Key=summary_key(args, MODEL, band, receiver, seed), Body=json.dumps(summary, default=str).encode())
        record.update(ok=True, n_origins=int(base.shape[0]))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
        try:
            client.put_object(Bucket=bucket, Key=(f"{args['prefix'].strip('/')}/{MODEL}/{band}/errors/seed-{seed}.json" if args.get("layout") == "model-first"
                                                  else f"{args['prefix'].strip('/')}/{band}/errors/{MODEL}-seed-{seed}.json"),
                              Body=json.dumps(record, default=str).encode())
        except Exception:
            pass
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def main(argv=None) -> None:
    import subprocess

    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=["600_800", "2400_2600"])
    p.add_argument("--models", nargs="+", default=[MODEL], choices=["stsprednet", "residualconvlstm", "dswinlstm_i"])
    p.add_argument("--layout", default="sts", choices=["sts", "model-first"],
                   help="sts: <prefix>/<band>/<receiver>/4d/<model>/seed-N (the STS-PredNet test); model-first: <prefix>/<model>/<band>/<receiver>/seed-N")
    p.add_argument("--discover-seeds", action="store_true", help="only seeds whose tuning-final weights exist in the bucket (a rerun picks up new seeds)")
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    p.add_argument("--offsets", type=float, nargs="+", help="subset of the offsets (smoke tests only; default all six)")
    p.add_argument("--stride", type=int, default=STRIDE, help="origin thinning (default 4)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda; with cuda pass --gpu-fraction")
    p.add_argument("--gpu-fraction", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=24.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="offset-response-sts-oct2026")
    p.add_argument("--tuning-prefix", default="tuning")
    p.add_argument("--code-revision", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = args.code_revision
    if revision is None:
        try:
            revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    def have_weights(model, band, seed):
        prefix = f"{args.tuning_prefix}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/"
        return any(o["Key"].endswith(".pt") for o in client.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []))

    jobs = [{"model": m, "band": b, "seed": s} for m in args.models for b in args.bands for s in args.seeds
            if not args.discover_seeds or have_weights(m, b, s)]
    if args.skip_done:
        def done(j):
            try:
                key = summary_key({"prefix": args.prefix, "layout": args.layout}, j["model"], j["band"], "humanities", j["seed"])
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    print(f"[offset-sts] {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "tuning_prefix": args.tuning_prefix, "code_revision": revision,
              "batch_size": args.batch_size, "device": args.device, "threads": args.threads, "offsets": args.offsets, "stride": args.stride,
              "layout": args.layout}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=args.threads, num_gpus=args.gpu_fraction, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.remote(queue.pop(0), shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[offset-sts] {row.get('model')} {row.get('band')} seed{row.get('seed')} ok={row.get('ok')} {row.get('error', '')} "
              f"zero-offset diff {row.get('zero_offset_max_abs_diff_db')} ({row.get('wall_seconds')}s) "
              f"[{len(queue) + len(pending)} left]", flush=True)
    print(f"[offset-sts] done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
