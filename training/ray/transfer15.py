"""Frozen zero-shot transfer of the completed 15-minute 1D/2D checkpoints (evaluation only, no training, no adaptation).

Stage A (this file): same-band transfer to the two POWDER receivers outside the training set (guest house, law 73) during the same
October test week as the core evaluation (target interval 2026-10-27T20:00Z to 2026-11-03T20:00Z, end exclusive). A 600-800 MHz
checkpoint is scored on 600-800 MHz data and a 2400-2600 MHz checkpoint on 2400-2600 MHz data; source and target bands are written
into every record, and combined site-and-band transfer is not attempted here.

Frozen protocol (what the code does, not just what we intend):
* weights and the source normalization (per-frequency mean and standard deviation of the core receivers' training partition) come
  from the checkpoint; no statistic is ever computed from target data, and no post-hoc centering is applied;
* lookback 96 x 15 minutes; endpoint error at steps 1, 4, 12, 24, 48 (the one-step models are rolled out by the shared
  autoregressive ``forecast``; horizon 1 is a rollout of length 1);
* one set of forecast origins per (band, receiver): every origin has the full 96-step history and the full 48-step future inside
  one gap-free segment, and every model and baseline is scored on exactly those origins, at every horizon;
* 1D models are flattened to one scalar series per frequency bin, and a series never crosses a segment boundary;
* per-receiver results only (a job is one checkpoint on one receiver).

Baselines on the same origins: persistence and the 96-step lookback mean (parameter-free). Source-validation-selected mean / median /
EWMA settings are NOT part of this stage (they need frozen validation selections from the core receivers; see CAMPAIGN).

Outputs (durable bucket), ``<prefix>/<band>/<receiver>/<rep>/<model>/seed-<n>/`` (baselines: ``.../baseline/<name>/``):
    summary.json    aggregate metrics per horizon (MAE, MSE, RMSE, MAPE linear, MAPE dBm, signed bias) and full provenance
    per_bin.csv     per (horizon, frequency bin): count, MAE, MSE, bias
    per_origin.csv  per (horizon, forecast-origin time): MAE, MSE, bias over bins (paired blocked intervals and per-day skill)
    forecasts/      the exported prediction and target arrays (dBm), as the other transfer evaluators write them

    python -m training.ray.transfer15 --bands 600_800 2400_2600 --skip-done
"""

from __future__ import annotations

import argparse
import hashlib
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
import pandas as pd

HORIZONS = (1, 4, 12, 24, 48)
LOOKBACK = 96
STEP = pd.Timedelta(minutes=15)
TARGET_START, TARGET_END = pd.Timestamp("2026-10-27T20:00:00Z"), pd.Timestamp("2026-11-03T20:00:00Z")
RAW_START = TARGET_START - LOOKBACK * STEP                      # history for the first origin
RECEIVERS = {"guesthouse": "guesthouse-nuc1", "law73": "law73-nuc1"}
BAND_LIMITS = {"600_800": (600.0, 800.0), "2400_2600": (2400.0, 2600.0)}
BASELINES = ("persistence", "lookbackmean")
NORMALIZATION_POLICY = "source_checkpoint_statistics_by_bin_position; never fitted on target; no post-hoc centering"


# ----------------------------------------------------------------------------------------------- scoring
def long_errors(prediction: np.ndarray, target: np.ndarray, bins: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Flatten (N, F) arrays into long vectors of error, target, frequency-bin index and origin time.

    2D rows carry all F bins (``bins`` empty: bin = column); flattened 1D rows carry one scalar each and ``bins`` is (N, 1).
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    n, f = prediction.shape
    bin_index = bins.reshape(n, f) if bins.size == n * f else np.broadcast_to(np.arange(f)[None, :], (n, f))
    return {"error": (prediction - target).reshape(-1).astype(np.float64), "target": target.reshape(-1).astype(np.float64),
            "bin": bin_index.reshape(-1).astype(np.int64), "time": np.repeat(times, f)}


def score_horizon(pred: np.ndarray, tgt: np.ndarray, bins: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Aggregate metrics, per-bin table and per-origin table for one horizon (dBm arrays of equal shape)."""
    from training.common.evaluation_integrated import _percentage_error_arrays

    long = long_errors(np.asarray(pred, dtype=np.float32), np.asarray(tgt, dtype=np.float32), bins, times)
    error = long["error"]
    pct = _percentage_error_arrays(np.asarray(pred, dtype=np.float32).reshape(-1, 1), np.asarray(tgt, dtype=np.float32).reshape(-1, 1),
                                   error.reshape(-1, 1).astype(np.float32))
    aggregate = {"n_values": int(error.size), "mae_db": float(np.abs(error).mean()), "mse_db2": float((error ** 2).mean()),
                 "rmse_db": float(np.sqrt((error ** 2).mean())), "bias_db": float(error.mean()),
                 "mape_linear_pct": float(pct["mape_linear_pct"].mean(dtype=np.float64)),
                 "mape_dbm_pct": float(pct["mape_dbm_pct"].mean(dtype=np.float64))}
    frame = pd.DataFrame({"bin": long["bin"], "time": long["time"], "abs": np.abs(error), "sq": error ** 2, "err": error})
    per_bin = frame.groupby("bin").agg(count=("abs", "size"), mae_db=("abs", "mean"), mse_db2=("sq", "mean"), bias_db=("err", "mean"))
    per_origin = frame.groupby("time").agg(mae_db=("abs", "mean"), mse_db2=("sq", "mean"), bias_db=("err", "mean"))
    aggregate["n_origins"] = int(len(per_origin))
    return {"aggregate": aggregate, "per_bin": per_bin.reset_index(), "per_origin": per_origin.reset_index()}


def score_all(predictions: dict[int, np.ndarray], targets: dict[int, np.ndarray], bins: dict[int, np.ndarray],
              times: dict[int, np.ndarray]) -> dict[str, Any]:
    horizons = sorted(predictions)
    scored = {h: score_horizon(predictions[h], targets[h], bins[h], times[h]) for h in horizons}
    per_bin = pd.concat([s["per_bin"].assign(horizon=h) for h, s in scored.items()], ignore_index=True)
    per_origin = pd.concat([s["per_origin"].assign(horizon=h) for h, s in scored.items()], ignore_index=True)
    per_origin["time"] = pd.to_datetime(per_origin["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"metrics": {str(h): s["aggregate"] for h, s in scored.items()}, "per_bin": per_bin, "per_origin": per_origin}


# --------------------------------------------------------------------------------------------- data
def target_task(band: str, receiver: str):
    from training.common.transfer_2d import TransferTask

    low, high = BAND_LIMITS[band]
    return TransferTask(task_id=f"powder15-{band}-{receiver}", evaluation="held-out receiver, same band, same October test week",
                        platform="powder", raw_start=RAW_START, raw_end=TARGET_END, target_start=TARGET_START, target_end=TARGET_END,
                        frequency_start_mhz=low, frequency_end_mhz=high, sites=(receiver,))


def target_file(repo_root: str, band: str, receiver: str) -> Path:
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, TEST_RUN

    return Path(repo_root) / OUT_ROOT / RECEIVERS[receiver] / TEST_RUN / band / OUT_NAME


def missingness(repo_root: str, band: str, receiver: str) -> dict[str, Any]:
    """Share of 15-minute blocks that were missing in the raw trace (and so interpolated) inside the raw window of this evaluation."""
    from training.ray.powder15 import RAW_NAME, RAW_ROOT, TEST_RUN, block_average_db

    raw = Path(repo_root) / RAW_ROOT / RECEIVERS[receiver] / TEST_RUN / band / RAW_NAME
    frame = pd.read_csv(raw)
    stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
    frame.index = stamps
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="min"))
    blocks = block_average_db(frame).loc[RAW_START:TARGET_END - STEP]
    return {"raw_trace": str(raw.relative_to(repo_root)), "blocks_in_window": int(len(blocks)),
            "missing_share_before_interpolation": float(blocks.isna().to_numpy().mean()),
            "imputation": "linear interpolation in dB over gaps of at most 16 blocks (the core experiments' 15-minute preprocessing)"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(repo_root: str, band: str, receiver: str, source_band: str, code_revision: str) -> dict[str, Any]:
    path = target_file(repo_root, band, receiver)
    return {"source_band": source_band, "target_band": band, "target_receiver": receiver, "target_node": RECEIVERS[receiver],
            "target_file": str(path.relative_to(repo_root)), "target_file_sha256": file_sha256(path),
            "raw_window": [RAW_START.isoformat(), TARGET_END.isoformat()],
            "target_interval": [TARGET_START.isoformat(), TARGET_END.isoformat()], "target_interval_end_exclusive": True,
            "lookback_steps": LOOKBACK, "step_minutes": 15, "horizons": list(HORIZONS), "code_revision": code_revision,
            "normalization_policy": NORMALIZATION_POLICY, "normalization_fitted_on_target": False,
            "preprocessing": "15-minute UTC-aligned blocks, mean in linear power converted back to dB, blocks need >= 12 of 15 valid minutes, "
                             "gaps up to 16 blocks interpolated in dB", **missingness(repo_root, band, receiver)}


def output_base(prefix: str, band: str, receiver: str, rep: str, model: str, seed: int | str) -> str:
    return f"{prefix.strip('/')}/{band}/{receiver}/{rep}/{model}/seed-{seed}"


def baseline_base(prefix: str, band: str, receiver: str, name: str) -> str:
    return f"{prefix.strip('/')}/{band}/{receiver}/baseline/{name}"


def upload(client, bucket: str, base: str, summary: dict[str, Any], scored: dict[str, Any], scratch: Path | None = None) -> None:
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(summary, default=str, indent=1).encode())
    for name in ("per_bin", "per_origin"):
        buffer = io.StringIO()
        scored[name].to_csv(buffer, index=False)
        client.put_object(Bucket=bucket, Key=f"{base}/{name}.csv", Body=buffer.getvalue().encode())
    if scratch is not None and (scratch / "forecasts").exists():
        for path in sorted((scratch / "forecasts").glob("*")):
            client.upload_file(str(path), bucket, f"{base}/forecasts/{path.name}")


# ----------------------------------------------------------------------------------------- model jobs
def origin_times(source, rows: np.ndarray, horizon: int) -> np.ndarray:
    """UTC time of the forecast origin (last observed step) for each target row."""
    return np.asarray((source.timestamps[rows] - horizon * STEP).values, dtype="datetime64[ns]")


def checkpoint_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one saved checkpoint on one held-out receiver."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry, receiver, band = job["entry"], job["receiver"], job["band"]
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    base = output_base(args["prefix"], band, receiver, entry["rep"], entry["model"], entry["seed"])
    record = {"kind": "transfer15", "source_band": entry["band"], "target_band": band, "target_receiver": receiver,
              "rep": entry["rep"], "model": entry["model"], "seed": entry["seed"], "checkpoint_key": entry["pt_key"], "key": base}
    started = time.time()
    try:
        from training.common import transfer_1d, transfer_2d
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        if entry["band"] != band:
            raise ValueError("combined site-and-band transfer is outside stage A")
        ensure_powder15(band, root, SPATIAL_NODES)
        meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
        config = meta["config"]
        config["windowing"] = {"lookback": LOOKBACK, "horizons": list(HORIZONS)}
        name = entry["model"]
        config[name].setdefault("train", {})["test_stride"] = 1
        module = transfer_2d if entry["rep"] == "2d" else transfer_1d
        task = target_task(band, receiver)
        with tempfile.TemporaryDirectory(prefix="transfer15-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint = scratch / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(checkpoint))
            source = module.load_target(task, [target_file(root, band, receiver)], impute=False, max_missing_gap=0)
            module.evaluate_transfer_model(task=task, source=source, model_name=name, checkpoint_path=checkpoint, config=config,
                                           output_dir=scratch, seed=int(entry["seed"]))
            stem = f"{task.task_id}_{name}"
            preds, tgts = np.load(scratch / "forecasts" / f"{stem}_predictions.npz"), np.load(scratch / "forecasts" / f"{stem}_targets.npz")
            predictions, targets, bins, times = {}, {}, {}, {}
            for h in HORIZONS:
                rows = preds[f"rows_t_plus_{h}"].astype(np.int64)
                predictions[h], targets[h] = preds[f"t_plus_{h}"], tgts[f"t_plus_{h}"]
                times[h] = origin_times(source, rows, h)
                bins[h] = (module._flattened_frequency_indices(source.segments, rows).reshape(-1, 1)
                           if entry["rep"] == "1d" else np.empty(0))
            scored = score_all(predictions, targets, bins, times)
            checkpoint_meta = {"checkpoint_sha256": meta.get("sha256"), "training_code_revision": meta.get("code_revision"),
                               "source_config_candidate_id": meta.get("candidate_id"), "source_parameters": meta.get("parameters"),
                               "source_train_files": [f["path"] for f in meta["config"]["data"]["files"] if f["partition"] == "train"]}
            summary = {**record, **provenance(root, band, receiver, entry["band"], args["code_revision"]), **checkpoint_meta,
                       "metrics": scored["metrics"], "n_origins": scored["metrics"][str(HORIZONS[0])]["n_origins"]}
            upload(client, bucket, base, summary, scored, scratch)
        record.update(ok=True, metrics=scored["metrics"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        client.put_object(Bucket=bucket, Key=f"{base}/error.json", Body=json.dumps(record, default=str).encode())
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def baseline_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: persistence and lookback mean on one receiver, on the origins every model uses."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    band, receiver = job["band"], job["receiver"]
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record = {"kind": "transfer15_baseline", "target_band": band, "target_receiver": receiver}
    started = time.time()
    try:
        from training.common import transfer_2d
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        ensure_powder15(band, root, SPATIAL_NODES)
        task = target_task(band, receiver)
        source = transfer_2d.load_target(task, [target_file(root, band, receiver)], impute=False, max_missing_gap=0)
        starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=max(HORIZONS),
                                                  stride=1, target_start=task.target_start, target_end=task.target_end)
        if not len(starts):
            raise ValueError("no origins")
        data = np.asarray(source.data, dtype=np.float32)
        last = data[starts + LOOKBACK - 1]
        mean = np.stack([data[s:s + LOOKBACK].mean(axis=0) for s in starts])
        forecasts = {"persistence": last, "lookbackmean": mean}
        for name, pred in forecasts.items():
            predictions, targets, bins, times = {}, {}, {}, {}
            for h in HORIZONS:
                rows = starts + LOOKBACK + h - 1
                predictions[h], targets[h], bins[h] = pred, data[rows], np.empty(0)
                times[h] = origin_times(source, rows, h)
            scored = score_all(predictions, targets, bins, times)
            summary = {**record, "baseline": name, **provenance(root, band, receiver, band, args["code_revision"]),
                       "metrics": scored["metrics"], "n_origins": int(len(starts))}
            upload(client, bucket, baseline_base(args["prefix"], band, receiver, name), summary, scored)
        record.update(ok=True, n_origins=int(len(starts)))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        client.put_object(Bucket=bucket, Key=f"{baseline_base(args['prefix'], band, receiver, 'error')}/error.json",
                          Body=json.dumps(record, default=str).encode())
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


# ------------------------------------------------------------------------------------------------ driver
def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=list(BAND_LIMITS))
    p.add_argument("--receivers", nargs="+", default=list(RECEIVERS), choices=list(RECEIVERS))
    p.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--gpu-fraction", type=float, default=0.1)
    p.add_argument("--memory-gb", type=float, default=12.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="transfer15-oct2026")
    p.add_argument("--code-revision", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = args.code_revision
    if revision is None:
        try:
            revision = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    entries = [e for e in discover_sources(client, bucket, ["p15tuned"], args.models)
               if e["band"] in args.bands and e["rep"] in args.reps and int(e["seed"]) in args.seeds]

    def done(key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/summary.json")
            return True
        except Exception:
            return False

    jobs = [("model", {"entry": e, "receiver": r, "band": e["band"]}) for e in entries for r in args.receivers]
    if not args.no_baselines:
        jobs = [("baseline", {"band": b, "receiver": r}) for b in args.bands for r in args.receivers] + jobs
    if args.skip_done:
        def job_key(kind, job):
            if kind == "baseline":
                return baseline_base(args.prefix, job["band"], job["receiver"], BASELINES[-1])
            e = job["entry"]
            return output_base(args.prefix, job["band"], job["receiver"], e["rep"], e["model"], e["seed"])
        jobs = [(k, j) for k, j in jobs if not done(job_key(k, j))]
    print(f"[transfer15] {len(jobs)} jobs ({sum(k == 'baseline' for k, _ in jobs)} baseline), revision {revision[:12]}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "code_revision": revision}
    ray.init(address=args.address, ignore_reinit_error=True)
    memory = int(args.memory_gb * 2 ** 30)
    model_task = ray.remote(num_cpus=2, num_gpus=args.gpu_fraction, memory=memory, max_retries=1)(checkpoint_job)
    base_task = ray.remote(num_cpus=2, num_gpus=0, memory=memory, max_retries=1)(baseline_job)
    queue, pending, ok, failed = list(jobs), [], 0, 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            kind, job = queue.pop(0)
            pending.append((base_task if kind == "baseline" else model_task).remote(job, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        h12 = (row.get("metrics") or {}).get("12", {})
        print(f"[transfer15] {row.get('target_band')} {row.get('target_receiver')} {row.get('rep', 'baseline')} {row.get('model', '')} "
              f"seed{row.get('seed', '')} ok={row.get('ok')} mae_t12={h12.get('mae_db')} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[transfer15] done: {ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
