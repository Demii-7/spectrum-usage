"""NCAR transfer batch, item 3: rescore the saved target-trained RevIN-MLP finals (AERPAW/ARA/COSMOS) with per-window arrays.

The original runs (``training.ray.transfer_external_target_trained``, prefix ``transfer-external-target-trained``) kept only aggregate
metrics. This loads each saved checkpoint (``weights_key`` of the record), rebuilds the identical evaluation config (same tuned winner, same
60/15/25 chronological split, test windows with history inside the test partition) and runs the same evaluator, without training. Per cell
it writes, in the ``ncar_cells`` format, ``forecasts.npz`` (origin = last observed step, epoch seconds) and ``per_bin.csv`` plus a
``summary.json`` that also carries the stored MAE/MSE and the difference (a reproduction check), under
``transfer-ncar-2026-11/item3/target-trained/<platform>/<site>/<run>/<band>/h<h>/seed-<n>/``.

    python -m training.ray.ncar_target_trained --skip-done --gpu-fraction 0.1 --max-parallel 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE = "transfer-external-target-trained"
PREFIX = "transfer-ncar-2026-11/item3/target-trained"


def cell_base(r: dict[str, Any]) -> str:
    return f"{PREFIX}/{r['platform']}/{r['site']}/{r['run']}/{r['band']}/h{int(r['horizon'])}/seed-{int(r['seed'])}"


def job(r: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray import ncar_cells
    from training.ray.storage import spectrum_s3_client
    from training.ray.transfer_external import convert, converted_path
    from training.ray.transfer_external_target_trained import LOOKBACK, apply_target, split_rows
    from training.ray.tune_direct import build_config

    client, bucket = spectrum_s3_client(), args["bucket"]
    t = {k: r[k] for k in ("platform", "site", "run", "band")}
    horizon, seed = int(r["horizon"]), int(r["seed"])
    out = {"target": f"{t['platform']}/{t['site']}/{t['run']}/{t['band']}", "horizon": horizon, "seed": seed}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(int(args.get("threads", 2)))
        convert(root, t)
        path = converted_path(root, t)
        frame = pd.read_csv(path)
        stamps = frame["timestamp_utc"]
        n = len(frame)
        _, b = split_rows(n)
        config, name = build_config("revinmlp", "15min", t["band"], horizon, seed, r["parameters"], root, train_stride=1, test_stride=1)
        config = apply_target(config, str(path.relative_to(root)), stamps, t["band"], horizon, n)
        starts = np.arange(b - LOOKBACK, n - LOOKBACK - horizon + 1)
        origins = pd.to_datetime(stamps.iloc[starts + LOOKBACK - 1].to_numpy(), utc=True)
        with tempfile.TemporaryDirectory(prefix="ncar-tt-") as scratch:
            ckpt = Path(scratch) / Path(r["weights_key"]).name
            client.download_file(bucket, r["weights_key"], str(ckpt))
            eval_dir = Path(scratch) / "eval"
            eval_dir.mkdir()
            evaluate_one_model(config, name, eval_dir, ckpt, skip_plots=True)
            meta = next((eval_dir / "forecasts").glob("*_metadata.json"))
            stem = meta.name.removesuffix("_metadata.json")
            with np.load(eval_dir / "forecasts" / f"{stem}_predictions.npz") as p, np.load(eval_dir / "forecasts" / f"{stem}_targets.npz") as q:
                pred = np.asarray(p[f"t_plus_{horizon}"], dtype=np.float32)
                target = np.asarray(q[f"t_plus_{horizon}"], dtype=np.float32)
        if len(pred) != len(starts):
            raise RuntimeError(f"{len(pred)} predictions for {len(starts)} test windows")
        mats = {horizon: {"origin": np.asarray(origins.values.astype("datetime64[s]").astype(np.int64)), "pred": pred, "target": target}}
        base = cell_base(r)
        table = ncar_cells.save_cell(client, bucket, base, mats, ncar_cells.band_frequencies(t["band"]))
        e = pred.astype(np.float64) - target
        out.update(ok=True, n_windows=int(len(pred)), mae_db=float(np.abs(e).mean()), mse_db2=float((e ** 2).mean()),
                   stored_mae_db=r.get("model_mae_db"), stored_mse_db2=r.get("model_mse_db"),
                   first_test_target_time=r.get("first_test_target_time"), weights_key=r["weights_key"],
                   mae_minus_stored=float(np.abs(e).mean()) - float(r.get("model_mae_db", np.nan)),
                   per_bin_mae_mean=float(table.mae_db.mean()), code_revision=args.get("code_revision"))
        client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(out, default=str, indent=1).encode())
    except Exception as error:
        import traceback

        out.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
    out["wall_seconds"] = round(time.time() - started, 1)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-fraction", type=float, default=0.1)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--code-revision", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    import ray

    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    keys = [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=SOURCE + "/")
            for o in pg.get("Contents", []) if o["Key"].endswith(".json") and "/weights/" not in o["Key"] and not o["Key"].endswith("-error.json")]
    records = [json.loads(client.get_object(Bucket=bucket, Key=k)["Body"].read()) for k in keys]
    records = [r for r in records if r.get("ok") and r.get("weights_key")]

    def done(r):
        try:
            client.head_object(Bucket=bucket, Key=f"{cell_base(r)}/summary.json")
            return True
        except Exception:
            return False

    todo = [r for r in records if not (a.skip_done and done(r))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[ncar-tt] {len(records)} records with weights, {len(todo)} to score", flush=True)
    ray.init(address=a.address, ignore_reinit_error=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "code_revision": a.code_revision, "threads": 2}
    queue, pending, ok, failed = list(todo), {}, 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            r = queue.pop(0)
            pending[ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, memory=6 * 2 ** 30, max_retries=1)(job).remote(r, shared)] = r
        finished, _ = ray.wait(list(pending), num_returns=1)
        pending.pop(finished[0])
        row = ray.get(finished[0])
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        print(f"[ncar-tt] {row['target']} h{row['horizon']} seed{row['seed']} ok={row.get('ok')} mae={row.get('mae_db')} "
              f"d_stored={row.get('mae_minus_stored')} {row.get('error', '')} ({row['wall_seconds']}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[ncar-tt] done: {ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
