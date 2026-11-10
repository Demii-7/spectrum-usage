"""Window-averaged ElectroSense metrics for saved ``tune_direct`` checkpoints (the Time-Series-Library / article convention).

The published ElectroSense numbers average the error over the whole forecast window (steps 1..H); our evaluation scores the single
endpoint step t+H. For a direct model trained for horizon H, the window-averaged MSE is the mean over steps h = 1..H of the per-step
MSE, so each saved checkpoint is evaluated with horizons 1..H (no retraining) and the per-step values are averaged. The endpoint value
of the same evaluation (step H) is compared with the number recorded when the final was trained, as a consistency check.

    python -m training.ray.es_window_rescore --models revinmlp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

KEY = re.compile(r"direct-tuning/electrosense/(?P<ds>spec\d+)/(?P<model>[a-z_0-9]+)/weights/h(?P<h>\d+)/seed-(?P<seed>\d+)/[^/]+\.pt$")


def window_average(per_step: dict[int, float], horizon: int) -> float:
    """Mean of the per-step values over steps 1..horizon (all must be present)."""
    missing = [h for h in range(1, horizon + 1) if h not in per_step]
    if missing:
        raise ValueError(f"missing steps {missing[:5]}")
    return sum(per_step[h] for h in range(1, horizon + 1)) / horizon


def rescore_one(spec: dict) -> dict:
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.electrosense import ensure_dataset, standardized_from_per_frequency
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), spec["bucket"]
    ds, model, horizon, seed = spec["ds"], spec["model"], int(spec["h"]), int(spec["seed"])
    record = {"dataset": ds, "model": model, "horizon": horizon, "seed": seed}
    started = time.time()
    try:
        meta = json.loads(client.get_object(Bucket=bucket, Key=spec["pt_key"] + ".meta.json")["Body"].read())
        config = meta["config"]
        name = meta.get("pipeline") or next(k for k in config if k in ("revinmlp", "nlinear", "dlinear", "patchtst", "residualnonlinear2d",
                                                                      "residuallinearar2d", "linearar2d", "ofa"))
        _, stats = ensure_dataset(ds, root, bucket, client)
        config["windowing"]["horizons"] = list(range(1, horizon + 1))
        config["training"] = {**config.get("training", {}), "models": [name], "device": "cpu"}
        config[name].setdefault("train", {})["test_stride"] = 1
        with tempfile.TemporaryDirectory(prefix="es-window-") as scratch:
            scratch = Path(scratch)
            checkpoint = scratch / Path(spec["pt_key"]).name
            client.download_file(bucket, spec["pt_key"], str(checkpoint))
            out = scratch / "eval"
            out.mkdir()
            evaluate_one_model(config, name, out, checkpoint, skip_plots=True)
            values = standardized_from_per_frequency(pd.read_csv(out / "per_frequency_metrics.csv"), stats)
        for kind in ("raw", "std"):
            step = {h: values[f"mse_{kind}_h{h}"] for h in range(1, horizon + 1)}
            record[f"window_mse_{kind}"] = window_average(step, horizon)
            record[f"endpoint_mse_{kind}"] = step[horizon]
        final = json.loads(client.get_object(Bucket=bucket, Key=f"direct-tuning/electrosense/{ds}/{model}/finals/h{horizon}/seed-{seed}.json")["Body"].read())
        record["recorded_endpoint_mse_raw"] = final.get(f"mse_raw_h{horizon}")
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{spec['prefix']}/{ds}/{model}/h{horizon}/seed-{seed}.json", Body=json.dumps(record).encode())
    return record


def main() -> None:
    import ray

    from training.ray.storage import spectrum_s3_client

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["revinmlp"])
    parser.add_argument("--datasets", nargs="+", default=["spec4", "spec25"])
    parser.add_argument("--prefix", default="es-window-rescore")
    parser.add_argument("--max-parallel", type=int, default=16)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = []
    for ds in args.datasets:
        for model in args.models:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"direct-tuning/electrosense/{ds}/{model}/weights/"):
                for obj in page.get("Contents", []):
                    match = KEY.match(obj["Key"])
                    if match:
                        specs.append({**match.groupdict(), "pt_key": obj["Key"], "repo_root": args.repo_root, "bucket": bucket,
                                      "prefix": args.prefix})
    print(f"[es-window] {len(specs)} checkpoints", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=int(6 * 2**30), max_retries=1)(rescore_one)
    queue, pending, rows = list(specs), [], []
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.remote(queue.pop(0)))
        done, pending = ray.wait(pending, num_returns=1)
        rows.append(ray.get(done[0]))
        r = rows[-1]
        print(f"[es-window] {r['dataset']} {r['model']} h{r['horizon']} seed{r['seed']} ok={r['ok']} "
              f"window={r.get('window_mse_raw')} endpoint={r.get('endpoint_mse_raw')} recorded={r.get('recorded_endpoint_mse_raw')} "
              f"{r.get('error', '')} ({r['wall_seconds']}s) [{len(queue) + len(pending)} left]", flush=True)


if __name__ == "__main__":
    main()
