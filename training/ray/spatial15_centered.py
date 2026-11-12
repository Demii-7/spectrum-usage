"""Per-window centring of the frozen neural 4D finals, scored at the four core receivers (evaluation only; task 3 of the neural-4D brief).

Is the neural 4D models' error a level failure? Each tuning final (``tuning/powder15_4d/<band>/4d/<model>/weights/seed-N``, the
``core4-p2`` map, the readout unchanged: ``sample_forecast_grid`` = IDW over the grid cells with the map's own power) is scored twice on the
same origins in one job: as trained (``geometry_correct``) and with per-window centring (``center_windows``: the lookback mean of every cell and
bin of the normalized input map window is removed; the map is linear in the receiver series, so this equals centring every receiver's series
before the map is built; the level is restored after the forecast, either the receiver's OWN window mean added to its sampled forecast
(``own``, the brief's definition) or the map's window mean added back to the forecast map before it is read (``map``)).

    <output-prefix>/<band>/core4-p2/<model>/seed-N/per_origin.npz   keys: horizons, "<receiver>|times", "<receiver>|mae" (as trained),
                                                                    "<receiver>|mae_center_own", "<receiver>|mae_center_map" (N, H)
    <output-prefix>/<band>/core4-p2/<model>/seed-N/summary.json     (seed, configuration label, origins identical, per-receiver means)

STS-PredNet and ResidualConvLSTM run on the CPU, DSwinLSTM-I on a GPU share. Resumable (--skip-done).

    python -m training.ray.spatial15_centered --models stsprednet residualconvlstm dswinlstm_i --skip-done
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
import yaml

GPU_MODELS = ("dswinlstm_i",)
CONFIG_LABEL = {"stsprednet": "tuned", "residualconvlstm": "tuned", "dswinlstm_i": "inherited:1min-r2"}
VARIANT = "core4-p2"
READOUT = "sample_forecast_grid: IDW over the grid cells with the map's own power (2)"


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray.powder15 import ensure_powder15
    from training.ray.spatial15 import (CORE, SIX, experiment_yaml, grid_for, model_config_for, six_receiver_frame, variants)
    from training.ray.spatial15_origins import per_origin_errors, weights_for
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    band, model, seed = spec["band"], spec["model"], int(spec["seed"])
    base = f"{args['output_prefix'].strip('/')}/{band}/{VARIANT}/{model}/seed-{seed}"
    record: dict[str, Any] = {"band": band, "variant": VARIANT, "model": model, "seed": seed, "config_label": CONFIG_LABEL.get(model, "tuned"),
                              "readout": READOUT}
    started = time.time()
    try:
        import torch  # noqa: F401

        variant = next(v for v in variants(["s2"]) if v["name"] == VARIANT)
        grid = grid_for(variant, six_receiver_frame(root))
        ensure_powder15(band, root, tuple(SIX))
        names = [SIX[n] for n in CORE]
        key = weights_for(client, bucket, args["prefix"], args["tuning_prefix"], band, VARIANT, model, seed)
        if key is None:
            raise FileNotFoundError("no saved weights for this cell")
        with tempfile.TemporaryDirectory(prefix="s15-centered-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint = scratch / Path(key).name
            client.download_file(bucket, key, str(checkpoint))
            meta = json.loads(client.get_object(Bucket=bucket, Key=key + ".meta.json")["Body"].read())
            _, normalization = _checkpoint_metadata(checkpoint)
            record["weights_key"] = key
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_config_for(meta["config"], model)))
            exp_path = Path(root) / "training/configs" / f".s15centered-{os.getpid()}-{model}-{seed}.yaml"
            exp_path.write_text(yaml.safe_dump(experiment_yaml(band, list(CORE), None, grid)))
            device = "cuda" if model in GPU_MODELS else "cpu"
            try:
                for label, center in (("plain", None), ("center", "own")):
                    evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint, model_name=model,
                                       seed=seed, condition="geometry_correct", output_directory=scratch / label,
                                       batch_size=args["batch_size"], device=device, center_windows=center)
            finally:
                exp_path.unlink(missing_ok=True)
            collected: dict[str, np.ndarray] = {}
            with np.load(scratch / "plain/receiver_forecasts.npz", allow_pickle=True) as plain, \
                    np.load(scratch / "center/receiver_forecasts.npz", allow_pickle=True) as centered:
                same = bool(np.array_equal(plain["origins"], centered["origins"]))
                record["origins_identical"] = same
                if not same:
                    raise RuntimeError("as-trained and centred runs scored different origins")
                for index, receiver in enumerate(names):
                    base_errors = per_origin_errors(plain, index, normalization)
                    collected[f"{receiver}|times"], collected[f"{receiver}|mae"] = base_errors["times"], base_errors["mae"]
                    for tag, array_key in (("own", "predictions_normalized_center_own"), ("map", "predictions_normalized_center_map")):
                        shim = {"predictions_normalized": centered[array_key], "targets_dbm": centered["targets_dbm"],
                                "origin_timestamps": centered["origin_timestamps"]}
                        collected[f"{receiver}|mae_center_{tag}"] = per_origin_errors(shim, index, normalization)["mae"]
            from training.ray.spatial15 import HORIZONS
            buffer = io.BytesIO()
            np.savez_compressed(buffer, horizons=np.asarray(HORIZONS), **collected)
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.npz", Body=buffer.getvalue())
            record["summary"] = {r: {k: float(collected[f"{r}|{k}"].mean()) for k in ("mae", "mae_center_own", "mae_center_map")} for r in names}
            record["n_origins"] = int(len(collected[f"{names[0]}|times"]))
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=("600_800", "2400_2600"))
    p.add_argument("--models", nargs="+", default=["stsprednet", "residualconvlstm", "dswinlstm_i"])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=12.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="spatial15-tuned-oct2026")
    p.add_argument("--tuning-prefix", default="tuning")
    p.add_argument("--output-prefix", default="spatial15-centered-oct2026")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = [{"band": b, "model": m, "seed": s} for b in args.bands for m in args.models for s in args.seeds]
    if args.skip_done:
        def done(s):
            try:
                key = f"{args.output_prefix.strip('/')}/{s['band']}/{VARIANT}/{s['model']}/seed-{s['seed']}/summary.json"
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        specs = [s for s in specs if not done(s)]
    print(f"[s15-centered] {len(specs)} cells", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "tuning_prefix": args.tuning_prefix,
              "output_prefix": args.output_prefix, "batch_size": args.batch_size}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    gpu_remote = ray.remote(num_cpus=2, num_gpus=args.gpu_fraction, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    queue, pending, ok = list(specs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            spec = queue.pop(0)
            pending.append((gpu_remote if spec["model"] in GPU_MODELS else remote).remote(spec, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[s15-centered] {row.get('band')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[s15-centered] done: {ok} of {len(specs)} ok", flush=True)


if __name__ == "__main__":
    main()
