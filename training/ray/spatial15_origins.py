"""Per-origin errors of the finished 15-minute spatial cells (evaluation only), so map constructions can be compared on identical origins.

``training.ray.spatial15`` keeps only aggregate metrics for each cell. For a finished cell ``<prefix>/<band>/<variant>/<model>/seed-N``
this job downloads its saved weights (the base variant ``core4-p2`` of a tuned model falls back to the tuning final of the same seed),
rebuilds the variant's maps (its grid, with the frozen variogram for ``core4-krige``), runs the ``geometry_correct`` condition of
``training.common.spatial_ablation_eval.evaluate_condition`` on the four core receivers, and keeps per-origin, per-horizon mean
absolute error (over the 200 bins) at each receiver with the origin times:

    <output-prefix>/<band>/<variant>/<model>/seed-N/per_origin.npz     keys: horizons, "<receiver>|times", "<receiver>|mae" (N, H)
    <output-prefix>/<band>/<variant>/<model>/seed-N/summary.json

CPU only. ``evaluation/analysis/map_construction_vs_per_receiver.py`` reads these files.

    python -m training.ray.spatial15_origins --bands 600_800 2400_2600 --models linearar4d residuallinearar4d lookbackmean4d --skip-done
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

SEEDS_PER_MODEL = {"lookbackmean4d": (42,), "dswinlstm_i": (42,)}       # dswinlstm_i: one seed, inherited configuration
GPU_MODELS = ("dswinlstm_i",)                                            # too slow to score 672 origins on the CPU


def per_origin_errors(npz, receiver_index: int, normalization) -> dict[str, np.ndarray]:
    """Per-origin, per-horizon mean absolute error (dB, mean over bins) at one receiver, with the origin times."""
    prediction = npz["predictions_normalized"][:, :, receiver_index, :].astype(np.float64)       # (N, H, F)
    target = npz["targets_dbm"][:, :, receiver_index, :].astype(np.float64)
    if normalization is not None:
        mean = np.asarray(normalization["mean_dbm"], dtype=np.float64).squeeze()
        std = np.asarray(normalization["std_dbm"], dtype=np.float64).squeeze()
        prediction = prediction * std + mean
    return {"times": np.asarray(npz["origin_timestamps"]).astype(str), "mae": np.abs(prediction - target).mean(axis=2)}


def weights_for(client, bucket: str, prefix: str, tuning_prefix: str, band: str, variant: str, model: str, seed: int):
    """The checkpoint of a finished cell, or (base variant only) the tuning final of the same seed; None if neither exists."""
    candidates = [f"{prefix.strip('/')}/{band}/{variant}/{model}/seed-{seed}/weights/"]
    if variant == "core4-p2":
        candidates.append(f"{tuning_prefix}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/")
    for where in candidates:
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=where)
                for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
        if keys:
            return keys[0]
    return None


def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.spatial_ablation_eval import _checkpoint_metadata, evaluate_condition
    from training.ray.powder15 import ensure_powder15
    from training.ray.spatial15 import (CORE, HORIZONS, SIX, experiment_yaml, grid_for, load_variogram, model_config_for,
                                        six_receiver_frame, variants)
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    band, variant_name, model, seed = spec["band"], spec["variant"], spec["model"], int(spec["seed"])
    base = f"{args['output_prefix'].strip('/')}/{band}/{variant_name}/{model}/seed-{seed}"
    record: dict[str, Any] = {"band": band, "variant": variant_name, "model": model, "seed": seed}
    started = time.time()
    try:
        import torch  # noqa: F401  (the evaluator imports torch lazily)

        variant = next(v for v in variants(["s2", "solo"]) if v["name"] == variant_name)
        grid = grid_for(variant, six_receiver_frame(root))
        if grid.get("method") == "kriging":
            grid = {**grid, "variogram": load_variogram(root, band)}
        ensure_powder15(band, root, tuple(SIX))
        names = [SIX[n] for n in variant["nodes"]]        # the map's receivers (the core four, or one receiver for a solo variant)
        with tempfile.TemporaryDirectory(prefix="s15-origins-") as scratch_dir:
            scratch = Path(scratch_dir)
            checkpoint, normalization = None, None
            if model == "lookbackmean4d":
                model_yaml = {"windowing": {"lookback": 96, "horizons": list(HORIZONS)}}
            else:
                key = weights_for(client, bucket, args["prefix"], args["tuning_prefix"], band, variant_name, model, seed)
                if key is None:
                    raise FileNotFoundError("no saved weights for this cell")
                checkpoint = scratch / Path(key).name
                client.download_file(bucket, key, str(checkpoint))
                meta = json.loads(client.get_object(Bucket=bucket, Key=key + ".meta.json")["Body"].read())
                _, normalization = _checkpoint_metadata(checkpoint)
                model_yaml = model_config_for(meta["config"], model)
                record["weights_key"] = key
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_yaml))
            exp_path = Path(root) / "training/configs" / f".s15origins-{os.getpid()}-{variant_name}-{model}-{seed}.yaml"
            exp_path.write_text(yaml.safe_dump(experiment_yaml(band, list(variant["nodes"]), None, grid)))
            try:
                out = scratch / "out"
                evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint, model_name=model,
                                   seed=seed, condition="geometry_correct", output_directory=out, batch_size=args["batch_size"],
                                   device="cuda" if model in GPU_MODELS else "cpu")
            finally:
                exp_path.unlink(missing_ok=True)
            collected: dict[str, np.ndarray] = {}
            with np.load(out / "receiver_forecasts.npz", allow_pickle=True) as npz:
                for index, receiver in enumerate(names):
                    for name, array in per_origin_errors(npz, index, normalization).items():
                        collected[f"{receiver}|{name}"] = array
            buffer = io.BytesIO()
            np.savez_compressed(buffer, horizons=np.asarray(HORIZONS), **collected)
            client.put_object(Bucket=bucket, Key=f"{base}/per_origin.npz", Body=buffer.getvalue())
            record["summary"] = {r: float(collected[f"{r}|mae"].mean()) for r in names}
            record["n_origins"] = int(len(collected[f"{names[0]}|times"]))
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    record["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(record, default=str).encode())
    return record


def finished_cells(client, bucket: str, prefix: str, band: str, models: list[str], variants_: list[str] | None) -> list[dict[str, Any]]:
    """Cells with an ok result.json under the spatial15 prefix (variants, models, seeds), for ``band``."""
    cells = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.strip('/')}/{band}/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) == 6 and parts[-1] == "result.json" and parts[3] in models and parts[2] != "permuted":
                if variants_ and parts[2] not in variants_ or not parts[4].startswith("seed-"):
                    continue
                cells.append({"band": band, "variant": parts[2], "model": parts[3], "seed": int(parts[4].split("-")[1]), "_key": obj["Key"]})
    keep = []
    for c in cells:
        try:
            if json.loads(client.get_object(Bucket=bucket, Key=c.pop("_key"))["Body"].read()).get("ok"):
                keep.append(c)
        except Exception:
            pass
    return keep


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"], choices=("600_800", "2400_2600"))
    p.add_argument("--models", nargs="+", default=["linearar4d", "residuallinearar4d", "lookbackmean4d"])
    p.add_argument("--variants", nargs="+", help="only these variant names (default: every finished S2 variant)")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=12.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--gpu-fraction", type=float, default=0.25, help="GPU share of a cell of a GPU model (dswinlstm_i)")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="spatial15-tuned-oct2026")
    p.add_argument("--tuning-prefix", default="tuning")
    p.add_argument("--output-prefix", default="spatial15-origin-errors-oct2026")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.spatial15 import variants as s2_variants
    from training.ray.storage import spectrum_s3_client

    s2_names = {v["name"] for v in s2_variants(["s2"])}
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    specs = []
    for band in args.bands:
        for cell in finished_cells(client, bucket, args.prefix, band, args.models, args.variants):
            if cell["variant"] in s2_names and (not args.seeds or cell["seed"] in args.seeds):
                specs.append(cell)
        if "core4-p2" in s2_names and (not args.variants or "core4-p2" in args.variants):      # base cells of tuned models come from the tuning finals
            for model in args.models:
                for seed in SEEDS_PER_MODEL.get(model, (40, 41, 42, 43, 44)):
                    cell = {"band": band, "variant": "core4-p2", "model": model, "seed": seed}
                    if cell not in specs and (not args.seeds or seed in args.seeds):
                        specs.append(cell)
    if args.skip_done:
        def done(s):
            try:
                key = f"{args.output_prefix.strip('/')}/{s['band']}/{s['variant']}/{s['model']}/seed-{s['seed']}/summary.json"
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        specs = [s for s in specs if not done(s)]
    print(f"[s15-origins] {len(specs)} cells", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "tuning_prefix": args.tuning_prefix,
              "output_prefix": args.output_prefix, "batch_size": args.batch_size}
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=2, num_gpus=0, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    gpu_remote = ray.remote(num_cpus=2, num_gpus=args.gpu_fraction, memory=int(args.memory_gb * 2 ** 30), max_retries=1)(job)
    queue, pending, ok = list(specs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            spec = queue.pop(0)
            pending.append((gpu_remote if spec['model'] in GPU_MODELS else remote).remote(spec, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        print(f"[s15-origins] {row.get('band')} {row.get('variant')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} "
              f"{row.get('error', '')} ({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[s15-origins] done: {ok} of {len(specs)} ok", flush=True)


if __name__ == "__main__":
    main()
