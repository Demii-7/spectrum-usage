"""Spatial evaluation of the October 2026 frozen 1-minute 4D checkpoints (evaluation only; no training).

Wraps ``training.common.spatial_ablation_eval.evaluate_condition`` with the October receivers (config
``training/configs/spatial_eval_oct2026_<band>.yaml``): checkpoints come from the durable bucket, each seed's exact model
settings are rebuilt from the checkpoint manifest (``cost_model_config_json``), and every output except the large forecast
arrays is written back to the durable bucket under ``<prefix>/<band>/<model>/seed-<n>/<condition>[-p<perm>]/``.

Conditions (names inherited from the July evaluator; see the config header):
    geometry_correct     core four in, core four scored: physical-receiver metrics plus full-grid and receiver-cell map metrics
    geometry_permuted    the same with the receivers' coordinates permuted before interpolation (frozen model)
    six_to_new           law 73 scored, not in the input map (extrapolation: it lies outside the core-four grid)
    eight_to_new         law 73 scored and added to the input map without retraining
    six_to_guesthouse    guest house scored, not in the input map (interpolation: inside the grid)
    seven_to_guesthouse  guest house scored and added to the input map

One-minute 4D windows for the full test week take about 50 GB of host memory, so each task reserves ``--memory-gb``.

    python -m training.ray.spatial_oct --models lookbackmean4d linearar4d residuallinearar4d convlstm residualconvlstm dswinlstm_i
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = ("geometry_correct", "geometry_permuted", "six_to_new", "eight_to_new", "six_to_guesthouse", "seven_to_guesthouse")
SEEDS = (40, 41, 42, 43, 44)
CHECKPOINT_PREFIX = {"dswinlstm_i": "checkpoints-oct2026-r2/4d/dswinlstm_i"}          # round-2 winner
UPLOAD = ("result.json", "metadata.json", "region_metrics.csv", "map_metrics.csv")


def checkpoint_prefix(model: str) -> str:
    return CHECKPOINT_PREFIX.get(model, f"checkpoints-oct2026/4d/{model}")


def model_section(client, bucket: str, model: str, seed: int) -> dict[str, Any]:
    """The {model: {model: ..., train: ...}} config section for one seed, from the checkpoint manifest."""
    manifest = json.loads(client.get_object(Bucket=bucket, Key=f"{checkpoint_prefix(model)}/MANIFEST.json")["Body"].read())
    for entry in manifest["entries"]:
        metrics = entry.get("metrics") or {}
        if int(entry.get("seed", -1)) == int(seed) and "cost_model_config_json" in metrics:
            train = json.loads(metrics.get("cost_train_config_json") or "{}")
            return {"model": json.loads(metrics["cost_model_config_json"]),
                    "train": {"batch_size": int(train.get("batch_size", 4)), "seed": int(seed)}}
    raise KeyError(f"no model config for {model} seed {seed} in {checkpoint_prefix(model)}/MANIFEST.json")


def output_base(prefix: str, band: str, cell: dict[str, Any]) -> str:
    name = cell["condition"] + (f"-p{cell['permutation_seed']}" if cell.get("permutation_seed") is not None else "")
    return f"{prefix.strip('/')}/{band}/{cell['model']}/seed-{cell['seed']}/{name}"


def run_cell(cell: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: evaluate one (model, seed, condition[, permutation]) and upload its outputs."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.spatial_ablation_eval import evaluate_condition
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    base = output_base(args["prefix"], args["band"], cell)
    row: dict[str, Any] = {**cell, "band": args["band"], "key": base}
    try:
        import torch

        with tempfile.TemporaryDirectory(prefix="spatial-oct-") as scratch:
            scratch = Path(scratch)
            model, seed = cell["model"], int(cell["seed"])
            checkpoint = None
            config = {"windowing": {"lookback": 60, "horizons": [1, 15, 60]}}
            if model != "lookbackmean4d":
                checkpoint = scratch / f"{model}-seed-{seed}.pt"
                client.download_file(bucket, f"{checkpoint_prefix(model)}/seed-{seed}/powder_{args['band']}_t4t6_{model}.pt",
                                     str(checkpoint))
                config[model] = model_section(client, bucket, model, seed)
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(config))
            stride = args["permutation_origin_stride"] if cell["condition"] == "geometry_permuted" else args["origin_stride"]
            result = evaluate_condition(
                experiment_path=Path(root) / args["experiment"], model_config_path=model_config, checkpoint_path=checkpoint,
                model_name=model, seed=seed, condition=cell["condition"], output_directory=scratch / "out",
                permutation_seed=cell.get("permutation_seed"), origin_stride=stride, batch_size=args["batch_size"],
                device="cuda" if torch.cuda.is_available() else "cpu")
            for name in UPLOAD:
                path = scratch / "out" / name
                if path.exists():
                    client.upload_file(str(path), bucket, f"{base}/{name}")
            if args["save_forecasts"] and (scratch / "out" / "receiver_forecasts.npz").exists():
                client.upload_file(str(scratch / "out" / "receiver_forecasts.npz"), bucket, f"{base}/receiver_forecasts.npz")
            row.update(ok=True, mean_horizon_mae_db=result.get("mean_horizon_mae_db"), n_origins=result.get("n_origins"),
                       seconds=round(result.get("elapsed_seconds", 0.0), 1))
    except Exception as error:
        import traceback

        row.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        client.put_object(Bucket=bucket, Key=f"{base}/error.json", Body=json.dumps(row, default=str).encode())
    return row


def cells(models, seeds, conditions, permutations: int) -> list[dict[str, Any]]:
    from training.ray.spatial_ablation import unique_permutation_seeds

    perm_seeds = unique_permutation_seeds(permutations, receiver_count=4) if "geometry_permuted" in conditions else []
    out = []
    for model in models:
        for seed in ([42] if model == "lookbackmean4d" else seeds):
            for condition in conditions:
                if condition == "geometry_permuted":
                    out.extend({"model": model, "seed": int(seed), "condition": condition, "permutation_seed": p} for p in perm_seeds)
                else:
                    out.append({"model": model, "seed": int(seed), "condition": condition, "permutation_seed": None})
    return out


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="600_800", choices=("600_800",))   # 2400-2600 4D checkpoints use another layout (not wired yet)
    p.add_argument("--models", nargs="+", default=["lookbackmean4d", "linearar4d", "residuallinearar4d", "convlstm", "residualconvlstm",
                                                     "dswinlstm_i"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--permutations", type=int, default=5, help="permuted geometries per (model, seed); 23 exist for four receivers")
    p.add_argument("--origin-stride", type=int, default=1)
    p.add_argument("--permutation-origin-stride", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=70.0)
    p.add_argument("--gpus", type=float, default=0.25)
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--save-forecasts", action="store_true", help="also upload receiver_forecasts.npz (large)")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="spatial-oct2026")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "band": args.band,
              "experiment": f"training/configs/spatial_eval_oct2026_{args.band}.yaml", "origin_stride": args.origin_stride,
              "permutation_origin_stride": args.permutation_origin_stride, "batch_size": args.batch_size,
              "save_forecasts": args.save_forecasts}
    todo = cells(args.models, args.seeds, args.conditions, args.permutations)
    if args.skip_done:
        def done(cell):
            try:
                client.head_object(Bucket=bucket, Key=f"{output_base(args.prefix, args.band, cell)}/result.json")
                return True
            except Exception:
                return False
        todo = [c for c in todo if not done(c)]
    print(f"[spatial-oct] {len(todo)} cells", flush=True)
    ray.init(address=args.address, ignore_reinit_error=True)
    remote = ray.remote(num_cpus=4, num_gpus=args.gpus, memory=int(args.memory_gb * 2**30), max_retries=2)(run_cell)
    pending, queue, results = [], list(todo), []
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            pending.append(remote.remote(queue.pop(0), shared))
        done_refs, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(done_refs[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        results.append(row)
        print(f"[spatial-oct] {row.get('model')} seed{row.get('seed')} {row.get('condition')} p={row.get('permutation_seed')} "
              f"ok={row.get('ok')} mae={row.get('mean_horizon_mae_db')} n={row.get('n_origins')} {row.get('error', '')} "
              f"({row.get('seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    failed = [r for r in results if not r.get("ok")]
    print(f"[spatial-oct] done: {len(results) - len(failed)} ok, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
