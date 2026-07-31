"""Launch the 48 evaluation-only 4D transfer cells with Ray Tune."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import yaml

from training.common.transfer_4d import (
    DEFAULT_MANIFEST,
    ROOT,
    SEED,
    TRANSFER_MODELS,
    campaign_cells,
    evaluate_transfer_task,
    load_transfer_manifest,
    manifest_digest,
    repository_path,
    resolve_checkpoint,
    transfer_tasks,
)
from training.ray.optional import require_ray
from training.ray.storage import MinIOConfig


def _trainable(args: argparse.Namespace, manifest: dict[str, Any], config: dict[str, Any]):
    require_ray()
    from ray import tune

    tasks = {task.task_id: task for task in transfer_tasks(manifest)}

    def run(values: dict[str, Any]) -> None:
        cell = values["cell"]
        task = tasks[str(cell["task_id"])]
        model = str(cell["model"])
        trial_directory = Path(tune.get_context().get_trial_dir())
        cache = trial_directory / "source-checkpoints"
        checkpoint_entry = manifest["checkpoints"][model]
        checkpoint = None if checkpoint_entry is None else resolve_checkpoint(
            checkpoint_entry,
            archive=args.archive,
            cache_dir=cache,
            endpoint=args.endpoint,
            default_bucket=args.bucket,
        )
        normalization_checkpoint = checkpoint
        if model == "lookbackmean4d":
            normalization_entry = manifest["checkpoints"][manifest["normalization_checkpoint"]]
            normalization_checkpoint = resolve_checkpoint(
                normalization_entry,
                archive=args.archive,
                cache_dir=cache,
                endpoint=args.endpoint,
                default_bucket=args.bucket,
            )
        evaluation_directory = trial_directory / "evaluation"
        result = evaluate_transfer_task(
            task=task,
            model_name=model,
            config=config,
            checkpoint_path=checkpoint,
            normalization_checkpoint_path=normalization_checkpoint,
            output_directory=evaluation_directory,
            seed=SEED,
            device=args.device,
            batch_size=args.batch_size,
            origin_stride=args.origin_stride,
            origin_limit=args.origin_limit,
            checkpoint_seed=(checkpoint_entry or {}).get("checkpoint_seed"),
        )
        tune.report(
            result,
            checkpoint=tune.Checkpoint.from_directory(str(evaluation_directory)),
        )

    return run


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig

    manifest = load_transfer_manifest(DEFAULT_MANIFEST)
    config = yaml.safe_load(repository_path(manifest["config"]).read_text(encoding="utf-8"))
    cells = campaign_cells(args.tasks, args.models)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.archive is not None:
        args.archive = args.archive.resolve()
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
    storage = MinIOConfig(args.bucket, args.prefix, args.endpoint)
    os.environ.update(storage.environment())
    plan = {
        "evaluation_only": True,
        "seed": SEED,
        "tasks": args.tasks or [task.task_id for task in transfer_tasks(manifest)],
        "models": args.models or list(TRANSFER_MODELS),
        "trial_count": len(cells),
        "gpu_per_trial": 0.5,
        "manifest_sha256": manifest_digest(manifest),
        "storage_path": storage.storage_path,
    }
    (args.output_dir / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tuner = tune.Tuner(
        tune.with_resources(_trainable(args, manifest, config), {"cpu": args.cpus_per_trial, "gpu": 0.5}),
        param_space={"cell": tune.grid_search(cells)},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=RunConfig(name="transfer-4d-eval", storage_path=storage.storage_path),
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"4D transfer trials failed: {failed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="Task IDs to run; defaults to all eight")
    parser.add_argument("--models", nargs="+", help="Models to run; defaults to all six")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="ray/transfer-4d")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum-usage"))
    parser.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000"))
    parser.add_argument("--archive", type=Path, default=ROOT / "checkpoints.tgz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--origin-limit", type=int)
    parser.add_argument("--cpus-per-trial", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.max_concurrent <= 0:
        parser.error("--max-concurrent must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
