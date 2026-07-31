"""Run evaluation-only spatial ablations on frozen 4D checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Sequence

from training.ray.optional import require_ray
from training.ray.storage import MinIOConfig


DEFAULT_SEEDS = (40, 41, 42, 43, 44)
CONDITIONS = (
    "geometry_correct",
    "geometry_permuted",
    "six_to_new",
    "eight_to_new",
    "six_to_guesthouse",
    "seven_to_guesthouse",
)


def unique_permutation_seeds(count: int, receiver_count: int = 6) -> list[int]:
    from math import factorial

    from training.common.spatial_ablation_eval import _permutation

    if count < 0 or count > factorial(receiver_count) - 1:
        raise ValueError("permutation count exceeds the number of non-identity assignments")
    seeds = []
    assignments = set()
    candidate = 10_000
    while len(seeds) < count:
        assignment = tuple(int(value) for value in _permutation(receiver_count, candidate))
        if assignment not in assignments:
            assignments.add(assignment)
            seeds.append(candidate)
        candidate += 1
    return seeds


def campaign_cells(
    models: Sequence[str],
    seeds: Sequence[int],
    conditions: Sequence[str],
    permutation_count: int,
) -> list[dict[str, Any]]:
    if permutation_count < 0:
        raise ValueError("permutation_count must be non-negative")
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    permutation_seeds = unique_permutation_seeds(permutation_count)
    cells = []
    for model in models:
        for seed in seeds:
            for condition in conditions:
                if condition == "geometry_permuted":
                    cells.extend(
                        {
                            "model": model,
                            "seed": int(seed),
                            "condition": condition,
                            "permutation_seed": permutation_seed,
                        }
                        for permutation_seed in permutation_seeds
                    )
                else:
                    cells.append(
                        {
                            "model": model,
                            "seed": int(seed),
                            "condition": condition,
                            "permutation_seed": None,
                        }
                    )
    return cells


def _checkpoint_member(model: str, seed: int) -> str:
    return (
        f"checkpoints/4d/{model}/seed-{seed}/"
        f"powder_600_800_t4t6_{model}.pt"
    )


def extract_checkpoint(archive: Path, model: str, seed: int, output: Path) -> Path:
    member = _checkpoint_member(model, seed)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / Path(member).name
    with tarfile.open(archive, "r:gz") as bundle:
        try:
            source = bundle.extractfile(member)
        except KeyError as exc:
            raise FileNotFoundError(f"Checkpoint archive has no {member}") from exc
        if source is None:
            raise FileNotFoundError(f"Checkpoint archive has no file {member}")
        with destination.open("wb") as handle:
            while block := source.read(1024 * 1024):
                handle.write(block)
    return destination


def _trainable(args: argparse.Namespace):
    require_ray()
    from ray import tune

    def run(values: dict[str, Any]) -> None:
        from training.common.spatial_ablation_eval import evaluate_condition

        cell = values["cell"]
        trial_directory = Path(tune.get_context().get_trial_dir())
        checkpoint = None
        if str(cell["model"]) != "lookbackmean4d":
            checkpoint = extract_checkpoint(
                args.checkpoint_archive,
                str(cell["model"]),
                int(cell["seed"]),
                trial_directory / "checkpoint",
            )
        condition = str(cell["condition"])
        stride = args.permutation_origin_stride if condition.startswith("geometry_") else args.new_receiver_origin_stride
        result = evaluate_condition(
            experiment_path=args.experiment,
            model_config_path=args.model_config,
            checkpoint_path=checkpoint,
            model_name=str(cell["model"]),
            seed=int(cell["seed"]),
            condition=condition,
            output_directory=trial_directory / "evaluation",
            permutation_seed=cell.get("permutation_seed"),
            origin_stride=stride,
            origin_limit=args.origin_limit,
            batch_size=args.batch_size,
            device=args.device,
        )
        tune.report(
            result,
            checkpoint=tune.Checkpoint.from_directory(
                str(trial_directory / "evaluation")
            ),
        )

    return run


def launch(args: argparse.Namespace) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig

    args.experiment = args.experiment.resolve()
    args.model_config = args.model_config.resolve()
    args.checkpoint_archive = args.checkpoint_archive.resolve()
    args.output_dir = args.output_dir.resolve()
    os.environ.update(MinIOConfig(args.bucket, args.prefix, args.endpoint).environment())
    cells = campaign_cells(args.models, args.seeds, args.conditions, args.permutations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "evaluation_only": True,
        "models": args.models,
        "seeds": args.seeds,
        "conditions": args.conditions,
        "permutations": args.permutations,
        "permutation_origin_stride": args.permutation_origin_stride,
        "new_receiver_origin_stride": args.new_receiver_origin_stride,
        "trial_count": len(cells),
    }
    (args.output_dir / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resources = {"cpu": args.cpus_per_task, "gpu": args.gpus_per_task}
    tuner = tune.Tuner(
        tune.with_resources(_trainable(args), resources),
        param_space={"cell": tune.grid_search(cells)},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=RunConfig(
            name=f"spatial-ablation-eval-{args.models[0]}",
            storage_path=args.storage_path,
        ),
    )
    results = tuner.fit()
    failed = [result.path for result in results if result.error is not None]
    if failed:
        raise RuntimeError(f"Spatial evaluation trials failed: {failed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("training/configs/spatial_ablation_eval.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("training/configs/spatial_ablation_models.yaml"))
    parser.add_argument("--checkpoint-archive", type=Path, default=Path("checkpoints.tgz"))
    parser.add_argument("--models", nargs="+", default=["convlstm"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--permutation-origin-stride", type=int, default=10)
    parser.add_argument("--new-receiver-origin-stride", type=int, default=1)
    parser.add_argument("--origin-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--storage-path")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=float, default=2.0)
    parser.add_argument("--gpus-per-task", type=float, default=0.5)
    args = parser.parse_args(argv)
    args.storage_path = args.storage_path or f"s3://{args.bucket}/{args.prefix.strip('/')}"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
