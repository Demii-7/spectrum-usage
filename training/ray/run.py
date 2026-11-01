"""Plan two-stage Ray tuning or launch its search stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

from .candidate import candidate_id
from .config import validate_tuning_config
from .manifests import provenance_manifest, write_manifest
from .optional import require_ray
from .registry import MODEL_REGISTRY, bundle_token, get_model_spec
from .robustness import DEFAULT_MAX_FAILURES, build_or_restore, run_config, summarize_results
from .scheduler import ASHAConfig
from .spaces import materialize_space
from .storage import MinIOConfig
from .trainable import run_integrated_trial


SEARCH_SEED = 42
RERANK_SEEDS = (41, 42, 43)
DEFAULT_CANDIDATE_BUDGET = 12
DEFAULT_MAX_CONCURRENT_TRIALS = 8
CAPACITIES = ("tiny", "small", "reference")


def candidate_budget(total: int = DEFAULT_CANDIDATE_BUDGET) -> dict[str, int]:
    if total < len(CAPACITIES):
        raise ValueError("Candidate budget must accommodate tiny, small, and reference anchors")
    return {"total": total, "anchors": len(CAPACITIES), "random": total - len(CAPACITIES)}


def _resources(spec, gpu_fraction: float | None) -> dict[str, float]:
    """Per-trial resources; ``gpu_fraction`` only applies to models that use a GPU."""
    resources = dict(spec.resources.as_ray())
    if gpu_fraction is not None and resources.get("gpu", 0) > 0:
        resources["gpu"] = float(gpu_fraction)
    return resources


def _grace_period(model_name: str, override: int | None) -> int:
    default = (12 if model_name == "temporalconvnet" else 1 if model_name == "stsprednet" else 5)
    if override is None:
        return default
    # Keep the longer per-model grace period where one was chosen deliberately.
    return max(int(override), default) if model_name == "temporalconvnet" else int(override)


def build_plan(config: dict[str, Any], models: list[str], *, candidates: int = DEFAULT_CANDIDATE_BUDGET,
               validate: bool = True, fixed_batch_size: int | None = None,
               grace_period: int | None = None, reduction_factor: int = 3, brackets: int = 1,
               max_concurrent: int = DEFAULT_MAX_CONCURRENT_TRIALS,
               gpu_fraction: float | None = None) -> dict[str, Any]:
    budget = candidate_budget(candidates)
    entries = []
    for requested_name in models:
        spec = get_model_spec(requested_name)
        entry: dict[str, Any] = {
            "model": spec.name,
            "status": spec.status,
            "policy": spec.policy,
            "reason": spec.reason,
            "resources": _resources(spec, gpu_fraction),
            "objective": spec.objective,
            "fallback_objective": spec.fallback_objective,
            "space": {name: domain.to_dict() for name, domain in spec.space.items()},
            "anchors": {capacity: spec.anchor(capacity) for capacity in CAPACITIES},
        }
        if spec.hpo_executable:
            try:
                validate_tuning_config(config, spec.name)
                entry["config_validation"] = {"valid": True, "error": None}
            except (ValueError, KeyError, RuntimeError) as exc:
                entry["config_validation"] = {"valid": False, "error": str(exc)}
                if validate:
                    raise
            historical = {
                name: dict(parameters)
                for name, parameters in spec.historical.items()
                if fixed_batch_size is None
                or parameters.get("train.batch_size") == fixed_batch_size
            }
            guaranteed = len(CAPACITIES) + len(historical)
            if candidates < guaranteed:
                raise ValueError(
                    f"{spec.name} requires at least {guaranteed} candidates for "
                    "architecture anchors and historical presets"
                )
            entry["historical_candidates"] = historical
            entry["search"] = {
                "seed": SEARCH_SEED,
                "candidate_budget": {
                    "total": candidates,
                    "anchors": len(CAPACITIES),
                    "historical": len(historical),
                    "random": candidates - guaranteed,
                },
                "anchor_candidate_ids": {
                    capacity: candidate_id(spec.name, spec.anchor(capacity))
                    for capacity in CAPACITIES
                },
                "historical_candidate_ids": {
                    name: candidate_id(spec.name, parameters)
                    for name, parameters in historical.items()
                },
                "search_algorithm": "BasicVariantGenerator(points_to_evaluate=anchors)",
                "max_concurrent_trials": max_concurrent,
                "grace_period": _grace_period(spec.name, grace_period),
                "reduction_factor": reduction_factor,
                "brackets": brackets,
            }
            entry["rerank"] = {
                "selection_methods": ["best", "best-simple", "reference"],
                "seeds": list(RERANK_SEEDS),
                "configurations": 3,
                "total_runs": 3 * len(RERANK_SEEDS),
                "one_standard_error": False,
                "simplicity_tie_fraction": 0.01,
            }
        entries.append(entry)
    return {"schema_version": 2, "search_seed": SEARCH_SEED,
            "rerank_seeds": list(RERANK_SEEDS), "models": entries}


def anchor_points(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the three guaranteed BasicVariantGenerator points."""
    return [{**dict(entry["anchors"][capacity]), "seed": SEARCH_SEED, "capacity": capacity}
            for capacity in CAPACITIES]


def ray_anchor_points(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Encode mapping-valued architecture bundles for Ray's preset merger."""
    architectures = {
        capacity: dict(entry["anchors"][capacity]["architecture"])
        for capacity in CAPACITIES
    }
    points = []
    for capacity, point in zip(CAPACITIES, anchor_points(entry), strict=True):
        encoded = dict(point)
        encoded["architecture"] = capacity
        points.append(encoded)
    for name, parameters in entry.get("historical_candidates", {}).items():
        token = f"historical:{name}"
        encoded = dict(parameters)
        architectures[token] = dict(encoded["architecture"])
        encoded.update(seed=SEARCH_SEED, capacity=token, architecture=token)
        points.append(encoded)
    return points, architectures


def extra_architecture_tokens(spec, existing: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tokens for registered architecture bundles beyond the anchors and historical presets.

    The token is a hash of the bundle's contents (``bundle_token``), so the replicate stage can
    resolve a winner back to its bundle even if the registry is reordered later.
    """
    known = [dict(bundle) for bundle in existing.values()]
    return {
        bundle_token(bundle): dict(bundle)
        for bundle in spec.space["architecture"].choices
        if dict(bundle) not in known
    }


def build_rerank_candidates(model_name: str,
                            selections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand the three selected configurations into nine deterministic runs."""
    required = {"best", "best-simple", "reference"}
    if set(selections) != required:
        raise ValueError(f"Rerank selections must be exactly {sorted(required)}")
    runs = []
    for selection_name in ("best", "best-simple", "reference"):
        parameters = dict(selections[selection_name])
        aggregate_id = candidate_id(model_name, parameters)
        for seed in RERANK_SEEDS:
            runs.append({
                "selection": selection_name,
                "candidate_id": aggregate_id,
                "trial_id": candidate_id(model_name, parameters, seed),
                "seed": seed,
                "parameters": parameters,
            })
    return runs


def _launch(config: dict[str, Any], plan: dict[str, Any], output: Path,
            storage_path: str | None, fixed_batch_size: int | None = None,
            *, resume: bool = False, max_failures: int = DEFAULT_MAX_FAILURES) -> None:
    require_ray()
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator

    for entry in plan["models"]:
        spec = get_model_spec(entry["model"], require_hpo=True)
        space = materialize_space(spec.space)
        if fixed_batch_size is not None:
            space["train.batch_size"] = tune.choice([fixed_batch_size])
        points, architectures = ray_anchor_points(entry)
        # Without this the random draws only ever pick the three anchors and historical presets.
        architectures.update(extra_architecture_tokens(spec, architectures))
        if fixed_batch_size is not None:
            for point in points:
                point["train.batch_size"] = fixed_batch_size
        # Ray 2.54 cannot merge a dictionary preset into a categorical domain.
        # Tokens preserve coupled bundles while keeping guaranteed anchors valid.
        space["architecture"] = tune.choice(list(architectures))
        space.update(seed=SEARCH_SEED, capacity="random")
        search = BasicVariantGenerator(
            random_state=SEARCH_SEED,
            points_to_evaluate=points,
        )

        def trainable(parameters, model=spec.name, architecture_bundles=architectures):
            values = dict(parameters)
            seed = int(values.pop("seed"))
            values.pop("capacity", None)
            values["architecture"] = architecture_bundles[str(values["architecture"])]
            trial_directory = Path(tune.get_context().get_trial_dir())
            run_integrated_trial(config, model, values, seed, trial_directory)

        train_config = config[spec.name].get("train", {})
        epochs = int(
            config[spec.name].get("epochs", 100)
            if spec.name == "stsprednet"
            else train_config.get("diffusion_epochs", 100)
            if spec.name == "tss_lcd"
            else train_config.get("epochs", 100)
        )
        # The extra iteration lets a surviving trial attach its completed
        # integrated checkpoint after the final forecasting epoch.
        scheduler = ASHAConfig(
            max_t=epochs + 1,
            grace_period=int(entry["search"]["grace_period"]),
            reduction_factor=int(entry["search"].get("reduction_factor", 3)),
            brackets=int(entry["search"].get("brackets", 1)),
        ).build()

        experiment_name = f"{spec.name}-search"
        root = storage_path or str(output.resolve())
        resourced = tune.with_resources(trainable, entry["resources"])

        def make_tuner(resourced=resourced, space=space, search=search, scheduler=scheduler,
                       entry=entry, experiment_name=experiment_name, root=root):
            return tune.Tuner(
                resourced,
                param_space=space,
                tune_config=tune.TuneConfig(
                    num_samples=int(entry["search"]["candidate_budget"]["total"]),
                    max_concurrent_trials=int(entry["search"]["max_concurrent_trials"]),
                    search_alg=search,
                    scheduler=scheduler,
                ),
                run_config=run_config(experiment_name, root, max_failures=max_failures),
            )

        tuner = build_or_restore(
            f"{root.rstrip('/')}/{experiment_name}", resourced, make_tuner, resume=resume
        )
        summarize_results(tuner.fit(), label=experiment_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATE_BUDGET,
                        help="Total seed-42 search candidates, including three anchors (default: 12)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ray"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minio-bucket")
    parser.add_argument("--minio-prefix", default="spectrum-usage/ray")
    parser.add_argument("--minio-endpoint")
    parser.add_argument("--fixed-batch-size", type=int,
                        help="Restrict HPO to one training batch size")
    parser.add_argument("--grace-period", type=int,
                        help="ASHA grace period in epochs (default: per-model, 5)")
    parser.add_argument("--reduction-factor", type=int, default=3,
                        help="ASHA reduction factor; lower is less aggressive (default: 3)")
    parser.add_argument("--brackets", type=int, default=1, help="ASHA brackets (default: 1)")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT_TRIALS,
                        help=f"Concurrent trials (default: {DEFAULT_MAX_CONCURRENT_TRIALS})")
    parser.add_argument("--gpu-fraction", type=float,
                        help="Override the per-trial GPU fraction for GPU models")
    parser.add_argument("--resume", action="store_true",
                        help="Restore an interrupted search and rerun errored trials")
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
                        help="Retries per failed trial (default: 2)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config, args.models, candidates=args.candidates,
                      validate=not args.dry_run, fixed_batch_size=args.fixed_batch_size,
                      grace_period=args.grace_period, reduction_factor=args.reduction_factor,
                      brackets=args.brackets, max_concurrent=args.max_concurrent,
                      gpu_fraction=args.gpu_fraction)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "plan.json", plan)
    write_manifest(args.output_dir / "provenance.json", provenance_manifest(
        command=sys.argv, config_path=args.config, extra={"models": args.models}))
    storage_path = None
    if args.minio_bucket:
        minio = MinIOConfig(args.minio_bucket, args.minio_prefix, args.minio_endpoint)
        os.environ.update(minio.environment())
        storage_path = minio.storage_path
    _launch(config, plan, args.output_dir, storage_path, args.fixed_batch_size,
            resume=args.resume, max_failures=args.max_failures)


if __name__ == "__main__":
    main()
