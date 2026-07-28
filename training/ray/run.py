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
from .registry import MODEL_REGISTRY, get_model_spec
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


def build_plan(config: dict[str, Any], models: list[str], *, candidates: int = DEFAULT_CANDIDATE_BUDGET,
               validate: bool = True) -> dict[str, Any]:
    budget = candidate_budget(candidates)
    entries = []
    for requested_name in models:
        spec = get_model_spec(requested_name)
        entry: dict[str, Any] = {
            "model": spec.name,
            "status": spec.status,
            "policy": spec.policy,
            "reason": spec.reason,
            "resources": spec.resources.as_ray(),
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
                "max_concurrent_trials": DEFAULT_MAX_CONCURRENT_TRIALS,
                "grace_period": 12 if spec.name == "temporalconvnet" else 5,
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
            storage_path: str | None) -> None:
    require_ray()
    from ray import tune
    from ray.tune import RunConfig
    from ray.tune.search.basic_variant import BasicVariantGenerator

    for entry in plan["models"]:
        spec = get_model_spec(entry["model"], require_hpo=True)
        space = materialize_space(spec.space)
        points, architectures = ray_anchor_points(entry)
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

        epochs = int(config[spec.name].get("train", {}).get("epochs", 100))
        # The extra iteration lets a surviving trial attach its completed
        # integrated checkpoint after the final forecasting epoch.
        scheduler = ASHAConfig(
            max_t=epochs + 1,
            grace_period=int(entry["search"]["grace_period"]),
        ).build()

        tuner = tune.Tuner(
            tune.with_resources(trainable, spec.resources.as_ray()),
            param_space=space,
            tune_config=tune.TuneConfig(
                num_samples=int(entry["search"]["candidate_budget"]["total"]),
                max_concurrent_trials=int(entry["search"]["max_concurrent_trials"]),
                search_alg=search,
                scheduler=scheduler,
            ),
            run_config=RunConfig(
                name=f"{spec.name}-search",
                storage_path=storage_path or str(output.resolve()),
            ),
        )
        tuner.fit()


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config, args.models, candidates=args.candidates, validate=not args.dry_run)
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
    _launch(config, plan, args.output_dir, storage_path)


if __name__ == "__main__":
    main()
