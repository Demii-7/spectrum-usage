"""15-minute spatial (4D) experiments on the six October receivers: map sensitivity (S2), held-out receivers (S3), receiver count (S5).

Every run trains one 4D model on maps interpolated from a chosen receiver set, on a grid fixed to the six-receiver area (origin and
extent of humanities, madsen, moran, sage point, guest house and law 73), so removing a receiver never moves the grid. Hyperparameters
are inherited from the October 1-minute 600-800 winners (seed-42 run_costs of ``checkpoints-oct2026/4d``), as for the other inherited
15-minute results. Each trained model is then scored

* on its own maps (``map__*``: full grid against the interpolated target, all 100 cells), and
* at the physical receivers with ``spatial_ablation_eval.evaluate_condition``: every receiver in its input set (``geometry_correct``)
  and each of the six receivers it did not see, both left out of the input map and added to it without retraining.

Variants (``--experiments``):
    s2   core four; IDW power 1, 2, 4, 16 (16 ~ nearest neighbour), linear-power interpolation, grids 5x5 and 20x20, and ordinary
         kriging (``core4-krige``: the variogram of each band is fitted on the training range by ``training.ray.fit_variogram``,
         frozen in ``evaluation/analysis/kriging_variograms.json`` and injected into the grid; the forecast is sampled back at the
         receivers with the default IDW power-2 weights, as in every other variant)
    s3   leave one receiver out of the six (six folds)
    s5   receiver count: 2 and 3 of the core four (three subsets each); evaluated at guest house and law 73
         (the full core four is s2's default)
    solo one core receiver alone (``solo-<receiver>``: nodes=[receiver], the grid frame unchanged, so the IDW map is constant in space); scored at that
         receiver only (geometry_correct); the matched 'own receiver only' control of the four-receiver model

Outputs (durable bucket): ``<prefix>/<band>/<variant>/<model>/seed-<n>/{result.json, eval/<condition>-<receiver>/...}`` plus
weights and the training config, saved with ``save_run``.

    python -m training.ray.spatial15 --experiments s2 s3 --models lookbackmean4d linearar4d residuallinearar4d residualconvlstm convlstm
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

SIX = {"humanities-nuc1": "humanities", "madsen-nuc1": "madsen", "moran-nuc1": "moran", "sagepoint-nuc1": "sage point",
       "guesthouse-nuc1": "guest house", "law73-nuc1": "law 73"}
CORE = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
HORIZONS = (4, 12, 24, 48)
SEEDS = (40, 41, 42)
CPU_MODELS = {"lookbackmean4d", "linearar4d", "residuallinearar4d"}
GRID_DEFAULT = {"height": 10, "width": 10, "power": 2.0, "power_domain": "db"}
VARIOGRAM_FILE = "evaluation/analysis/kriging_variograms.json"


def load_variogram(repo_root: str, band: str) -> dict[str, Any]:
    """The frozen ordinary-kriging variogram of ``band`` (fitted on the training range before any scoring)."""
    data = json.loads((Path(repo_root) / VARIOGRAM_FILE).read_text())["bands"][band]["variogram"]
    return {"model": data["model"], "nugget": float(data["nugget"]), "sill": float(data["sill"]), "range": float(data["range"])}


def six_receiver_frame(repo_root: str) -> dict[str, float]:
    """Origin and extent of the six receivers (the fixed grid every variant shares)."""
    import numpy as np

    from training.common.map_builder import _local_xy

    entries = json.loads((Path(repo_root) / "data/locations/powder.json").read_text())["endpoints"]
    by_name = {e["name"]: e for e in entries}
    lons = np.asarray([by_name[name]["longitude"] for name in SIX.values()])
    lats = np.asarray([by_name[name]["latitude"] for name in SIX.values()])
    lon0, lat0 = float(lons.mean()), float(lats.mean())
    x, y = _local_xy(lons, lats, lon0, lat0)
    return {"origin_lon": lon0, "origin_lat": lat0, "width_meters": float(np.ptp(x)), "height_meters": float(np.ptp(y))}


def tuned_winner(client, bucket: str, tuning_prefix: str, band: str, model: str) -> dict[str, Any]:
    """The best completed trial of the 15-minute 4D search for ``model`` (best so far if the search is still running)."""
    from training.ray.tune_fixed import group_prefix, pick_winner, read_records

    records = [r for r in read_records(client, bucket, group_prefix(tuning_prefix, "powder15_4d", band, "4d", model))
               if r.get("kind") == "trial"]
    best = pick_winner(records)
    return {**best, "n_trials": sum(1 for r in records if r.get("ok"))}


def variants(experiments: list[str]) -> list[dict[str, Any]]:
    """(name, receiver nodes, grid overrides) for each requested experiment."""
    out = []
    if "s2" in experiments:
        out.append({"name": "core4-p2", "nodes": list(CORE), "grid": {}})
        for power in (1.0, 4.0, 16.0):
            out.append({"name": f"core4-p{power:g}", "nodes": list(CORE), "grid": {"power": power}})
        out.append({"name": "core4-p2-linear", "nodes": list(CORE), "grid": {"power_domain": "linear"}})
        for size in (5, 20):
            out.append({"name": f"core4-p2-g{size}", "nodes": list(CORE), "grid": {"height": size, "width": size}})
        out.append({"name": "core4-krige", "nodes": list(CORE), "grid": {"method": "kriging"}})      # variogram injected per band in main()
    if "s3" in experiments:
        for held in SIX:
            out.append({"name": f"loo-{SIX[held].replace(' ', '')}", "nodes": [n for n in SIX if n != held], "grid": {}})
    if "s5" in experiments:
        for k in (2, 3):
            for index, subset in enumerate(itertools.combinations(CORE, k)):
                if index < 3:
                    out.append({"name": f"k{k}-{index}", "nodes": list(subset), "grid": {}})
    if "solo" in experiments:
        for node in CORE:
            out.append({"name": f"solo-{SIX[node].replace(' ', '')}", "nodes": [node], "grid": {}, "solo": True})
    return out


def grid_for(variant: dict[str, Any], frame: dict[str, float]) -> dict[str, Any]:
    return {**GRID_DEFAULT, **frame, **variant["grid"]}


def experiment_yaml(band: str, nodes: list[str], held: str | None, grid: dict[str, Any]) -> dict[str, Any]:
    """An evaluate_condition experiment: ``nodes`` are the map's receivers, ``held`` the receiver scored outside the map."""
    from training.ray.powder15 import OUT_NAME, OUT_ROOT, TEST_RUN

    test = {"start": "2026-10-27T20:00:00Z", "end": "2026-11-03T20:00:00Z"}
    files = {SIX[n]: f"{OUT_ROOT}/{n}/{TEST_RUN}/{band}/{OUT_NAME}" for n in SIX}
    names = [SIX[n] for n in nodes]
    exp = {"locations": "data/locations/powder.json", "collection_key": "endpoints",
           "regions": f"evaluation/analysis/plan_regions_{band}_wholeband.csv", "baseline_receivers": names, "added_receivers": [],
           "grid": grid, "lookback": 96, "horizons": list(HORIZONS), "max_missing_gap": 16,
           "geometry_timing": test, "geometry_files": {name: files[name] for name in names}}
    if held:
        exp["guesthouse_receiver"] = SIX[held]
        exp["guesthouse_timing"] = test
        exp["guesthouse_files"] = {name: files[name] for name in names + [SIX[held]]}
    return exp


STS_BRANCH_KEYS = ("use_closeness", "use_period", "use_trend", "lc", "lp", "lq", "period_interval", "trend_interval",
                   "prediction_offset", "share_branch_weights", "allow_unavailable_branches")


def model_config_for(config: dict[str, Any], model: str) -> dict[str, Any]:
    """The {windowing, model section} file ``evaluate_condition`` reads. STS-PredNet keeps its branch settings flat in training, but the
    shared model factory reads them from ``branches``, so they are copied there."""
    section = copy.deepcopy(config[model])
    if model == "stsprednet" and "branches" not in section:
        section["branches"] = {k: section[k] for k in STS_BRANCH_KEYS if k in section}
    return {"windowing": config["windowing"], model: section}


def physical_scores(client, bucket: str, base: str, root: str, args: dict[str, Any], band: str, variant: dict[str, Any],
                    grid: dict[str, Any], model: str, seed: int, checkpoint: Path | None, config: dict[str, Any],
                    scratch: Path) -> dict[str, Any]:
    """Score one trained model at the physical receivers (geometry_correct plus each held-out receiver, left out of and added to the map)."""
    from training.common.spatial_ablation_eval import evaluate_condition

    model_config = scratch / "models.yaml"
    model_config.write_text(yaml.safe_dump(model_config_for(config, model)))
    held_out = [] if variant.get("solo") else [n for n in SIX if n not in variant["nodes"]]       # solo: its own receiver only
    scored: dict[str, Any] = {}
    for held in held_out or [None]:
        exp_path = Path(root) / "training/configs" / f".spatial15-{os.getpid()}-{variant['name']}.yaml"
        exp_path.write_text(yaml.safe_dump(experiment_yaml(band, variant["nodes"], held, grid)))
        try:
            conditions = ["geometry_correct"] if held is None or held == held_out[0] else []
            if held:
                conditions += ["six_to_guesthouse", "seven_to_guesthouse"]
            for condition in conditions:
                label = condition if condition == "geometry_correct" else f"{condition}-{SIX[held].replace(' ', '')}"
                out = scratch / "cond" / label
                summary = evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint,
                                             model_name=model, seed=seed, condition=condition, output_directory=out,
                                             batch_size=args["batch_size"], device="cpu" if model in CPU_MODELS else "cuda")
                scored[label] = {k: summary.get(k) for k in ("mean_horizon_mae_db", "n_origins", "target_receivers",
                                                               *[f"mae_db_t{h}" for h in HORIZONS])}
                for name in ("result.json", "metadata.json", "region_metrics.csv", "map_metrics.csv"):
                    if (out / name).exists():
                        client.upload_file(str(out / name), bucket, f"{base}/eval/{label}/{name}")
        finally:
            exp_path.unlink(missing_ok=True)
    return scored


def eval_only_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: score an already trained cell at the physical receivers from its saved weights (no training).

    Weights come from the cell's own ``<base>/weights`` or, with ``eval_from_tuning`` and the base variant ``core4-p2``, from the tuning
    finals of the same seed (the same model: tuned winner, core-four default map). The cell's earlier result.json (map metrics and
    provenance from the training run) is kept and extended.
    """
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.ray.storage import spectrum_s3_client

    client, bucket, band = spectrum_s3_client(), args["bucket"], args["band"]
    variant, model, seed = job["variant"], job["model"], int(job["seed"])
    base = f"{args['prefix'].strip('/')}/{band}/{variant['name']}/{model}/seed-{seed}"
    try:
        result = json.loads(client.get_object(Bucket=bucket, Key=f"{base}/result.json")["Body"].read())
    except Exception:
        result = {"band": band, "variant": variant["name"], "nodes": variant["nodes"], "model": model, "seed": seed, "grid": job["grid"]}
    started = time.time()
    try:
        from training.ray.powder15 import ensure_powder15

        ensure_powder15(band, root, tuple(SIX))
        if args.get("eval_from_tuning") and variant["name"] == "core4-p2":
            prefix = f"{args['tuning_prefix']}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/"
            result["hyperparameters"] = "tuned 15-minute 4D winner (tuning final of this seed; weights not retrained)"
            try:      # the tuning final's own scoring on its map; a fixed-configuration final (--fixed-config) names its configuration
                final = json.loads(client.get_object(Bucket=bucket, Key=f"{args['tuning_prefix']}/powder15_4d/{band}/4d/{model}/final/"
                                                                        f"seed-{seed}.json")["Body"].read())
            except Exception:
                final = {}
            if str(final.get("hyperparameters", "")).startswith("inherited:"):
                result["hyperparameters"] = f"{final['hyperparameters']} (no 15-minute search; weights from the tuning-prefix final)"
                result["single_seed"] = True
            if "map__test_mean_horizon_mae_db" not in result:
                result.update({f"map__{k}": v for k, v in final.items() if k.startswith(("test_", "validation_"))})
        else:
            prefix = f"{base}/weights/"
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
        if not keys:
            raise FileNotFoundError(f"no saved weights under {prefix}")
        with tempfile.TemporaryDirectory(prefix="spatial15-eval-") as scratch:
            scratch = Path(scratch)
            checkpoint = scratch / Path(keys[0]).name
            client.download_file(bucket, keys[0], str(checkpoint))
            meta = json.loads(client.get_object(Bucket=bucket, Key=keys[0] + ".meta.json")["Body"].read())
            config = meta["config"]
            result["receivers"] = physical_scores(client, bucket, base, root, args, band, variant, job["grid"], model, seed,
                                                  checkpoint, config, scratch)
            result.update(eval_only=True, eval_weights_key=keys[0], eval_weights_sha256=meta.get("sha256"))
        result.pop("error", None)
        result.pop("trace", None)
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    result["eval_wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/result.json", Body=json.dumps(result, default=str).encode())
    return result


def permuted_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker (S4): the frozen base model scored with the four core receivers' coordinates permuted before interpolation.

    One job per (model, seed, permutation seed). The weights are the tuning finals' (the same tuned model as ``core4-p2``);
    ``lookbackmean4d`` has none. Output: ``<prefix>/<band>/permuted/<model>/seed-<n>/p<perm>/result.json`` (+ the evaluator's files).
    """
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    from training.common.spatial_ablation_eval import evaluate_condition
    from training.ray.powder15 import ensure_powder15
    from training.ray.storage import spectrum_s3_client

    client, bucket, band = spectrum_s3_client(), args["bucket"], args["band"]
    variant, model, seed, perm = job["variant"], job["model"], int(job["seed"]), int(job["permutation_seed"])
    base = f"{args['prefix'].strip('/')}/{band}/permuted/{model}/seed-{seed}/p{perm}"
    result: dict[str, Any] = {"band": band, "model": model, "seed": seed, "permutation_seed": perm, "condition": "geometry_permuted"}
    started = time.time()
    try:
        ensure_powder15(band, root, tuple(SIX))
        with tempfile.TemporaryDirectory(prefix="spatial15-perm-") as scratch:
            scratch = Path(scratch)
            checkpoint, config = None, None
            if model == "lookbackmean4d":
                config = {"windowing": {"lookback": 96, "horizons": list(HORIZONS)}}
            else:
                prefix = f"{args['tuning_prefix']}/powder15_4d/{band}/4d/{model}/weights/seed-{seed}/"
                keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                        for o in page.get("Contents", []) if o["Key"].endswith(".pt")]
                if not keys:
                    raise FileNotFoundError(f"no tuning final weights under {prefix}")
                checkpoint = scratch / Path(keys[0]).name
                client.download_file(bucket, keys[0], str(checkpoint))
                meta = json.loads(client.get_object(Bucket=bucket, Key=keys[0] + ".meta.json")["Body"].read())
                config = meta["config"]
                result.update(weights_key=keys[0], weights_sha256=meta.get("sha256"))
            model_config = scratch / "models.yaml"
            model_config.write_text(yaml.safe_dump(model_config_for(config, model) if model != "lookbackmean4d" else config))
            exp_path = Path(root) / "training/configs" / f".spatial15-perm-{os.getpid()}-{perm}.yaml"
            exp_path.write_text(yaml.safe_dump(experiment_yaml(band, variant["nodes"], None, job["grid"])))
            try:
                summary = evaluate_condition(experiment_path=exp_path, model_config_path=model_config, checkpoint_path=checkpoint,
                                             model_name=model, seed=seed, condition="geometry_permuted", permutation_seed=perm,
                                             output_directory=scratch / "out", batch_size=args["batch_size"],
                                             device="cpu" if model in CPU_MODELS else "cuda")
            finally:
                exp_path.unlink(missing_ok=True)
            result.update({k: summary.get(k) for k in ("mean_horizon_mae_db", "n_origins", "target_receivers",
                                                        *[f"mae_db_t{h}" for h in HORIZONS])})
            for name in ("result.json", "metadata.json", "region_metrics.csv", "map_metrics.csv"):
                if (scratch / "out" / name).exists():
                    client.upload_file(str(scratch / "out" / name), bucket, f"{base}/eval/{name}")
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    result["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/result.json", Body=json.dumps(result, default=str).encode())
    return result


def run_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one (variant, model, seed), score it on its maps and at the physical receivers, upload everything."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.artifacts import save_run
    from training.ray.final_eval import build_eval_config, config_file_for, summarize_metrics
    from training.ray.powder15 import apply_powder15_4d, ensure_powder15, set_lookback_and_horizon
    from training.ray.storage import spectrum_s3_client

    client, bucket, band = spectrum_s3_client(), args["bucket"], args["band"]
    variant, model, seed = job["variant"], job["model"], int(job["seed"])
    base = f"{args['prefix'].strip('/')}/{band}/{variant['name']}/{model}/seed-{seed}"
    result: dict[str, Any] = {"band": band, "variant": variant["name"], "nodes": variant["nodes"], "model": model, "seed": seed,
                              "grid": job["grid"], "hyperparameters": "inherited (1-minute 600-800 seed-42 winner)"}
    started = time.time()
    try:
        ensure_powder15(band, root, tuple(SIX))
        if model == "lookbackmean4d":
            config = yaml.safe_load(Path(config_file_for(model, "4d")).read_text())
            config["training"] = {**config.get("training", {}), "models": [model], "device": "cpu"}
        elif args.get("fixed_config"):       # a FIXED configuration (no search), e.g. DSwinLSTM-I's inherited:1min-r2 (training/configs/fixed/*.json)
            from training.ray.tune_fixed import apply_fixed_config, tuning_base

            fixed = json.loads(Path(args["fixed_config"]).read_text())
            result["hyperparameters"] = f"{fixed.get('label', 'fixed')} (fixed configuration, no 15-minute search; {Path(args['fixed_config']).name})"
            config = apply_fixed_config(tuning_base(model, "4d"), model, fixed, seed)
        elif args.get("tuned"):
            from training.ray.config import inject_parameters
            from training.ray.tune_fixed import tuning_base

            winner = tuned_winner(client, bucket, args["tuning_prefix"], band, model)
            result.update(hyperparameters=f"tuned 15-minute 4D winner ({winner['n_trials']} trials, search seed 42)",
                          tuned_candidate_id=winner["candidate_id"], tuned_parameters=winner["parameters"],
                          tuned_search_objective=winner["objective"], tuned_n_trials=winner["n_trials"])
            config = inject_parameters(tuning_base(model, "4d"), model, winner["parameters"], seed=seed)
            if model in CPU_MODELS:
                config["training"]["device"] = "cpu"
        else:
            config_base = yaml.safe_load(Path(config_file_for(model, "4d")).read_text())
            costs = json.loads(client.get_object(Bucket=bucket, Key=f"checkpoints-oct2026/4d/{model}/seed-42/"
                                                 f"powder_600_800_t4t6_run_costs.json")["Body"].read())
            config = build_eval_config(config_base, model, costs, test_start="2026-10-27T20:00:00Z", test_end="2026-11-03T20:00:00Z",
                                       device="cpu" if model in CPU_MODELS else None)
        map_name = f"p15_{band}_{variant['name']}"
        config = apply_powder15_4d(config, band, list(HORIZONS), variant["nodes"], job["grid"], map_name)
        config = set_lookback_and_horizon(config, model, max(HORIZONS) if args.get("fixed_config") else None)      # a direct multi-step model emits 48 steps
        if args.get("smoke_epochs"):          # a few minutes end to end (training, map scoring, all five physical-receiver evaluations)
            section = config[model]
            (section["train"] if "train" in section else section)["epochs"] = int(args["smoke_epochs"])
            result["smoke_epochs"] = int(args["smoke_epochs"])
        if "train" in config[model]:
            config[model]["train"]["seed"] = seed          # STS-PredNet's flat section already carries it (inject_parameters)
        with tempfile.TemporaryDirectory(prefix="spatial15-") as scratch:
            scratch = Path(scratch)
            run_dir, eval_dir = scratch / "run", scratch / "eval"
            train_one_model(config, model, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"), None)
            eval_dir.mkdir()
            evaluate_one_model(config, model, eval_dir, checkpoint, skip_plots=True)
            result.update({f"map__{k}": v for k, v in summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv")
                                                                        .to_dict("records")).items()})
            if checkpoint is not None:
                result.update(save_run(client, bucket, f"{base}/weights", run_dir, config,
                                       {"result_kind": "spatial15", "variant": variant, "seed": seed}))
            result["receivers"] = physical_scores(client, bucket, base, root, args, band, variant, job["grid"], model, seed,
                                                  checkpoint, config, scratch)
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1500:])
    result["wall_seconds"] = round(time.time() - started, 1)
    client.put_object(Bucket=bucket, Key=f"{base}/result.json", Body=json.dumps(result, default=str).encode())
    return result


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="600_800", choices=("600_800", "2400_2600"))
    p.add_argument("--experiments", nargs="+", default=["s2"], choices=("s2", "s3", "s5", "solo"))
    p.add_argument("--variants", nargs="+", help="only these variant names")
    p.add_argument("--models", nargs="+", default=["lookbackmean4d", "linearar4d", "residuallinearar4d", "residualconvlstm"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=24.0)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--gpu-fraction", type=float, default=0.5, help="GPU share of each neural cell")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--tuned", action="store_true",
                   help="train with each model's tuned 15-minute 4D winner (tuning/powder15_4d) instead of the inherited 1-minute settings")
    p.add_argument("--fixed-config", help="JSON of a fixed configuration (no search), e.g. training/configs/fixed/dswinlstm_i_inherited_1min_r2.json")
    p.add_argument("--tuning-prefix", default="tuning")
    p.add_argument("--permuted", type=int, default=0,
                   help="S4: score the frozen base models (tuning finals) with this many permutations of the four core receivers' coordinates")
    p.add_argument("--eval-only", action="store_true", help="score saved weights at the physical receivers; no training")
    p.add_argument("--eval-from-tuning", action="store_true",
                   help="with --eval-only: the base variant core4-p2 is scored from the tuning finals' weights (same tuned model, same seed)")
    p.add_argument("--smoke-epochs", type=int, default=0, help="override the epoch count for a quick end-to-end test (use another --prefix)")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="spatial15-oct2026")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    frame = six_receiver_frame(args.repo_root)
    chosen = [v for v in variants(args.experiments) if not args.variants or v["name"] in args.variants]
    jobs = [{"variant": v, "grid": grid_for(v, frame), "model": m, "seed": s}
            for v in chosen for m in args.models for s in ([42] if m == "lookbackmean4d" else args.seeds)]
    if args.permuted:
        from training.ray.spatial_ablation import unique_permutation_seeds

        base_variant = {"name": "core4-p2", "nodes": list(CORE), "grid": {}}
        perms = unique_permutation_seeds(args.permuted, receiver_count=4)
        jobs = [{"variant": base_variant, "grid": grid_for(base_variant, frame), "model": m, "seed": s, "permutation_seed": ps}
                for m in args.models for s in ([42] if m == "lookbackmean4d" else args.seeds) for ps in perms]
    for job in jobs:                                   # the frozen variogram of this band goes into every kriging grid
        if job["grid"].get("method") == "kriging":
            job["grid"] = {**job["grid"], "variogram": load_variogram(args.repo_root, args.band)}
    if args.skip_done:
        def done(job):
            key = (f"{args.prefix.strip('/')}/{args.band}/permuted/{job['model']}/seed-{job['seed']}/p{job['permutation_seed']}/result.json"
                   if args.permuted else
                   f"{args.prefix.strip('/')}/{args.band}/{job['variant']['name']}/{job['model']}/seed-{job['seed']}/result.json")
            try:
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    print(f"[spatial15] {len(jobs)} runs over {len(chosen)} variants; grid frame {frame}", flush=True)
    shared = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "band": args.band, "batch_size": args.batch_size,
              "tuned": args.tuned, "fixed_config": args.fixed_config, "tuning_prefix": args.tuning_prefix, "smoke_epochs": args.smoke_epochs,
              "eval_from_tuning": args.eval_from_tuning}
    ray.init(address=args.address, ignore_reinit_error=True)
    memory = int(args.memory_gb * 2**30)
    worker = permuted_job if args.permuted else (eval_only_job if args.eval_only else run_job)
    cpu_task = ray.remote(num_cpus=4, num_gpus=0, memory=memory, max_retries=2)(worker)
    gpu_task = ray.remote(num_cpus=4, num_gpus=args.gpu_fraction, memory=memory, max_retries=2)(worker)
    queue, pending, ok = list(jobs), [], 0
    while queue or pending:
        while queue and len(pending) < args.max_parallel:
            job = queue.pop(0)
            pending.append((cpu_task if job["model"] in CPU_MODELS else gpu_task).remote(job, shared))
        done_refs, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(done_refs[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        receivers = {k: round(v["mean_horizon_mae_db"], 3) for k, v in (row.get("receivers") or {}).items()
                     if v.get("mean_horizon_mae_db") is not None}
        print(f"[spatial15] {row.get('variant')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} "
              f"map={row.get('map__test_mean_horizon_mae_db')} perm_mae={row.get('mean_horizon_mae_db')} receivers={receivers} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[spatial15] done: {ok} of {len(jobs)} ok", flush=True)


if __name__ == "__main__":
    main()
