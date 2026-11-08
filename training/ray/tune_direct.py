"""Equal-budget tuning of direct multi-step models on POWDER (1-minute and 15-minute), same protocol as our other core models.

Models: ``ofa``, ``timellm`` (Time-LLM, GPT-2), ``patchtst``, ``dlinear`` and ``residualnonlinear2d`` (ResidualLinearAR plus a nonlinear
MLP correction head). Each is a *direct* model trained for one horizon at a time, channel by channel, on the pooled four core
receivers, using the repo's training and evaluation pipeline (so the metrics, split and windows are exactly those of our other models).

Protocol: 16 candidates per (resolution, band, model) -- the model's default settings plus 15 seeded random draws -- each trained with
seed 42 and scored on the *validation* split at a short and a long horizon (15-minute: 4 and 48 steps; 1-minute: 1 and 60); the candidate
with the lowest mean validation objective wins (written to ``winners.json`` before any test-scored run). The winner is then trained for
every horizon with seeds 40-44 and scored on the final test range 2026-10-27T20:00Z to 2026-11-03T20:00Z (MAE, MSE, RMSE and both MAPEs,
per horizon). Weights and the exact config are uploaded with every final. Test data is never used for selection.

Disclosed deviations (set by the user): training windows are thinned by ``--train-stride`` (default: Time-LLM 2 at 15 minutes; every model
15 at 1 minute, where consecutive windows overlap in 59 of 60 minutes), and 1-minute test windows are thinned by ``--test-stride-1min``.

    python -m training.ray.tune_direct --resolution 15min --models ofa dlinear patchtst residualnonlinear2d timellm
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

SEARCH_SEED, BUDGET = 42, 16
FINAL_SEEDS = (40, 41, 42, 43, 44)
TUNE_HORIZONS = {"15min": (4, 48), "1min": (1, 60)}
FINAL_HORIZONS = {"15min": (4, 12, 24, 48), "1min": (1, 15, 60)}
BANDS = ("600_800", "2400_2600")
PIPELINE_NAME = {"ofa": "ofa", "timellm": "spectrumllm", "patchtst": "patchtst", "dlinear": "dlinear",
                 "residualnonlinear2d": "residualnonlinear2d", "ofa_rand": "ofa", "ofa_scratch": "ofa"}
# Controls for what the pretrained language model contributes: the same OFA architecture, tuned with the same 16 candidates.
#   ofa_rand    random initialization, only layer norms and positional embeddings train (OFA's rule, so the frozen rest is random)
#   ofa_scratch random initialization, every backbone weight trains (a transformer trained from scratch, like PatchTST)
CONTROLS = {"ofa_rand": {"backbone_init": "random", "backbone_trainable": "ofa"},
            "ofa_scratch": {"backbone_init": "random", "backbone_trainable": "all"}}
OFA_FAMILY = ("ofa", "ofa_rand", "ofa_scratch")


def family(model: str) -> str:
    """The model whose settings and candidate list ``model`` shares (the controls use OFA's)."""
    return "ofa" if model in OFA_FAMILY else model
# (CPUs, GPU share). Time-LLM peaks near 45-50 GB at the pipeline batch size, so it needs a share of 0.6 of a 93 GB H100; each task is also
# capped to its share (torch memory fraction) so a task that outgrows it fails alone instead of starving its neighbours.
RESOURCES = {"ofa": (4, 0.25), "ofa_rand": (4, 0.25), "ofa_scratch": (4, 0.25), "timellm": (4, 0.6), "patchtst": (4, 0.25), "dlinear": (2, 0.125), "residualnonlinear2d": (2, 0.125)}
EPOCHS = {"ofa": 10, "ofa_rand": 10, "ofa_scratch": 10, "timellm": 10, "patchtst": 20, "dlinear": 20, "residualnonlinear2d": 10}
PATIENCE = 5
MODEL_KEYS = {"patch_length", "patch_stride", "d_model", "d_ff", "llm_layers", "n_heads", "e_layers", "dropout", "moving_avg",
              "mlp_hidden", "mlp_layers", "mlp_dropout", "ridge_alpha"}
TRAIN_KEYS = {"learning_rate": "learning_rate", "batch_size": "batch_size", "weight_decay": "weight_decay"}

DEFAULTS: dict[str, dict[str, Any]] = {
    "ofa": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "ofa_rand": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "ofa_scratch": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "timellm": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8, "d_model": 32, "d_ff": 32},
    "patchtst": {"learning_rate": 5e-4, "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256, "dropout": 0.1,
                 "patch_length": 16, "patch_stride": 8, "batch_size": 16},
    "dlinear": {"learning_rate": 1e-3, "moving_avg": 25, "batch_size": 32, "weight_decay": 0.0},
    "residualnonlinear2d": {"learning_rate": 1e-3, "ridge_alpha": 0.01, "mlp_hidden": 64, "mlp_layers": 2, "mlp_dropout": 0.0,
                            "batch_size": 64},
}


def _loguniform(rng: random.Random, low: float, high: float) -> float:
    return round(math.exp(rng.uniform(math.log(low), math.log(high))), 6)


def draw(model: str, rng: random.Random) -> dict[str, Any]:
    """One random candidate for ``model`` (the knobs each model exposes)."""
    patch = rng.choice([8, 16, 32])
    llm = {"learning_rate": _loguniform(rng, 5e-5, 1e-3), "llm_layers": rng.choice([3, 6]), "patch_length": patch,
           "patch_stride": rng.choice([patch // 2, patch])}
    if model in OFA_FAMILY:
        return llm
    if model == "timellm":
        return {**llm, "d_model": rng.choice([16, 32, 64]), "d_ff": rng.choice([32, 64, 128])}
    if model == "patchtst":
        patch = rng.choice([8, 16])
        return {"learning_rate": _loguniform(rng, 1e-4, 2e-3), "d_model": rng.choice([64, 128, 256]), "n_heads": rng.choice([4, 8]),
                "e_layers": rng.choice([1, 2, 3]), "d_ff": rng.choice([128, 256, 512]), "dropout": rng.choice([0.0, 0.1, 0.2]),
                "patch_length": patch, "patch_stride": rng.choice([patch // 2, patch]), "batch_size": rng.choice([8, 16, 32])}
    if model == "dlinear":
        return {"learning_rate": _loguniform(rng, 1e-4, 1e-2), "moving_avg": rng.choice([13, 25, 49]),
                "batch_size": rng.choice([16, 32, 64]), "weight_decay": rng.choice([0.0, 1e-5, 1e-4])}
    if model == "residualnonlinear2d":
        return {"learning_rate": _loguniform(rng, 1e-4, 1e-2), "ridge_alpha": rng.choice([1e-3, 1e-2, 1e-1]),
                "mlp_hidden": rng.choice([32, 64, 128]), "mlp_layers": rng.choice([1, 2]), "mlp_dropout": rng.choice([0.0, 0.1]),
                "batch_size": rng.choice([32, 64, 128])}
    raise ValueError(f"unknown model {model!r}")


def candidate_list(model: str, budget: int = BUDGET, seed: int = SEARCH_SEED) -> list[dict[str, Any]]:
    """Candidate 0 is the model's default; the rest are distinct seeded draws (same list for every resolution and band)."""
    rng = random.Random(f"{seed}-{family(model)}")                    # the controls draw exactly OFA's candidates
    out = [dict(DEFAULTS[model])]
    seen = {json.dumps(out[0], sort_keys=True)}
    attempts = 0
    while len(out) < budget and attempts < budget * 100:
        attempts += 1
        candidate = draw(model, rng)
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def default_train_stride(model: str, resolution: str) -> int:
    """Disclosed thinning of the training windows: Time-LLM 2 at 15 minutes; 15 for every model at 1 minute."""
    if resolution == "1min":
        return 15
    return 2 if model == "timellm" else 1


def apply_params(config: dict[str, Any], model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    section = config[model_name]
    for key, value in params.items():
        if key in MODEL_KEYS:
            section["model"][key] = value
        elif key in TRAIN_KEYS:
            section["train"][TRAIN_KEYS[key]] = value
        else:
            raise ValueError(f"unknown parameter {key!r} for {model_name}")
    return config


def template(model: str, repo_root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(model section, train section) defaults for ``model``."""
    from training.ray.llm_powder import ARMS

    if family(model) in ARMS:
        name, path = ARMS[family(model)]
        arm = yaml.safe_load((Path(repo_root) / path).read_text())[name]
        return copy.deepcopy(arm["model"]), copy.deepcopy(arm["train"])
    train = {"val_fraction": 0.1, "weight_decay": 0.0, "optimizer": "adam", "gradient_clip_norm": 1.0, "batch_size": 32,
             "learning_rate": 1e-3}
    return {}, train


def build_config(model: str, resolution: str, band: str, horizon: int, seed: int, params: dict[str, Any], repo_root: str, *,
                 train_stride: int, test_stride: int, device: str | None = None) -> tuple[dict[str, Any], str]:
    """Pipeline config for one run of ``model``; returns (config, pipeline model name)."""
    from training.ray.llm_powder import BASE_CONFIG, FINAL_TEST, RESOLUTIONS

    name, spec = PIPELINE_NAME[model], RESOLUTIONS[resolution]
    config = yaml.safe_load((Path(repo_root) / BASE_CONFIG).read_text())
    model_section, train_section = template(model, repo_root)
    config[name] = {"model": model_section, "train": train_section}
    config["training"] = {"models": [name], "device": device or "auto"}
    config["data"]["split"]["ranges"]["test"] = dict(FINAL_TEST)
    if resolution == "15min":
        from training.ray.powder15 import apply_powder15

        config = apply_powder15(config, band, repo_root, [horizon])
    else:
        for entry in config["data"]["files"]:
            entry["path"] = entry["path"].replace("/600_800/", f"/{band}/")
        config["data"]["chunks"] = [{"id": f"powder_{band}", "start_mhz": float(band.split("_")[0]), "end_mhz": float(band.split("_")[1])}]
        config["windowing"] = {"lookback": spec["lookback"], "horizons": [int(horizon)]}
    config[name]["model"].update(input_sequence_length=spec["lookback"], prediction_horizon=int(horizon))
    if model in OFA_FAMILY or model == "timellm":
        config[name]["model"]["channel_independent"] = True
        config[name]["model"].update(CONTROLS.get(model, {}))
    config[name]["train"].update(train_stride=int(train_stride), val_stride=int(train_stride), test_stride=int(test_stride),
                                 epochs=EPOCHS[model], seed=int(seed), early_stopping=True, early_stopping_patience=PATIENCE)
    if model in OFA_FAMILY or model == "timellm":
        config[name]["train"]["batch_size"] = 4                          # the LLM runs' memory-safe default (llm_powder)
    return apply_params(config, name, params), name


def pick_winner(scores: dict[int, dict[int, float]], horizons) -> tuple[int, float] | None:
    """scores[candidate][horizon] -> (candidate, mean validation objective) of the best candidate scored at every tuning horizon."""
    complete = {c: sum(v[h] for h in horizons) / len(horizons)
                for c, v in scores.items() if all(h in v and math.isfinite(v[h]) for h in horizons)}
    if not complete:
        return None
    best = min(complete, key=lambda c: (complete[c], c))
    return best, complete[best]


def group(prefix: str, resolution: str, band: str, model: str) -> str:
    return f"{prefix.strip('/')}/{resolution}/{band}/{model}"


def trial_key(base: str, index: int, horizon: int) -> str:
    return f"{base}/trials/c{index:02d}_h{horizon}.json"


def final_key(base: str, horizon: int, seed: int) -> str:
    return f"{base}/finals/h{horizon}/seed-{seed}.json"


def job(spec: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one search trial or one final run of one model."""
    root = spec["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    import pandas as pd

    from training.common.evaluation_integrated import evaluate_one_model
    from training.ray.artifacts import save_artifact
    from training.ray.final_eval import summarize_metrics
    from training.ray.llm_powder import use_isolated_libs
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_fixed import train_prepared

    use_isolated_libs()
    import torch

    client, bucket = spectrum_s3_client(), spec["bucket"]
    model, resolution, band, horizon, seed = spec["model"], spec["resolution"], spec["band"], int(spec["horizon"]), int(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(0.95, RESOURCES[model][1]))
        torch.cuda.reset_peak_memory_stats()
    base = group(spec["output_prefix"], resolution, band, model)
    record: dict[str, Any] = {"kind": spec["kind"], "model": model, "resolution": resolution, "band": band, "horizon": horizon,
                              "seed": seed, "index": spec.get("index"), "params": spec["params"],
                              "train_stride": spec["train_stride"], "test_stride": spec["test_stride"]}
    started = time.time()
    try:
        if resolution == "15min":
            from training.ray.powder15 import ensure_powder15

            ensure_powder15(band, root)
        config, name = build_config(model, resolution, band, horizon, seed, spec["params"], root, train_stride=spec["train_stride"],
                                    test_stride=spec["test_stride"])
        with tempfile.TemporaryDirectory(prefix="tune-direct-") as scratch:
            result = train_prepared(config, name, Path(scratch) / "run")
            record.update(objective=result["objective"], objective_name=result["objective_name"])
            if spec["kind"] == "final":
                from training.common.run_costs import read_run_costs

                eval_dir = Path(scratch) / "eval"
                eval_dir.mkdir()
                evaluate_one_model(config, name, eval_dir, result["checkpoint"], skip_plots=True)
                record.update(summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")))
                import io

                from training.ray.final_eval import per_bin_test_rows

                table = per_bin_test_rows(pd.read_csv(eval_dir / "per_frequency_metrics.csv"),
                                          {"model": model, "resolution": resolution, "band": band, "trained_horizon": horizon, "seed": seed})
                buffer = io.StringIO()
                table.to_csv(buffer, index=False)
                record["per_bin_key"] = f"{base}/per-bin/h{horizon}/seed-{seed}.csv"
                client.put_object(Bucket=bucket, Key=record["per_bin_key"], Body=buffer.getvalue().encode())
                record["run_costs"] = read_run_costs(Path(scratch) / "run" / "integrated")
                record.update(save_artifact(client, bucket, f"{base}/weights/h{horizon}/seed-{seed}", result["checkpoint"],
                                            {"config": config, "params": spec["params"], "seed": seed, "horizon": horizon, "model": model,
                                             "resolution": resolution, "band": band, "kind": "tuned_final"}, repo_root=root))
        if torch.cuda.is_available():
            record["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        record["ok"] = math.isfinite(float(record["objective"]))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-800:])
    record["wall_seconds"] = round(time.time() - started, 1)
    key = trial_key(base, int(spec["index"]), horizon) if spec["kind"] == "trial" else final_key(base, horizon, seed)
    try:
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def _done(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        record = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        return record if record.get("ok") else None
    except Exception:
        return None


def _run_all(ray, specs: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    pending, results = [], []
    for s in specs:
        cpus, gpus = RESOURCES[s["model"]]
        pending.append(ray.remote(num_cpus=cpus, num_gpus=gpus, max_retries=1)(job).remote(s))
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        results.append(r)
        print(f"[tune-direct] {label} {r['resolution']} {r['band']} {r['model']} c{r.get('index')} h{r['horizon']} seed{r['seed']} "
              f"ok={r['ok']} obj={r.get('objective')} {r.get('error', '')} ({r['wall_seconds']}s) [{len(pending)} left]", flush=True)
    return results


def main(argv=None) -> None:
    import ray

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", choices=("15min", "1min"), required=True)
    p.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    p.add_argument("--models", nargs="+", default=list(PIPELINE_NAME), choices=list(PIPELINE_NAME))
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    p.add_argument("--horizons", type=int, nargs="+", help="final horizons (default: all of the resolution's)")
    p.add_argument("--train-stride", action="append", default=[], metavar="MODEL=N", help="override a model's training-window stride")
    p.add_argument("--test-stride-1min", type=int, default=15, help="thinning of the 1-minute test windows")
    p.add_argument("--trials-only", action="store_true")
    p.add_argument("--max-retries", type=int, default=3, help="Ray retries for a task whose node died (a lease ending mid-run)")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="direct-tuning")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    client = spectrum_s3_client()
    overrides = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.train_stride}
    test_stride = 1 if args.resolution == "15min" else args.test_stride_1min
    tune_horizons = TUNE_HORIZONS[args.resolution]
    final_horizons = tuple(args.horizons) if args.horizons else FINAL_HORIZONS[args.resolution]
    base_spec = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix, "resolution": args.resolution,
                 "test_stride": test_stride}

    def stride(model: str) -> int:
        return overrides.get(model, default_train_stride(model, args.resolution))

    # Pipelined: every (model, band) group goes trials -> winner -> finals on its own, so a deadline that arrives mid-run still leaves
    # complete results for the groups that finished (an earlier version waited for ALL trials before starting any final).
    scores: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    pending: dict[Any, tuple[str, tuple[str, str]]] = {}
    trials_left: collections.Counter = collections.Counter()
    finals_total = 0

    def submit(spec: dict[str, Any]):
        cpus, gpus = RESOURCES[spec["model"]]
        return ray.remote(num_cpus=cpus, num_gpus=gpus, max_retries=args.max_retries)(job).remote(spec)

    def finalize(model: str, band: str) -> None:
        nonlocal finals_total
        table = scores.get((model, band), {})
        picked = pick_winner(table, tune_horizons)
        if picked is None:
            print(f"[tune-direct] no complete candidate for {model} {band}", flush=True)
            return
        index, value = picked
        record = {"index": index, "mean_val_objective": value, "params": candidate_list(model, args.budget)[index],
                  "default_mean_val_objective": (sum(table[0][h] for h in tune_horizons) / len(tune_horizons)
                                                 if all(h in table.get(0, {}) for h in tune_horizons) else None),
                  "tune_horizons": list(tune_horizons), "budget": args.budget}
        base = group(args.output_prefix, args.resolution, band, model)
        client.put_object(Bucket=bucket, Key=f"{base}/winners.json", Body=json.dumps(record, indent=1).encode())
        launched = 0
        for horizon in final_horizons:
            for seed in args.seeds:
                if not _done(client, bucket, final_key(base, horizon, seed)):
                    pending[submit({**base_spec, "kind": "final", "model": model, "band": band, "index": index, "horizon": horizon,
                                    "seed": seed, "params": record["params"], "train_stride": stride(model)})] = ("final", (model, band))
                    launched += 1
        finals_total += launched
        print(f"[tune-direct] winner {model}/{band}: candidate {index}; {launched} finals launched", flush=True)

    trials_total = 0
    for model in args.models:
        for band in args.bands:
            base = group(args.output_prefix, args.resolution, band, model)
            for index, params in enumerate(candidate_list(model, args.budget)):
                for horizon in tune_horizons:
                    record = _done(client, bucket, trial_key(base, index, horizon))
                    if record:
                        scores.setdefault((model, band), {}).setdefault(index, {})[horizon] = float(record["objective"])
                    else:
                        pending[submit({**base_spec, "kind": "trial", "model": model, "band": band, "index": index, "horizon": horizon,
                                        "seed": SEARCH_SEED, "params": params, "train_stride": stride(model)})] = ("trial", (model, band))
                        trials_left[(model, band)] += 1
                        trials_total += 1
    print(f"[tune-direct] {trials_total} trials to run (training stride {({m: stride(m) for m in args.models})}), pipelined per group",
          flush=True)
    if not args.trials_only:
        for model in args.models:
            for band in args.bands:
                if trials_left[(model, band)] == 0:
                    finalize(model, band)                     # nothing to wait for: its trials were all done already
    while pending:
        done_refs, _ = ray.wait(list(pending), num_returns=1)
        ref = done_refs[0]
        kind, (model, band) = pending.pop(ref)
        try:
            r = ray.get(ref)
        except Exception as error:                            # a task lost to a node dying more often than max_retries allows
            print(f"[tune-direct] {kind} {args.resolution} {band} {model} LOST: {type(error).__name__}: {str(error)[:120]}", flush=True)
            r = None
        if r is not None:
            print(f"[tune-direct] {kind} {r['resolution']} {r['band']} {r['model']} c{r.get('index')} h{r['horizon']} seed{r['seed']} "
                  f"ok={r['ok']} obj={r.get('objective')} {r.get('error', '')} ({r['wall_seconds']}s) [{len(pending)} left]", flush=True)
            if kind == "trial" and r["ok"]:
                scores.setdefault((model, band), {}).setdefault(r["index"], {})[r["horizon"]] = float(r["objective"])
        if kind == "trial":
            trials_left[(model, band)] -= 1
            if trials_left[(model, band)] == 0 and not args.trials_only:
                finalize(model, band)
    print(f"[tune-direct] done ({finals_total} finals launched)", flush=True)

if __name__ == "__main__":
    main()
