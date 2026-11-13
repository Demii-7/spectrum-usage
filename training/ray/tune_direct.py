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
TUNE_HORIZONS = {"15min": (4, 48), "1min": (1, 60), "electrosense": (12, 192)}
FINAL_HORIZONS = {"15min": (4, 12, 24, 48), "1min": (1, 15, 60), "electrosense": (12, 24, 48, 96, 192)}
ES_DATASETS = ("spec4", "spec25", "spec151")
BANDS = ("600_800", "2400_2600")
PIPELINE_NAME = {"ofa": "ofa", "timellm": "spectrumllm", "patchtst": "patchtst", "dlinear": "dlinear",
                 "residualnonlinear2d": "residualnonlinear2d", "ofa_rand": "ofa", "ofa_scratch": "ofa",
                 # simple baselines and PatchTST ablations (cheap tests of what makes OFA and PatchTST work)
                 "residuallinearar2d": "residuallinearar2d", "linearar2d": "linearar2d", "nlinear": "nlinear", "revinmlp": "revinmlp",
                 "revinmlp_mix": "revinmlp", "revinmlp_nbr": "revinmlp", "itransformer": "itransformer",
                 "patchtst_noinorm": "patchtst", "patchtst_joint": "patchtst", "patchtst_nopatch": "patchtst",
                 # TimeRAN (MOMENT continued-pretrained on RAN telemetry) and its controls; all MOMENT-1-small, 512-step padded context
                 "timeran_lp": "timeran", "timeran_ff": "timeran", "moment_lp": "timeran", "moment_ff": "timeran", "moment_rand": "timeran",
                 # 3D-SwinSTB (DeepSPred) on colour frames of the same windows; colours are inverted to dB for validation and scoring
                 "deepspred2d": "deepspred2d"}
# Controls for what the pretrained language model contributes: the same OFA architecture, tuned with the same 16 candidates.
#   ofa_rand    random initialization, only layer norms and positional embeddings train (OFA's rule, so the frozen rest is random)
#   ofa_scratch random initialization, every backbone weight trains (a transformer trained from scratch, like PatchTST)
# PatchTST ablations, each with PatchTST's own 16 candidates and exactly one ingredient removed:
#   patchtst_noinorm  no per-window normalization
#   patchtst_joint    every patch token spans all channels (instead of one shared model per channel)
#   patchtst_nopatch  one token per time step (patch length 1, stride 1)
CONTROLS = {"revinmlp_nbr": {"neighbor_inputs": True, "n_receivers": 4},   # each (receiver, bin) series also sees the same bin at the other three core receivers
            "revinmlp_mix": {"channel_mixing": True},      # RevIN-MLP plus one residual linear mixing across the bins
            "ofa_rand": {"backbone_init": "random", "backbone_trainable": "ofa"},
            "ofa_scratch": {"backbone_init": "random", "backbone_trainable": "all"},
            "patchtst_noinorm": {"instance_norm": False},
            "patchtst_joint": {"channel_mode": "joint"},
            "patchtst_nopatch": {"patch_length": 1, "patch_stride": 1},
            # TimeRAN's two regimes (linear probing = head only, full fine-tuning) on the TimeRAN backbone, the same on plain MOMENT (what
            # the RAN pretraining adds), and MOMENT's architecture from random initialization trained fully (what any pretraining adds)
            "timeran_lp": {"backbone_init": "timeran", "backbone_trainable": "head"},
            "timeran_ff": {"backbone_init": "timeran", "backbone_trainable": "all"},
            "moment_lp": {"backbone_init": "moment", "backbone_trainable": "head"},
            "moment_ff": {"backbone_init": "moment", "backbone_trainable": "all"},
            "moment_rand": {"backbone_init": "random", "backbone_trainable": "all"}}
OFA_FAMILY = ("ofa", "ofa_rand", "ofa_scratch")
PATCHTST_FAMILY = ("patchtst", "patchtst_noinorm", "patchtst_joint", "patchtst_nopatch")
LINEAR_FAMILY = ("residuallinearar2d", "linearar2d")
TIMERAN_FAMILY = ("timeran_lp", "timeran_ff", "moment_lp", "moment_ff", "moment_rand")
TIMERAN_SIZE = "small"
TIMERAN_WEIGHTS_KEY = f"pretrained/timeran/TimeRAN_{TIMERAN_SIZE}.pth"           # in the source bucket (copied from the authors' release)
TIMERAN_LOCAL = f"/workspace/pretrained/timeran/TimeRAN_{TIMERAN_SIZE}.pth"     # node-local copy


def family(model: str) -> str:
    """The model whose settings and candidate list ``model`` shares (each control uses its parent's)."""
    if model in OFA_FAMILY:
        return "ofa"
    if model in TIMERAN_FAMILY:
        return "timeran"
    if model in ("revinmlp_mix", "revinmlp_nbr"):
        return "revinmlp"
    return "patchtst" if model in PATCHTST_FAMILY else model
# (CPUs, GPU share). Time-LLM peaks near 45-50 GB at the pipeline batch size, so it needs a share of 0.6 of a 93 GB H100; each task is also
# capped to its share (torch memory fraction) so a task that outgrows it fails alone instead of starving its neighbours.
RESOURCES = {"ofa": (4, 0.25), "ofa_rand": (4, 0.25), "ofa_scratch": (4, 0.25), "timellm": (4, 0.6), "patchtst": (4, 0.25), "dlinear": (2, 0.125),
             "residualnonlinear2d": (2, 0.125), "residuallinearar2d": (2, 0.125), "linearar2d": (2, 0.125), "nlinear": (2, 0.125),
             "revinmlp": (2, 0.125), "revinmlp_mix": (2, 0.125), "revinmlp_nbr": (2, 0.125), "itransformer": (2, 0.125), "patchtst_noinorm": (4, 0.25), "patchtst_joint": (4, 0.25), "patchtst_nopatch": (4, 0.25),
             **{m: (4, 1.0) for m in ("timeran_lp", "timeran_ff", "moment_lp", "moment_ff", "moment_rand")},  # one 24 GB RTX 6000 each
             "deepspred2d": (4, 0.25)}                                        # small frames: a quarter GPU (capped to that share by torch)
EPOCHS = {"ofa": 10, "ofa_rand": 10, "ofa_scratch": 10, "timellm": 10, "patchtst": 20, "dlinear": 20, "residualnonlinear2d": 10,
          "residuallinearar2d": 10, "linearar2d": 10, "nlinear": 20, "revinmlp": 20, "revinmlp_mix": 20, "revinmlp_nbr": 20, "itransformer": 20, "patchtst_noinorm": 20, "patchtst_joint": 20,
          "patchtst_nopatch": 20, "deepspred2d": 20, **{m: 10 for m in ("timeran_lp", "timeran_ff", "moment_lp", "moment_ff", "moment_rand")}}
PATIENCE = 5
MODEL_KEYS = {"patch_length", "patch_stride", "d_model", "d_ff", "llm_layers", "n_heads", "e_layers", "dropout", "moving_avg",
              "mlp_hidden", "mlp_layers", "mlp_dropout", "ridge_alpha", "head_dropout", "channel_mixing",
              # deepspred2d: architecture plus the two image-construction choices (colormap and frame geometry); the rendering's
              # normalization is fixed (the pipeline's per-frequency z-score), and selection is on validation MAE in dB
              "embed_dim", "depths", "num_heads", "window_size", "drop_path_rate", "colormap", "input_frames"}
TRAIN_KEYS = {"learning_rate": "learning_rate", "batch_size": "batch_size", "weight_decay": "weight_decay"}

DEFAULTS: dict[str, dict[str, Any]] = {
    "ofa": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "ofa_rand": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "ofa_scratch": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8},
    "timellm": {"learning_rate": 1e-4, "llm_layers": 6, "patch_length": 16, "patch_stride": 8, "d_model": 32, "d_ff": 32},
    "patchtst": {"learning_rate": 5e-4, "d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256, "dropout": 0.1,
                 "patch_length": 16, "patch_stride": 8, "batch_size": 16},
    "dlinear": {"learning_rate": 1e-3, "moving_avg": 25, "batch_size": 32, "weight_decay": 0.0},
    "residuallinearar2d": {"learning_rate": 1e-3, "ridge_alpha": 0.01, "batch_size": 64},
    "linearar2d": {"learning_rate": 1e-3, "ridge_alpha": 0.01, "batch_size": 64},
    "nlinear": {"learning_rate": 1e-3, "batch_size": 32, "weight_decay": 0.0},
    "revinmlp": {"learning_rate": 5e-4, "mlp_hidden": 128, "mlp_layers": 2, "mlp_dropout": 0.1, "batch_size": 32},
    "itransformer": {"learning_rate": 5e-4, "d_model": 128, "n_heads": 8, "e_layers": 2, "d_ff": 256, "dropout": 0.1, "batch_size": 16},
    "residualnonlinear2d": {"learning_rate": 1e-3, "ridge_alpha": 0.01, "mlp_hidden": 64, "mlp_layers": 2, "mlp_dropout": 0.0,
                            "batch_size": 64},
}
# The paper's 3D-SwinSTB settings (C 96, depths 2-4-2, heads 4-8-16, window 2-7-7, lr 1e-3) as the anchor; T = 4 frames per lookback.
DEFAULTS["deepspred2d"] = {"learning_rate": 1e-3, "embed_dim": 96, "depths": [2, 4, 2], "num_heads": [4, 8, 16], "window_size": [2, 7, 7],
                           "drop_path_rate": 0.0, "colormap": "jet", "input_frames": 4, "batch_size": 8, "weight_decay": 0.0}
for _variant in ("timeran_lp", "timeran_ff", "moment_lp", "moment_ff", "moment_rand"):
    DEFAULTS[_variant] = {"learning_rate": 1e-4, "batch_size": 16, "head_dropout": 0.1, "weight_decay": 0.0}
DEFAULTS["revinmlp_mix"] = dict(DEFAULTS["revinmlp"])
DEFAULTS["revinmlp_nbr"] = dict(DEFAULTS["revinmlp"])
for _control in ("patchtst_noinorm", "patchtst_joint", "patchtst_nopatch"):
    DEFAULTS[_control] = dict(DEFAULTS["patchtst"])
    if _control == "patchtst_nopatch":
        DEFAULTS[_control].update(patch_length=1, patch_stride=1)


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
    if model in PATCHTST_FAMILY:
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
    if model in LINEAR_FAMILY:                                       # the registry's space for the linear models (training/ray/registry.py)
        return {"learning_rate": _loguniform(rng, 1e-5, 1e-2), "ridge_alpha": rng.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
                "batch_size": rng.choice([32, 64, 128])}
    if model == "nlinear":
        return {"learning_rate": _loguniform(rng, 1e-4, 1e-2), "batch_size": rng.choice([16, 32, 64]),
                "weight_decay": rng.choice([0.0, 1e-5, 1e-4])}
    if model in TIMERAN_FAMILY:
        # batch 16 already uses 20.6 GB of a 24 GB RTX 6000 under full fine-tuning (smoke test 2026-11-09), so 32 is not offered
        return {"learning_rate": _loguniform(rng, 1e-5, 1e-3), "batch_size": rng.choice([8, 16]),
                "head_dropout": rng.choice([0.0, 0.1, 0.2]), "weight_decay": rng.choice([0.0, 1e-5, 1e-4])}
    if model == "itransformer":
        return {"learning_rate": _loguniform(rng, 1e-4, 2e-3), "d_model": rng.choice([64, 128, 256]), "n_heads": rng.choice([4, 8]),
                "e_layers": rng.choice([1, 2, 3]), "d_ff": rng.choice([128, 256, 512]), "dropout": rng.choice([0.0, 0.1, 0.2]),
                "batch_size": rng.choice([8, 16, 32])}
    if model == "deepspred2d":
        size = rng.choice([(32, [2, 2, 2]), (48, [2, 2, 2]), (48, [2, 4, 2]), (96, [2, 2, 2]), (96, [2, 4, 2])])
        return {"learning_rate": _loguniform(rng, 1e-4, 2e-3), "embed_dim": size[0], "depths": size[1], "num_heads": [4, 8, 16],
                "window_size": rng.choice([[2, 4, 4], [2, 7, 7], [2, 8, 8]]), "drop_path_rate": rng.choice([0.0, 0.1, 0.2]),
                "colormap": "jet", "input_frames": rng.choice([1, 2, 3, 4]), "batch_size": rng.choice([4, 8, 16]),
                "weight_decay": rng.choice([0.0, 1e-5, 1e-4])}
    if model in ("revinmlp", "revinmlp_mix", "revinmlp_nbr"):
        return {"learning_rate": _loguniform(rng, 1e-4, 2e-3), "mlp_hidden": rng.choice([64, 128, 256]), "mlp_layers": rng.choice([1, 2, 3]),
                "mlp_dropout": rng.choice([0.0, 0.1, 0.2]), "batch_size": rng.choice([16, 32, 64])}
    raise ValueError(f"unknown model {model!r}")


def candidate_list(model: str, budget: int = BUDGET, seed: int = SEARCH_SEED) -> list[dict[str, Any]]:
    """Candidate 0 is the model's default; the rest are distinct seeded draws (same list for every resolution and band)."""
    rng = random.Random(f"{seed}-{family(model)}")                    # the controls draw exactly OFA's candidates
    out = [dict(DEFAULTS[model])]
    if model == "patchtst_nopatch":
        out[0].update(patch_length=1, patch_stride=1)
    seen = {json.dumps(out[0], sort_keys=True)}
    attempts = 0
    while len(out) < budget and attempts < budget * 100:
        attempts += 1
        candidate = draw(model, rng)
        if model == "patchtst_nopatch":                              # patching is removed: the patch draws are replaced, the rest is PatchTST's
            candidate.update(patch_length=1, patch_stride=1)
        if model == "deepspred2d":                                   # half the budget per colormap: candidate 1 is the paper anchor in gray
            candidate["colormap"] = ("jet", "gray")[len(out) % 2]
            if len(out) == 1:
                candidate = {**DEFAULTS[model], "colormap": "gray"}
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def default_train_stride(model: str, resolution: str) -> int:
    """Disclosed thinning of the training windows: Time-LLM 2 at 15 minutes; 15 for every model at 1 minute."""
    if resolution == "1min":
        return 15
    if resolution == "electrosense":
        return 1                                                      # a tiny dataset: every window is used
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
    if model in TIMERAN_FAMILY:
        model_section = {"checkpoint_size": TIMERAN_SIZE, "context_length": 512, "timeran_checkpoint_path": TIMERAN_LOCAL}
    else:
        model_section = {}
    train = {"val_fraction": 0.1, "weight_decay": 0.0, "optimizer": "adam", "gradient_clip_norm": 1.0, "batch_size": 32,
             "learning_rate": 1e-3}
    return model_section, train


def build_config(model: str, resolution: str, band: str, horizon: int, seed: int, params: dict[str, Any], repo_root: str, *,
                 train_stride: int, test_stride: int, device: str | None = None, client=None, bucket: str | None = None,
                 train_start: str | None = None) -> tuple[dict[str, Any], str]:
    """Pipeline config for one run of ``model``; returns (config, pipeline model name).

    ``resolution`` is "15min" or "1min" (POWDER, ``band`` = "600_800" or "2400_2600") or "electrosense" (``band`` = "spec4", "spec25" or
    "spec151", the SpectrumLLM paper's protocol: 70/10/20 split, lookback 192; needs ``client`` and ``bucket`` to fetch the data).
    """
    from training.ray.llm_powder import BASE_CONFIG, FINAL_TEST, RESOLUTIONS

    name = PIPELINE_NAME[model]
    spec = RESOLUTIONS.get(resolution, {"lookback": 192})
    config = yaml.safe_load((Path(repo_root) / BASE_CONFIG).read_text())
    model_section, train_section = template(model, repo_root)
    config[name] = {"model": model_section, "train": train_section}
    config["training"] = {"models": [name], "device": device or "auto"}
    config["data"]["split"]["ranges"]["test"] = dict(FINAL_TEST)
    if train_start:                                                    # training-length study: a later start of the same training run
        if resolution == "electrosense":
            raise ValueError("--train-start applies to the POWDER resolutions only")
        config["data"]["split"]["ranges"]["train"]["start"] = str(train_start)
    if resolution == "electrosense":
        from training.ray.electrosense import apply_dataset, ensure_dataset

        series_path, stats = ensure_dataset(band, repo_root, bucket, client)
        config = apply_dataset(config, series_path, stats, band, [horizon])
    elif resolution == "15min":
        from training.ray.powder15 import apply_powder15

        config = apply_powder15(config, band, repo_root, [horizon])
        if model == "revinmlp_nbr":
            # the four core receivers side by side on their common timestamps (one 4 x 200 column row per time), receiver order = file order
            config["data"]["concat"] = "columns"
            order = {n: i for i, n in enumerate(("humanities", "madsen", "moran", "sagepoint"))}
            config["data"]["files"] = sorted(config["data"]["files"], key=lambda e: ({"train": 0, "validation": 1, "test": 2}[e["partition"]],
                                                                                    order[next(n for n in order if f"/{n}-nuc1/" in e["path"])]))
    else:
        for entry in config["data"]["files"]:
            entry["path"] = entry["path"].replace("/600_800/", f"/{band}/")
        config["data"]["chunks"] = [{"id": f"powder_{band}", "start_mhz": float(band.split("_")[0]), "end_mhz": float(band.split("_")[1])}]
        config["windowing"] = {"lookback": spec["lookback"], "horizons": [int(horizon)]}
    config[name]["model"].update(input_sequence_length=spec["lookback"], prediction_horizon=int(horizon))
    if model in OFA_FAMILY or model == "timellm":
        config[name]["model"]["channel_independent"] = True
    config[name]["train"].update(train_stride=int(train_stride), val_stride=int(train_stride), test_stride=int(test_stride),
                                 epochs=EPOCHS[model], seed=int(seed), early_stopping=True, early_stopping_patience=PATIENCE)
    if model in OFA_FAMILY or model == "timellm":
        config[name]["train"]["batch_size"] = 4                          # the LLM runs' memory-safe default (llm_powder)
    config = apply_params(config, name, params)
    config[name]["model"].update(CONTROLS.get(model, {}))              # ablation flags last, so they win over a candidate's value
    return config, name


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

    if spec["model"] in TIMERAN_FAMILY:
        ensure_timeran_weights(spectrum_s3_client(), spec["bucket"])

    client, bucket = spectrum_s3_client(), spec["bucket"]
    model, resolution, band, horizon, seed = spec["model"], spec["resolution"], spec["band"], int(spec["horizon"]), int(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(0.95, RESOURCES[model][1]))
        torch.cuda.reset_peak_memory_stats()
    base = group(spec["output_prefix"], resolution, band, model)
    record: dict[str, Any] = {"kind": spec["kind"], "model": model, "resolution": resolution, "band": band, "horizon": horizon,
                              "seed": seed, "index": spec.get("index"), "params": spec["params"],
                              "train_stride": spec["train_stride"], "test_stride": spec["test_stride"]}
    if spec.get("train_start"):
        record["train_start"] = spec["train_start"]
    started = time.time()
    try:
        stats = None
        if resolution == "15min":
            from training.ray.powder15 import ensure_powder15

            ensure_powder15(band, root)
        elif resolution == "electrosense":
            from training.ray.electrosense import ensure_dataset

            _, stats = ensure_dataset(band, root, bucket, client)
        config, name = build_config(model, resolution, band, horizon, seed, spec["params"], root, train_stride=spec["train_stride"],
                                    test_stride=spec["test_stride"], client=client, bucket=bucket, train_start=spec.get("train_start"))
        with tempfile.TemporaryDirectory(prefix="tune-direct-") as scratch:
            result = train_prepared(config, name, Path(scratch) / "run")
            record.update(objective=result["objective"], objective_name=result["objective_name"])
            if spec["kind"] == "final":
                from training.common.run_costs import read_run_costs

                eval_dir = Path(scratch) / "eval"
                eval_dir.mkdir()
                evaluate_one_model(config, name, eval_dir, result["checkpoint"], skip_plots=True)
                record.update(summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")))
                if stats is not None:                                    # ElectroSense: the paper's metric is raw MSE (and standardized, per bin)
                    from training.ray.electrosense import standardized_from_per_frequency

                    record.update(standardized_from_per_frequency(pd.read_csv(eval_dir / "per_frequency_metrics.csv"), stats))
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


def ensure_timeran_weights(client, bucket: str, local: str = TIMERAN_LOCAL, key: str = TIMERAN_WEIGHTS_KEY) -> str:
    """Download the TimeRAN checkpoint to this node once (atomic rename, so concurrent tasks never read a partial file)."""
    path = Path(local)
    if path.is_file() and path.stat().st_size > 0:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    client.download_file(bucket, key, str(partial))
    os.replace(partial, path)
    return str(path)


MOMENTFM_VERSION = "0.1.4"


def setup_timeran_node(bucket: str) -> dict[str, Any]:
    """Remote worker: install momentfm (no dependencies, beside the isolated transformers in PYLIBS, so the system NumPy is untouched),
    cache MOMENT-1-small from Hugging Face and the TimeRAN checkpoint from the bucket, and build one model of each backbone."""
    import socket
    import subprocess

    from training.ray.llm_powder import PYLIBS, setup_node, use_isolated_libs
    from training.ray.storage import spectrum_s3_client

    row: dict[str, Any] = {"host": socket.gethostname()}
    try:
        base = setup_node()                                              # transformers in PYLIBS (also caches GPT-2; harmless)
        if not base.get("ok"):
            return {**row, **base}
        marker = Path(PYLIBS) / f".installed-momentfm-{MOMENTFM_VERSION}"
        if not marker.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", PYLIBS,
                            f"momentfm=={MOMENTFM_VERSION}"], check=True, capture_output=True, text=True, timeout=1800)
            marker.write_text("ok")
        use_isolated_libs()
        ensure_timeran_weights(spectrum_s3_client(), bucket)
        import numpy

        from models.TimeRAN import TimeRANForecaster

        for init in ("timeran", "moment", "random"):
            model = TimeRANForecaster({"model": {"input_sequence_length": 60, "prediction_horizon": 15, "input_size": 3,
                                                 "checkpoint_size": TIMERAN_SIZE, "backbone_init": init, "backbone_trainable": "head",
                                                 "timeran_checkpoint_path": TIMERAN_LOCAL}})
            row[f"{init}_report"] = model.load_report
        row.update(ok=True, numpy=numpy.__version__, parameters=sum(p.numel() for p in model.parameters()))
    except subprocess.CalledProcessError as error:
        row.update(ok=False, error=f"pip failed: {(error.stderr or '')[-400:]}")
    except Exception as error:
        row.update(ok=False, error=f"{type(error).__name__}: {error}"[:400])
    return row


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
    p.add_argument("--resolution", choices=("15min", "1min", "electrosense"))
    p.add_argument("--setup-timeran", action="store_true", help="install momentfm and cache MOMENT/TimeRAN weights on every GPU node, then exit")
    p.add_argument("--bands", nargs="+", help="POWDER bands (default both) or, with --resolution electrosense, datasets (default spec4 spec25)")
    p.add_argument("--models", nargs="+", default=list(PIPELINE_NAME), choices=list(PIPELINE_NAME))
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    p.add_argument("--horizons", type=int, nargs="+", help="final horizons (default: all of the resolution's)")
    p.add_argument("--train-stride", action="append", default=[], metavar="MODEL=N", help="override a model's training-window stride")
    p.add_argument("--test-stride-1min", type=int, default=15, help="thinning of the 1-minute test windows")
    p.add_argument("--trials-only", action="store_true")
    p.add_argument("--fixed-params", help="JSON object of hyperparameters: skip the search and run the finals of every model with exactly these "
                   "settings (an equal-configuration comparison, or a pipeline check against a winner found elsewhere); use a separate "
                   "--output-prefix")
    p.add_argument("--winners-prefix", help="skip the search and train the finals with each (band, model) winner already recorded under this "
                   "output prefix (its winners.json); requires a different --output-prefix")
    p.add_argument("--train-start", help="ISO timestamp: start the training range here instead of at the start of the training run "
                   "(training-length study; validation and test ranges unchanged); requires --winners-prefix or --fixed-params")
    p.add_argument("--max-parallel", type=int, default=0, help="most tasks in flight at once (0 = no limit); keeps the queue short for other jobs")
    p.add_argument("--max-retries", type=int, default=3, help="Ray retries for a task whose node died (a lease ending mid-run)")
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="direct-tuning")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    if args.setup_timeran:
        sys.path.insert(0, args.repo_root)
        ray.init(address=args.address, ignore_reinit_error=True)
        bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]

        def on_node(bucket_name: str) -> dict[str, Any]:
            os.chdir(args.repo_root)
            sys.path.insert(0, args.repo_root)
            return setup_timeran_node(bucket_name)

        refs = [ray.remote(num_cpus=1, num_gpus=0)(on_node).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(bucket) for n in nodes]
        for row in ray.get(refs):
            print(json.dumps(row, default=str), flush=True)
        return
    if not args.resolution:
        p.error("--resolution is required unless --setup-timeran is given")
    args.bands = args.bands or (["spec4", "spec25"] if args.resolution == "electrosense" else list(BANDS))
    allowed = ES_DATASETS if args.resolution == "electrosense" else BANDS
    if any(b not in allowed for b in args.bands):
        p.error(f"--bands must be among {allowed} for {args.resolution}")
    if args.winners_prefix and (args.fixed_params or args.winners_prefix.strip("/") == args.output_prefix.strip("/")):
        p.error("--winners-prefix excludes --fixed-params and needs a different --output-prefix")
    if args.train_start and (not (args.winners_prefix or args.fixed_params) or args.output_prefix.strip("/") == "direct-tuning"):
        p.error("--train-start needs --winners-prefix or --fixed-params, and its own --output-prefix")
    sys.path.insert(0, args.repo_root)
    from training.ray.storage import spectrum_s3_client

    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    client = spectrum_s3_client()
    overrides = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.train_stride}
    test_stride = args.test_stride_1min if args.resolution == "1min" else 1
    tune_horizons = TUNE_HORIZONS[args.resolution]
    final_horizons = tuple(args.horizons) if args.horizons else FINAL_HORIZONS[args.resolution]
    base_spec = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix, "resolution": args.resolution,
                 "test_stride": test_stride, "train_start": args.train_start}

    def stride(model: str) -> int:
        return overrides.get(model, default_train_stride(model, args.resolution))

    # Pipelined: every (model, band) group goes trials -> winner -> finals on its own, so a deadline that arrives mid-run still leaves
    # complete results for the groups that finished (an earlier version waited for ALL trials before starting any final).
    scores: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    pending: dict[Any, tuple[str, tuple[str, str]]] = {}
    trials_left: collections.Counter = collections.Counter()
    finals_total = 0

    queue: list[tuple[str, tuple[str, str], dict[str, Any]]] = []

    def submit(spec: dict[str, Any]):
        cpus, gpus = RESOURCES[spec["model"]]
        return ray.remote(num_cpus=cpus, num_gpus=gpus, max_retries=args.max_retries)(job).remote(spec)

    def pump() -> None:
        while queue and (args.max_parallel <= 0 or len(pending) < args.max_parallel):
            kind, grp, spec = queue.pop(0)
            pending[submit(spec)] = (kind, grp)

    def enqueue(kind: str, grp: tuple[str, str], spec: dict[str, Any]) -> None:
        queue.append((kind, grp, spec))
        pump()

    def finalize(model: str, band: str) -> None:
        nonlocal finals_total
        table = scores.get((model, band), {})
        if args.fixed_params:
            index = "fixed"
            record = {"index": index, "params": json.loads(args.fixed_params), "tune_horizons": None, "budget": None,
                      "note": "no search: every model uses exactly these hyperparameters"}
        elif args.winners_prefix:
            source_key = f"{group(args.winners_prefix, args.resolution, band, model)}/winners.json"
            try:
                source = json.loads(client.get_object(Bucket=bucket, Key=source_key)["Body"].read())
            except Exception as error:
                print(f"[tune-direct] no winner at {source_key} ({type(error).__name__}); {model} {band} skipped", flush=True)
                return
            index = source["index"]
            record = {**source, "note": f"no search: winner copied from {source_key}", "train_start": args.train_start}
        else:
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
                    enqueue("final", (model, band), {**base_spec, "kind": "final", "model": model, "band": band, "index": index,
                                                     "horizon": horizon, "seed": seed, "params": record["params"],
                                                     "train_stride": stride(model)})
                    launched += 1
        finals_total += launched
        print(f"[tune-direct] winner {model}/{band}: candidate {index}; {launched} finals launched", flush=True)

    trials_total = 0
    for model in args.models:
        for band in args.bands:
            base = group(args.output_prefix, args.resolution, band, model)
            for index, params in enumerate([] if (args.fixed_params or args.winners_prefix) else candidate_list(model, args.budget)):
                for horizon in tune_horizons:
                    record = _done(client, bucket, trial_key(base, index, horizon))
                    if record:
                        scores.setdefault((model, band), {}).setdefault(index, {})[horizon] = float(record["objective"])
                    else:
                        enqueue("trial", (model, band), {**base_spec, "kind": "trial", "model": model, "band": band, "index": index,
                                                         "horizon": horizon, "seed": SEARCH_SEED, "params": params,
                                                         "train_stride": stride(model)})
                        trials_left[(model, band)] += 1
                        trials_total += 1
    print(f"[tune-direct] {trials_total} trials to run (training stride {({m: stride(m) for m in args.models})}), pipelined per group",
          flush=True)
    if not args.trials_only:
        for model in args.models:
            for band in args.bands:
                if trials_left[(model, band)] == 0:
                    finalize(model, band)                     # nothing to wait for: its trials were all done already
    while pending or queue:
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
        pump()
    print(f"[tune-direct] done ({finals_total} finals launched)", flush=True)

if __name__ == "__main__":
    main()
