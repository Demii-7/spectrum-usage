"""Train Time-LLM (our ``spectrumllm`` port) and OFA on POWDER and score them like our other models.

The two LLM-based models from the SpectrumLLM paper beat our ResidualLinearAR on ElectroSense only in channel-
independent mode with 25 channels, and they have never been run on POWDER. This runner trains them on the same data,
split, horizons and per-bin metrics as the rest of the benchmark:

* ``--resolution 1min``: 600-800 MHz, the four core receivers pooled (2D), lookback 60, horizons 1/15/60 minutes.
* ``--resolution 15min``: block-averaged data (``powder15``), lookback 96, horizons 4/12/24/48 steps (1-12 h).

Unlike our own models these are trained *directly* for each horizon (one model per horizon, as on ElectroSense) and run
channel-independently: every frequency bin is its own sequence, sharing one set of weights. Hyperparameters are the
SpectrumLLM reproduction's regime (lr 1e-4, patch 16/8, d_model 32, 6 GPT-2 layers, 10 epochs, patience 5), not tuned.
Because a channel-independent batch holds ``batch_size x 200`` sequences, windows are subsampled with ``--train-stride``.

    python -m training.ray.llm_powder --resolution 15min --arms ofa --horizons 4 12 24 48 --seeds 40 41 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

BASE_CONFIG = "training/configs/oct2026/ray_tuning_2d.yaml"        # data files, split, preprocessing (1-minute POWDER)
ARMS = {"ofa": ("ofa", "training/configs/config_ofa.yaml"),
        "timellm": ("spectrumllm", "training/configs/config_spectrumllm.yaml")}
FINAL_TEST = {"start": "2026-10-27T20:00:00Z", "end": "2026-11-03T20:00:00Z"}
RESOLUTIONS = {"1min": {"lookback": 60, "horizons": (1, 15, 60)}, "15min": {"lookback": 96, "horizons": (4, 12, 24, 48)}}


def build_config(arm: str, resolution: str, band: str, horizon: int, seed: int, repo_root: str, *,
                 train_stride: int = 1, test_stride: int = 1, epochs: int = 10, batch_size: int = 4, learning_rate: float = 1e-4,
                 patience: int = 5, device: str | None = None) -> tuple[dict[str, Any], str]:
    """Pipeline config for one (arm, resolution, band, horizon, seed) run; returns (config, model_name)."""
    model_name, arm_file = ARMS[arm]
    spec = RESOLUTIONS[resolution]
    config = yaml.safe_load((Path(repo_root) / BASE_CONFIG).read_text())
    arm_config = yaml.safe_load((Path(repo_root) / arm_file).read_text())[model_name]
    config[model_name] = {"model": copy.deepcopy(arm_config["model"]), "train": copy.deepcopy(arm_config["train"])}
    config["training"] = {"models": [model_name], "device": device or "auto"}
    config["data"]["split"]["ranges"]["test"] = dict(FINAL_TEST)
    if resolution == "15min":
        from training.ray.powder15 import apply_powder15

        config = apply_powder15(config, band, repo_root, [horizon])
    else:
        for entry in config["data"]["files"]:
            entry["path"] = entry["path"].replace("/600_800/", f"/{band}/")
        config["data"]["chunks"] = [{"id": f"powder_{band}", "start_mhz": float(band.split("_")[0]),
                                     "end_mhz": float(band.split("_")[1])}]
        config["windowing"] = {"lookback": spec["lookback"], "horizons": [int(horizon)]}
    model, train = config[model_name]["model"], config[model_name]["train"]
    model.update(input_sequence_length=spec["lookback"], prediction_horizon=int(horizon), channel_independent=True)
    train.update(train_stride=int(train_stride), val_stride=int(train_stride), test_stride=int(test_stride), epochs=int(epochs),
                 batch_size=int(batch_size), learning_rate=float(learning_rate), seed=int(seed),
                 early_stopping=True, early_stopping_patience=int(patience))
    return config, model_name


PYLIBS = "/workspace/pylibs"
HF_HOME = "/workspace/hf-cache"
TRANSFORMERS_VERSION = "4.44.2"
BACKBONE = "openai-community/gpt2"


def use_isolated_libs() -> None:
    """Put the isolated ``transformers`` install first on the path and point Hugging Face at the node-local cache."""
    import sys

    if PYLIBS not in sys.path:
        sys.path.insert(0, PYLIBS)
    os.environ.setdefault("HF_HOME", HF_HOME)


def setup_node(_: dict[str, Any] | None = None) -> dict[str, Any]:
    """Remote worker: install ``transformers`` under PYLIBS (system packages untouched) and cache GPT-2 weights."""
    import shutil
    import socket
    import subprocess
    import sys

    row: dict[str, Any] = {"host": socket.gethostname()}
    try:
        marker = Path(PYLIBS) / f".installed-{TRANSFORMERS_VERSION}"
        if not marker.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", PYLIBS,
                            f"transformers=={TRANSFORMERS_VERSION}"], check=True, capture_output=True, text=True,
                           timeout=1800)
            for leftover in list(Path(PYLIBS).glob("numpy*")):   # keep the system NumPy
                shutil.rmtree(leftover, ignore_errors=True)
            marker.write_text("ok")
        use_isolated_libs()
        os.makedirs(HF_HOME, exist_ok=True)
        from transformers import AutoConfig, AutoModel

        AutoConfig.from_pretrained(BACKBONE)
        backbone = AutoModel.from_pretrained(BACKBONE)       # downloads (safetensors) into HF_HOME on first use
        import transformers

        row.update(transformers=transformers.__version__, backbone_parameters=sum(p.numel() for p in backbone.parameters()),
                   ok=True)
    except subprocess.CalledProcessError as error:
        row.update(ok=False, error=f"pip failed: {(error.stderr or '')[-400:]}")
    except Exception as error:
        row.update(ok=False, error=f"{type(error).__name__}: {error}"[:400])
    return row


def result_key(prefix: str, resolution: str, band: str, arm: str, horizon: int, seed: int) -> str:
    return f"{prefix.strip('/')}/{resolution}/{band}/{arm}/h{horizon}/seed-{seed}.json"


def llm_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one model and score it on the final test range."""
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    use_isolated_libs()
    from training.common.evaluation_integrated import evaluate_one_model
    from training.common.train_integrated import train_one_model
    from training.ray.final_eval import summarize_metrics
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    arm, resolution, band, horizon, seed = job["arm"], job["resolution"], job["band"], int(job["horizon"]), int(job["seed"])
    result = {"arm": arm, "resolution": resolution, "band": band, "horizon": horizon, "seed": seed, "kind": "llm",
              "channel_independent": True, "train_stride": args["train_stride"], "test_stride": args["test_stride"], "epochs_max": args["epochs"],
              "batch_size": args["batch_size"], "learning_rate": args["learning_rate"],
              "hyperparameters": "SpectrumLLM reproduction regime (untuned)"}
    started = time.time()
    try:
        if resolution == "15min":
            from training.ray.powder15 import ensure_powder15

            result["conversion"] = json.dumps(ensure_powder15(band, args["repo_root"]))
        config, model_name = build_config(arm, resolution, band, horizon, seed, args["repo_root"],
                                          train_stride=args["train_stride"], test_stride=args["test_stride"], epochs=args["epochs"],
                                          batch_size=args["batch_size"], learning_rate=args["learning_rate"],
                                          patience=args["patience"], device=args.get("device"))
        with tempfile.TemporaryDirectory(prefix="llm-powder-") as scratch:
            run_dir, eval_dir = Path(scratch) / "run", Path(scratch) / "eval"
            train_one_model(config, model_name, run_dir)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            eval_dir.mkdir()
            evaluate_one_model(config, model_name, eval_dir, checkpoint, skip_plots=True)
            result.update(summarize_metrics(pd.read_csv(eval_dir / "aggregate_metrics.csv").to_dict("records")))
        result["ok"] = True
    except Exception as error:
        import traceback

        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-800:])
    result["wall_seconds"] = round(time.time() - started, 1)
    try:
        client.put_object(Bucket=bucket, Key=result_key(args["output_prefix"], resolution, band, arm, horizon, seed),
                          Body=json.dumps(result, default=str).encode())
    except Exception:
        pass
    return result


def launch(args: argparse.Namespace) -> None:
    import ray

    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    if args.setup:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        futures = [ray.remote(num_cpus=1, scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(
            setup_node).remote() for n in nodes]
        for row in ray.get(futures):
            print("[llm-powder setup]", json.dumps(row), flush=True)
        return
    horizons = args.horizons or list(RESOLUTIONS[args.resolution]["horizons"])
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix,
                   "train_stride": args.train_stride, "test_stride": args.test_stride, "epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.learning_rate, "patience": args.patience, "device": args.device}
    jobs = []
    for arm in args.arms:
        for horizon in horizons:
            for seed in args.seeds:
                key = result_key(args.output_prefix, args.resolution, args.band, arm, horizon, seed)
                try:
                    if json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"):
                        continue
                except Exception:
                    pass
                jobs.append({"arm": arm, "resolution": args.resolution, "band": args.band, "horizon": horizon,
                             "seed": seed})
    print(f"[llm-powder] {len(jobs)} runs ({args.resolution}, {args.band})", flush=True)
    remote = ray.remote(num_cpus=2, num_gpus=args.gpu_fraction, max_retries=1)(llm_job)
    pending = [remote.remote(j, worker_args) for j in jobs]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        print(f"[llm-powder] {r['arm']} h{r['horizon']} seed {r['seed']} ok={r['ok']} "
              f"{r.get('test_mean_horizon_mae_db', r.get('error'))} ({r['wall_seconds']}s)", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", action="store_true", help="install transformers + cache GPT-2 on every GPU node, then exit")
    p.add_argument("--resolution", choices=list(RESOLUTIONS))
    p.add_argument("--band", default="600_800")
    p.add_argument("--arms", nargs="+", default=["ofa", "timellm"], choices=list(ARMS))
    p.add_argument("--horizons", nargs="+", type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--test-stride", type=int, default=1, help="score every n-th test window (use 1 for real results)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--bucket", default=None)
    p.add_argument("--output-prefix", default="powder-llm-results")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    if not args.setup and not args.resolution:
        p.error("--resolution is required unless --setup is given")
    return args


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
