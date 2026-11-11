"""Run the SpectrumLLM paper's baselines with the Time-Series-Library on the ElectroSense-Madrid datasets.

The library (https://github.com/thuml/Time-Series-Library, MIT) is cloned on each worker at a pinned commit and run
in an isolated virtualenv that reuses the system packages, so its dependencies cannot disturb our pipeline.
Protocol: ``--data custom`` (70/10/20 split, StandardScaler on train), ``--features M``, input 192, horizons
12/24/48/96/192, library-default hyperparameters (what a reproduction without tuning would use). Each run reports
standardized-unit MSE/MAE (the library default) from training, and raw-unit MSE/MAE from a second ``--inverse``
test pass on the saved checkpoint.

    python -m training.ray.tsl_runs setup
    python -m training.ray.tsl_runs run --models DLinear PatchTST --seeds 2021 2022 2023
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .electrosense import DATASETS, HORIZONS, RAW_PREFIX, SEQ_LEN

TSL_URL = "https://github.com/thuml/Time-Series-Library"
TSL_DIR = "/workspace/tsl"
VENV = "/workspace/tsl/venv"
DEFAULT_MODELS = ("DLinear", "PatchTST", "Informer", "Reformer", "Autoformer", "TimesNet", "LightTS")
CHANNELS = {"spec4": 4, "spec25": 25, "spec151": 151}
LABEL_LEN = SEQ_LEN // 2


def run_command(model: str, dataset: str, horizon: int, seed: int, root: str, target: str, *,
                training: bool, inverse: bool, has_seed_flag: bool, extra: list[str] | None = None,
                des: str = "electrosense") -> list[str]:
    """The run.py command line for one cell (library defaults except data/shape arguments and ``extra``).

    ``des`` names the run (the library derives its checkpoint directory from it), so runs with different
    hyperparameters must use different values."""
    channels = CHANNELS[dataset]
    command = [
        f"{VENV}/bin/python", "-u", "run.py", "--task_name", "long_term_forecast",
        "--is_training", "1" if training else "0",
        "--root_path", root, "--data_path", DATASETS[dataset],
        "--model_id", f"{dataset}_{SEQ_LEN}_{horizon}_s{seed}", "--model", model,
        "--data", "custom", "--features", "M", "--target", target, "--freq", "t",
        "--seq_len", str(SEQ_LEN), "--label_len", str(LABEL_LEN), "--pred_len", str(horizon),
        "--enc_in", str(channels), "--dec_in", str(channels), "--c_out", str(channels),
        "--des", des, "--itr", "1",
    ]
    command += list(extra or [])
    if has_seed_flag:
        command += ["--seed", str(seed)]
    if inverse:
        command.append("--inverse")
    return command


_VALI = re.compile(r"Vali Loss:\s*([-+0-9.eE]+)")


def parse_best_validation(output: str) -> float | None:
    """Lowest per-epoch 'Vali Loss' (standardized MSE) printed by run.py's training loop."""
    values = [float(v) for v in _VALI.findall(output)]
    return min(values) if values else None


_METRIC = re.compile(r"mse:\s*([-+0-9.eE]+),\s*mae:\s*([-+0-9.eE]+)")


def parse_metrics(output: str) -> tuple[float, float] | None:
    """Last 'mse:..., mae:...' line printed by run.py's test pass."""
    matches = _METRIC.findall(output)
    return (float(matches[-1][0]), float(matches[-1][1])) if matches else None


PINNED_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"  # thuml/Time-Series-Library main, 2026-04-18
MODULE_TO_PACKAGE = {"patoolib": "patool", "sklearn": "scikit-learn", "reformer_pytorch": "reformer-pytorch",
                     "skimage": "scikit-image", "PIL": "pillow", "yaml": "pyyaml"}


SEED_LINE = "fix_seed = 2021"
SEED_PATCH = "fix_seed = int(os.environ.get('TSL_SEED', 2021))"


def patch_seed(text: str) -> tuple[str, bool]:
    """Make the library's run.py take its random seed from the environment variable ``TSL_SEED``.

    The pinned library hardcodes ``fix_seed = 2021`` (random, torch and numpy are seeded with it before the arguments are parsed) and its
    ``--seed`` flag only reaches the augmentation helpers, so five "seeds" of one cell were five identical runs. The patch changes only that
    one line; with ``TSL_SEED`` unset the default is still 2021, so every earlier result stays reproducible. ``run.py`` already imports ``os``.
    Returns (new text, whether anything changed); an already patched text is returned unchanged.
    """
    if SEED_PATCH in text:
        return text, False
    if SEED_LINE not in text:
        raise ValueError("run.py has no 'fix_seed = 2021' line: the pinned library changed")
    return text.replace(SEED_LINE, SEED_PATCH, 1), True


def download_library(commit: str, repo: str) -> None:
    """Fetch the pinned commit as a tarball (workers have no git) and extract it, refusing unsafe paths."""
    import tarfile
    import urllib.request

    data = urllib.request.urlopen(f"https://codeload.github.com/thuml/Time-Series-Library/tar.gz/{commit}",
                                  timeout=120).read()
    tmp = Path(repo).parent / "_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    root = os.path.realpath(tmp)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            target = os.path.realpath(os.path.join(root, member.name))
            if target != root and not target.startswith(root + os.sep):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                continue
            archive.extract(member, root)
    os.rename(os.path.join(root, os.listdir(root)[0]), repo)


def setup_node(args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: fetch the pinned library, build the venv, import its training entry point, fetch the data."""
    f = lambda c, t=1800: subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    out: dict[str, Any] = {"host": f("hostname").stdout.strip()}
    os.makedirs(TSL_DIR, exist_ok=True)
    repo = f"{TSL_DIR}/Time-Series-Library"
    commit = args.get("commit") or PINNED_COMMIT
    if not os.path.isfile(f"{repo}/run.py"):
        download_library(commit, repo)
    out["commit"] = commit
    run_py = Path(f"{repo}/run.py")
    patched_text, changed = patch_seed(run_py.read_text())
    if changed:
        run_py.write_text(patched_text)
    out["seed_patch"] = "applied" if changed else "already applied"
    if not os.path.exists(f"{VENV}/bin/python"):
        f(f"python -m venv --system-site-packages {VENV}")
    # install only what the library's imports are missing, one at a time (never its pinned requirements)
    installed = []
    probe = "from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast; print('IMPORT_OK')"
    for _ in range(20):
        r = f(f"cd {repo} && {VENV}/bin/python -c \"{probe}\" 2>&1 | tail -3")
        match = re.search(r"No module named '([A-Za-z0-9_]+)", r.stdout)
        if "IMPORT_OK" in r.stdout or not match:
            break
        package = MODULE_TO_PACKAGE.get(match.group(1), match.group(1))
        f(f"{VENV}/bin/pip install -q {package} 2>&1 | tail -1")
        installed.append(package)
    out["pip_installed"] = installed
    out["import_ok"] = "IMPORT_OK" in f(f"cd {repo} && {VENV}/bin/python -c \"{probe}\" 2>&1 | tail -1").stdout
    help_text = f(f"cd {repo} && {VENV}/bin/python run.py --help 2>&1").stdout
    out["has_seed_flag"] = "--seed" in help_text
    out["has_inverse_flag"] = "--inverse" in help_text
    import sys

    sys.path.insert(0, args["repo_root"])
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    root = f"{args['repo_root']}/data/electrosense_raw"
    os.makedirs(root, exist_ok=True)
    for filename in DATASETS.values():
        if not os.path.exists(f"{root}/{filename}"):
            client.download_file(args["bucket"], f"{RAW_PREFIX}/{filename}", f"{root}/{filename}")
    out["data"] = sorted(os.listdir(root))
    return out


def tsl_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: train one (model, dataset, horizon, seed) cell and report standardized and raw metrics."""
    import sys

    sys.path.insert(0, args["repo_root"])
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    model, dataset, horizon, seed = job["model"], job["dataset"], int(job["horizon"]), int(job["seed"])
    result = {"dataset": dataset, "model": model, "horizon": horizon, "seed": seed, "kind": "tsl",
              "hyperparameters": job.get("hp_label", "library defaults")}
    if job.get("params"):
        result["params"] = job["params"]
    started = time.time()
    try:
        repo = f"{TSL_DIR}/Time-Series-Library"
        root = f"{args['repo_root']}/data/electrosense_raw/"
        header = Path(root, DATASETS[dataset]).read_text().splitlines()[0].split(",")
        target = header[-1]
        common = dict(root=root, target=target, has_seed_flag=args["has_seed_flag"],
                      extra=job.get("extra"), des=job.get("des", "electrosense"))
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TSL_SEED": str(seed)}     # the patched run.py seeds from this (see patch_seed)
        if job.get("threads"):                               # CPU runs: stay inside the reserved cores
            env.update(OMP_NUM_THREADS=str(job["threads"]), MKL_NUM_THREADS=str(job["threads"]))
        train = subprocess.run(run_command(model, dataset, horizon, seed, training=True, inverse=False, **common),
                               cwd=repo, env=env, capture_output=True, text=True, timeout=args["timeout"])
        metrics = parse_metrics(train.stdout + train.stderr)
        if metrics is None:
            raise RuntimeError("no metrics in run.py output: " + (train.stdout + train.stderr)[-400:])
        result["mse_std"], result["mae_std"] = metrics
        raw = subprocess.run(run_command(model, dataset, horizon, seed, training=False, inverse=True, **common),
                             cwd=repo, env=env, capture_output=True, text=True, timeout=args["timeout"])
        raw_metrics = parse_metrics(raw.stdout + raw.stderr)
        if raw_metrics is not None:
            result["mse_raw"], result["mae_raw"] = raw_metrics
        result["ok"] = True
    except Exception as error:
        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:500])
    result["wall_seconds"] = round(time.time() - started, 1)
    try:
        key = f"{args['output_prefix'].strip('/')}/tsl/{dataset}/{model}/h{horizon}/seed-{seed}.json"
        client.put_object(Bucket=args["bucket"], Key=key, Body=json.dumps(result).encode())
    except Exception:
        pass
    return result


def launch(args: argparse.Namespace) -> None:
    import pandas as pd
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    base = {"repo_root": args.repo_root, "bucket": bucket, "output_prefix": args.output_prefix}
    if args.command == "setup":
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        futures = [ray.remote(num_cpus=2, scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(
            setup_node).remote({**base, "commit": args.commit}) for n in nodes]
        for row in ray.get(futures):
            print("[tsl setup]", json.dumps(row), flush=True)
        return
    worker_args = {**base, "timeout": args.timeout, "has_seed_flag": not args.no_seed_flag}
    jobs = [{"model": m, "dataset": d, "horizon": h, "seed": s}
            for d in args.datasets for m in args.models for h in args.horizons for s in args.seeds]
    if args.skip_done:
        def done(j):
            try:
                key = f"{args.output_prefix.strip('/')}/tsl/{j['dataset']}/{j['model']}/h{j['horizon']}/seed-{j['seed']}.json"
                return bool(json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read()).get("ok"))
            except Exception:
                return False
        jobs = [j for j in jobs if not done(j)]
    if args.cpu:
        jobs = [{**j, "threads": args.cpus, "extra": ["--no_use_gpu"]} for j in jobs]
    print(f"[tsl] {len(jobs)} runs", flush=True)
    remote = ray.remote(num_cpus=args.cpus if args.cpu else 2, num_gpus=0 if args.cpu else args.gpu_fraction, max_retries=1)(tsl_job)
    from .tune_tsl import run_bounded

    rows = []
    for job, result in run_bounded(ray, remote, jobs, worker_args, args.max_parallel):
        rows.append(result)
        print(f"[tsl] {job['dataset']}/{job['model']}/h{job['horizon']}/seed{job['seed']} ok={result.get('ok')} "
              f"mse_raw={result.get('mse_raw')} ({result.get('wall_seconds')}s) [{len(jobs) - len(rows)} left]", flush=True)
    frame = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    key = f"{args.output_prefix.strip('/')}/summary_tsl_{stamp}.csv"
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode())
    failed = frame[~frame["ok"].astype(bool)] if "ok" in frame else frame
    print(f"[tsl] wrote s3://{bucket}/{key}; {len(frame)} rows, {len(failed)} failed", flush=True)
    for _, row in failed.head(8).iterrows():
        print(f"[tsl] FAILED {row.get('dataset')}/{row.get('model')}/h{row.get('horizon')}: {row.get('error')}",
              flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "run"):
        p = sub.add_parser(name)
        p.add_argument("--bucket", default=None)
        p.add_argument("--output-prefix", default="electrosense-results")
        p.add_argument("--address", default="auto")
        p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    sub.choices["setup"].add_argument("--commit", default=None, help=f"Library commit (default: {PINNED_COMMIT[:10]})")
    run = sub.choices["run"]
    run.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    run.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    run.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    run.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    run.add_argument("--gpu-fraction", type=float, default=0.25)
    run.add_argument("--timeout", type=int, default=5400)
    run.add_argument("--no-seed-flag", action="store_true", help="run.py has no --seed flag (set by setup output)")
    run.add_argument("--max-parallel", type=int, default=None, help="cap on runs in flight (default: all at once)")
    run.add_argument("--skip-done", action="store_true", help="skip cells that already have an ok record under the output prefix")
    run.add_argument("--cpu", action="store_true", help="train on CPU (library --no_use_gpu) and reserve no GPU")
    run.add_argument("--cpus", type=int, default=2, help="CPU cores reserved per run with --cpu")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
