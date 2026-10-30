"""One model, end to end: search -> best configuration -> five seeds -> checkpoint export.

Run one driver per model as a Ray job. Every stage is idempotent: a stage that has
already finished is skipped (a marker is written to MinIO), and an interrupted
search or replicate stage is restored and its unfinished/errored trials rerun. It is
therefore always safe to resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

from .registry import get_model_spec

DEFAULT_SEEDS = (40, 41, 42, 43, 44)
STAGES = ("search", "replicate", "export")


# --- completion markers and status, stored next to the Ray experiments in MinIO ------------

def _marker_path(bucket: str, prefix: str, model: str, name: str) -> str:
    return f"{bucket}/{prefix.strip('/')}/_campaign/{model}.{name}"


def _exists(fs, path: str) -> bool:
    from pyarrow.fs import FileType

    return fs.get_file_info(path).type == FileType.File


def _write_json(fs, path: str, payload: dict[str, Any]) -> None:
    from pyarrow.fs import LocalFileSystem

    if isinstance(fs, LocalFileSystem):
        fs.create_dir(path.rsplit("/", 1)[0], recursive=True)
    with fs.open_output_stream(path) as stream:
        stream.write((json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode())


def run_stage(fs, bucket: str, prefix: str, model: str, stage: str,
              action: Callable[[], Any]) -> bool:
    """Run ``action`` unless its done-marker exists. Returns True if it ran."""
    done = _marker_path(bucket, prefix, model, f"{stage}.done")
    status = _marker_path(bucket, prefix, model, "status.json")
    if _exists(fs, done):
        print(f"[campaign] {model}: {stage} already complete, skipping", flush=True)
        return False
    _write_json(fs, status, {"model": model, "stage": stage, "state": "running",
                             "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    try:
        action()
    except Exception as error:  # record why, then let the job fail visibly
        _write_json(fs, status, {"model": model, "stage": stage, "state": "failed",
                                 "error": f"{type(error).__name__}: {error}",
                                 "trace": traceback.format_exc()[-4000:],
                                 "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise
    _write_json(fs, done, {"model": model, "stage": stage,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _write_json(fs, status, {"model": model, "stage": stage, "state": "done",
                             "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return True


# --- stages ----------------------------------------------------------------------------------

def _search_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--config", str(args.config), "--models", args.model,
            "--candidates", str(args.candidates),
            "--output-dir", str(args.output_dir),
            "--minio-bucket", args.bucket, "--minio-prefix", args.prefix,
            "--minio-endpoint", args.endpoint,
            "--grace-period", str(args.grace_period),
            "--reduction-factor", str(args.reduction_factor),
            "--brackets", str(args.brackets),
            "--max-concurrent", str(args.max_concurrent_search),
            "--max-failures", str(args.max_failures), "--resume"]
    if args.gpu_fraction is not None:
        argv += ["--gpu-fraction", str(args.gpu_fraction)]
    return argv


def _replicate_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--config", str(args.config), "--model", args.model,
            "--search-path", f"{args.bucket}/{args.prefix.strip('/')}/{args.model}-search",
            "--output-dir", str(args.output_dir),
            "--bucket", args.bucket, "--prefix", args.prefix, "--endpoint", args.endpoint,
            "--max-concurrent", str(args.max_concurrent_replicate),
            "--seeds", *[str(seed) for seed in args.seeds],
            "--max-failures", str(args.max_failures), "--resume"]
    if args.gpu_fraction is not None:
        argv += ["--gpu-fraction", str(args.gpu_fraction)]
    return argv


def _reference_argv(args: argparse.Namespace) -> list[str]:
    return ["--model", args.model, "--config", str(args.config),
            "--output-dir", str(args.output_dir), "--bucket", args.bucket,
            "--prefix", args.prefix, "--endpoint", args.endpoint,
            "--max-failures", str(args.max_failures), "--resume"]


def export_checkpoints(args: argparse.Namespace, fs, representation: str, experiment: str) -> int:
    """Copy final seed checkpoints (+ cost JSON) from MinIO to the Chameleon bucket.

    MinIO sits on a single node's disk, so this keeps a second copy under a new prefix;
    it never writes to the existing ``checkpoints/`` prefix.
    """
    from ray import tune

    from .storage import spectrum_s3_client

    client = spectrum_s3_client()
    bucket = args.export_bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    analysis = tune.ExperimentAnalysis(
        f"{args.bucket}/{args.prefix.strip('/')}/{experiment}", storage_filesystem=fs)
    manifest, uploaded = [], 0
    for result in tune.ResultGrid(analysis):
        if result.error is not None or result.checkpoint is None:
            continue
        seed = int(result.config["seed"])
        with tempfile.TemporaryDirectory(prefix="campaign-export-") as scratch:
            local = Path(result.checkpoint.to_directory(scratch))
            for path in sorted(local.rglob("*")):
                if path.is_file() and (path.suffix == ".pt" or path.name.endswith("_run_costs.json")):
                    key = f"{args.export_prefix.strip('/')}/{representation}/{args.model}/seed-{seed}/{path.name}"
                    client.upload_file(str(path), bucket, key)
                    uploaded += 1
                    manifest.append({"seed": seed, "key": key, "bytes": path.stat().st_size})
        metrics = {k: v for k, v in (result.metrics or {}).items()
                   if isinstance(v, (int, float, str, bool)) and (
                       k.startswith(("test_", "val_", "cost_")) or k in {"objective", "seed"})}
        manifest.append({"seed": seed, "metrics": metrics})
    client.put_object(
        Bucket=bucket,
        Key=f"{args.export_prefix.strip('/')}/{representation}/{args.model}/MANIFEST.json",
        Body=(json.dumps({"model": args.model, "source": f"{args.bucket}/{args.prefix}/{experiment}",
                          "entries": manifest}, indent=2, sort_keys=True, default=str) + "\n").encode(),
    )
    print(f"[campaign] {args.model}: exported {uploaded} files to s3://{bucket}/{args.export_prefix}",
          flush=True)
    return uploaded


def run_campaign(args: argparse.Namespace) -> None:
    from . import lookback_mean_eval, replicate
    from . import run as search_run
    from .replicate import _s3_filesystem

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    representation = str(config["data"]["representation"]).lower()
    spec = get_model_spec(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Credentials come from the job environment; MinIO endpoint/region from the flags.
    from .storage import MinIOConfig

    os.environ.update(MinIOConfig(args.bucket, args.prefix, args.endpoint).environment())
    fs = _s3_filesystem(args.endpoint)

    if spec.status == "reference_only":
        run_stage(fs, args.bucket, args.prefix, args.model, "replicate",
                  lambda: lookback_mean_eval.main(_reference_argv(args)))
        return

    run_stage(fs, args.bucket, args.prefix, args.model, "search",
              lambda: search_run.main(_search_argv(args)))
    run_stage(fs, args.bucket, args.prefix, args.model, "replicate",
              lambda: replicate.main(_replicate_argv(args)))
    if not args.skip_export:
        try:
            run_stage(fs, args.bucket, args.prefix, args.model, "export",
                      lambda: export_checkpoints(args, fs, representation, f"{args.model}-replicate"))
        except Exception as error:  # checkpoints are still safe in MinIO
            print(f"[campaign] {args.model}: export failed ({error}); checkpoints remain in MinIO",
                  flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", required=True, help="MinIO prefix for this campaign")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Put this under the repo bind mount so manifests survive container restarts")
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--grace-period", type=int, default=8)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--brackets", type=int, default=2)
    parser.add_argument("--max-concurrent-search", type=int, default=8)
    parser.add_argument("--max-concurrent-replicate", type=int, default=5)
    parser.add_argument("--gpu-fraction", type=float, default=None)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--export-bucket", default=None,
                        help="Chameleon bucket for the checkpoint copy (default: $SPECTRUM_S3_BUCKET or 'spectrum')")
    parser.add_argument("--export-prefix", default="checkpoints-oct2026")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run_campaign(parse_args(argv))


if __name__ == "__main__":
    main()
