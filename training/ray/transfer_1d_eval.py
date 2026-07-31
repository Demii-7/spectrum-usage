"""Launch independent zero-shot 1D transfer tasks on Ray with MinIO artifacts."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import yaml

from training.common.transfer_1d import (
    DEFAULT_MANIFEST,
    ROOT,
    SEED,
    TRANSFER_MODELS,
    evaluate_transfer_model,
    load_target,
    load_transfer_manifest,
    manifest_digest,
    preflight_checkpoints,
    repository_path,
    resolve_checkpoint,
    transfer_tasks,
)
from training.ray.optional import require_ray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument(
        "--task",
        action="append",
        dest="task_ids",
        default=None,
        help="Run only this stable task ID; repeat for a smoke subset.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_names",
        default=None,
        help="Run only this manifest model; repeat to select a subset.",
    )
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum-usage"))
    parser.add_argument("--prefix", default="spectrum-usage/transfer-1d")
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _s3_client():
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_S3_ENDPOINT_URL") \
        or os.environ.get("MINIO_ENDPOINT")
    return boto3.client("s3", endpoint_url=endpoint)


def _artifact_key(prefix: str, digest: str, task_id: str) -> str:
    return f"{prefix.strip('/')}/{digest}/{task_id}.zip"


def _object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _zip_directory(directory: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return payload.getvalue()


def _run_task(
    task,
    entry,
    models,
    model_names,
    config_path: str,
    bucket: str,
    prefix: str,
    digest: str,
    seed: int,
):
    client = _s3_client()
    key = _artifact_key(prefix, digest, task.task_id)
    if _object_exists(client, bucket, key):
        return {"task_id": task.task_id, "status": "skipped", "artifact_key": key}
    config = yaml.safe_load(repository_path(config_path).read_text(encoding="utf-8"))
    source = load_target(
        task,
        [repository_path(value) for value in entry["files"]],
        impute=bool(entry.get("impute", True)),
        max_missing_gap=int(entry.get("max_missing_gap", 10)),
    )
    with tempfile.TemporaryDirectory(prefix=f"transfer-1d-{task.task_id}-") as temporary:
        task_dir = Path(temporary)
        all_metrics = []
        for model_name in model_names:
            model_dir = task_dir / model_name
            model_dir.mkdir(parents=True)
            all_metrics.extend(evaluate_transfer_model(
                task=task,
                source=source,
                model_name=model_name,
                checkpoint_path=resolve_checkpoint(models[model_name]),
                config=config,
                output_dir=model_dir,
                seed=seed,
            ))
        pd.DataFrame(all_metrics).to_csv(task_dir / "aggregate_metrics.csv", index=False)
        (task_dir / "complete.json").write_text(
            json.dumps(
                {"task_id": task.task_id, "seed": seed, "manifest_sha256": digest},
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=_zip_directory(task_dir),
            ContentType="application/zip",
        )
    return {"task_id": task.task_id, "status": "completed", "artifact_key": key}


def _download_artifacts(results, output_dir: Path, bucket: str) -> pd.DataFrame:
    client = _s3_client()
    frames = []
    for result in results:
        payload = client.get_object(Bucket=bucket, Key=result["artifact_key"])["Body"].read()
        task_dir = output_dir / result["task_id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(task_dir)
        frames.append(pd.read_csv(task_dir / "aggregate_metrics.csv"))
    return pd.concat(frames, ignore_index=True)


def launch(args):
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = load_transfer_manifest(manifest_path)
    if args.seed != SEED:
        raise ValueError("1D transfer uses only seed 42 checkpoints")
    model_paths = manifest["models"]
    config = yaml.safe_load(repository_path(manifest["config"]).read_text(encoding="utf-8"))
    preflight_checkpoints(
        config,
        {name: resolve_checkpoint(path) for name, path in model_paths.items()},
    )
    digest = manifest_digest(manifest)
    all_tasks = {task.task_id: task for task in transfer_tasks()}
    unknown = set(args.task_ids or ()) - set(all_tasks)
    if unknown:
        raise ValueError(f"Unknown transfer task IDs: {sorted(unknown)}")
    selected = (
        [all_tasks[task_id] for task_id in args.task_ids]
        if args.task_ids
        else list(all_tasks.values())
    )
    model_names = tuple(args.model_names or TRANSFER_MODELS)
    unknown_models = set(model_names) - set(TRANSFER_MODELS)
    if unknown_models:
        raise ValueError(f"Unknown transfer model names: {sorted(unknown_models)}")
    ray = require_ray()
    ray.init(address=args.address, ignore_reinit_error=True)
    remote_task = ray.remote(
        num_cpus=1,
        num_gpus=0.5,
        max_retries=1 if args.retry_failed else 0,
    )(_run_task)
    pending, results = [], []
    limit = args.max_concurrent or len(selected)
    for task in selected:
        pending.append(remote_task.remote(
            task,
            manifest["tasks"][task.task_id],
            model_paths,
            model_names,
            manifest["config"],
            args.bucket,
            args.prefix,
            digest,
            args.seed,
        ))
        if len(pending) >= limit:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(ray.get(ready))
    results.extend(ray.get(pending))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _download_artifacts(results, args.output_dir, args.bucket)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    (args.output_dir / "transfer_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": args.seed,
                "manifest_sha256": digest,
                "bucket": args.bucket,
                "prefix": args.prefix,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    launch(parse_args(argv))


if __name__ == "__main__":
    main()
