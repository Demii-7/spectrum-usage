"""Evaluate matched POWDER-trained TimeRAN and MOMENT heads on transfer tasks."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import boto3
import pandas as pd
import ray
import yaml

from training.common.transfer_2d import (
    ROOT,
    SEED,
    evaluate_transfer_model,
    load_target,
    repository_path,
    transfer_tasks,
)
from training.common.config import load_config


RUN_ROOT = ROOT.parent
CHECKPOINTS = {
    "timeran": RUN_ROOT / "checkpoints/source_timeran.pt",
    "moment": RUN_ROOT / "checkpoints/source_moment.pt",
}
TRAIN_CONFIGS = {
    "timeran": ROOT / "training/configs/timeran_powder_one_seed.yaml",
    "moment": ROOT / "training/configs/moment_powder_one_seed.yaml",
}


def _s3_client():
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")
    )
    return boto3.client("s3", endpoint_url=endpoint)


def _zip_bytes(directory: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return payload.getvalue()


@ray.remote(num_gpus=1, num_cpus=4)
def _run_variant(variant: str, bucket: str, prefix: str) -> dict[str, str]:
    train_config = load_config(TRAIN_CONFIGS[variant])
    transfer_config = yaml.safe_load(
        (ROOT / "training/configs/transfer_2d.yaml").read_text(encoding="utf-8")
    )
    transfer_config["timeran"] = train_config["timeran"]
    manifest = json.loads(
        (ROOT / "training/configs/transfer_2d_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    task_by_id = {
        task.task_id: task
        for task in transfer_tasks()
        if task.platform in {"cosmos", "ara", "aerpaw"}
    }
    checkpoint = CHECKPOINTS[variant]

    with tempfile.TemporaryDirectory(prefix=f"timeran-transfer-{variant}-") as temporary:
        output_dir = Path(temporary) / variant
        output_dir.mkdir()
        rows: list[dict[str, object]] = []
        for task in task_by_id.values():
            entry = manifest["tasks"][task.task_id]
            source = load_target(
                task,
                [repository_path(value) for value in entry["files"]],
                impute=bool(entry.get("impute", True)),
                max_missing_gap=int(entry.get("max_missing_gap", 10)),
            )
            task_dir = output_dir / task.task_id
            task_dir.mkdir()
            rows.extend(
                evaluate_transfer_model(
                    task=task,
                    source=source,
                    model_name="timeran",
                    checkpoint_path=checkpoint,
                    config=transfer_config,
                    output_dir=task_dir,
                    seed=SEED,
                )
            )
            shutil.rmtree(task_dir)
        pd.DataFrame(rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)
        key = f"{prefix.strip('/')}/transfer-{variant}.zip"
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=_zip_bytes(output_dir),
            ContentType="application/zip",
        )
        return {
            "variant": variant,
            "artifact_key": key,
            "node_ip": ray.util.get_node_ip_address(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum-usage"))
    parser.add_argument("--prefix", default="spectrum-usage/timeran-fast-v1")
    args = parser.parse_args()

    ray.init(address=args.address, ignore_reinit_error=True)
    results = ray.get(
        [
            _run_variant.remote(variant, args.bucket, args.prefix)
            for variant in ("timeran", "moment")
        ]
    )
    output_dir = RUN_ROOT / "transfer-results"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _s3_client()
    for result in results:
        payload = client.get_object(
            Bucket=args.bucket,
            Key=result["artifact_key"],
        )["Body"].read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(output_dir / result["variant"])
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps({"seed": SEED, "variants": results}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
