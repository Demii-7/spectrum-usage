"""Train and evaluate matched TimeRAN and raw MOMENT heads on POWDER."""

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
import ray

from training.common.config import load_config
from training.common.evaluation_integrated import evaluate_one_model
from training.common.train_integrated import train_one_model


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent
CONFIGS = {
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


def _zip_directory(directory: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return payload.getvalue()


@ray.remote(num_gpus=1, num_cpus=4)
def _run_variant(variant: str, bucket: str, prefix: str) -> dict[str, str]:
    config_path = CONFIGS[variant]
    config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix=f"timeran-{variant}-") as temporary:
        output_dir = Path(temporary) / variant
        train_one_model(config, "timeran", output_dir)
        checkpoint = next((output_dir / "checkpoints").glob("*.pt"))
        evaluate_one_model(
            config,
            "timeran",
            output_dir,
            checkpoint,
            skip_plots=True,
        )
        payload = _zip_directory(output_dir)
        key = f"{prefix.strip('/')}/{variant}.zip"
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
        )
        return {
            "variant": variant,
            "artifact_key": key,
            "node_ip": ray.util.get_node_ip_address(),
        }


def _download_artifact(client, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "spectrum-usage"))
    parser.add_argument("--prefix", default="spectrum-usage/timeran-fast-v1")
    args = parser.parse_args()

    ray.init(address=args.address, ignore_reinit_error=True)
    futures = [
        _run_variant.remote(variant, args.bucket, args.prefix)
        for variant in ("timeran", "moment")
    ]
    results = ray.get(futures)

    output_root = RUN_ROOT / "results"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    client = _s3_client()
    for result in results:
        _download_artifact(
            client,
            args.bucket,
            result["artifact_key"],
            output_root / result["variant"],
        )
    (output_root / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "variants": results,
                "configs": {name: str(path) for name, path in CONFIGS.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
