"""End-to-end acceptance probe for the spectrum Ray cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import uuid
from typing import Any

from .optional import require_ray


def _worker_snapshot(check_imports: bool = False, delay: float = 0.0) -> dict[str, Any]:
    import torch

    if delay:
        time.sleep(delay)
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        result.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            gpu_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
        )
    if check_imports:
        import einops
        import scipy
        import timm
        from training.common import model_factory
        from training.common.specialized_models import SPECIALIZED_MODELS

        result["imports"] = {
            "einops": einops.__version__,
            "scipy": scipy.__version__,
            "timm": timm.__version__,
            "registered_models": sorted(model_factory.SUPPORTED_MODELS - {"timeran"}),
            "specialized_models": sorted(SPECIALIZED_MODELS),
        }
    return result


def _minio_round_trip(action: str, key: str, payload: bytes | None = None) -> str:
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT")
    bucket = os.environ.get("S3_BUCKET", "spectrum-usage")
    client = boto3.client("s3", endpoint_url=endpoint)
    if action == "put":
        assert payload is not None
        client.put_object(Bucket=bucket, Key=key, Body=payload)
        return hashlib.sha256(payload).hexdigest()
    if action == "get":
        value = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return hashlib.sha256(value).hexdigest()
    if action == "delete":
        client.delete_object(Bucket=bucket, Key=key)
        return "deleted"
    raise ValueError(action)


def probe(
    address: str = "auto",
    *,
    expected_gpu_nodes: dict[str, str] | None = None,
    fractional_gpu: float | None = None,
    check_model_imports: bool = False,
    check_minio: bool = False,
) -> dict[str, Any]:
    ray = require_ray()
    ray.init(address=address, ignore_reinit_error=True)
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpu_nodes = [
        node for node in alive_nodes if float(node.get("Resources", {}).get("GPU", 0)) >= 1
    ]
    remote_snapshot = ray.remote(_worker_snapshot)
    snapshots: list[dict[str, Any]] = []
    for node in gpu_nodes:
        snapshots.append(
            ray.get(
                remote_snapshot.options(
                    num_cpus=1,
                    num_gpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"], soft=False
                    ),
                ).remote(check_model_imports)
            )
        )

    normalize_uuid = lambda value: str(value).lower().removeprefix("gpu-")
    observed = {snapshot["hostname"]: snapshot.get("gpu_uuid", "") for snapshot in snapshots}
    if expected_gpu_nodes:
        missing = set(expected_gpu_nodes) - set(observed)
        mismatched = {
            host: (expected_gpu_nodes[host], observed.get(host))
            for host in expected_gpu_nodes
            if host in observed
            and normalize_uuid(expected_gpu_nodes[host]) != normalize_uuid(observed[host])
        }
        if missing or mismatched:
            raise RuntimeError(
                f"GPU inventory mismatch: missing={sorted(missing)}, mismatched={mismatched}"
            )

    fractional_results: list[dict[str, Any]] = []
    if fractional_gpu is not None:
        if not 0 < fractional_gpu <= 1:
            raise ValueError("fractional GPU request must be in (0, 1]")
        pending = []
        per_node = max(1, int(1 / fractional_gpu))
        for node in gpu_nodes:
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            for _ in range(per_node):
                pending.append(
                    remote_snapshot.options(
                        num_cpus=1,
                        num_gpus=fractional_gpu,
                        scheduling_strategy=strategy,
                    ).remote(False, 1.0)
                )
        fractional_results = ray.get(pending)

    minio_result = None
    if check_minio:
        if len(gpu_nodes) < 2:
            raise RuntimeError("MinIO cross-worker check requires at least two GPU workers")
        remote_minio = ray.remote(_minio_round_trip)
        key = f"acceptance/{uuid.uuid4().hex}.bin"
        payload = os.urandom(4096)
        first = NodeAffinitySchedulingStrategy(node_id=gpu_nodes[0]["NodeID"], soft=False)
        second = NodeAffinitySchedulingStrategy(node_id=gpu_nodes[1]["NodeID"], soft=False)
        written = ray.get(remote_minio.options(scheduling_strategy=first).remote("put", key, payload))
        read = ray.get(remote_minio.options(scheduling_strategy=second).remote("get", key))
        ray.get(remote_minio.options(scheduling_strategy=first).remote("delete", key))
        if written != read:
            raise RuntimeError(f"MinIO checksum mismatch: wrote {written}, read {read}")
        minio_result = {"key": key, "sha256": written, "cross_worker": True}

    return {
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
        "alive_node_count": len(alive_nodes),
        "gpu_worker_count": len(gpu_nodes),
        "gpu_workers": snapshots,
        "fractional_gpu_tasks": fractional_results,
        "minio": minio_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--expected-gpu-node", action="append", default=[], metavar="HOST=UUID")
    parser.add_argument("--fractional-gpu-check", type=float)
    parser.add_argument("--check-model-imports", action="store_true")
    parser.add_argument("--check-minio-cross-worker", action="store_true")
    args = parser.parse_args()
    expected = {}
    for item in args.expected_gpu_node:
        if "=" not in item:
            parser.error("--expected-gpu-node requires HOST=UUID")
        host, gpu_uuid = item.split("=", 1)
        expected[host] = gpu_uuid
    result = probe(
        args.address,
        expected_gpu_nodes=expected or None,
        fractional_gpu=args.fractional_gpu_check,
        check_model_imports=args.check_model_imports,
        check_minio=args.check_minio_cross_worker,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
