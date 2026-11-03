"""Acceptance checks for a standalone Ray worker exposing multiple GPUs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import uuid

from .cluster_probe import _minio_round_trip, _spectrum_read, _worker_snapshot
from .optional import require_ray


class GPUProbe:
    def snapshot(self, imports: bool = False):
        import torch

        value = torch.ones((32, 32), device="cuda")
        assert (value @ value)[0, 0].item() == 32
        shared = torch.zeros(256 * 1024).share_memory_()
        assert shared.is_shared()
        result = _worker_snapshot(check_imports=imports)
        shm = os.statvfs("/dev/shm")
        result["shm_capacity_bytes"] = shm.f_frsize * shm.f_blocks
        result["pytorch_shared_memory"] = True
        return result

    def storage(self, action: str, key: str, payload: bytes | None = None):
        return _minio_round_trip(action, key, payload)

    def source(self):
        return _spectrum_read()

    def shared_file(self, path: str, payload: bytes | None = None):
        file = Path(path)
        if payload is not None:
            file.write_bytes(payload)
        return hashlib.sha256(file.read_bytes()).hexdigest()


def _all_gpus(expected_count: int):
    import torch

    assert torch.cuda.device_count() == expected_count
    result = []
    for index in range(expected_count):
        value = torch.ones((32, 32), device=f"cuda:{index}")
        assert (value @ value)[0, 0].item() == 32
        result.append(str(torch.cuda.get_device_properties(index).uuid))
    return result


def probe(expected_uuids: list[str], gpu_count: int = 4):
    ray = require_ray()
    ray.init(address="auto", ignore_reinit_error=True)
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    workers = [node for node in nodes if node.get("Resources", {}).get("GPU", 0)]
    assert len(nodes) == 2 and len(workers) == 1, "Expected one head and one GPU worker"
    assert workers[0]["Resources"]["GPU"] == gpu_count
    head = next(node for node in nodes if not node.get("Resources", {}).get("GPU", 0))
    assert head.get("Resources", {}).get("CPU", 0) == 0
    normalize = lambda value: value.lower().removeprefix("gpu-")
    actor = ray.remote(num_cpus=1, num_gpus=1)(GPUProbe)
    actors = [actor.remote() for _ in range(gpu_count)]
    repo = Path("/workspace/spectrum-usage")
    token = uuid.uuid4().hex
    data_file = repo / "data/.spectrum-sync" / f"acceptance-{token}.bin"
    checkpoint_file = repo / "checkpoints" / f".acceptance-{token}.bin"
    payload = token.encode()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        snapshots = ray.get([item.snapshot.remote(True) for item in actors], timeout=180)
        observed = {normalize(item["gpu_uuid"]) for item in snapshots}
        assert len(observed) == gpu_count, "One-GPU actors did not receive distinct GPUs"
        assert all(item["shm_capacity_bytes"] >= 256 * 1024**3 for item in snapshots)
        if expected_uuids:
            assert observed == {normalize(value) for value in expected_uuids}

        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_bytes(payload)
        assert all(value == digest for value in ray.get(
            [item.shared_file.remote(str(data_file)) for item in actors], timeout=60
        ))
        assert ray.get(actors[0].shared_file.remote(str(checkpoint_file), payload), timeout=60) == digest
        assert checkpoint_file.read_bytes() == payload
        assert all(value == digest for value in ray.get(
            [item.shared_file.remote(str(checkpoint_file)) for item in actors], timeout=60
        ))
        source = ray.get(actors[0].source.remote(), timeout=60)
        key = f"acceptance/standalone-{token}.bin"
        written = ray.get(actors[0].storage.remote("put", key, payload), timeout=60)
        try:
            assert all(value == written for value in ray.get(
                [item.storage.remote("get", key) for item in actors], timeout=60
            ))
        finally:
            ray.get(actors[0].storage.remote("delete", key), timeout=60)
    finally:
        data_file.unlink(missing_ok=True)
        checkpoint_file.unlink(missing_ok=True)
        for item in actors:
            ray.kill(item)

    all_gpu_uuids = ray.get(ray.remote(num_cpus=1, num_gpus=gpu_count)(_all_gpus).remote(gpu_count), timeout=180)
    assert {normalize(value) for value in all_gpu_uuids} == observed

    fractional_results = {}
    for fraction, per_gpu in [(0.5, 2), (0.25, 4)]:
        fractional_actor = ray.remote(num_cpus=1, num_gpus=fraction)(GPUProbe)
        fractional_actors = [fractional_actor.remote() for _ in range(gpu_count * per_gpu)]
        try:
            fractional = ray.get([item.snapshot.remote() for item in fractional_actors], timeout=180)
            counts = Counter(normalize(item["gpu_uuid"]) for item in fractional)
            assert set(counts) == observed and set(counts.values()) == {per_gpu}
            fractional_results[str(fraction)] = dict(counts)
        finally:
            for item in fractional_actors:
                ray.kill(item)

    inventories = {}
    for kind, filename in [("data", "inventory.json"), ("checkpoints", "checkpoint-inventory.json")]:
        inventory = json.loads((repo / "data/.spectrum-sync" / filename).read_text())
        inventories[kind] = {"objects": len(inventory["objects"]),
                             "bytes": sum(item["Size"] for item in inventory["objects"].values())}
    assert json.loads((repo / "data/locations/powder.json").read_text())
    result = {"alive_ray_nodes": len(nodes), "gpu_workers": len(workers),
              "cluster_resources": ray.cluster_resources(), "one_gpu_actors": snapshots,
              "four_gpu_task": all_gpu_uuids, "fractional_gpu_actors_per_uuid": fractional_results,
              "shared_data_and_checkpoints": True, "source_read": source,
              "minio_cross_gpu_actors": True, "inventories": inventories}
    ray.shutdown()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu-uuid", action="append", default=[])
    parser.add_argument("--gpu-count", type=int, default=4)
    args = parser.parse_args()
    if args.gpu_count < 1 or (args.expected_gpu_uuid and len(args.expected_gpu_uuid) != args.gpu_count):
        parser.error("GPU count must be positive and match the number of supplied UUIDs")
    print(json.dumps(probe(args.expected_gpu_uuid, args.gpu_count), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
