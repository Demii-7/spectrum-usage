"""Upload local checkpoint files that the durable bucket does not have yet (additive, never overwrites, verifies sizes).

Each node's checkpoints live under /workspace/spectrum-usage/checkpoints (the host's /home/cc/spectrum-usage/checkpoints). For every
file there the key ``checkpoints/<relative path>`` is checked in the bucket: identical size -> already backed up (skipped); missing ->
uploaded, then re-checked by size; present with a *different* size -> left alone and reported as a conflict (nothing is overwritten).
One Ray task runs on each requested node (matched by its private address), so each node uploads its own files.

    python -m training.ray.upload_local_checkpoints --node-ips 192.168.1.39 192.168.1.185
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def classify(local_size: int, remote_size: int | None) -> str:
    """'present' (same size), 'missing' (upload it) or 'conflict' (the bucket has a different object under that key)."""
    if remote_size is None:
        return "missing"
    return "present" if remote_size == local_size else "conflict"


def upload_node(args: dict) -> dict:
    import socket

    from botocore.exceptions import ClientError

    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    root = Path(args["root"])
    counts = {"present": 0, "uploaded": 0, "conflict": 0, "failed": 0}
    bytes_up, conflicts, failures = 0, [], []
    started = time.time()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        key = f"{args['prefix'].strip('/')}/{path.relative_to(root).as_posix()}"
        size = path.stat().st_size
        try:
            try:
                remote = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
            except ClientError as error:
                if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                    raise
                remote = None
            state = classify(size, remote)
            if state == "present":
                counts["present"] += 1
            elif state == "conflict":
                counts["conflict"] += 1
                conflicts.append({"key": key, "local": size, "remote": remote})
            else:
                client.upload_file(str(path), bucket, key)
                after = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
                if after != size:
                    raise RuntimeError(f"size after upload {after} != {size}")
                counts["uploaded"] += 1
                bytes_up += size
                print(f"[upload] {socket.gethostname()} uploaded {key} ({size / 1e6:.0f} MB)", flush=True)
        except Exception as error:
            counts["failed"] += 1
            failures.append({"key": key, "error": f"{type(error).__name__}: {error}"[:200]})
    return {"host": socket.gethostname(), "counts": counts, "uploaded_gb": round(bytes_up / 1e9, 2), "conflicts": conflicts[:20],
            "failures": failures[:20], "seconds": round(time.time() - started)}


def main() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node-ips", nargs="+", required=True)
    parser.add_argument("--root", default="/workspace/spectrum-usage/checkpoints")
    parser.add_argument("--prefix", default="checkpoints")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--address", default="auto")
    args = parser.parse_args()
    ray.init(address=args.address, ignore_reinit_error=True)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    nodes = {n["NodeManagerAddress"]: n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) >= 1}
    refs = []
    for ip in args.node_ips:
        if ip not in nodes:
            print(f"[upload] no alive node with address {ip}; skipping", flush=True)
            continue
        strategy = NodeAffinitySchedulingStrategy(node_id=nodes[ip]["NodeID"], soft=False)
        refs.append(ray.remote(num_cpus=1, num_gpus=0, max_retries=0, scheduling_strategy=strategy)(upload_node)
                    .remote({"root": args.root, "prefix": args.prefix, "bucket": bucket}))
    for result in ray.get(refs):
        print("[upload] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
