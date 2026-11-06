"""Copy the Time-Series-Library baselines' checkpoints from the Ray workers' container disks to the results bucket.

``tsl_runs.py`` and ``tune_tsl.py`` ran the library inside the worker containers, which keep each run's weights under
``/workspace/tsl/Time-Series-Library/checkpoints/<setting>/checkpoint.pth``. The setting name encodes dataset, horizon, seed,
model and the run label (``--des``), so runs do not overwrite each other. This job runs one task on every alive node that has
such a directory and uploads the *final* runs: the library defaults (label ``electrosense``) and the tuned finals
(``tuned-<Model>``). Search trials (``tune-<Model>-c<N>``) are skipped; their hyperparameters are in the trial records.

Objects: ``<prefix>/<setting>/checkpoint.pth`` plus ``<prefix>/<setting>/meta.json`` (size, mtime, host, parsed fields).
Re-running skips files already uploaded with the same size.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CHECKPOINT_ROOT = "/workspace/tsl/Time-Series-Library/checkpoints"
SETTING = re.compile(r"^long_term_forecast_(?P<dataset>spec\d+)_192_(?P<horizon>\d+)_s(?P<seed>\d+)_(?P<model>[A-Za-z0-9]+)_custom_.*_dtTrue_(?P<label>.+?)(?:_\d+)?$")


def parse_setting(name: str) -> dict[str, Any] | None:
    match = SETTING.match(name)
    if not match:
        return None
    fields = match.groupdict()
    return {"dataset": fields["dataset"], "horizon": int(fields["horizon"]), "seed": int(fields["seed"]),
            "model": fields["model"], "label": fields["label"]}


def is_final(name: str) -> bool:
    """Library defaults (``electrosense``) and tuned finals (``tuned-...``); not search trials (``tune-...-cN``)."""
    parsed = parse_setting(name)
    return bool(parsed) and (parsed["label"] == "electrosense" or parsed["label"].startswith("tuned-"))


def upload_node(args: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, args["repo_root"])
    from training.ray.storage import spectrum_s3_client

    client, bucket, prefix = spectrum_s3_client(), args["bucket"], args["prefix"].strip("/")
    root = Path(CHECKPOINT_ROOT)
    out = {"host": socket.gethostname(), "found": 0, "uploaded": 0, "skipped": 0, "failed": [], "bytes": 0}
    if not root.is_dir():
        return out
    todo = [d for d in sorted(root.iterdir()) if d.is_dir() and is_final(d.name) and (d / "checkpoint.pth").is_file()]
    out["found"] = len(todo)

    def one(directory: Path) -> tuple[str, int]:
        path = directory / "checkpoint.pth"
        key = f"{prefix}/{directory.name}/checkpoint.pth"
        size = path.stat().st_size
        try:
            if client.head_object(Bucket=bucket, Key=key)["ContentLength"] == size:
                return "skipped", 0
        except Exception:
            pass
        client.upload_file(str(path), bucket, key)
        if client.head_object(Bucket=bucket, Key=key)["ContentLength"] != size:
            raise RuntimeError("size mismatch after upload")
        meta = {**(parse_setting(directory.name) or {}), "setting": directory.name, "bytes": size, "mtime_unix": path.stat().st_mtime,
                "host": socket.gethostname(), "recovered_at_unix": round(time.time(), 1)}
        client.put_object(Bucket=bucket, Key=f"{prefix}/{directory.name}/meta.json", Body=json.dumps(meta).encode())
        return "uploaded", size

    with ThreadPoolExecutor(args["threads"]) as pool:
        futures = {pool.submit(one, d): d.name for d in todo}
        for future, name in futures.items():
            try:
                state, size = future.result()
                out[state] += 1
                out["bytes"] += size
            except Exception as error:
                out["failed"].append(f"{name}: {type(error).__name__}: {error}"[:200])
    return out


def main(argv=None) -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="tsl-checkpoints")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--bucket", default=None)
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    bucket = args.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    ray.init(address=args.address, ignore_reinit_error=True)
    worker_args = {"repo_root": args.repo_root, "bucket": bucket, "prefix": args.prefix, "threads": args.threads}
    futures = []
    for node in ray.nodes():
        if node.get("Alive"):
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            futures.append(ray.remote(num_cpus=1, scheduling_strategy=strategy, max_retries=0)(upload_node).remote(worker_args))
    for result in ray.get(futures):
        print("[recover]", json.dumps({**result, "failed": result["failed"][:5], "n_failed": len(result["failed"]),
                                       "gb": round(result["bytes"] / 1e9, 2)}), flush=True)


if __name__ == "__main__":
    main()
