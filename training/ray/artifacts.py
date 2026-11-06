"""Upload a trained model's weights, with a metadata sidecar, to the results bucket.

Every runner that trains a model should call :func:`save_artifact` before its scratch directory is deleted, so the weights
and the exact settings that produced them survive the Ray lease. Two objects are written::

    <key_prefix>/<name>            the weights file, unchanged
    <key_prefix>/<name>.meta.json  config, seed, code revision, host, size, sha256, timestamp

A failure to upload never fails the run (the metrics matter more), but it is recorded in the returned dictionary so the
caller can store ``artifact_error`` next to the metrics.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

REVISION_FILES = ("/workspace/spectrum-source-revision", "/home/cc/spectrum-source-revision")


def code_revision(repo_root: str = "/workspace/spectrum-usage") -> str:
    """Commit of the synced source: the sync script's revision file if present, else ``git rev-parse``, else ``unknown``."""
    for path in REVISION_FILES + (f"{repo_root}/../spectrum-source-revision",):
        try:
            text = Path(path).read_text().strip()
            if text:
                return text.splitlines()[0]
        except OSError:
            pass
    try:
        return subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return "unknown"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_keys(key_prefix: str, name: str) -> tuple[str, str]:
    prefix = key_prefix.strip("/")
    return f"{prefix}/{name}", f"{prefix}/{name}.meta.json"


def save_artifact(client, bucket: str, key_prefix: str, path: str | Path, meta: dict[str, Any] | None = None,
                  name: str | None = None, repo_root: str = "/workspace/spectrum-usage") -> dict[str, Any]:
    """Upload ``path`` and its sidecar; return ``{"weights_key", "meta_key", "bytes", "sha256"}`` or ``{"artifact_error"}``."""
    path = Path(path)
    try:
        weights_key, meta_key = artifact_keys(key_prefix, name or path.name)
        sidecar = {**(meta or {}), "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path),
                   "code_revision": code_revision(repo_root), "host": socket.gethostname(),
                   "saved_at_unix": round(time.time(), 1)}
        client.upload_file(str(path), bucket, weights_key)
        client.put_object(Bucket=bucket, Key=meta_key, Body=json.dumps(sidecar, default=str, indent=1).encode())
        return {"weights_key": weights_key, "meta_key": meta_key, "bytes": sidecar["bytes"], "sha256": sidecar["sha256"]}
    except Exception as error:
        return {"artifact_error": f"{type(error).__name__}: {error}"[:300]}


def save_run(client, bucket: str, key_prefix: str, run_dir: str | Path, config: dict[str, Any] | None = None,
             extra: dict[str, Any] | None = None, repo_root: str = "/workspace/spectrum-usage") -> dict[str, Any]:
    """Upload the checkpoint(s) of an ``integrated`` training run with the resolved config and its ``run_costs``."""
    run_dir = Path(run_dir)
    meta: dict[str, Any] = {"config": config, **(extra or {})}
    try:
        from training.common.run_costs import read_run_costs

        meta["run_costs"] = read_run_costs(run_dir)
    except Exception as error:
        meta["run_costs_error"] = f"{type(error).__name__}: {error}"[:200]
    saved = [save_artifact(client, bucket, key_prefix, path, meta, repo_root=repo_root)
             for path in sorted((run_dir / "checkpoints").glob("*.pt"))]
    errors = [s["artifact_error"] for s in saved if "artifact_error" in s]
    out: dict[str, Any] = {"artifact_keys": [s["weights_key"] for s in saved if "weights_key" in s]}
    if errors or not saved:
        out["artifact_error"] = "; ".join(errors) or "no checkpoint found"
    return out
