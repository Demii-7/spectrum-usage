"""Provenance and result manifest writers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping


SCHEMA_VERSION = 1


def provenance_manifest(*, command: list[str], config_path: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                                text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "command": list(command), "config_path": str(config_path.resolve()), "git_commit": commit,
            "python": platform.python_version(), **dict(extra or {})}


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def result_manifest(*, experiment_id: str, selections: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(), "selections": dict(selections),
            "candidates": [dict(candidate) for candidate in candidates]}
