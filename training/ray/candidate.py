"""Deterministic candidate identity and plan construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def candidate_id(model_name: str, parameters: Mapping[str, Any], seed: int | None = None,
                 *, length: int = 16) -> str:
    payload = {"model": model_name.lower(), "parameters": dict(parameters)}
    if seed is not None:
        payload["seed"] = int(seed)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]
    return f"{model_name.lower()}-{digest}"
