"""Optional dependency helpers."""

from __future__ import annotations

import importlib


class RayNotInstalledError(RuntimeError):
    """Raised when a Ray-backed operation is requested without Ray."""


def require_ray():
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise RayNotInstalledError(
            "Ray is required for this operation. Install it with `pip install "
            "\"ray[tune]\"` on the driver and workers. Dry-run planning and "
            "selection do not require Ray."
        ) from exc
