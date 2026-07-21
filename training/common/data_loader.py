"""Shared PyTorch DataLoader configuration and worker initialization."""

from __future__ import annotations

import random
import multiprocessing as mp
from typing import Any, Mapping

import numpy as np
import torch


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_loader_kwargs(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a ``data_loader`` config block and return DataLoader kwargs."""
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise TypeError("data_loader must be a mapping")

    allowed = {
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory",
        "multiprocessing_context",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown data_loader option(s): {', '.join(sorted(unknown))}")

    num_workers = config.get("num_workers", 0)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("data_loader.num_workers must be a non-negative integer")

    persistent_workers = config.get("persistent_workers", False)
    if not isinstance(persistent_workers, bool):
        raise ValueError("data_loader.persistent_workers must be a boolean")
    if persistent_workers and num_workers == 0:
        raise ValueError("data_loader.persistent_workers requires num_workers > 0")

    prefetch_factor = config.get("prefetch_factor", None)
    if (
        prefetch_factor is not None
        and (isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int) or prefetch_factor <= 0)
    ):
        raise ValueError("data_loader.prefetch_factor must be a positive integer or null")

    pin_memory = config.get("pin_memory", "auto")
    if pin_memory == "auto":
        pin_memory = torch.cuda.is_available()
    elif not isinstance(pin_memory, bool):
        raise ValueError("data_loader.pin_memory must be a boolean or 'auto'")

    multiprocessing_context = config.get("multiprocessing_context", None)
    if multiprocessing_context is not None and not isinstance(multiprocessing_context, str):
        raise ValueError("data_loader.multiprocessing_context must be a string or null")
    if multiprocessing_context is not None and multiprocessing_context not in mp.get_all_start_methods():
        raise ValueError(
            "data_loader.multiprocessing_context must be one of: "
            + ", ".join(mp.get_all_start_methods())
        )

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_data_loader_worker
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context
    return kwargs
