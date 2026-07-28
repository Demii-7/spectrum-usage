"""Lazy dispatch for models whose training contract is not a plain forecaster."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from training.common.training_events import TrainingCallback

SPECIALIZED_MODELS = {"stsprednet", "tss_lcd", "deepspred"}
_MODEL_DIRECTORIES = {
    "stsprednet": "STS-PredNet",
    "tss_lcd": "TSS-LCD",
    "deepspred": "DeepSPred",
}
_LOCAL_MODULE_NAMES = {
    "dataset", "model", "utils", "config_support", "train_integrated"
}


def _load_module(model_name: str, phase: str) -> ModuleType:
    """Load a legacy runner without leaking its generic local imports."""
    directory = Path(__file__).resolve().parents[1] / _MODEL_DIRECTORIES[model_name]
    path = directory / f"{phase}_integrated.py"
    module_name = f"_spectrum_usage_{model_name}_{phase}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    saved = {name: sys.modules.get(name) for name in _LOCAL_MODULE_NAMES}
    for name in _LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load specialized runner {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))
        for name in _LOCAL_MODULE_NAMES:
            sys.modules.pop(name, None)
        for name, previous in saved.items():
            if previous is not None:
                sys.modules[name] = previous


def train_specialized_chunk(
    model_name: str,
    config: dict[str, Any],
    chunk,
    data,
    output_directory: Path,
    checkpoint_directory: Path,
    callback: TrainingCallback | None = None,
) -> Path:
    module = _load_module(model_name, "train")
    if hasattr(module, "train_chunk"):
        return module.train_chunk(
            config, chunk, data, output_directory, checkpoint_directory,
            callback=callback,
        )

    split = data.splits[data.train_split]
    validation = data.splits.get(data.validation_split)
    module.train_one_model(
        config,
        split.model_input,
        split.segments,
        checkpoint_directory,
        output_directory,
        chunk.chunk_id,
        frequencies=data.frequencies,
        normalization=data.normalization,
        validation_data=None if validation is None else validation.model_input,
        validation_segments=() if validation is None else validation.segments,
        callback=callback,
    )
    return checkpoint_directory / f"{chunk.chunk_id}_{model_name}.pt"


def evaluate_specialized_chunk(
    model_name: str,
    config: dict[str, Any],
    chunk,
    bands,
    output_directory: Path,
    checkpoint_path: Path,
):
    module = _load_module(model_name, "evaluate")
    function = getattr(module, "evaluate_chunk", None)
    if function is None and model_name == "stsprednet":
        function = module.evaluate_csv_chunk
    if function is None:
        raise AttributeError(f"{model_name} has no callable integrated evaluator")
    return function(config, chunk, bands, output_directory, checkpoint_path)
