"""
Shared data-selection interface for frequency-chunk training and evaluation.

This module translates the common project configuration into calls to the
appropriate dataset loader. It provides a small model-agnostic interface that
allows training and evaluation scripts to request one configured frequency
chunk without implementing loader-specific branching.

Primary responsibilities include:

- defining the immutable frequency-chunk specification used across the pipeline;
- converting configured chunk dictionaries into validated ChunkSpec objects;
- resolving project-relative data paths;
- selecting the configured AERPAW or POWDER loading path;
- distinguishing POWDER CSV inputs from POWDER map archives;
- validating that paired train and test map paths are provided together;
- forwarding chunk frequency bounds and preprocessing settings;
- forwarding validation-fraction information used for leakage-safe fitting;
- forwarding max-row limits used for reduced test runs;
- forwarding data.prediction_start_row to both POWDER CSV and map loaders; and
- returning a LoadedSpectrumData object through one consistent interface.

This module does not parse source files or perform normalization directly.
Those operations are delegated to training.common.dataset_loader and the shared
preprocessing utilities.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.common.config import resolve_path
from training.common.preprocessing import LoadedSpectrumData
from training.common.dataset_loader import load_aerpaw_data, load_powder_data, load_powder_map_data


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    start_mhz: float
    end_mhz: float


def chunk_specs(config: dict[str, Any]) -> list[ChunkSpec]:
    return [
        ChunkSpec(str(chunk["id"]), float(chunk["start_mhz"]), float(chunk["end_mhz"]))
        for chunk in config["data"]["chunks"]
    ]


def load_chunk( config: dict[str, Any], chunk: ChunkSpec, val_fraction: float,) -> LoadedSpectrumData:
    
    normalize = bool(config["preprocessing"].get("normalize", True))
    data_cfg = config["data"]

    prediction_start_row = data_cfg.get(
        "prediction_start_row"
    )
    
    if prediction_start_row is not None:
        prediction_start_row = int(
            prediction_start_row
        )
    
    loader = str(data_cfg.get("loader", "aerpaw")).lower()
    max_rows = config["data"].get("max_rows")
    
    if loader == "powder":
        reference_site = str(
            data_cfg.get("reference_site", "POWDER")
        )
    
        train_map_path = data_cfg.get("train_map_path")
        test_map_path = data_cfg.get("test_map_path")
    
        if train_map_path or test_map_path:
            if not train_map_path or not test_map_path:
                raise ValueError(
                    "Both data.train_map_path and data.test_map_path "
                    "must be provided for POWDER map data."
                )
    
            return load_powder_map_data(
                train_path=resolve_path(train_map_path),
                test_path=resolve_path(test_map_path),
                map_key=str(data_cfg.get("map_key", "map_db")),
                chunk_start_mhz=chunk.start_mhz,
                chunk_end_mhz=chunk.end_mhz,
                normalize=normalize,
                reference_site=reference_site,
                max_rows=max_rows,
                val_fraction=val_fraction,
                prediction_start_row=prediction_start_row,
            )
    
        train_files = [
            resolve_path(path)
            for path in data_cfg.get("train_files", [])
        ]
        test_files = [
            resolve_path(path)
            for path in data_cfg.get("test_files", [])
        ]
    
        return load_powder_data(
            train_files,
            test_files,
            chunk.start_mhz,
            chunk.end_mhz,
            normalize=normalize,
            reference_site=reference_site,
            max_rows=max_rows,
            val_fraction=val_fraction,
            prediction_start_row=prediction_start_row
        )

    data_dir = resolve_path(data_cfg["data_dir"])
    reference_site = str(data_cfg.get("reference_site", "CC2"))
    test_rows = int(data_cfg.get("test_rows", 2880))
    
    return load_aerpaw_data(
        data_dir,
        chunk.start_mhz,
        chunk.end_mhz,
        normalize=normalize,
        reference_site=reference_site,
        max_rows=max_rows,
        test_rows=test_rows,
        val_fraction=val_fraction,
    )
