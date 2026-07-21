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
- validating that paired train and test paths are provided together;
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
from training.common.dataset_loader import load_aerpaw_data, load_powder_data


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
    """ Load and process data chunk automatically routing by file extension. """
    
    # Load data config
    normalize = bool(config["preprocessing"].get("normalize", True))
    impute = bool(config["preprocessing"].get("impute", False))
    data_cfg = config["data"]

    # Prediction_start_row is ONLY VALID FOR STS-PREDNET
    prediction_start_row = data_cfg.get(
        "prediction_start_row"
    )
    
    if prediction_start_row is not None:
        prediction_start_row = int(
            prediction_start_row
        )

    # Extract the loader so prgram know which one to call
    loader = str(data_cfg.get("loader", "aerpaw")).lower()
    max_rows = config["data"].get("max_rows")
    
    if loader == "powder" or loader == "cosmos" or loader == "ara":
        reference_site = str(
            data_cfg.get("reference_site", "Not provided")
        )
        # Get raw config inputs (can be a string or a list)
        raw_train = data_cfg.get("train_files")
        raw_test = data_cfg.get("test_files")

        if isinstance(raw_train, list) or isinstance(raw_test, list):
            raise ValueError(
                "The combined POWDER loader expects one train file "
                "and one test file."
            )
        
        if raw_train is None or raw_test is None:
            raise ValueError(
                "Both data.train_files and data.test_files must be provided."
            )
        
        # Resolve all paths provided
        train_path = resolve_path(raw_train) 
        test_path = resolve_path(raw_test)

        if not train_path or not test_path:
            raise ValueError("Both data.train_paths and data.test_paths must be provided.")
            
        # Check if the path is an NPZ map file
        train_suffix = train_path.suffix.lower()
        test_suffix = test_path.suffix.lower()
        
        if train_suffix != test_suffix:
            raise ValueError(
                "POWDER train and test files must use the same "
                "input representation."
            )
        # Check for map or csv input
        if train_suffix == ".npz":
            input_format = "map"
        elif train_suffix == ".csv":
            input_format = "csv"
        else:
            raise ValueError(
                "POWDER train and test files must be CSV or NPZ files."
            )
    
        return load_powder_data(
            train_path=train_path,
            test_path=test_path,
            chunk_start_mhz=chunk.start_mhz,
            chunk_end_mhz=chunk.end_mhz,
            input_format=input_format,
            map_key=str(data_cfg.get("map_key", "map_db")),
            normalize=normalize,
            reference_site=reference_site,
            max_rows=max_rows,
            val_fraction=val_fraction,
            prediction_start_row=prediction_start_row,
            impute=impute,
        )

    data_dir = resolve_path(data_cfg["data_dir"])
    reference_site = str(data_cfg.get("reference_site", "CC2"))
    test_rows = int(data_cfg.get("test_rows", 2880))
    
    if loader == "aerpaw":
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

    raise ValueError(
        f"Unsupported data loader: {loader!r}."
    )
