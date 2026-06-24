from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from training.common.aerpaw_loader import LoadedSpectrumData, load_aerpaw_data
from training.common.config import ROOT, resolve_path


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


def load_chunk(config: dict[str, Any], chunk: ChunkSpec) -> LoadedSpectrumData:
    data_dir = resolve_path(config["data"]["data_dir"])
    normalize = bool(config["preprocessing"].get("normalize", True))
    reference_site = str(config["data"].get("reference_site", "CC2"))
    return load_aerpaw_data(
        data_dir,
        chunk.start_mhz,
        chunk.end_mhz,
        normalize=normalize,
        reference_site=reference_site,
    )


def model_matrix_to_convlstm_frames(x: np.ndarray) -> np.ndarray:
    """Convert (T, F) model input to ConvLSTM frame layout (T, 1, 200)."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix (time, frequency), got shape {x.shape}")
    return x[:, None, :].astype(np.float32)
