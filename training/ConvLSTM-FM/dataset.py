"""Datasets for IQ recordings and precomputed ConvLSTM-FM tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from preprocessing import SpectrogramConfig, recordings_to_sentences, sentences_to_tokens


def load_token_dataset(paths: list[str | Path], config: SpectrogramConfig) -> TensorDataset:
    """Load complex .npy recordings or precomputed .npy/.pt sentence/token tensors."""
    token_batches = []
    iq_recordings = []
    for path_value in paths:
        path = Path(path_value)
        value = torch.load(path, map_location="cpu", weights_only=True) if path.suffix == ".pt" else np.load(path)
        tensor = torch.as_tensor(value)
        if torch.is_complex(tensor):
            iq_recordings.append(tensor)
        elif tensor.ndim == 5:
            token_batches.append(tensor.float())
        elif tensor.ndim in (3, 4):
            token_batches.append(sentences_to_tokens(tensor.float(), config.token_count))
        else:
            raise ValueError(f"Unsupported tensor shape in {path}: {tuple(tensor.shape)}")
    if iq_recordings:
        sentences, _ = recordings_to_sentences(iq_recordings, config)
        token_batches.append(sentences_to_tokens(sentences, config.token_count))
    if not token_batches:
        raise ValueError("No complete sentences were loaded")
    return TensorDataset(torch.cat(token_batches))
