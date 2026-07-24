"""Masked-token pretraining entrypoint for paper-native ConvLSTM-FM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_token_dataset
from preprocessing import SpectrogramConfig


def mask_tokens(tokens: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(1, round(tokens.shape[1] * ratio))
    mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    for row in mask:
        row[torch.randperm(tokens.shape[1], device=tokens.device)[:count]] = True
    mean = tokens.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = tokens.std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    corrupted = torch.where(mask[:, :, None, None, None], torch.randn_like(tokens) * std + mean, tokens)
    return corrupted, mask


def pretrain(model: nn.Module, loader: DataLoader, epochs: int, learning_rate: float, mask_ratio: float, device: str) -> None:
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for (tokens,) in loader:
            tokens = tokens.to(device)
            masked, mask = mask_tokens(tokens, mask_ratio)
            reconstruction = model.reconstruct(masked)
            mask_values = mask[:, :, None, None, None].expand_as(tokens)
            loss = (reconstruction - tokens).square()[mask_values].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    options = json.loads(Path(args.config).read_text())
    config = SpectrogramConfig(**options["preprocessing"])
    dataset = load_token_dataset(options["inputs"], config)
    from models.ConvLSTM_FM import ConvLSTMFMForecaster
    model_config = {"model": options["model"]}
    model = ConvLSTMFMForecaster(model_config)
    pretrain(model, DataLoader(dataset, batch_size=options.get("batch_size", 16), shuffle=True),
             options.get("epochs", 10), options.get("learning_rate", 1e-3),
             options.get("mask_ratio", 0.2), options.get("device", "cpu"))
    checkpoint = {
        "encoder_state_dict": model.encoder.state_dict(),
        "reconstruction_state_dict": model.head.state_dict(),
        "preprocessing": config.metadata(),
    }
    torch.save(checkpoint, options["output"])


if __name__ == "__main__":
    main()
