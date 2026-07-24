"""Frozen-backbone paper-style ConvLSTM-FM segmentation head."""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMFMSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, feature_channels: int = 64, classes: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, classes, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            encoded = self.backbone.encoder(tokens)
            sequence = encoded[0][-1] if isinstance(encoded, tuple) else encoded
        if sequence.ndim != 5:
            raise ValueError("backbone must return (B, T, C, H, Wtoken) features")
        features = sequence.permute(0, 2, 3, 1, 4).reshape(
            sequence.shape[0], sequence.shape[2], sequence.shape[3], sequence.shape[1] * sequence.shape[4]
        )
        return self.classifier(features)
