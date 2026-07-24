"""
ConvLSTM-FM: a ConvLSTM backbone with masked-reconstruction self-supervised
pretraining, adapted to this repo's spectrum-map forecasting pipeline.

Reimplements the core idea of "Self-Supervised Radio Pre-training: Toward
Foundational Models for Spectrogram Learning" (Aboulfotouh, Eshaghbeigi,
Karslidis, Abou-Zeid -- IEEE GLOBECOM 2024): a multi-layer ConvLSTM backbone is
first pretrained to reconstruct randomly masked timesteps of its own input
window (Masked Spectrogram Modeling / MSM), then reused -- optionally frozen --
as the encoder for downstream forecasting.

Full paper spec (including the IQ-capture spectrogram pipeline, token/sentence
tokenization, and the segmentation downstream task) lives in
training/ConvLSTM-FM/info.md. This implementation intentionally keeps only what
maps onto data this repo already has (spectrum maps used by ConvLSTM /
ResidualConvLSTM): the ConvLSTM backbone, masked-reconstruction pretraining,
and a one-step forecasting head that can freeze the pretrained backbone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.ConvLSTM import ConvLSTM, _get_activation


def mask_sequence(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace a random subset of timesteps with white noise matching each
    sequence's own mean/std, following the paper's "mask ~20% of tokens with
    white noise" procedure (applied here at per-timestep granularity rather
    than per spectrogram-token).

    Returns (masked_x, mask) where mask[b, t] is True for a masked timestep.
    """
    b, t, c, h, w = x.shape
    n_masked = max(1, int(round(t * mask_ratio)))
    mask = torch.zeros(b, t, dtype=torch.bool, device=x.device)
    for i in range(b):
        idx = torch.randperm(t, device=x.device)[:n_masked]
        mask[i, idx] = True

    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = x.std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(x) * std + mean

    masked_x = torch.where(mask.view(b, t, 1, 1, 1), noise, x)
    return masked_x, mask


def masked_reconstruction_loss(model: "ConvLSTMFMForecaster", x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Masked MSE (Eq. 1 of the paper): unmasked timesteps contribute zero loss."""
    masked_x, mask = mask_sequence(x, mask_ratio)
    recon = model.reconstruct(masked_x)
    mask_weight = mask.view(*mask.shape, 1, 1, 1).float()
    squared_error = (recon - x) ** 2 * mask_weight
    denominator = mask_weight.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


class ConvLSTMFMForecaster(nn.Module):
    """
    Multi-layer ConvLSTM backbone (the "foundation" encoder) + a lightweight
    Conv2d head, used in two modes:

    - ``reconstruct(x)``: reconstructs every timestep of the input sequence,
      used for masked self-supervised pretraining (paper Stage A).
    - ``forward(x)``: predicts the single next timestep, matching the shared
      one-step forecasting interface used across this repo (`prediction_horizon`
      must be 1 -- multi-step rollout is handled by
      `training/common/forecasting.py`, same as ConvLSTM/ResidualConvLSTM).

    Setting ``freeze_backbone: true`` freezes the ConvLSTM encoder so only the
    head keeps training, matching the paper's Stage B fine-tuning ("only the
    final layer is fine-tuned").
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        c = config["model"]

        self.input_channels = int(c["input_channels"])
        self.spatial_h = int(c["grid_height"])
        self.spatial_w = int(c["grid_width"])
        self.prediction_horizon = int(c.get("prediction_horizon", 1))
        if self.prediction_horizon != 1:
            raise ValueError(
                "Error! ConvLSTM-FM only supports prediction_horizon=1; "
                "multi-step rollout is handled by the shared forecasting pipeline."
            )

        # Paper defaults: 5 ConvLSTM layers, 64 kernels each, 3x3 kernels, ReLU.
        hidden = list(c.get("hidden_channels", [64, 64, 64, 64, 64]))
        num_layers = int(c.get("num_layers", len(hidden)))
        kernels = [tuple(k) for k in c.get("kernel_size", [[3, 3]] * num_layers)]
        activation = _get_activation(c.get("cell_activation", "relu"))
        dropout = float(c.get("dropout", 0.0))

        self.encoder = ConvLSTM(
            input_dim=self.input_channels,
            hidden_dim=hidden,
            kernel_size=kernels,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            activation=activation,
        )
        # Approximates the paper's Conv3D head with a per-timestep Conv2d,
        # since the backbone already models the time axis recurrently.
        self.head = nn.Conv2d(hidden[-1], self.input_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        pretrained_backbone = c.get("pretrained_backbone")
        if pretrained_backbone:
            checkpoint = torch.load(pretrained_backbone, map_location="cpu")
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

        if c.get("freeze_backbone", False):
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze the ConvLSTM encoder so only the head keeps training (paper Stage B)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct every input timestep, for masked self-supervised pretraining."""
        b, t, c_in, h, w = x.shape
        layer_outputs, _ = self.encoder(x)
        sequence = layer_outputs[-1]  # (B, T, hidden, H, W) -- full sequence, top layer
        sequence = self.dropout(sequence)
        flat = sequence.reshape(b * t, sequence.shape[2], h, w)
        return self.head(flat).reshape(b, t, c_in, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One-step forecast: predict the timestep immediately after the input window."""
        if x.dim() != 5:
            raise ValueError(f"Error! Expected 5D input (B, T, C, H, W), got {x.dim()}D")
        b, t_in, c_in, h, w = x.shape
        if c_in != self.input_channels:
            raise ValueError(f"Error! Input channels {c_in} != model input_channels {self.input_channels}")
        if h != self.spatial_h or w != self.spatial_w:
            raise ValueError(f"Error! Input spatial ({h}, {w}) != model spatial ({self.spatial_h}, {self.spatial_w})")

        _, last_states = self.encoder(x)
        h_last, _ = last_states[-1]  # final hidden state of the top layer, (B, hidden, H, W)
        h_last = self.dropout(h_last)
        out = self.head(h_last)
        return out.unsqueeze(1)


def pretrain_backbone(
    model: ConvLSTMFMForecaster,
    train_data: np.ndarray,
    input_sequence_length: int,
    epochs: int,
    mask_ratio: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> None:
    """
    Masked-reconstruction (MSM) self-supervised pretraining of the backbone
    (paper Stage A / Algorithm 1), run directly on the same map data used for
    forecasting, before the shared pipeline fine-tunes the model on next-step
    prediction. Windows are built the same way as the shared forecasting
    windows, just without needing separate future targets.
    """
    from training.common.windowing import to_model_layout

    frames = torch.from_numpy(to_model_layout(train_data)).float()  # (T, C, H, W)
    n_windows = frames.shape[0] - input_sequence_length + 1
    if n_windows < 1:
        raise ValueError(
            f"Error! Not enough timesteps ({frames.shape[0]}) to build a single "
            f"pretraining window of length {input_sequence_length}."
        )
    windows = torch.stack(
        [frames[start : start + input_sequence_length] for start in range(n_windows)],
        dim=0,
    )  # (N, T, C, H, W)

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(windows.shape[0])
        total_loss = 0.0
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[permutation[start : start + batch_size]].to(device)
            optimizer.zero_grad()
            loss = masked_reconstruction_loss(model, batch, mask_ratio)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
        print(
            f"[ConvLSTM-FM pretrain] epoch {epoch:03d}/{epochs} "
            f"masked_recon_loss={total_loss / windows.shape[0]:.6f}"
        )
