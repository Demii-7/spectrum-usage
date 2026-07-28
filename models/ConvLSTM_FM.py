"""
ConvLSTM-FM: a ConvLSTM backbone with masked-reconstruction self-supervised
pretraining, adapted to this repo's spectrum-map forecasting pipeline.

Reimplements the core idea of "Self-Supervised Radio Pre-training: Toward
Foundational Models for Spectrogram Learning" (Aboulfotouh, Eshaghbeigi,
Karslidis, Abou-Zeid -- IEEE GLOBECOM 2024): a multi-layer ConvLSTM backbone is
first pretrained to reconstruct randomly masked timesteps of its own input
window (Masked Spectrogram Modeling / MSM), then reused -- optionally frozen --
as the encoder for downstream forecasting.

    The shared path adapts the method to spectrum maps. Paper-native IQ
    preprocessing, radio-sentence tokenization, and segmentation support live in
    training/ConvLSTM-FM/.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ConvLSTM import ConvLSTM, _get_activation
from training.common.training_events import TrainingCallback, emit_training_event


def mask_sequence(
    x: torch.Tensor,
    mask_ratio: float,
    mask_mode: str = "tokens",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace a random subset of timesteps with white noise matching each
    sequence's own mean/std, following the paper's "mask ~20% of tokens with
    white noise" procedure (applied here at per-timestep granularity rather
    than per spectrogram-token).

    Returns (masked_x, mask) where mask[b, t] is True for a masked timestep.
    """
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
    if mask_mode not in ("tokens", "sequence_elements", "whole_sequence_elements", "timesteps"):
        raise ValueError(
            "mask_mode must name whole tokens/sequence elements; "
            f"got {mask_mode!r}. Pixel masking is not supported."
        )

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


def masked_reconstruction_loss(
    model: "ConvLSTMFMForecaster",
    x: torch.Tensor,
    mask_ratio: float,
    mask_mode: str = "tokens",
) -> torch.Tensor:
    """Masked MSE (Eq. 1 of the paper): unmasked timesteps contribute zero loss."""
    masked_x, mask = mask_sequence(x, mask_ratio, mask_mode)
    recon = model.reconstruct(masked_x)
    mask_weight = mask.view(*mask.shape, 1, 1, 1).float()
    squared_error = (recon - x) ** 2 * mask_weight
    denominator = (mask_weight.sum() * x.shape[2] * x.shape[3] * x.shape[4]).clamp_min(1.0)
    return squared_error.sum() / denominator


class ConvLSTMFMForecaster(nn.Module):
    """
    Multi-layer ConvLSTM backbone (the "foundation" encoder) + a lightweight
        Conv3d head, used in two modes:

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
        self.head = nn.Conv3d(hidden[-1], self.input_channels, kernel_size=3, padding=1)
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

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return top-layer hidden states in oldest-to-newest order."""
        layer_outputs, _ = self.encoder(x)
        return self.dropout(layer_outputs[-1])

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct every input timestep, for masked self-supervised pretraining."""
        sequence = self._encode_sequence(x)
        reconstruction = self.head(sequence.permute(0, 2, 1, 3, 4))
        return reconstruction.permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One-step forecast: predict the timestep immediately after the input window."""
        if x.dim() != 5:
            raise ValueError(f"Error! Expected 5D input (B, T, C, H, W), got {x.dim()}D")
        b, t_in, c_in, h, w = x.shape
        if c_in != self.input_channels:
            raise ValueError(f"Error! Input channels {c_in} != model input_channels {self.input_channels}")
        if h != self.spatial_h or w != self.spatial_w:
            raise ValueError(f"Error! Input spatial ({h}, {w}) != model spatial ({self.spatial_h}, {self.spatial_w})")

        # WindowDataset and ConvLSTM both preserve chronological order, so the
        # final Conv3D output is the one aligned with the newest input token.
        sequence = self._encode_sequence(x)
        outputs = self.head(sequence.permute(0, 2, 1, 3, 4))
        return outputs[:, :, -1:].permute(0, 2, 1, 3, 4)


def pretrain_backbone(
    model: ConvLSTMFMForecaster,
    train_loader: DataLoader,
    epochs: int,
    mask_ratio: float = 0.2,
    learning_rate: float = 1e-3,
    mask_mode: str = "tokens",
    callback: TrainingCallback | None = None,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """
    Masked-reconstruction (MSM) self-supervised pretraining of the backbone
    (paper Stage A / Algorithm 1), run directly on the same map data used for
    forecasting, before the shared pipeline fine-tunes the model on next-step
    prediction. The supplied forecasting loader is already segment-safe; its
    targets are intentionally ignored.
    """
    device = next(model.parameters()).device
    configured_batch_size = int(
        model.config["model"].get("pretrain_batch_size", train_loader.batch_size)
    )
    if configured_batch_size <= 0:
        raise ValueError("pretrain_batch_size must be greater than zero")
    if configured_batch_size != train_loader.batch_size:
        loader_kwargs: dict[str, Any] = {
            "num_workers": train_loader.num_workers,
            "collate_fn": train_loader.collate_fn,
            "pin_memory": train_loader.pin_memory,
            "drop_last": train_loader.drop_last,
            "timeout": train_loader.timeout,
            "worker_init_fn": train_loader.worker_init_fn,
            "generator": train_loader.generator,
        }
        if train_loader.num_workers > 0:
            loader_kwargs.update({
                "persistent_workers": train_loader.persistent_workers,
                "prefetch_factor": train_loader.prefetch_factor,
                "multiprocessing_context": train_loader.multiprocessing_context,
            })
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=configured_batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        sample_count = 0
        model.train()
        for x, _ in train_loader:
            batch = x.to(device)
            optimizer.zero_grad()
            loss = masked_reconstruction_loss(model, batch, mask_ratio, mask_mode)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            sample_count += batch.shape[0]
        epoch_loss = total_loss / max(sample_count, 1)
        epoch_losses.append(float(epoch_loss))
        print(
            f"[ConvLSTM-FM pretrain] epoch {epoch:03d}/{epochs} "
            f"masked_recon_loss={epoch_loss:.6f}"
        )
        emit_training_event(
            callback,
            model_name="convlstmfm",
            chunk_id=chunk_id,
            stage="pretrain",
            epoch=epoch,
            epochs=epochs,
            metrics={"masked_reconstruction_loss": float(epoch_loss)},
            selection_metric="masked_reconstruction_loss",
            selection_mode="min",
            duration=time.perf_counter() - epoch_start,
            prunable=False,
            is_best=epoch_loss == min(epoch_losses),
        )

    return {
        "epochs": int(epochs),
        "mask_ratio": float(mask_ratio),
        "mask_mode": mask_mode,
        "learning_rate": float(learning_rate),
        "batch_size": configured_batch_size,
        "samples": int(len(train_loader.dataset)),
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
    }
