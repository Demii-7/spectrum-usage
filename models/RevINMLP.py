"""RevIN + MLP: the simplest nonlinear channel-independent forecaster.

Each channel's window is instance-normalized (mean and standard deviation over the lookback, RevIN without affine terms), one MLP
(weights shared by every channel) maps the normalized window to the normalized forecast, and the statistics are put back. It has the
two ingredients PatchTST and OFA share (per-window normalization and weights shared across channels) without patching or attention.

With ``channel_mixing`` true, one residual linear layer across the C channels (``input_size``, e.g. the 200 bins) is applied to the
hidden representation after the per-channel MLP body (zero-initialised, so training starts from the channel-independent model). With it
off no parameter is created and the state_dict is the original one.

With ``neighbor_inputs`` true the input holds ``n_receivers`` receivers' spectra side by side (``input_size`` = n_receivers x bins, receiver-major,
the layout of ``data.concat: columns``). Every (receiver, bin) series then also receives the same bin of the other receivers as extra input channels:
each of the n_receivers series is instance-normalized over its own window, the MLP input is [own, the other receivers in their original order]
along time (width n_receivers x seq_len), and the forecast of each receiver (de-normalized with its OWN statistics) is returned in the same
layout, so the loss and the evaluation see one forecast per receiver and bin. With it off nothing changes (no extra parameter, same state_dict).

Config keys (``config["model"]``): ``input_sequence_length``, ``prediction_horizon``, ``mlp_hidden`` (128), ``mlp_layers`` (2 hidden
layers), ``mlp_dropout`` (0.1), ``channel_mixing`` (false), ``neighbor_inputs`` (false), ``n_receivers`` (4, used with neighbor_inputs). Input ``(B, T, C)``, output ``(B, H, C)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NORM_EPS = 1e-5


class RevINMLPForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        hidden, layers, dropout = int(mc.get("mlp_hidden", 128)), int(mc.get("mlp_layers", 2)), float(mc.get("mlp_dropout", 0.1))
        self.neighbor_inputs = bool(mc.get("neighbor_inputs", False))
        self.n_receivers = int(mc.get("n_receivers", 4)) if self.neighbor_inputs else 1
        if self.neighbor_inputs and bool(mc.get("channel_mixing", False)):
            raise ValueError("neighbor_inputs and channel_mixing are separate controls; use one")
        if self.neighbor_inputs and int(mc["input_size"]) % self.n_receivers:
            raise ValueError(f"input_size {mc['input_size']} is not a multiple of n_receivers {self.n_receivers}")
        blocks: list[nn.Module] = []
        width = self.seq_len * self.n_receivers
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.pred_len)
        self.mixer = None
        if bool(mc.get("channel_mixing", False)):
            channels = int(mc["input_size"])
            self.mixer = nn.Linear(channels, channels)
            nn.init.zeros_(self.mixer.weight)
            nn.init.zeros_(self.mixer.bias)

    def _forward_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, columns = x.shape
        r = self.n_receivers
        bins = columns // r
        series = x.reshape(batch, steps, r, bins).permute(0, 2, 3, 1)       # (B, R, F, T)
        mean = series.mean(dim=-1, keepdim=True)
        std = (series.var(dim=-1, keepdim=True, unbiased=False) + NORM_EPS).sqrt()
        z = (series - mean) / std
        order = torch.tensor([[k] + [j for j in range(r) if j != k] for k in range(r)], device=x.device)   # own first, others in original order
        stacked = z[:, order]                                              # (B, R(target), R(position), F, T)
        stacked = stacked.permute(0, 1, 3, 2, 4).reshape(batch, r, bins, r * steps)
        forecast = self.head(self.body(stacked)) * std + mean              # (B, R, F, H) with the target receiver's own statistics
        return forecast.permute(0, 3, 1, 2).reshape(batch, self.pred_len, columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.neighbor_inputs:
            return self._forward_neighbors(x)
        series = x.permute(0, 2, 1)                                      # (B, C, T)
        mean = series.mean(dim=-1, keepdim=True)
        std = (series.var(dim=-1, keepdim=True, unbiased=False) + NORM_EPS).sqrt()
        hidden = self.body((series - mean) / std)                        # (B, C, hidden)
        if self.mixer is not None:                                       # mix across channels, residually
            hidden = hidden + self.mixer(hidden.transpose(1, 2)).transpose(1, 2)
        forecast = self.head(hidden) * std + mean
        return forecast.permute(0, 2, 1)
