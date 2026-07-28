"""
STSPredNet: Spatio-Temporal Spectrum Prediction Network.

Combines multiple temporal branches (closeness, period, trend) each processed
by a PredRNN, then fuses their outputs via learned per-location or per-tensor
weights followed by an optional activation.

The forward interface accepts a single contiguous history window and internally
extracts the branch-specific subsequences.  This keeps the shared pipeline
model-agnostic — the generic trainer and evaluator call model(window) without
knowing about branch structure.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class STSConvLSTMCell(nn.Module):
    """A ConvLSTM cell with dual memory (cell state + cross-layer memory).

    The standard LSTM gating (input, forget, cell update) is applied to both
    c and m. The output gate additionally conditions on the concatenation of
    x, h, c_next, and m_next. The final hidden state is the output gate
    multiplied by a tanh over a 1x1 fusion of c and m.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv_standard = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self.conv_memory = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self.conv_output = nn.Conv2d(
            in_channels=input_dim + 3 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self.conv_1x1 = nn.Conv2d(
            in_channels=2 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x, h, c, m):
        """Perform one cell update.

        Args:
            x: Input tensor (B, input_dim, H, W).
            h: Previous hidden state (B, hidden_dim, H, W).
            c: Previous cell state (B, hidden_dim, H, W).
            m: Previous memory state (B, hidden_dim, H, W).

        Returns:
            Tuple of (h_next, c_next, m_next) each shaped (B, hidden_dim, H, W).
        """
        g, i, f = torch.split(
            self.conv_standard(torch.cat([x, h], dim=1)),
            self.hidden_dim, dim=1,
        )
        g = torch.tanh(g)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_next = f * c + i * g

        g_m, i_m, f_m = torch.split(
            self.conv_memory(torch.cat([x, m], dim=1)),
            self.hidden_dim, dim=1,
        )
        g_m = torch.tanh(g_m)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        m_next = f_m * m + i_m * g_m

        o_input = torch.cat([x, h, c_next, m_next], dim=1)
        o = torch.sigmoid(self.conv_output(o_input))

        h_next = o * torch.tanh(self.conv_1x1(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next


class PredRNN(nn.Module):
    """A stacked ConvLSTM network with cross-layer memory connections.

    Processes an input sequence T steps long. The hidden state of one layer
    becomes the input to the next, while a dedicated memory state (m) is
    passed from the top layer back to the bottom at each time step.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, bias=True, output_channels=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.bias = bias
        self.output_channels = output_channels

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(
                STSConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.output_proj = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x):
        """Process a temporal sequence through stacked ConvLSTM cells.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Output tensor of shape (B, output_channels, H, W) from the last
            layer at the final time step.
        """
        B, T, C, H, W = x.shape
        device = x.device

        h_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]
        m_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t]
            for layer in range(self.num_layers):
                if layer == 0:
                    m_in = m_states[self.num_layers - 1]
                else:
                    m_in = m_states[layer - 1]

                h_next, c_next, m_next = self.cell_list[layer](
                    inp, h_states[layer], c_states[layer], m_in,
                )

                h_states[layer] = h_next
                c_states[layer] = c_next
                m_states[layer] = m_next

                inp = h_next

        out = self.output_proj(h_states[self.num_layers - 1])
        return out


class STSPredNetForecaster(nn.Module):
    """Multi-branch spatio-temporal prediction model for spectrum data.

    Accepts a single contiguous history window of shape (B, T, F, H, W) and
    internally extracts the closeness, period, and trend subsequences required
    by each branch.  Each branch is processed by a PredRNN (or a shared one)
    and their outputs are linearly fused with learnable weights before an
    output activation.

    The input window length must be at least ``input_sequence_length``, which
    is calculated from the enabled branch lengths and intervals.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        c = config["model"]
        b = config["branches"]

        self.num_layers = c["num_layers"]
        self.hidden_dim = c["hidden_dim"]
        self.kernel_size = tuple(c["kernel_size"])
        self.H = c["map_height"]
        self.W = c["map_width"]
        self.input_channels = c.get("input_channels", 1)
        self.output_activation = c.get("output_activation", "identity")
        self.fusion_shape = c.get("fusion_weight_shape", "per_location")

        self.use_closeness = b["use_closeness"]
        self.use_period = b["use_period"]
        self.use_trend = b["use_trend"]
        self.share_weights = b.get("share_branch_weights", False)

        if not any((self.use_closeness, self.use_period, self.use_trend)):
            raise ValueError(
                "STS-PredNet requires at least one enabled temporal branch."
            )

        self.closeness_length = int(b.get("closeness_length", 10))
        self.period_length = int(b.get("period_length", 3))
        self.trend_length = int(b.get("trend_length", 0))
        self.period_interval = int(b.get("period_interval", 10))
        self.trend_interval = int(b.get("trend_interval", 40320))

        required_history = 0
        if self.use_closeness:
            required_history = max(required_history, self.closeness_length)
        if self.use_period:
            required_history = max(
                required_history, self.period_length * self.period_interval
            )
        if self.use_trend:
            required_history = max(
                required_history, self.trend_length * self.trend_interval
            )

        self.input_sequence_length = required_history
        self.prediction_horizon = 1

        common_kwargs = {
            "input_dim": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "output_channels": self.input_channels,
        }

        if self.share_weights:
            self.branch = PredRNN(**common_kwargs)
        else:
            if self.use_closeness:
                self.predrnn_c = PredRNN(**common_kwargs)
            if self.use_period:
                self.predrnn_p = PredRNN(**common_kwargs)
            if self.use_trend:
                self.predrnn_q = PredRNN(**common_kwargs)

        n_branches = sum([self.use_closeness, self.use_period, self.use_trend])

        if self.fusion_shape == "per_location":
            fusion_shape = (1, 1, self.H, self.W)
        else:
            fusion_shape = (1, 1, 1, 1)

        init_weight = 1.0 / n_branches
        self.W_c = nn.Parameter(torch.ones(fusion_shape) * init_weight) if self.use_closeness else None
        self.W_p = nn.Parameter(torch.ones(fusion_shape) * init_weight) if self.use_period else None
        self.W_q = nn.Parameter(torch.ones(fusion_shape) * init_weight) if self.use_trend else None

    def _extract_regular_branch(
        self,
        x: torch.Tensor,
        *,
        length: int,
        interval: int,
    ) -> torch.Tensor:
        """Extract a regularly-spaced subsequence from the end of x.

        Selects ``length`` frames spaced ``interval`` steps apart, in
        chronological order.  For ``length=3``, ``interval=10``, selects
        frames at ``t-30, t-20, t-10``.
        """
        indices = [
            x.size(1) - interval * step
            for step in range(length, 0, -1)
        ]

        if indices[0] < 0:
            raise ValueError(
                "STS-PredNet input window is too short for the configured "
                f"branch: need at least {interval * length} timesteps, "
                f"got window length {x.size(1)}."
            )

        return x[:, indices]

    def _extract_branches(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Split a contiguous window into the three branch inputs."""
        if x.ndim != 5:
            raise ValueError(
                "STS-PredNet expects input shaped (B, T, F, H, W), "
                f"got {tuple(x.shape)}."
            )

        closeness_seq = None
        period_seq = None
        trend_seq = None

        if self.use_closeness:
            closeness_seq = x[:, -self.closeness_length:]

        if self.use_period:
            period_seq = self._extract_regular_branch(
                x,
                length=self.period_length,
                interval=self.period_interval,
            )

        if self.use_trend:
            trend_seq = self._extract_regular_branch(
                x,
                length=self.trend_length,
                interval=self.trend_interval,
            )

        return closeness_seq, period_seq, trend_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the multi-branch prediction on one contiguous history window.

        Args:
            x: Input tensor of shape (B, T, F, H, W) where T must be at
               least ``input_sequence_length``.

        Returns:
            Fused prediction tensor of shape (B, 1, F, H, W).
        """
        (
            closeness_seq,
            period_seq,
            trend_seq,
        ) = self._extract_branches(x)

        branch_outputs: dict[str, torch.Tensor] = {}

        if self.share_weights:
            if closeness_seq is not None:
                branch_outputs["closeness"] = self.branch(closeness_seq)
            if period_seq is not None:
                branch_outputs["period"] = self.branch(period_seq)
            if trend_seq is not None:
                branch_outputs["trend"] = self.branch(trend_seq)
        else:
            if closeness_seq is not None:
                branch_outputs["closeness"] = self.predrnn_c(closeness_seq)
            if period_seq is not None:
                branch_outputs["period"] = self.predrnn_p(period_seq)
            if trend_seq is not None:
                branch_outputs["trend"] = self.predrnn_q(trend_seq)

        fused = torch.zeros_like(next(iter(branch_outputs.values())))

        if "closeness" in branch_outputs:
            fused = fused + self.W_c * branch_outputs["closeness"]
        if "period" in branch_outputs:
            fused = fused + self.W_p * branch_outputs["period"]
        if "trend" in branch_outputs:
            fused = fused + self.W_q * branch_outputs["trend"]

        if self.output_activation == "tanh":
            fused = torch.tanh(fused)
        elif self.output_activation == "sigmoid":
            fused = torch.sigmoid(fused)
        elif self.output_activation != "identity":
            raise ValueError(
                "Unsupported STS-PredNet output "
                f"activation: {self.output_activation!r}"
            )

        return fused.unsqueeze(1)
