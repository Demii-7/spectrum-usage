"""DLinear -- an exact port of Time-Series-Library's models/DLinear.py.

Line-for-line equivalent of the reference (github.com/thuml/Time-Series-Library,
main): series_decomp with a replicate-padded moving average (kernel 25), two
shared seq_len->pred_len linear maps over the time axis, both initialised to
(1/seq_len)*ones so the model starts as the lookback-mean predictor, no
normalisation inside the model.

Deliberately NOT ported: TSLib's `--individual` per-channel heads -- in TSLib
main that flag is dead code (exp_basic calls Model(args) and DLinear's
__init__ never reads configs.individual), so "their DLinear" is always
shared-head. This port matches that behaviour.

Verified against the real library (2026-11-04): running TSLib's own run.py on
the ElectroSense CSVs and this port under the identical protocol
(training/SpectrumLLM/run_dlinear.py) reproduces 8 of the 10 published DLinear
cells within 0.5-3%; both published K=96 cells are unreproducible under any
single protocol. See repro/RESULTS_PAPER_REGIME.md and repro/dlinear/.

Interface follows the repo: config dict in, (batch, time, channels) ->
(batch, horizon, channels).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Replicate-padded moving average, as Time-Series-Library's series_decomp."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, T, C)
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:].repeat(1, self.kernel_size // 2, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.seq_len = int(mc["input_sequence_length"])
        self.pred_len = int(mc["prediction_horizon"])
        self.decomposition = MovingAverage(int(mc.get("moving_avg", 25)))

        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        # Time-Series-Library initialises both maps to (1/seq_len)*ones
        for head in (self.linear_seasonal, self.linear_trend):
            head.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(x)
        seasonal = x - trend
        s = self.linear_seasonal(seasonal.permute(0, 2, 1))
        t = self.linear_trend(trend.permute(0, 2, 1))
        return (s + t).permute(0, 2, 1)
