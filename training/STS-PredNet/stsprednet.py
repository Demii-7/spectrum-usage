import torch
import torch.nn as nn

from predrnn import PredRNN


class STSPredNet(nn.Module):
    def __init__(self, config):
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
        self.output_activation = c.get("output_activation", "tanh")
        self.fusion_shape = c.get("fusion_weight_shape", "per_location")

        self.use_closeness = b["use_closeness"]
        self.use_period = b["use_period"]
        self.use_trend = b["use_trend"]
        self.share_weights = b.get("share_branch_weights", False)

        common_kwargs = {
            "input_dim": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
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

        self.W_c = nn.Parameter(torch.ones(fusion_shape) / n_branches) if self.use_closeness else None
        self.W_p = nn.Parameter(torch.ones(fusion_shape) / n_branches) if self.use_period else None
        self.W_q = nn.Parameter(torch.ones(fusion_shape) / n_branches) if self.use_trend else None

    def forward(self, closeness_seq, period_seq=None, trend_seq=None):
        if self.share_weights:
            branch_out = []
            if self.use_closeness:
                branch_out.append(self.branch(closeness_seq))
            if self.use_period and period_seq is not None:
                branch_out.append(self.branch(period_seq))
            if self.use_trend and trend_seq is not None:
                branch_out.append(self.branch(trend_seq))
        else:
            branch_out = []
            if self.use_closeness:
                branch_out.append(self.predrnn_c(closeness_seq))
            if self.use_period and period_seq is not None:
                branch_out.append(self.predrnn_p(period_seq))
            if self.use_trend and trend_seq is not None:
                branch_out.append(self.predrnn_q(trend_seq))

        fused = 0.0
        idx = 0
        if self.use_closeness:
            fused = fused + self.W_c * branch_out[idx]
            idx += 1
        if self.use_period and period_seq is not None:
            fused = fused + self.W_p * branch_out[idx]
            idx += 1
        if self.use_trend and trend_seq is not None:
            fused = fused + self.W_q * branch_out[idx]

        if self.output_activation == "tanh":
            fused = torch.tanh(fused)
        elif self.output_activation == "sigmoid":
            fused = torch.sigmoid(fused)

        return fused
