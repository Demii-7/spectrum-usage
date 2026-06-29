"""
STSConvLSTMCell: Spatio-Temporal Spectral ConvLSTM cell.

Extends a standard ConvLSTM cell with dual memory pathways:
  - A standard cell state (c) for per-layer temporal memory.
  - A separate memory state (m) that flows across layers and time steps,
    enabling the network to capture long-range spatio-temporal patterns.
"""
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

        # Gates for the standard cell state (c): input, forget, cell update
        self.conv_standard = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        # Gates for the cross-layer memory state (m)
        self.conv_memory = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        # Output gate: conditions on input, hidden, and both updated states
        self.conv_output = nn.Conv2d(
            in_channels=input_dim + 3 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

        # 1x1 convolution to fuse c and m before the output activation
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
        # Standard LSTM gates on (x, h) for cell state update
        g, i, f = torch.split(
            self.conv_standard(torch.cat([x, h], dim=1)),
            self.hidden_dim, dim=1,
        )
        g = torch.tanh(g)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_next = f * c + i * g

        # Memory LSTM gates on (x, m) for cross-layer memory update
        g_m, i_m, f_m = torch.split(
            self.conv_memory(torch.cat([x, m], dim=1)),
            self.hidden_dim, dim=1,
        )
        g_m = torch.tanh(g_m)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        m_next = f_m * m + i_m * g_m

        # Output gate combines all available information
        o_input = torch.cat([x, h, c_next, m_next], dim=1)
        o = torch.sigmoid(self.conv_output(o_input))

        # Fuse c and m through 1x1 conv, then modulate with output gate
        h_next = o * torch.tanh(self.conv_1x1(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next
