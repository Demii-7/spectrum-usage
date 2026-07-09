"""
PredRNN: Stacked spatio-temporal LSTM with memory flow between layers.

Each layer uses an STSConvLSTMCell that incorporates both a standard
cell state (c) and a separate memory state (m) that flows across layers
and time steps, enabling long-range spatio-temporal dependencies.
"""
import torch
import torch.nn as nn

from sts_convlstm_cell import STSConvLSTMCell


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

        # 1x1 convolution to project hidden dim down to output_channels
        self.output_proj = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x):
        """Process a temporal sequence through stacked ConvLSTM cells.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Output tensor of shape (B, 1, H, W) from the last layer at the
            final time step.
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialize hidden (h), cell (c), and memory (m) states to zero
        h_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]
        m_states = [torch.zeros(B, self.hidden_dim, H, W, device=device)
                    for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t]
            for layer in range(self.num_layers):
                # Memory flows from the previous layer (or the top layer for layer 0)
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

        # Project the final hidden state of the top layer to output channels
        out = self.output_proj(h_states[self.num_layers - 1])
        return out
