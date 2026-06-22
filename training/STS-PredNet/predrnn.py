import torch
import torch.nn as nn

from sts_convlstm_cell import STSConvLSTMCell


class PredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.bias = bias

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

        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
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
