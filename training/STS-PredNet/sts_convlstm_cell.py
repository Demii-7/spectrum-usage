import torch
import torch.nn as nn


class STSConvLSTMCell(nn.Module):
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
