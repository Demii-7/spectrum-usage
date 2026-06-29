import torch
import torch.nn as nn


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, activation=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        self.activation = activation

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = self.activation(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * self.activation(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False,
                 activation=nn.ReLU()):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    activation=activation,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(e, tuple) for e in kernel_size))):
            raise ValueError("kernel_size must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param


class ConvLSTMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config["model"]
        d = config["data"]
        self.input_channels = c.get("input_channels", 1)
        self.n_nodes = d.get("n_nodes", 1)
        self.n_bins = d.get("n_bins_per_node", 1)
        self.grid_h = d.get("grid_height", self.n_nodes)
        self.grid_w = d.get("grid_width", self.n_bins)
        self.spatial_h = self.grid_h
        self.spatial_w = self.grid_w
        self.t_in = config["windowing"]["input_sequence_length"]
        self.t_out = config["windowing"]["prediction_horizon"]
        cell_act = c.get("cell_activation", "relu")
        fc_act = c.get("fc_intermediate_activation", "relu")
        self.activation = _get_activation(cell_act)
        self.fc_activation = _get_activation(fc_act)

        enc_input_dim = self.input_channels
        use_proj = c.get("use_channel_projection", False)
        if use_proj:
            proj_dim = c.get("channel_projection_dim", 16)
            self.channel_proj = nn.Conv2d(self.input_channels, proj_dim, kernel_size=1)
            enc_input_dim = proj_dim
        else:
            self.channel_proj = nn.Identity()

        hidden = c["hidden_channels"]
        kernels = [tuple(k) for k in c["kernel_size"]]
        num_enc = c["num_encoder_layers"]
        dec_hidden = c["decoder_hidden_channels"]
        dec_kernel = tuple(c["decoder_kernel_size"])
        dec_lstm_hidden = c["decoder_lstm_hidden"]
        fc_hidden = c.get("fc_hidden_channels", 0)
        fc_kernel = tuple(c.get("fc_kernel_size", [3, 3]))
        dropout = c.get("dropout", 0.0)
        use_bn = c.get("use_batch_norm", False)

        self.encoder = ConvLSTM(
            input_dim=enc_input_dim,
            hidden_dim=hidden,
            kernel_size=kernels,
            num_layers=num_enc,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            activation=self.activation,
        )

        enc_flat_dim = hidden[-1] * self.spatial_h * self.spatial_w
        self.transfer_lstm = nn.LSTM(
            input_size=enc_flat_dim,
            hidden_size=dec_lstm_hidden,
            batch_first=True,
        )
        self.transfer_proj = nn.Linear(dec_lstm_hidden, dec_hidden * self.spatial_h * self.spatial_w)

        self.decoder_cell = ConvLSTMCell(
            input_dim=self.input_channels,
            hidden_dim=dec_hidden,
            kernel_size=dec_kernel,
            bias=True,
            activation=self.activation,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.batch_norm = nn.BatchNorm2d(dec_hidden) if use_bn else nn.Identity()

        if fc_hidden > 0:
            self.output_head = nn.Sequential(
                nn.Conv2d(dec_hidden, fc_hidden, kernel_size=fc_kernel, padding=(fc_kernel[0] // 2, fc_kernel[1] // 2)),
                self.fc_activation,
                nn.Conv2d(fc_hidden, self.input_channels, kernel_size=1),
            )
        else:
            self.output_head = nn.Conv2d(dec_hidden, self.input_channels, kernel_size=1)

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0):
        b, t_in, c_in, h, w = x.shape
        x_2d = x.reshape(b * t_in, c_in, h, w)
        x_proj = self.channel_proj(x_2d)
        _, c_proj, _, _ = x_proj.shape
        x = x_proj.reshape(b, t_in, c_proj, h, w)

        _, enc_states = self.encoder(x)
        h_enc, c_enc = enc_states[0]

        h_enc_flat = h_enc.reshape(b, 1, -1)
        c_enc_flat = c_enc.reshape(b, 1, -1)
        lstm_out, (h_lstm, c_lstm) = self.transfer_lstm(h_enc_flat)
        h_dec_init = self.transfer_proj(h_lstm.squeeze(0)).reshape(
            b, self.decoder_cell.hidden_dim, self.spatial_h, self.spatial_w)
        c_dec_init = self.transfer_proj(c_lstm.squeeze(0)).reshape(
            b, self.decoder_cell.hidden_dim, self.spatial_h, self.spatial_w)

        h_dec, c_dec = h_dec_init, c_dec_init
        outputs = []
        decoder_input = torch.zeros(b, self.input_channels, self.spatial_h, self.spatial_w, device=x.device)

        for t in range(self.t_out):
            if y_teacher is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y_teacher[:, t, :, :, :]
            h_dec, c_dec = self.decoder_cell(decoder_input, (h_dec, c_dec))
            h_dropped = self.dropout(h_dec)
            h_normed = self.batch_norm(h_dropped)
            out = self.output_head(h_normed)
            outputs.append(out)
            decoder_input = out

        return torch.stack(outputs, dim=1)
