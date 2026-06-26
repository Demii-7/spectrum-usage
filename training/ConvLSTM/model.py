"""
ConvLSTM model definitions for spatiotemporal spectrum prediction.

Architecture overview:
1. Encoder: A multi-layer ConvLSTM that processes the input sequence and compresses
   it into a latent state capturing spatiotemporal dynamics.
2. Transfer: The encoder's final hidden/cell states are flattened, passed through
   a standard LSTM, then projected back to spatial dimensions to initialize the decoder.
3. Decoder: A single ConvLSTM cell that iteratively predicts future time steps,
   optionally using teacher forcing during training.

The model operates on data shaped as (batch, time, channels, height=n_nodes, width=n_bins),
where each spatial location (node, frequency bin) carries a time series.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell with convolutional gates.

    Unlike a standard LSTM where the state-to-state transition is a matrix multiply,
    ConvLSTM uses convolution operations, allowing it to capture local spatial patterns.
    The cell follows the formulation:

        i = sigmoid(conv([x, h_prev]))
        f = sigmoid(conv([x, h_prev]))
        o = sigmoid(conv([x, h_prev]))
        g = activation(conv([x, h_prev]))
        c = f * c_prev + i * g
        h = o * activation(c)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, activation=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        self.activation = activation

        # Single convolution produces all four gates (input, forget, output, cell).
        # Concatenating input and hidden along the channel axis enables the convolution
        # to learn both input-to-state and state-to-state transitions jointly.
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
        # Split the 4*hidden_dim output into the four gates.
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = self.activation(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * self.activation(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """Initialize hidden and cell states as zeros matching the spatial dimensions."""
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM that processes a full sequence of 2D spatial maps.

    Each layer's hidden state at every time step is the output for that layer;
    the next layer receives the full sequence of hidden states from the previous layer.
    This creates a stacked recurrent architecture similar to stacked RNNs/LSTMs.

    By default, only the last layer's outputs and states are returned
    (``return_all_layers=False``), which is typical when the ConvLSTM is used as an encoder.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False,
                 activation=nn.ReLU()):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Allow scalar hyper-parameters to be broadcast to all layers for convenience.
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
        """
        Args:
            input_tensor: (B, T, C, H, W) if batch_first else (T, B, C, H, W).
            hidden_state: Optional initial hidden state per layer. If None, zero-initialized.

        Returns:
            layer_output_list: Hidden states for the last time step of each layer
                               (or just the top layer if return_all_layers=False).
            last_state_list:   (h, c) for the last time step of each layer.
        """
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
            # Unroll over the time dimension for this layer.
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            # The next layer receives the full output sequence of this layer.
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
        """Broadcast a scalar parameter to a list of length num_layers for multi-layer convenience."""
        if not isinstance(param, list):
            return [param] * num_layers
        return param


class ConvLSTMPredictor(nn.Module):
    """
    Encoder–Transfer–Decoder architecture for multi-step spectrum prediction.

    Architecture:
    - Encoder: Multi-layer ConvLSTM that reads ``t_in`` time steps.
    - Transfer: Flattens the encoder state, runs it through a standard LSTM
      (to model temporal dependencies in the compressed space), then projects
      it back into spatial format to initialize the decoder state.
    - Decoder: Auto-regressive ConvLSTM cell (single layer) that predicts
      ``t_out`` future time steps one by one.

    Teacher forcing: During training, the decoder can optionally receive the
    ground-truth previous step as input (with probability ``teacher_forcing_ratio``)
    instead of its own prediction, which stabilizes and accelerates convergence.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config["model"]
        self.input_channels = c.get("input_channels", 1)
        self.n_nodes = config["data"]["n_nodes"]
        self.n_bins = config["data"]["n_bins_per_node"]
        self.t_in = config["windowing"]["input_sequence_length"]
        self.t_out = config["windowing"]["prediction_horizon"]
        self.activation = nn.ReLU()

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
            input_dim=self.input_channels,
            hidden_dim=hidden,
            kernel_size=kernels,
            num_layers=num_enc,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            activation=self.activation,
        )

        # The transfer LSTM operates on the flattened spatial state so it can
        # learn temporal dynamics in a compact representation before expanding back.
        enc_flat_dim = hidden[-1] * self.n_nodes * self.n_bins
        self.transfer_lstm = nn.LSTM(
            input_size=enc_flat_dim,
            hidden_size=dec_lstm_hidden,
            batch_first=True,
        )
        self.transfer_proj = nn.Linear(dec_lstm_hidden, dec_hidden * self.n_nodes * self.n_bins)

        self.decoder_cell = ConvLSTMCell(
            input_dim=1,
            hidden_dim=dec_hidden,
            kernel_size=dec_kernel,
            bias=True,
            activation=self.activation,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.batch_norm = nn.BatchNorm2d(dec_hidden) if use_bn else nn.Identity()

        # Optional extra convolutional layers before the final 1x1 output conv
        # to increase capacity when predicting fine-grained frequency structure.
        if fc_hidden > 0:
            self.output_head = nn.Sequential(
                nn.Conv2d(dec_hidden, fc_hidden, kernel_size=fc_kernel, padding=(fc_kernel[0] // 2, fc_kernel[1] // 2)),
                nn.ReLU(),
                nn.Conv2d(fc_hidden, 1, kernel_size=1),
            )
        else:
            self.output_head = nn.Conv2d(dec_hidden, 1, kernel_size=1)

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.0):
        """
        Args:
            x: Input sequence, shape (B, t_in, C, n_nodes, n_bins).
            y_teacher: Ground-truth target sequence for teacher forcing, shape (B, t_out, C, n_nodes, n_bins).
            teacher_forcing_ratio: Probability (0–1) of using ground truth vs. own prediction at each step.

        Returns:
            Predictions, shape (B, t_out, C, n_nodes, n_bins).
        """
        # Encode the input sequence into a compressed latent state.
        _, enc_states = self.encoder(x)
        h_enc, c_enc = enc_states[0]
        b = x.size(0)

        # Flatten spatial dimensions and pass through the transfer LSTM.
        h_enc_flat = h_enc.reshape(b, 1, -1)
        c_enc_flat = c_enc.reshape(b, 1, -1)
        lstm_out, (h_lstm, c_lstm) = self.transfer_lstm(h_enc_flat)
        # Project LSTM output back to spatial ConvLSTM decoder state.
        h_dec_init = self.transfer_proj(h_lstm.squeeze(0)).reshape(b, self.decoder_cell.hidden_dim, self.n_nodes, self.n_bins)
        c_dec_init = self.transfer_proj(c_lstm.squeeze(0)).reshape(b, self.decoder_cell.hidden_dim, self.n_nodes, self.n_bins)

        h_dec, c_dec = h_dec_init, c_dec_init
        outputs = []
        # Start with a zero input; the decoder will use its own output as the next input.
        decoder_input = torch.zeros(b, 1, self.n_nodes, self.n_bins, device=x.device)

        for t in range(self.t_out):
            # Teacher forcing: replace decoder input with ground truth with given probability.
            if y_teacher is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y_teacher[:, t, :, :, :]
            h_dec, c_dec = self.decoder_cell(decoder_input, (h_dec, c_dec))
            h_dropped = self.dropout(h_dec)
            h_normed = self.batch_norm(h_dropped)
            out = self.output_head(h_normed)
            outputs.append(out)
            decoder_input = out  # Feed the prediction as the next step's input (autoregressive).

        return torch.stack(outputs, dim=1)
