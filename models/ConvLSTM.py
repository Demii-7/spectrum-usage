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
        raise ValueError(f"Error! Unsupported activation: {name}")


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

        # Single convolution produces the input/forget/output gates plus the cell candidate.
        # Concatenating input and hidden along the channel axis enables the convolution
        # to learn both input-to-state and state-to-state transitions jointly.
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        # Peephole connections are learned Hadamard weights from the previous cell
        # state into the input/forget/output gates, matching the paper's formulas.
        self.w_ci = nn.Parameter(torch.zeros(1, self.hidden_dim, 1, 1))
        self.w_cf = nn.Parameter(torch.zeros(1, self.hidden_dim, 1, 1))
        self.w_co = nn.Parameter(torch.zeros(1, self.hidden_dim, 1, 1))

    def forward(self, input_tensor, cur_state):
        """
        Single time-step forward of the ConvLSTM cell.

        Args:
            input_tensor: Input at current step, shape (B, input_dim, H, W).
            cur_state: Tuple (h, c) from previous step, each (B, hidden_dim, H, W).

        Returns:
            h_next: Hidden state for current step, (B, hidden_dim, H, W).
            c_next: Cell state for current step, (B, hidden_dim, H, W).
        """
        h_cur, c_cur = cur_state
        # Concatenate input and previous hidden along channel dim so the conv
        # can learn both input-to-state and state-to-state transitions jointly.
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split the 4*hidden_dim output into the four gates: input, forget, output, cell modulation.
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i + self.w_ci * c_cur)
        f = torch.sigmoid(cc_f + self.w_cf * c_cur)
        o = torch.sigmoid(cc_o + self.w_co * c_cur)
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
        
        # Ensure that teh number of defined dimensions and kernels is consistent with the number of stacked layers
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Error! Inconsistent list length")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Set up each ConvLSTM Layer
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

        # Convert tensor shape to batch first format
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Read input dimensions
        b, _, _, h, w = input_tensor.size()
        #Initialise states when none are supplied
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))

        # Prepare results containers
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # Loop through each LSTMlayer
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


class ConvLSTMForecaster(nn.Module):
    """
    Encoder–Transfer–Decoder architecture for multi-step spectrum prediction.

    Architecture:
    - (Optional) Channel projection: 1×1 Conv2d to reduce F → channel_projection_dim.
    - Encoder: Multi-layer ConvLSTM that reads ``input_sequence_length`` time steps.
    - Transfer: Flattens the encoder's final hidden state, projects it through
      a standard LSTM into the decoder's hidden dimensionality, then reshapes
      it back to spatial format to initialize the decoder state.
    - Decoder: Auto-regressive ConvLSTM cell (single layer) that predicts
      ``prediction_horizon`` future time steps one by one.
    """

    def __init__(self, config):
        super().__init__()
    
        self.config = config
        c = config["model"]
    
        self.input_channels = int(c["input_channels"])
        self.spatial_h = int(c["grid_height"])
        self.spatial_w = int(c["grid_width"])
    
        self.input_sequence_length = int(c["input_sequence_length"])
        self.prediction_horizon = int(c["prediction_horizon"])
        
        cell_act = c.get("cell_activation", "relu")
        fc_act = c.get("fc_intermediate_activation", "relu")
        
        self.activation = _get_activation(cell_act)
        self.fc_activation = _get_activation(fc_act)

        enc_input_dim = self.input_channels # 
        use_proj = c.get("use_channel_projection", False)
        
        # Sets up an optional channel compressor (or channel expander) using a 1x1 Convolutional layer.
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

        # Initialise encoder which accepts input data
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

        # The transfer LSTM projects the flattened encoder hidden state into the
        # decoder's hidden dimensionality before reshaping back to spatial format.
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

        # Optional extra convolutional layers before the final 1x1 output conv
        # to increase capacity when predicting fine-grained frequency structure.
        if fc_hidden > 0:
            self.output_head = nn.Sequential(
                nn.Conv2d(dec_hidden, fc_hidden, kernel_size=fc_kernel, padding=(fc_kernel[0] // 2, fc_kernel[1] // 2)),
                self.fc_activation,
                nn.Conv2d(fc_hidden, self.input_channels, kernel_size=1),
            )
        else:
            self.output_head = nn.Conv2d(dec_hidden, self.input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode, transfer, then autoregressively decode.

        The encoder compresses the input into a latent state, the transfer LSTM
        transforms the flattened encoder state into the decoder initialization, and the decoder
        unrolls future predictions one step at a time.

        Args:
            x: Input feature sequence tensor, shape (B, T_in, C, H, W)
               where C=input_channels, H=spatial_h, W=spatial_w.
               
        Returns:
            Projected future forecast tensor, shape (B, T_out, C, H, W)
        """

        # Check if input shape is as expected
        if x.dim() != 5:
            raise ValueError(f"Error! Expected 5D input (B, T, C, H, W), got {x.dim()}D")
            
        # Unpack input shape separately    
        b, input_sequence_length, c_in, h, w = x.shape
        
        # Check if input and spatial shapes are as expected
        if input_sequence_length != self.input_sequence_length:
            raise ValueError(
                f"Error! Input sequence length {input_sequence_length} != "
                f"configured length {self.input_sequence_length}"
            )
            
        if c_in != self.input_channels:
            raise ValueError(f"Error! Input channels {c_in} != model input_channels {self.input_channels}")
            
        if h != self.spatial_h or w != self.spatial_w:
            raise ValueError(f"Error!Input spatial ({h}, {w}) != model spatial ({self.spatial_h}, {self.spatial_w})")

        # Time-Step Channel Projection (If active)
        x_2d = x.reshape(b * input_sequence_length, c_in, h, w)
        x_proj = self.channel_proj(x_2d)
        
        _, c_proj, _, _ = x_proj.shape
        x = x_proj.reshape(b, input_sequence_length, c_proj, h, w)

        # Spatiotemporally encode the input sequence into a compressed latent state
        _, enc_states = self.encoder(x)
        h_enc, _ = enc_states[-1]     # Extracting final top layer hidden state and ignore cell state

        # Flatten spatial dimensions and pass through the transfer LSTM.
        h_enc_flat = h_enc.reshape(b, 1, -1)
        _, (h_lstm, c_lstm) = self.transfer_lstm(h_enc_flat)
        
        # Project LSTM output back to spatial ConvLSTM decoder state.
        # h_dec_init and c_dec_init will initialise decoder from encoder ouput
        h_dec = self.transfer_proj(h_lstm.squeeze(0)).reshape(
            b, self.decoder_cell.hidden_dim, self.spatial_h, self.spatial_w)
        c_dec = self.transfer_proj(c_lstm.squeeze(0)).reshape(
            b, self.decoder_cell.hidden_dim, self.spatial_h, self.spatial_w)

        # Autoregressive Future Sequence Unrolling Loop (both multistep or one step depending on prediction_horizon)
        outputs = []
        decoder_input = torch.zeros(b, self.input_channels, self.spatial_h, self.spatial_w, device=x.device)

        for _ in range(self.prediction_horizon):
            h_dec, c_dec = self.decoder_cell(decoder_input, (h_dec, c_dec))
            h_dropped = self.dropout(h_dec)
            h_normed = self.batch_norm(h_dropped)
            out = self.output_head(h_normed)
            outputs.append(out)

            # Clean and self-contained: Always use current output step as next step input
            decoder_input = out

        # Restore complete forecasting horizon time dimension and return predictions
        return torch.stack(outputs, dim=1)
