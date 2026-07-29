"""
DSwinLSTM-I model definitions for spatiotemporal spectrum prediction.

Architecture overview:
1. A patch-embedding stem projects each input frame into a sequence of tokens.
2. A multi-stage SwinLSTM encoder (SwinLSTMCellI with imputation) compresses
   the input sequence into a latent spatiotemporal state.
3. A multi-stage SwinLSTM decoder (SwinLSTMCell) iteratively predicts future
   frames in an autoregressive manner.
4. A reconstruction head projects decoder tokens back to the original map
   dimensions.

The model operates on data shaped as:

    (batch, time, channels, height, width)

and returns:

    (batch, prediction_horizon, channels, height, width)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Multilayer perceptron with two linear layers, GELU activation, and dropout.
    Used as the feed-forward network inside each Swin Transformer block.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass: fc1 -> activation -> dropout -> fc2 -> dropout.

        Args:
            x: Input tensor (B, L, C).

        Returns:
            Transformed tensor with the same shape.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition a spatial feature map into non-overlapping windows.

    Args:
        x: Tensor shaped (B, H, W, C).
        window_size: Integer window size.

    Returns:
        Windows reshaped to (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window_partition: merge non-overlapping windows back into a spatial map.

    Args:
        windows: Tensor shaped (num_windows * B, window_size, window_size, C).
        window_size: Integer window size.
        H: Original spatial height.
        W: Original spatial width.

    Returns:
        Reconstructed tensor shaped (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Operates on local windows rather than the full spatial map, which is
    computationally efficient for high-resolution inputs.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table parameterising distinct
        # attention priors for each pair of positions within a window.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection, attention dropout, output projection, and output dropout.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass through window attention.

        Args:
            x: Input tensor (B_, N, C) where B_ = B * num_windows, N = window_size^2.
            mask: Optional attention mask for shifted windowing (B_, N, N).

        Returns:
            Attended output with the same shape as input.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add learned relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    A single Swin Transformer block with shifted window attention.

    The block applies LayerNorm, window-based multi-head attention (with
    optional cyclic shift), residual connection, then MLP with another
    residual connection. When a hidden state (hx) is provided, it is
    concatenated along the channel dimension before attention for
    recurrent conditioning.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Linear projection to fuse the hidden state with the input when hx is provided
        self.red = nn.Linear(2 * dim, dim)

        # Pre-compute attention mask for shifted windowing
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, hx=None):
        """
        Forward pass: norm -> optional hidden fusion -> window attention -> MLP.

        Args:
            x: Input tokens (B, L, C).
            hx: Optional hidden state tokens (B, L, C) for recurrent conditioning.

        Returns:
            Output tokens with the same shape as input.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)

        # When a hidden state is provided, concatenate it and
        # project back down to the model dimension.
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)

        x = x.view(B, H, W, C)

        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window-based attention
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connections with stochastic depth
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Stack of SwinTransformerBlocks forming the core of the SwinLSTM cell.

    Alternates between regular and shifted-window blocks. The first block
    receives the pair (input_tokens, hidden_state); even-indexed subsequent
    blocks receive (output, input_tokens) as a residual skip, and odd-indexed
    blocks receive (output, None).
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, xt, hx):
        """
        Forward pass through the stack of SwinTransformerBlocks.

        Args:
            xt: Input tokens (B, L, C) from the current time step.
            hx: Hidden state tokens (B, L, C) from the previous time step.

        Returns:
            Output tokens (B, L, C).
        """
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
            else:
                if index % 2 == 0:
                    x = layer(x, xt)
                else:
                    x = layer(x, None)
        return x


class SwinLSTMCell(nn.Module):
    """
    SwinLSTM cell (decoder variant) with mask projection.

    Applies the Swin Transformer to the input and hidden state, then
    computes LSTM-style gates from the result. An optional mask bias
    can modulate the forget-gate pre-activation and the cell state.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.Swin = SwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path, norm_layer=norm_layer)
        # Learned mask projection that modulates the gate pre-activations
        self.mask_proj = nn.Linear(dim, dim)
        self.mask_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, xt, hidden_states, mask=None):
        """
        Single time-step forward of the SwinLSTM decoder cell.

        Args:
            xt: Input tokens (B, L, C) at the current step.
            hidden_states: Tuple (hx, cx) from the previous step. If None,
                           zero-initialised.
            mask: Optional mask tokens (B, L, C) biasing the gates.

        Returns:
            hy: Hidden state (B, L, C).
            (hy, cy): Tuple of hidden and cell state for the next step.
        """
        if hidden_states is None:
            B, L, C = xt.shape
            hx = xt.new_zeros(B, L, C)
            cx = xt.new_zeros(B, L, C)
        else:
            hx, cx = hidden_states

        Ft = self.Swin(xt, hx)

        if mask is not None:
            Ft = Ft + self.mask_proj(mask) + self.mask_bias

        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * (cx + cell)

        if mask is not None:
            cy = cy + self.mask_proj(mask) + self.mask_bias

        hy = gate * torch.tanh(cy)
        return hy, (hy, cy)


class SwinLSTMCellI(nn.Module):
    """
    SwinLSTM cell with imputation (encoder variant).

    Extends SwinLSTMCell by predicting a filling value for masked (missing)
    input tokens. The input is imputed as:

        x_filled = mask * x + (1 - mask) * P_hat

    where P_hat is a learned prediction based on the current hidden and
    cell states. This allows the encoder to handle missing data gracefully.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.Swin = SwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path, norm_layer=norm_layer)
        # Imputation parameters: predict missing values from hidden/cell state
        self.W_p = nn.Linear(dim, dim)
        self.U_p = nn.Linear(dim, dim)
        self.b_p = nn.Parameter(torch.zeros(dim))

    def forward(self, xt, mask, hidden_states):
        """
        Single time-step forward of the SwinLSTM encoder cell with imputation.

        Args:
            xt: Input tokens (B, L, C) at the current step.
            mask: Imputation mask (B, L, C); 1 = observed, 0 = missing.
            hidden_states: Tuple (hx, cx) from the previous step. If None,
                           zero-initialised.

        Returns:
            hy: Hidden state (B, L, C).
            (hy, cy): Tuple of hidden and cell state for the next step.
        """
        if hidden_states is None:
            B, L, C = xt.shape
            hx = xt.new_zeros(B, L, C)
            cx = xt.new_zeros(B, L, C)
        else:
            hx, cx = hidden_states

        # Predict filling value for missing positions
        P_hat = torch.sigmoid(self.W_p(cx) + self.U_p(hx) + self.b_p)
        xt_filled = mask * xt + (1 - mask) * P_hat

        Ft = self.Swin(xt_filled, hx)

        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        return hy, (hy, cy)


class PatchEmbed(nn.Module):
    """
    Image patch-embedding stem.

    Uses a strided convolution to project input frames into a sequence
    of non-overlapping patch tokens, followed by LayerNorm.
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Single convolution projects each patch to the embedding dimension
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Patch-embed a batch of 2D frames.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Patch tokens (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Spatial down-sampling layer for patch tokens.

    Groups 2x2 neighbouring patches, concatenates their features, normalises,
    and projects to 2x the input dimension (reducing spatial resolution by 2x).
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Merge 2x2 patches into one.

        Args:
            x: Input tokens (B, L, C) where L = H * W.

        Returns:
            Down-sampled tokens (B, L/4, 2*C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    """
    Spatial up-sampling layer for patch tokens.

    Expands each token into 2x2 spatial positions by up-projecting
    and rearranging, effectively increasing spatial resolution by 2x
    while halving the channel dimension.
    """
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.expand = nn.Linear(dim, 4 * self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        """
        Expand each token into 2x2 spatial positions.

        Args:
            x: Input tokens (B, L, C) where L = H * W.

        Returns:
            Up-sampled tokens (B, 4*L, C/2).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = self.expand(x).view(B, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, self.out_dim)
        x = x.view(B, -1, self.out_dim)
        x = self.norm(x)
        return x


class MaskPool(nn.Module):
    """
    Average-pool a spatiotemporal mask to a target token resolution.

    Accepts a mask shaped (B, T, H, W, F) and returns pooled mask
    tokens at the reduced spatial resolution, collapsed across frequency.
    """
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, mask):
        """
        Pool the mask to patch resolution.

        Args:
            mask: Input mask (B, T, H, W, F).

        Returns:
            Pooled mask (B, T, H_p * W_p, 1) where H_p, W_p = H // patch_size, W // patch_size.
        """
        B, T, H, W, F = mask.shape
        mask_5d = mask.permute(0, 1, 4, 2, 3).contiguous()
        B, T, F, H, W = mask_5d.shape
        mask_4d = mask_5d.view(B * T, F, H, W)
        pooled = self.pool(mask_4d)
        pooled = pooled.mean(dim=1, keepdim=True)
        _, _, H_p, W_p = pooled.shape
        pooled = pooled.view(B, T, H_p, W_p, 1)
        pooled = pooled.view(B, T, -1, 1)
        return pooled


class Reconstruction(nn.Module):
    """
    Reconstruction head that projects decoder tokens back to a full-resolution map.

    Each token is linearly projected to a patch-sized tile, then the tiles
    are assembled into the original spatial dimensions.
    """
    def __init__(self, in_dim, out_channels, map_size, patch_size):
        super().__init__()
        self.map_size = to_2tuple(map_size)
        self.patch_size = to_2tuple(patch_size)
        pH = self.map_size[0] // self.patch_size[0]
        pW = self.map_size[1] // self.patch_size[1]
        self.proj = nn.Linear(in_dim, self.patch_size[0] * self.patch_size[1] * out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Reconstruct a full-resolution map from decoder tokens.

        Args:
            x: Decoder tokens (B, L, in_dim).

        Returns:
            Reconstructed map (B, out_channels, map_H, map_W).
        """
        B, L, C = x.shape
        x = self.proj(x)
        pH = int(L ** 0.5) if int(L ** 0.5) ** 2 == L else -1
        pH_actual = self.map_size[0] // self.patch_size[0]
        pW_actual = self.map_size[1] // self.patch_size[1]
        x = x.view(B, pH_actual, pW_actual, self.patch_size[0], self.patch_size[1], self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.map_size[0], self.map_size[1])
        return x


class DSwinLSTM_IForecaster(nn.Module):
    """
    Spatiotemporal forecasting model combining Swin Transformer with LSTM.

    Encoder processes the input lookback sequence through two SwinLSTMCellI stages
    (with imputation), then the decoder autoregressively generates future frames
    through two SwinLSTMCell stages. The model outputs all prediction steps in a
    single forward pass.
    """
    def __init__(self, config):
        super().__init__()
        #Load configuration file
        model_cfg = config["model"]

        #Read configuration file and extract critical variables
        self.orig_H = model_cfg["map_height"]
        self.orig_W = model_cfg["map_width"]
        self.F = model_cfg["input_channels"]
        self.patch_shape = tuple(model_cfg.get("patch_shape") or [2, 2])
        self.base_dim = model_cfg.get("embed_dim", 128)
        self.hidden_dims = model_cfg.get("hidden_dims") or [self.base_dim, self.base_dim * 2]
        self.encoder_units = model_cfg.get("encoder_units", 2)
        self.decoder_units = model_cfg.get("decoder_units", 2)
        self.swin_depths = model_cfg.get("swin_depths", [2, 6, 6, 2])
        self.num_heads = model_cfg.get("num_heads", [4, 8, 8, 4])
        self.window_size = model_cfg.get("window_size", 4)
        self.drop_rate = model_cfg.get("drop_rate", 0.)
        self.attn_drop_rate = model_cfg.get("attn_drop_rate", 0.)
        self.drop_path_rate = model_cfg.get("drop_path_rate", 0.1)
        self.decoder_feedback = model_cfg.get("decoder_feedback", "pixel_feedback")
        self.T_out = model_cfg["prediction_horizon"]
        self.padding_mode = model_cfg.get("padding_mode", "reflect")
        self.output_activation = model_cfg.get("output_activation", "none")
        self.use_imputation_unit = bool(model_cfg.get("use_imputation_unit", True))
        self.mask_as_input_channel = bool(model_cfg.get("mask_as_input_channel", False))
        self.use_patch_merging = model_cfg.get("use_patch_merging", True)
        self.use_patch_expanding = model_cfg.get("use_patch_expanding", True)
        if self.encoder_units != 2 or self.decoder_units != 2:
            raise ValueError("DSwinLSTM-I currently requires exactly two encoder and decoder units")
        if len(self.hidden_dims) != 2 or len(self.swin_depths) != 4 or len(self.num_heads) != 4:
            raise ValueError("DSwinLSTM-I requires two hidden dimensions and four depth/head entries")
        if not self.use_patch_merging or not self.use_patch_expanding:
            raise ValueError("DSwinLSTM-I currently requires patch merging and patch expanding")
        if self.decoder_feedback != "pixel_feedback":
            raise ValueError("DSwinLSTM-I currently supports only pixel_feedback decoding")
        if self.mask_as_input_channel:
            raise ValueError("DSwinLSTM-I uses its observation mask in the imputation unit, not as an input channel")
        if self.output_activation not in {"none", "linear", "tanh"}:
            raise ValueError("DSwinLSTM-I output_activation must be one of: none, linear, tanh")
        self.num_merge_stages = 1

        #Compute padded spatial dimensions compatible with patch/window partitioning
        self.padded_H, self.padded_W = self._compute_padded_shape(self.orig_H, self.orig_W)

        # Patch-embedding stem: project input frames into token sequences
        self.patch_embed = PatchEmbed(
            img_size=(self.padded_H, self.padded_W), 
            patch_size=self.patch_shape, 
            in_chans=self.F, 
            embed_dim=self.hidden_dims[0],
        )
        #Mask pooling layers for imputation token generation at each stage
        self.mask_pool_stage0 = MaskPool(patch_size=self.patch_shape)
        self.stage0_resolution = tuple(self.patch_embed.patches_resolution)
        self.stage1_resolution = (self.stage0_resolution[0] // 2, self.stage0_resolution[1] // 2)
        self.mask_pool_stage1 = MaskPool(patch_size=(self.patch_shape[0] * 2, self.patch_shape[1] * 2))

        #Patch merging (downsample) and expansion (upsample) between stages
        self.merge = PatchMerging(self.stage0_resolution, self.hidden_dims[0])
        self.expand = PatchExpand(self.stage1_resolution, self.hidden_dims[1], out_dim=self.hidden_dims[0])

        #Encoder cells with imputation for processing the input sequence
        encoder_cell = SwinLSTMCellI if self.use_imputation_unit else SwinLSTMCell
        self.enc_cell0 = encoder_cell(
            self.hidden_dims[0], 
            self.stage0_resolution, 
            self.num_heads[0], 
            self.window_size, 
            self.swin_depths[0], 
            drop=self.drop_rate, 
            attn_drop=self.attn_drop_rate, 
            drop_path=self.drop_path_rate,
        )
        
        self.enc_cell1 = encoder_cell(
            self.hidden_dims[1], 
            self.stage1_resolution, 
            self.num_heads[1], 
            self.window_size, 
            self.swin_depths[1], 
            drop=self.drop_rate, 
            attn_drop=self.attn_drop_rate, 
            drop_path=self.drop_path_rate,
        )
        #Decoder cells without imputation for autoregressive future prediction
        self.dec_cell1 = SwinLSTMCell(
            self.hidden_dims[1], 
            self.stage1_resolution, 
            self.num_heads[2], 
            self.window_size, 
            self.swin_depths[2], 
            drop=self.drop_rate, 
            attn_drop=self.attn_drop_rate, 
            drop_path=self.drop_path_rate,
        )
        self.dec_cell0 = SwinLSTMCell(
            self.hidden_dims[0], 
            self.stage0_resolution, 
            self.num_heads[3], 
            self.window_size, 
            self.swin_depths[3], 
            drop=self.drop_rate, 
            attn_drop=self.attn_drop_rate, 
            drop_path=self.drop_path_rate,
        )

        #Reconstruction head: project decoder tokens back to the original map
        self.reconstruction = Reconstruction(
            in_dim=self.hidden_dims[0], 
            out_channels=self.F, 
            map_size=(self.padded_H,self.padded_W), 
            patch_size=self.patch_shape,
        )

    def _compute_padded_shape(self, H, W):
        """
        Compute the smallest spatial dimensions >= (H, W) that are divisible
        by patch_shape * (2 ** num_merge_stages) * window_size.

        Padding ensures that all patch merging, window partitioning, and
        attention operations cover the input without remainder.
        """
        h_factor = self.patch_shape[0] * (2 ** self.num_merge_stages) * self.window_size
        w_factor = self.patch_shape[1] * (2 ** self.num_merge_stages) * self.window_size
        padded_H = ((H + h_factor - 1) // h_factor) * h_factor
        padded_W = ((W + w_factor - 1) // w_factor) * w_factor
        return padded_H, padded_W

    def _pad_frames(self, frames):
        """
        Pad spatial dimensions of frames to the pre-computed padded shape.

        Falls back from reflect to replicate padding when the input is
        too small for reflect mode to be valid.
        """
        pad_h = self.padded_H - frames.shape[-2]
        pad_w = self.padded_W - frames.shape[-1]
        if pad_h == 0 and pad_w == 0:
            return frames
        mode = self.padding_mode
        if mode == "reflect" and (frames.shape[-2] <= 1 or frames.shape[-1] <= 1 or pad_h >= frames.shape[-2] or pad_w >= frames.shape[-1]):
            mode = "replicate"
        return F.pad(frames, (0, pad_w, 0, pad_h), mode=mode)

    def _crop_frames(self, frames):
        """
        Remove spatial padding, returning the original (H, W) dimensions.
        """
        return frames[:, :, :self.orig_H, :self.orig_W]

    def _mask_tokens(self, mask):
        """
        Average-pool the imputation mask to each stage's token resolution
        and expand the result to match the hidden dimension of that stage.

        Returns:
            Tuple (stage0_mask, stage1_mask), each shaped
            (B, T, H_p * W_p, hidden_dim).
        """
        stage0 = self.mask_pool_stage0(mask).squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dims[0])
        stage1 = self.mask_pool_stage1(mask).squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dims[1])
        return stage0, stage1

    def _embed_frame(self, frame):
        """
        Patch-embed a single frame and merge to stage-1 resolution.

        Returns:
            Tuple (tokens0, tokens1) at the two encoder stage resolutions.
        """
        tokens0 = self.patch_embed(frame)
        tokens1 = self.merge(tokens0)
        return tokens0, tokens1

    def forward(
        self,
        x: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode the input sequence then autoregressively decode.

        The encoder compresses the lookback window into a latent state. When
        no observation mask is supplied, all values are treated as observed.
        The decoder generates future frames one step at a time and feeds each
        prediction back as the next input.

        Args:
            x: Input tensor shaped (B, T_in, F, H, W) in channel-first layout.
            observation_mask: Optional tensor with the same shape as x, where
                              1 denotes observed and 0 denotes missing.

        Returns:
            Predicted frames shaped (B, T_out, F, H, W) in channel-first layout.
        """
        # Ensure input is 5-dimensional and has the expected feature and spatial dimensions
        if x.dim() != 5:
            raise ValueError(
                f"Error! DSwinLSTM-I expects 5D input (B, T, F, H, W), "
                f"got {x.dim()}D"
            )

        B, T_in, F_ch, H, W = x.shape

        if F_ch != self.F:
            raise ValueError(
                f"Error! Input channels {F_ch} != model input_channels {self.F}"
            )

        if H != self.orig_H or W != self.orig_W:
            raise ValueError(
                f"Error! Input spatial ({H}, {W}) != model spatial "
                f"({self.orig_H}, {self.orig_W})"
            )

        if observation_mask is None:
            observation_mask = torch.ones_like(x)
        elif observation_mask.shape != x.shape:
            raise ValueError(
                "Error! observation_mask must have the same shape as input; "
                f"got {tuple(observation_mask.shape)} and {tuple(x.shape)}"
            )
        else:
            observation_mask = observation_mask.to(device=x.device, dtype=x.dtype)

        if not torch.isfinite(observation_mask).all():
            raise ValueError("Error! observation_mask contains non-finite values")
        if torch.any((observation_mask < 0) | (observation_mask > 1)):
            raise ValueError("Error! observation_mask values must be between 0 and 1")
        if not torch.isfinite(x).logical_or(observation_mask == 0).all():
            raise ValueError("Error! observed DSwinLSTM-I inputs must be finite")

        # Missing entries use a finite placeholder until token-level imputation.
        x = torch.where(observation_mask > 0, x, torch.zeros_like(x))

        #Pad input frames to the model's required spatial dimensions
        x = self._pad_frames(x.view(B * T_in, F_ch, H, W)).view(B, T_in, F_ch, self.padded_H, self.padded_W)

        mask_cf = observation_mask
        mask_cf = self._pad_frames(mask_cf.view(B * T_in, F_ch, H, W)).view(B, T_in, F_ch, self.padded_H, self.padded_W)
        mask_ch_last = mask_cf.permute(0, 1, 3, 4, 2).contiguous()
        mask0, mask1 = self._mask_tokens(mask_ch_last)

        #Encoder: iterate over the input sequence, updating encoder states
        enc0_state = None
        enc1_state = None
        last_frame = x[:, -1]

        for t in range(max(T_in - 1, 0)):
            tokens0 = self.patch_embed(x[:, t])
            if self.use_imputation_unit:
                tokens0, enc0_state = self.enc_cell0(tokens0, mask0[:, t], enc0_state)
            else:
                tokens0, enc0_state = self.enc_cell0(tokens0, enc0_state)
            tokens1 = self.merge(tokens0)
            if self.use_imputation_unit:
                tokens1, enc1_state = self.enc_cell1(tokens1, mask1[:, t], enc1_state)
            else:
                tokens1, enc1_state = self.enc_cell1(tokens1, enc1_state)

        #Decoder: autoregressively generate future frames
        dec1_state = enc1_state
        dec0_state = enc0_state
        feedback_frame = last_frame
        outputs = []

        for t in range(self.T_out):
            _, input_tokens1 = self._embed_frame(feedback_frame)
            dec1_tokens, dec1_state = self.dec_cell1(input_tokens1, dec1_state)
            dec0_input = self.expand(dec1_tokens) if self.use_patch_expanding else dec1_tokens
            dec0_tokens, dec0_state = self.dec_cell0(dec0_input, dec0_state)
            y_hat = self.reconstruction(dec0_tokens)
            if self.output_activation == "tanh":
                y_hat = torch.tanh(y_hat)
            outputs.append(self._crop_frames(y_hat).unsqueeze(1))
            feedback_frame = y_hat

        return torch.cat(outputs, dim=1)
