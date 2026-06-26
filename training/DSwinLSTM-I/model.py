import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
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
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
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
        self.red = nn.Linear(2 * dim, dim)
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
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
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
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.Swin = SwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path, norm_layer=norm_layer)

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C, device=xt.device)
            cx = torch.zeros(B, L, C, device=xt.device)
        else:
            hx, cx = hidden_states
        Ft = self.Swin(xt, hx)
        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        return hy, (hy, cy)


class SwinLSTMCellI(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.Swin = SwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                    num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path, norm_layer=norm_layer)
        self.W_p = nn.Linear(dim, dim)
        self.U_p = nn.Linear(dim, dim)
        self.b_p = nn.Parameter(torch.zeros(dim))

    def forward(self, xt, mask, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C, device=xt.device)
            cx = torch.zeros(B, L, C, device=xt.device)
        else:
            hx, cx = hidden_states
        P_hat = torch.sigmoid(self.W_p(cx) + self.U_p(hx) + self.b_p)
        xt_filled = mask * xt + (1 - mask) * P_hat
        Ft = self.Swin(xt_filled, hx)
        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        return hy, (hy, cy)


class PatchEmbed(nn.Module):
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
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MaskPool(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, mask):
        B, T, H, W, F = mask.shape
        mask_5d = mask.permute(0, 1, 4, 2, 3).contiguous()
        B, T, F, H, W = mask_5d.shape
        mask_4d = mask_5d.view(B * T, F, H, W)
        pooled = self.pool(mask_4d)
        _, F_p, H_p, W_p = pooled.shape
        pooled = pooled.view(B, T, F_p, H_p, W_p)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()
        pooled = pooled.view(B, T, -1, 1)
        return pooled


class Reconstruction(nn.Module):
    def __init__(self, in_dim, out_channels, map_size, patch_size):
        super().__init__()
        self.map_size = to_2tuple(map_size)
        self.patch_size = to_2tuple(patch_size)
        pH = self.map_size[0] // self.patch_size[0]
        pW = self.map_size[1] // self.patch_size[1]
        self.proj = nn.Linear(in_dim, self.patch_size[0] * self.patch_size[1] * out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        B, L, C = x.shape
        x = self.proj(x)
        pH = int(L ** 0.5) if int(L ** 0.5) ** 2 == L else -1
        pH_actual = self.map_size[0] // self.patch_size[0]
        pW_actual = self.map_size[1] // self.patch_size[1]
        x = x.view(B, pH_actual, pW_actual, self.patch_size[0], self.patch_size[1], self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.map_size[0], self.map_size[1])
        return x


class Encoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads_list, window_size, depths,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = SwinLSTMCellI(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads_list[i] if num_heads_list is not None else 4,
                window_size=window_size, depth=depths[i],
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate)
            self.layers.append(layer)

    def forward(self, x, mask, hidden_states_list):
        new_hidden = []
        for i, layer in enumerate(self.layers):
            h_prev = hidden_states_list[i] if i < len(hidden_states_list) else None
            x, hs = layer(x, mask, h_prev)
            new_hidden.append(hs)
        return x, new_hidden


class Decoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads_list, window_size, depths,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = SwinLSTMCell(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads_list[i] if num_heads_list is not None else 4,
                window_size=window_size, depth=depths[i],
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate)
            self.layers.append(layer)

    def forward(self, x, hidden_states_list):
        new_hidden = []
        for i, layer in enumerate(self.layers):
            h_prev = hidden_states_list[i] if i < len(hidden_states_list) else None
            x, hs = layer(x, h_prev)
            new_hidden.append(hs)
        return x, new_hidden


class DSwinLSTM_I(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        H = model_cfg["map_height"]
        W = model_cfg["map_width"]
        F = model_cfg["input_channels"]
        patch_shape = model_cfg.get("patch_shape", [1, 2])
        embed_dim = model_cfg.get("embed_dim", 128)
        encoder_units = model_cfg.get("encoder_units", 2)
        decoder_units = model_cfg.get("decoder_units", 2)
        swin_depths = model_cfg.get("swin_depths", [2, 6, 6, 2])
        num_heads = model_cfg.get("num_heads", [4, 8, 8, 4])
        window_size = model_cfg.get("window_size", 4)
        drop_rate = model_cfg.get("drop_rate", 0.)
        attn_drop_rate = model_cfg.get("attn_drop_rate", 0.)
        drop_path_rate = model_cfg.get("drop_path_rate", 0.1)
        self.decoder_feedback = model_cfg.get("decoder_feedback", "hidden_state")
        self.T_out = config["windowing"]["prediction_horizon"]

        enc_depths = swin_depths[:encoder_units]
        dec_depths = swin_depths[encoder_units:encoder_units + decoder_units]
        enc_heads = num_heads[:encoder_units] if len(num_heads) >= encoder_units else num_heads[:1] * encoder_units
        dec_heads = num_heads[encoder_units:encoder_units + decoder_units] if len(num_heads) >= encoder_units + decoder_units else num_heads[:1] * decoder_units

        self.patch_embed = PatchEmbed(img_size=(H, W), patch_size=patch_shape, in_chans=F, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution

        self.mask_pool = MaskPool(patch_size=patch_shape)

        self.encoder = Encoder(dim=embed_dim, input_resolution=tuple(patches_resolution),
                               num_heads_list=enc_heads, window_size=window_size, depths=enc_depths,
                               drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

        self.decoder = Decoder(dim=embed_dim, input_resolution=tuple(patches_resolution),
                               num_heads_list=dec_heads, window_size=window_size, depths=dec_depths,
                               drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

        self.reconstruction = Reconstruction(in_dim=embed_dim, out_channels=F,
                                             map_size=(H, W), patch_size=patch_shape)

    def forward(self, x, mask):
        B, T_in, F, H, W = x.shape

        pooled_mask = self.mask_pool(mask)
        pooled_mask = pooled_mask.squeeze(-1)
        pooled_mask = pooled_mask.unsqueeze(-1).expand(-1, -1, -1, self.patch_embed.embed_dim)

        enc_hidden = [None] * len(self.encoder.layers)

        for t in range(T_in):
            xt = x[:, t]
            xt_tokens = self.patch_embed(xt)
            mt = pooled_mask[:, t]
            xt_tokens, enc_hidden = self.encoder(xt_tokens, mt, enc_hidden)

        dec_hidden = [None] * len(self.decoder.layers)
        decoder_input = xt_tokens
        outputs = []

        for t in range(self.T_out):
            dec_tokens, dec_hidden = self.decoder(decoder_input, dec_hidden)
            y_hat = self.reconstruction(dec_tokens)
            y_hat = torch.tanh(y_hat)
            outputs.append(y_hat.unsqueeze(1))

            if self.decoder_feedback == "pixel_feedback":
                decoder_input = self.patch_embed(y_hat)
            else:
                decoder_input = dec_tokens

        return torch.cat(outputs, dim=1)
