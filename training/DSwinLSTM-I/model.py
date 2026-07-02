import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.mask_proj = nn.Linear(dim, dim)
        self.mask_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, xt, hidden_states, mask=None):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C, device=xt.device)
            cx = torch.zeros(B, L, C, device=xt.device)
        else:
            hx, cx = hidden_states
        Ft = self.Swin(xt, hx)
        if mask is not None:
            Ft = Ft + self.mask_proj(mask) + self.mask_bias
        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)
        cy = gate * cell + cx
        if mask is not None:
            cy = cy + self.mask_proj(mask) + self.mask_bias
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


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
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
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.expand = nn.Linear(dim, 4 * self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        x = self.expand(x).view(B, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, self.out_dim)
        x = x.view(B, -1, self.out_dim)
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
        pooled = pooled.mean(dim=1, keepdim=True)
        _, _, H_p, W_p = pooled.shape
        pooled = pooled.view(B, T, H_p, W_p, 1)
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


class DSwinLSTM_I(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
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
        self.teacher_forcing_ratio = config.get("training", {}).get("teacher_forcing_ratio", 1.0)
        self.T_out = config["windowing"]["prediction_horizon"]
        self.padding_mode = model_cfg.get("padding_mode", "reflect")
        self.output_activation = model_cfg.get("output_activation", "tanh")
        self.use_patch_merging = model_cfg.get("use_patch_merging", True)
        self.use_patch_expanding = model_cfg.get("use_patch_expanding", True)
        self.num_merge_stages = 2 if self.use_patch_merging else 0

        self.padded_H, self.padded_W = self._compute_padded_shape(self.orig_H, self.orig_W)
        self.patch_embed = PatchEmbed(img_size=(self.padded_H, self.padded_W), patch_size=self.patch_shape, in_chans=self.F, embed_dim=self.hidden_dims[0])
        self.mask_pool_stage0 = MaskPool(patch_size=self.patch_shape)
        self.stage0_resolution = tuple(self.patch_embed.patches_resolution)
        self.stage1_resolution = (self.stage0_resolution[0] // 2, self.stage0_resolution[1] // 2)
        self.mask_pool_stage1 = MaskPool(patch_size=(self.patch_shape[0] * 2, self.patch_shape[1] * 2))

        self.merge = PatchMerging(self.stage0_resolution, self.hidden_dims[0])
        self.expand = PatchExpand(self.stage1_resolution, self.hidden_dims[1], out_dim=self.hidden_dims[0])

        self.enc_cell0 = SwinLSTMCellI(self.hidden_dims[0], self.stage0_resolution, self.num_heads[0], self.window_size, self.swin_depths[0], drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=self.drop_path_rate)
        self.enc_cell1 = SwinLSTMCellI(self.hidden_dims[1], self.stage1_resolution, self.num_heads[1], self.window_size, self.swin_depths[1], drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=self.drop_path_rate)
        self.dec_cell1 = SwinLSTMCell(self.hidden_dims[1], self.stage1_resolution, self.num_heads[2], self.window_size, self.swin_depths[2], drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=self.drop_path_rate)
        self.dec_cell0 = SwinLSTMCell(self.hidden_dims[0], self.stage0_resolution, self.num_heads[3], self.window_size, self.swin_depths[3], drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=self.drop_path_rate)

        self.reconstruction = Reconstruction(in_dim=self.hidden_dims[0], out_channels=self.F, map_size=(self.padded_H, self.padded_W), patch_size=self.patch_shape)

    def _compute_padded_shape(self, H, W):
        h_factor = self.patch_shape[0] * (2 ** self.num_merge_stages) * self.window_size
        w_factor = self.patch_shape[1] * (2 ** self.num_merge_stages) * self.window_size
        padded_H = ((H + h_factor - 1) // h_factor) * h_factor
        padded_W = ((W + w_factor - 1) // w_factor) * w_factor
        return padded_H, padded_W

    def _pad_frames(self, frames):
        pad_h = self.padded_H - frames.shape[-2]
        pad_w = self.padded_W - frames.shape[-1]
        if pad_h == 0 and pad_w == 0:
            return frames
        mode = self.padding_mode
        if mode == "reflect" and (frames.shape[-2] <= 1 or frames.shape[-1] <= 1 or pad_h >= frames.shape[-2] or pad_w >= frames.shape[-1]):
            mode = "replicate"
        return F.pad(frames, (0, pad_w, 0, pad_h), mode=mode)

    def _crop_frames(self, frames):
        return frames[:, :, :self.orig_H, :self.orig_W]

    def _mask_tokens(self, mask):
        stage0 = self.mask_pool_stage0(mask).squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dims[0])
        stage1 = self.mask_pool_stage1(mask).squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dims[1])
        return stage0, stage1

    def _embed_frame(self, frame):
        tokens0 = self.patch_embed(frame)
        tokens1 = self.merge(tokens0)
        return tokens0, tokens1

    def forward(self, x, mask, y_teacher=None, teacher_forcing_ratio=None):
        teacher_forcing_ratio = self.teacher_forcing_ratio if teacher_forcing_ratio is None else teacher_forcing_ratio
        B, T_in, F_ch, H, W = x.shape
        x = self._pad_frames(x.view(B * T_in, F_ch, H, W)).view(B, T_in, F_ch, self.padded_H, self.padded_W)
        mask_cf = mask.permute(0, 1, 4, 2, 3).contiguous()
        mask_cf = self._pad_frames(mask_cf.view(B * T_in, F_ch, H, W)).view(B, T_in, F_ch, self.padded_H, self.padded_W)
        mask_ch_last = mask_cf.permute(0, 1, 3, 4, 2).contiguous()
        mask0, mask1 = self._mask_tokens(mask_ch_last)

        enc0_state = None
        enc1_state = None
        last_frame = x[:, -1]

        for t in range(T_in):
            tokens0 = self.patch_embed(x[:, t])
            tokens0, enc0_state = self.enc_cell0(tokens0, mask0[:, t], enc0_state)
            tokens1 = self.merge(tokens0)
            tokens1, enc1_state = self.enc_cell1(tokens1, mask1[:, t], enc1_state)

        dec1_state = enc1_state
        dec0_state = enc0_state
        feedback_frame = last_frame
        outputs = []

        if y_teacher is not None:
            B_y, T_y, C_y, H_y, W_y = y_teacher.shape
            y_teacher = self._pad_frames(y_teacher.view(B_y * T_y, C_y, H_y, W_y)).view(B_y, T_y, C_y, self.padded_H, self.padded_W)

        for t in range(self.T_out):
            teacher_frame = None
            if y_teacher is not None and torch.rand(1).item() < teacher_forcing_ratio:
                teacher_frame = y_teacher[:, t]
            input_frame = teacher_frame if teacher_frame is not None else feedback_frame
            _, input_tokens1 = self._embed_frame(input_frame)
            dec1_tokens, dec1_state = self.dec_cell1(input_tokens1, dec1_state)
            dec0_input = self.expand(dec1_tokens) if self.use_patch_expanding else dec1_tokens
            dec0_tokens, dec0_state = self.dec_cell0(dec0_input, dec0_state)
            y_hat = self.reconstruction(dec0_tokens)
            if self.output_activation == "tanh":
                y_hat = torch.tanh(y_hat)
            outputs.append(self._crop_frames(y_hat).unsqueeze(1))
            feedback_frame = y_hat

        return torch.cat(outputs, dim=1)
