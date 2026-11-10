"""
DeepSPred / 3D-SwinSTB: spectrum prediction via a 3D Swin Transformer with a pyramid structure.

Single shared architecture module for reproduction and benchmark, per
training/REPRODUCTION_VS_BENCHMARK.md section 2. Defaults are PAPER-FAITHFUL (section 3):
every value the paper states carries a `[P:...]` source tag, every value it is silent about
carries a `[gap]` tag pointing at training/DeepSPred/repro/DEEPSPRED_REFERENCE.md section 4.

Paper: G. Pan et al., "Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning",
IEEE Trans. Wireless Commun. 24(1):509-525, Jan 2025 (arXiv:2408.06870v3).

Architecture (encoder-bottleneck-decoder with skip connections):
  PatchEmbed3D -> [SwinBlocks + PatchMerging] x 3 -> Bottleneck
               -> [skip concat + SwinBlocks + PatchExpanding] x 3 -> ProjectionHead

Input:  (B, T_in, 3, H, W_pad)
Output: (B, T_out, 3, H, w_orig)  values in [0,1] (sigmoid; [gap 4.6])

Two projection heads:
  head="paper" (DEFAULT) - eq. (22) of the paper. One 3D inverse convolution with kernel and
      stride (Tp,Hp,Wp) maps the token grid back to (T, H, W) at C channels, then
      n = ceil(log2(C/3)) 3D inverse convolutions with kernel and stride 1 reduce C -> 3.
      The output length equals the lookback: K = T is a property of the architecture
      [P:eq. 22], so any other K is rejected.
  head="temporal" - our benchmark adaptation: a learned Linear(T_in, K) decouples the horizon
      from the lookback plus a GELU-MLP channel reduction. Used only by the benchmark configs,
      which request it explicitly (REPRODUCTION_VS_BENCHMARK.md section 3).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth regularisation (per-sample drop)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Bernoulli(keep) per sample: the keep probability must be added
        # BEFORE flooring, otherwise mask == keep for every sample and the
        # layer degenerates to the identity.
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x / keep * mask


def window_partition(x, window_size):
    """
    Partition (B, T, H, W, C) into non-overlapping 3D windows.
    Returns (num_windows * B, Pt*Ph*Pw, C).
    """
    B, T, H, W, C = x.shape
    Pt, Ph, Pw = window_size
    x = x.view(B, T // Pt, Pt, H // Ph, Ph, W // Pw, Pw, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, Pt * Ph * Pw, C)


def window_reverse(windows, window_size, B, T, H, W):
    """Reverse of window_partition."""
    Pt, Ph, Pw = window_size
    nT, nH, nW = T // Pt, H // Ph, W // Pw
    x = windows.view(B, nT, nH, nW, Pt, Ph, Pw, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, T, H, W, -1)


def _region_slices(P, S):
    """Region slices along one axis for the shifted-window mask.

    With S == 0 nothing is rolled along this axis, so the whole axis is one
    region. Using the three-slice form in that case is wrong: ``slice(-P, -0)``
    is empty and ``slice(-0, None)`` covers the entire axis, which overwrites
    every earlier region label and yields a mask that forbids almost all
    attention.
    """
    if S == 0:
        return (slice(0, None),)
    return (slice(0, -P), slice(-P, -S), slice(-S, None))


def crop_tokens(x, T, H, W, T_out, H_out, W_out):
    """Crop a flattened token grid (B, T*H*W, C) down to (T_out, H_out, W_out).

    PatchExpanding3D always doubles the spatial extent, so after a merge of an
    odd-sized grid (which rounds up) the decoder grid is one token larger than
    the encoder grid it must be concatenated with.
    """
    if (T, H, W) == (T_out, H_out, W_out):
        return x
    B, _, C = x.shape
    x = x.view(B, T, H, W, C)[:, :T_out, :H_out, :W_out, :]
    return x.contiguous().view(B, T_out * H_out * W_out, C)


def compute_attn_mask(T, H, W, window_size, shift_size, device):
    """Attention mask for shifted-window attention (zero where attention is allowed).

    T, H, W must already be multiples of the window size.
    """
    Pt, Ph, Pw = window_size
    St, Sh, Sw = shift_size
    img = torch.zeros(1, T, H, W, 1, device=device)
    t_slices = _region_slices(Pt, St)
    h_slices = _region_slices(Ph, Sh)
    w_slices = _region_slices(Pw, Sw)
    cnt = 0
    for ts in t_slices:
        for hs in h_slices:
            for ws in w_slices:
                img[:, ts, hs, ws, :] = cnt
                cnt += 1
    mask_windows = window_partition(img, window_size).squeeze(-1)   # (nW, Pt*Ph*Pw)
    mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)    # (nW, N, N)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


# ---------------------------------------------------------------------------
# 3D Window Attention  [P:section IV-D, eq. 15-18]
# ---------------------------------------------------------------------------

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size          # (Pt, Ph, Pw)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        Pt, Ph, Pw = window_size
        # 3D relative position bias, parameterised as the smaller table
        # B-hat in R^{(2P-1)x(2Mh-1)x(2Mw-1) x heads}  [P:footnote 3]
        self.rel_bias = nn.Parameter(
            torch.zeros((2*Pt-1) * (2*Ph-1) * (2*Pw-1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        # Pre-compute relative position indices and register as buffer.
        coords = torch.stack(torch.meshgrid(
            torch.arange(Pt), torch.arange(Ph), torch.arange(Pw), indexing="ij"
        ))                                                      # (3, Pt, Ph, Pw)
        coords_flat = coords.flatten(1)                         # (3, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]# (3, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                # (N, N, 3)
        rel[:, :, 0] += Pt - 1
        rel[:, :, 1] += Ph - 1
        rel[:, :, 2] += Pw - 1
        rel[:, :, 0] *= (2*Ph-1) * (2*Pw-1)
        rel[:, :, 1] *= (2*Pw-1)
        self.register_buffer("rel_idx", rel.sum(-1))            # (N, N)

        self.qkv      = nn.Linear(dim, dim * 3)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (B_win, N, C)
        B_, N, C = x.shape
        nh = self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                               # each (B_, nh, N, d)

        attn = (q * self.scale) @ k.transpose(-2, -1)         # (B_, nh, N, N)

        bias = self.rel_bias[self.rel_idx.view(-1)]            # (N*N, nh)
        bias = bias.view(N, N, nh).permute(2, 0, 1)           # (nh, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, nh, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer Block (one W-MSA or SW-MSA block)  [P:eq. 13]
# ---------------------------------------------------------------------------

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size   # (0,0,0) for W-MSA, (Pt//2,Ph//2,Pw//2) for SW-MSA

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(dim, window_size, num_heads,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        # [gap 4.12] eq. (14) writes W1 in R^{C x 2C} (hidden 2C), but Table I's
        # parameter count (16.32 M at C=96) is only consistent with the standard
        # Swin ratio 4 (hidden 4C): mlp_ratio 4 gives ~16.2-17.1 M params, the
        # literal 2C reading gives ~12.2 M. Kept at 4.0.
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _pad_to_window(self, x, T, H, W):
        """Pad T/H/W so they are divisible by window sizes."""
        Pt, Ph, Pw = self.window_size
        pad_t = (Pt - T % Pt) % Pt
        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw
        if pad_t + pad_h + pad_w > 0:
            x = x.view(x.shape[0], T, H, W, -1)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        return x, T + pad_t, H + pad_h, W + pad_w

    def forward(self, x, T, H, W, attn_mask):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad if needed.
        x, Tp, Hp, Wp = self._pad_to_window(x, T, H, W)
        x = x.view(B, Tp, Hp, Wp, C)

        # Compute effective shift (no shift along a dim if window covers entire dim).
        Pt, Ph, Pw = self.window_size
        eff_shift = (
            self.shift_size[0] if Tp > Pt else 0,
            self.shift_size[1] if Hp > Ph else 0,
            self.shift_size[2] if Wp > Pw else 0,
        )

        if any(s > 0 for s in eff_shift):
            x = torch.roll(x, shifts=(-eff_shift[0], -eff_shift[1], -eff_shift[2]), dims=(1, 2, 3))
            mask = attn_mask
        else:
            mask = None

        x_win = window_partition(x, self.window_size)         # (nW*B, N, C)
        x_win = self.attn(x_win, mask=mask)
        x     = window_reverse(x_win, self.window_size, B, Tp, Hp, Wp)

        if any(s > 0 for s in eff_shift):
            x = torch.roll(x, shifts=(eff_shift[0], eff_shift[1], eff_shift[2]), dims=(1, 2, 3))

        # Crop back to original T, H, W.
        x = x[:, :T, :H, :W, :].contiguous().view(B, T * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# BasicLayer: a sequence of SwinTransformerBlock3D pairs
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        Pt, Ph, Pw = window_size
        self.window_size = window_size
        self.shift_size  = (Pt // 2, Ph // 2, Pw // 2)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
            )
            for i in range(depth)
        ])

    def forward(self, x, T, H, W):
        Pt, Ph, Pw = self.window_size
        St, Sh, Sw = self.shift_size
        # Compute the shifted-window mask once for this (T, H, W) grid.
        # The mask must be built on the SAME grid the blocks window-partition,
        # i.e. each dimension rounded UP to a multiple of the window size —
        # max(T, Pt) is not window-aligned whenever T is not a multiple of Pt
        # (e.g. H=15 with Ph=7) and window_partition() then fails to view.
        Tp = -(-T // Pt) * Pt
        Hp = -(-H // Ph) * Ph
        Wp = -(-W // Pw) * Pw
        attn_mask = compute_attn_mask(Tp, Hp, Wp, self.window_size,
                                      (St if Tp > Pt else 0,
                                       Sh if Hp > Ph else 0,
                                       Sw if Wp > Pw else 0),
                                       x.device)
        for blk in self.blocks:
            x = blk(x, T, H, W, attn_mask)
        return x


# ---------------------------------------------------------------------------
# Patch Embed / Merge / Expand
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """3D patch partition + linear embedding via Conv3d  [P:section IV-B]."""
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, T, H, W)
        x = self.proj(x)                           # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, T'*H'*W', embed_dim)
        return self.norm(x), T, H, W


class PatchMerging3D(nn.Module):
    """Merge 2x2 spatial patches: halve H and W, double channels  [P:eq. 11]."""
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, T, H, W):
        B, _, C = x.shape
        x = x.view(B, T, H, W, C)
        # Pad if H or W is odd.
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, T, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        # Report the grid of the tensor actually returned. When H or W was odd
        # it was padded up first, so the merged extent is ceil(H/2), not H//2;
        # returning H//2 leaves the token count and the reported grid out of
        # step for every later layer.
        return self.reduction(self.norm(x)), T, (H + 1) // 2, (W + 1) // 2


class PatchExpanding3D(nn.Module):
    """Upsample 2x in H and W: halve channels (inverse of PatchMerging3D)  [P:eq. 12]."""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x, T, H, W):
        B, _, C = x.shape
        x = self.expand(x)                         # (B, T*H*W, 2C)
        x = x.view(B, T, H, W, 2 * C)
        # rearrange: (B, T, H, W, 2C) -> (B, T, 2H, 2W, C//2)
        x = rearrange(x, "b t h w (p1 p2 c) -> b t (h p1) (w p2) c", p1=2, p2=2, c=C // 2)
        x = x.reshape(B, -1, C // 2)
        return self.norm(x), T, H * 2, W * 2


# ---------------------------------------------------------------------------
# Projection heads: token grid -> output frames
# ---------------------------------------------------------------------------

def _channel_reduction_schedule(dim, out=3):
    """Channel schedule of eq. (22)'s R_en stack: n = ceil(log2(C/out)) 1x1x1
    3D inverse convolutions  [P:eq. 21-22].

    Each step halves the channels and the last step lands on `out` (3 for the
    paper's RGB model; 1 for its Appendix-A grayscale ablation). For C of the
    form 3*2^k (the paper's 96 and 48) every step is an exact halving and the
    count equals the paper's n. For other C (the paper's own 128 ablation)
    ceil(log2(C/3)) steps cannot land on 3 by halving alone, so the final step
    maps the remainder to `out`; the count stays the paper's.
    """
    n = max(1, math.ceil(math.log2(dim / out)))
    sched, c = [], dim
    for i in range(n):
        nxt = out if i == n - 1 else max(out, c // 2)
        sched.append((c, nxt))
        c = nxt
    return sched


class PaperProjectionHead(nn.Module):
    """The paper's 3D projection layer, eq. (21)-(22)  [P:L 1244-1280].

    S_Tr^3 (T/Tp, H/Hp, W/Wp, C)
        -- 3DConv^-1, kernel and stride (Tp,Hp,Wp) -->  (T, H, W, C)
        -- R_en: n = ceil(log2(C/3)) 3DConv^-1, kernel and stride 1 --> (T, H, W, 3)

    The output frame count equals the lookback T: K is not a free parameter.
    A sigmoid bounds the output to [0,1] ([gap 4.6]: the paper's eq. (21)-(22)
    is silent on the output nonlinearity; without it the head is unbounded).
    """

    def __init__(self, dim, patch_size, w_orig, out_chans=3):
        super().__init__()
        self.patch_size = patch_size
        self.w_orig = w_orig
        # kernel and stride (tk,hk,wk) = (ts,hs,ws) = (Tp,Hp,Wp); C -> C  [P:eq. 22]
        self.deconv = nn.ConvTranspose3d(dim, dim, kernel_size=patch_size,
                                         stride=patch_size)
        # R_en: n stacked 1x1x1 3D inverse convolutions halving C down to out_chans.
        layers = []
        for c_in, c_out in _channel_reduction_schedule(dim, out_chans):
            layers.append(nn.ConvTranspose3d(c_in, c_out, kernel_size=1, stride=1))
        self.reduce = nn.ModuleList(layers)

    def forward(self, x, T_tok, H_tok, W_tok):
        B, _, C = x.shape
        x = x.view(B, T_tok, H_tok, W_tok, C).permute(0, 4, 1, 2, 3)  # (B, C, T', H', W')
        x = self.deconv(x)                                # (B, C, T, H, W_pad)
        for conv in self.reduce:
            x = conv(x)                                   # (B, 3, T, H, W_pad)
        x = torch.sigmoid(x)                              # [gap 4.6]
        x = x[:, :, :, :, : self.w_orig]                  # crop W_pad -> w_orig
        return x.permute(0, 2, 1, 3, 4)                   # (B, T, 3, H, w_orig)


class TemporalProjectionHead(nn.Module):
    """Benchmark adaptation: learned Linear(T_in, K) decouples the horizon K
    from the lookback T_in, plus a GELU + 1x1x1 conv channel reduction.

    NOT the paper's head: eq. (22) ties the output length to the lookback. Used
    only by benchmark configs that set ``head: temporal`` explicitly.
    """

    def __init__(self, dim, patch_size, w_orig, input_frames, output_frames):
        super().__init__()
        self.patch_size = patch_size
        self.w_orig = w_orig
        self.input_frames = input_frames
        self.output_frames = output_frames
        mid = max(dim * 4, 64)
        self.deconv = nn.ConvTranspose3d(dim, mid, kernel_size=patch_size, stride=patch_size)
        self.temporal_proj = nn.Linear(input_frames, output_frames)
        self.act    = nn.GELU()
        self.conv   = nn.Conv3d(mid, 3, kernel_size=1)

    def forward(self, x, T_tok, H_tok, W_tok):
        B, _, C = x.shape
        x = x.view(B, T_tok, H_tok, W_tok, C).permute(0, 4, 1, 2, 3)  # (B, C, T', H', W')
        x = self.act(self.deconv(x))                # (B, mid, T_in, H, W_pad)
        # Learned one-pass temporal projection decouples horizon K from lookback T_in.
        x = x.permute(0, 3, 4, 1, 2)
        x = self.temporal_proj(x)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, mid, T_out, H, W_pad)
        x = torch.sigmoid(self.conv(x))             # (B, 3, T_out, H, W_pad), values in [0,1]
        x = x[:, :, :, :, : self.w_orig]            # crop W_pad -> W_orig
        x = x.permute(0, 2, 1, 3, 4)               # (B, T_out, 3, H, W_orig)
        return x


# Historical name used by the benchmark pipeline and tests.
ProjectionHead = TemporalProjectionHead


# ---------------------------------------------------------------------------
# Full 3D-SwinSTB Model
# ---------------------------------------------------------------------------

class SwinSTB3D(nn.Module):
    """
    3D Swin Transformer with pyramid structure for spectrum prediction
    [P:section IV, Fig. 3].

    Encoder-bottleneck-decoder with skip connections.
    Input:  (B, T_in, 3, H, W_pad)
    Output: (B, T_out, 3, H, w_orig)  in [0, 1]

    Config (dict):
      model:
        head:           "paper" (default; eq. 22, forces K == T) or "temporal"
        embed_dim:      C = 96            [P:L 1812]
        depths:         [2, 4, 2]         [P:L 1813]
        num_heads:      [4, 8, 16]        [P:L 1813]
        patch_size:     (Tp, Hp, Wp) = (2, 4, 4)   [P:L 1819]
        window_size:    (P, M, M) = (2, 7, 7)      [P:L 1819]
        mlp_ratio:      4.0   [gap 4.12: eq. 14 says 2C; Table I params say 4C]
        drop_path_rate: 0.0   [gap 4.3: the paper names no regulariser]
      frames:    {minutes_per_frame: H, w_pad: W, w_orig: original width}
      windowing: {input_frames: T, output_frames: K}
    """

    def __init__(self, config):
        super().__init__()
        mcfg  = config["model"]
        fcfg  = config["frames"]

        embed_dim   = mcfg["embed_dim"]
        depths      = mcfg["depths"]          # e.g. [2, 4, 2]
        num_heads   = mcfg["num_heads"]       # e.g. [4, 8, 16]
        patch_size  = tuple(mcfg["patch_size"])  # (Tp, Hp, Wp)
        window_size = tuple(mcfg["window_size"]) # (P,  Mh, Mw)
        mlp_ratio   = mcfg.get("mlp_ratio", 4.0)
        drop        = mcfg.get("drop_rate", 0.0)
        attn_drop   = mcfg.get("attn_drop_rate", 0.0)
        # [gap 4.3] The paper names no stochastic depth; the paper-faithful
        # default is 0.0. Benchmark configs set it explicitly.
        drop_path_r = mcfg.get("drop_path_rate", 0.0)

        self.head_kind = str(mcfg.get("head", "paper"))
        # 3 for the paper's RGB model; 1 in / 1 out for its Appendix-A grayscale ablation.
        self.in_chans  = int(mcfg.get("in_chans", 3))
        self.out_chans = int(mcfg.get("out_chans", 3))
        self.w_orig  = fcfg.get("w_orig", 250)  # original W before padding
        self.input_frames = config["windowing"]["input_frames"]
        self.output_frames = config["windowing"]["output_frames"]
        if self.head_kind == "paper" and self.output_frames != self.input_frames:
            # Eq. (22): the projection layer emits T frames; K = T is a
            # property of the architecture, not a knob.
            raise ValueError(
                f"head='paper' (eq. 22) requires output_frames == input_frames, "
                f"got T={self.input_frames}, K={self.output_frames}. "
                f"Use head='temporal' to decouple them (benchmark adaptation)."
            )

        # Channel dimensions at each stage.
        C0, C1, C2 = embed_dim, embed_dim * 2, embed_dim * 4

        # Stochastic depth schedule.
        # encoder + decoder (sum(depths) each) + bottleneck (depths[-1] blocks).
        total_depth = sum(depths) * 2 + depths[-1]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_r, total_depth)]
        dp = iter(dp_rates)

        def _dp(n):
            return [next(dp) for _ in range(n)]

        # --- Encoder [P:eq. 9] ---
        self.patch_embed = PatchEmbed3D(self.in_chans, embed_dim, patch_size)

        self.enc1 = BasicLayer(C0, depths[0], num_heads[0], window_size,
                               mlp_ratio, drop, attn_drop, _dp(depths[0]))
        self.merge1 = PatchMerging3D(C0)

        self.enc2 = BasicLayer(C1, depths[1], num_heads[1], window_size,
                               mlp_ratio, drop, attn_drop, _dp(depths[1]))
        self.merge2 = PatchMerging3D(C1)

        self.enc3 = BasicLayer(C2, depths[2], num_heads[2], window_size,
                               mlp_ratio, drop, attn_drop, _dp(depths[2]))

        # --- Bottleneck: same number of blocks as the third encoder stage ---
        # [P:L 509, L 1814]
        self.bottleneck = BasicLayer(C2, depths[2], num_heads[2], window_size,
                                     mlp_ratio, drop, attn_drop, _dp(depths[2]))

        # --- Decoder [P:eq. 10]: symmetric, skip connections in C, heads reversed ---
        # Stage 1: concat with S3 (C2 + C2 = 2*C2) -> project to C2 -> blocks -> expand
        # [gap 4.5] eq. (10) concatenates in C without a projection; without one the
        # channel count doubles at every skip, so a Linear(2C -> C) follows each concat.
        self.dec_proj1  = nn.Linear(2 * C2, C2)
        self.dec1       = BasicLayer(C2, depths[2], num_heads[2], window_size,
                                     mlp_ratio, drop, attn_drop, _dp(depths[2]))
        self.expand1    = PatchExpanding3D(C2)

        # Stage 2: concat with S2 (C1 + C1 = 2*C1) -> project to C1 -> blocks -> expand
        self.dec_proj2  = nn.Linear(2 * C1, C1)
        self.dec2       = BasicLayer(C1, depths[1], num_heads[1], window_size,
                                     mlp_ratio, drop, attn_drop, _dp(depths[1]))
        self.expand2    = PatchExpanding3D(C1)

        # Stage 3: concat with S1 (C0 + C0 = 2*C0) -> project to C0 -> blocks
        self.dec_proj3  = nn.Linear(2 * C0, C0)
        self.dec3       = BasicLayer(C0, depths[0], num_heads[0], window_size,
                                     mlp_ratio, drop, attn_drop, _dp(depths[0]))

        # --- Head ---
        if self.head_kind == "paper":
            self.head = PaperProjectionHead(C0, patch_size, self.w_orig, self.out_chans)
        elif self.head_kind == "temporal":
            self.head = TemporalProjectionHead(C0, patch_size, self.w_orig,
                                               self.input_frames, self.output_frames)
        else:
            raise ValueError(f"unknown head {self.head_kind!r}; expected 'paper' or 'temporal'")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T_in, 3, H, W_pad)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)              # (B, 3, T, H, W) for Conv3d

        # Encoder
        x, Tt, Ht, Wt = self.patch_embed(x)       # (B, L, C0)
        x = self.enc1(x, Tt, Ht, Wt)
        S1, T1, H1, W1 = x, Tt, Ht, Wt

        x, Tt, Ht, Wt = self.merge1(x, Tt, Ht, Wt)
        x = self.enc2(x, Tt, Ht, Wt)
        S2, T2, H2, W2 = x, Tt, Ht, Wt

        x, Tt, Ht, Wt = self.merge2(x, Tt, Ht, Wt)
        x = self.enc3(x, Tt, Ht, Wt)
        S3, T3, H3, W3 = x, Tt, Ht, Wt

        # Bottleneck
        x = self.bottleneck(x, T3, H3, W3)

        # Decoder stage 1 - skip from enc3
        x = torch.cat([x, S3], dim=-1)
        x = self.dec_proj1(x)
        x = self.dec1(x, T3, H3, W3)
        x, Tt, Ht, Wt = self.expand1(x, T3, H3, W3)
        x = crop_tokens(x, Tt, Ht, Wt, T2, H2, W2)

        # Decoder stage 2 - skip from enc2
        x = torch.cat([x, S2], dim=-1)
        x = self.dec_proj2(x)
        x = self.dec2(x, T2, H2, W2)
        x, Tt, Ht, Wt = self.expand2(x, T2, H2, W2)
        x = crop_tokens(x, Tt, Ht, Wt, T1, H1, W1)

        # Decoder stage 3 - skip from enc1
        x = torch.cat([x, S1], dim=-1)
        x = self.dec_proj3(x)
        x = self.dec3(x, T1, H1, W1)

        # Project back to frames
        return self.head(x, T1, H1, W1)
