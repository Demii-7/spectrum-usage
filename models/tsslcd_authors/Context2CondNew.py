"""TSS-LCD context autoencoder, from the authors' released code.

Source: github.com/Xlab2024/TSS-LCD, commit b156d0a007 (``Context2CondNew.py``; the model code is byte-identical in the first
commit d2d4f29). Only the model classes are kept (their training and data helpers, plotting and the ``__main__`` block are
dropped), and ``ContextTransformerAE`` takes ``n_locations`` so the 3 x 250-column layout of the authors' data can be
4 x 200 for POWDER (4 receivers x 200 bins). With ``n_locations=3`` and ``input_dim=750`` it is identical to the original.
Edits, all in ``ContextTransformerAE``: ``n_locations``/``loc_dim`` replace the hard-coded 3 and 250 (positional-encoding
lengths, ``freq_pool`` input size, ``torch.chunk`` and the flatten ``view``) and the shape assertion message.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== Context Transformer Autoencoder ==================
# class ContextTransformerAE(nn.Module):
#     def __init__(self, T, D, latent_dim, nhead=4, num_layers=3):
#         super().__init__()
#         self.T, self.D = T, D
#         # 时间分支
#         self.time_proj = nn.Linear(D, latent_dim)
#         encoder_layer_t = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
#         self.time_encoder = nn.TransformerEncoder(encoder_layer_t, num_layers=num_layers)
#         # 频率分支
#         self.freq_proj = nn.Linear(T, latent_dim)
#         encoder_layer_f = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
#         self.freq_encoder = nn.TransformerEncoder(encoder_layer_f, num_layers=num_layers)
#         # 融合与解码
#         self.fuse = nn.Linear(latent_dim*2, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, T*D)
#         )
#
#     def forward(self, x):
#         B = x.size(0)
#         # 时间编码
#         xt = self.time_proj(x)       # [B, T, latent]
#         xt = xt.permute(1,0,2)       # [T, B, latent]
#         ht = self.time_encoder(xt)   # [T, B, latent]
#         ht = ht.mean(dim=0)          # [B, latent]
#         # 频率编码
#         xf = x.permute(0,2,1)        # [B, D, T]
#         xf = self.freq_proj(xf)      # [B, D, latent]
#         xf = xf.permute(1,0,2)       # [D, B, latent]
#         hf = self.freq_encoder(xf)   # [D, B, latent]
#         hf = hf.mean(dim=0)          # [B, latent]
#         # 融合
#         z = self.fuse(torch.cat([ht, hf], dim=1))  # [B, latent]
#         recon = self.decoder(z).view(B, self.T, self.D)
#         return recon, z

# =============================================================
# >>>>>>>>>>>>  Common Building Blocks  <<<<<<<<<<<<<<<
# =============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as in Vaswani et al. 2017)"""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi‑head self‑attention without external deps"""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.W_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.W_qkv(x).chunk(3, dim=-1)  # 3 × (B, T, D)
        Q, K, V = [
            t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            for t in qkv
        ]  # each: (B, H, T, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer (self‑attn + FFN)"""
    def __init__(self, dim: int, num_heads: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention + residual
        x = self.norm1(x + self.dropout(self.self_attn(x)))
        # FFN + residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N TransformerEncoderLayers"""
    def __init__(self, dim: int, num_heads: int, num_layers: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# class CrossAttention(nn.Module):
#     """Simple additive cross‑attention: treat q & kv separately then self‑attend"""
#     def __init__(self, dim_q: int, dim_kv: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         self.query_proj = nn.Linear(dim_q, dim_q)
#         self.key_proj = nn.Linear(dim_kv, dim_q)
#         self.value_proj = nn.Linear(dim_kv, dim_q)
#         self.self_attn = MultiHeadSelfAttention(dim_q, num_heads, dropout)
#         self.norm = nn.LayerNorm(dim_q)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
#         """q: (B, Tq, Dq), kv: (B, Tkv, Dkv)"""
#         q_proj = self.query_proj(q)
#         k_proj = self.key_proj(kv)
#         v_proj = self.value_proj(kv)
#         # Combine q, k, v by simple addition before self‑attn
#         fused = q_proj + k_proj + v_proj  # broadcast along T dims
#         out = self.self_attn(fused)
#         out = self.norm(q_proj + self.dropout(out))  # residual connection
#         return out

# new cross-Attention module
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, dropout=0.1, ffn_dim=2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_q)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim_q)
        )
        self.norm2 = nn.LayerNorm(dim_q)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        attn_out, _ = self.cross_attn(query=x_q, key=x_kv, value=x_kv)
        x = self.norm1(x_q + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        output = self.norm2(ffn_out + self.dropout2(ffn_out))
        return output


# class MultiHeadPooling(nn.Module):
#     def __init__(self, input_dim, output_dim, n_heads=4):
#         super(MultiHeadPooling, self).__init__()
#         self.n_heads = n_heads
#         self.attn_weights = nn.Parameter(torch.randn(n_heads, input_dim))  # [n_heads, D]
#         self.linear = nn.Linear(n_heads * input_dim, output_dim)
#
#     def forward(self, x):  # x: [B, L, D]
#         B, L, D = x.shape
#         outputs = []
#
#         for i in range(self.n_heads):
#             # [D] → [1, D] → [B, L, 1]
#             weight = self.attn_weights[i].unsqueeze(0).unsqueeze(0)  # [1, 1, D]
#             attn_scores = (x * weight).sum(dim=2)  # [B, L]
#             attn_probs = F.softmax(attn_scores, dim=1).unsqueeze(2)  # [B, L, 1]
#             pooled = (x * attn_probs).sum(dim=1)  # [B, D]
#             outputs.append(pooled)
#
#         out = torch.cat(outputs, dim=1)  # [B, n_heads * D]
#         return self.linear(out)  # [B, output_dim]
#
#
# class AttentionPooling(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attn = nn.Linear(dim, 1)
#
#     def forward(self, x):  # x: [B, T, D]
#         weights = self.attn(x)             # [B, T, 1]
#         attn_weights = torch.softmax(weights, dim=1)  # [B, T, 1]
#         pooled = (x * attn_weights).sum(dim=1)        # [B, D]
#         return pooled


# =============================================================
# >>>>>>>>>>>>   ContextTransformerAE   <<<<<<<<<<<<<<<
# =============================================================
class ContextTransformerAE(nn.Module):
    """Auto‑encoder specialised for [B, 50, 750] context sequences.

    encode():   (B, 50, 750) → (B, latent_dim)
    decode():   (B, latent_dim) → (B, 50, 750)
    forward():  returns (z, recon)
    """

    def __init__(
        self,
        seq_len: int = 70,
        input_dim: int = 750,
        num_heads: int = 5,
        num_layers: int = 2,
        latent_dim: int = 16,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_locations: int = 3,
    ):
        super().__init__()
        assert input_dim % n_locations == 0, "input_dim must be n_locations x bins"
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_locations = n_locations
        self.loc_dim = input_dim // n_locations

        # -----  Time branch  -----
        self.time_pos_enc = PositionalEncoding(input_dim)
        self.time_encoder = TransformerEncoder(input_dim, num_heads, num_layers, dim_feedforward, dropout)

        # -----  Frequency branch  -----
        # First‑stage: per‑location frequency attention
        #   For each of the 3 locations we treat 250 frequency points as tokens (length = seq_len)
        # self.freq_token_proj = nn.Linear(seq_len, input_dim)  # project 50‑length slice → dim
        self.freq_token_pos_enc = PositionalEncoding(dim=self.seq_len, max_len=self.loc_dim)
        self.freq_local_encoder = TransformerEncoder(seq_len, num_heads, num_layers, dim_feedforward, dropout)
        # self.freq_pool = AttentionPooling(input_dim // 3)
        # self.freq_pool = nn.Sequential(
        #     nn.Linear(seq_len * input_dim // 3, input_dim),
        #     nn.Dropout(dropout)
        # )
        self.freq_pool = nn.Linear(seq_len * self.loc_dim, input_dim)
        self.freq_token_proj = nn.Linear(input_dim, input_dim)  # project 50‑length slice → dim
        # self.freq_token_proj = nn.Linear(seq_len, input_dim)  # project 50‑length slice → dim
        # Second‑stage: location‑level attention over the 3 location tokens
        self.freq_pos_enc = PositionalEncoding(input_dim, max_len=n_locations)
        self.freq_global_encoder = TransformerEncoder(input_dim, num_heads, num_layers, dim_feedforward, dropout)

        # -----  Fusion  -----
        self.cross_attn = CrossAttention(dim_q=input_dim, dim_kv=input_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)
        # self.cross_attn3 =
        # self.fusion_pool = AttentionPooling(input_dim)
        # self.fusion_pool = nn.Sequential(
        #     nn.Linear(seq_len * input_dim, input_dim),
        #     nn.Dropout(dropout)
        # )
        self.fusion_pool = nn.Linear(seq_len * input_dim, input_dim)

        # -----  Latent mapping  -----
        self.to_latent = nn.Linear(input_dim, latent_dim)
        # self.to_embed = nn.Linear(latent_dim, input_dim)  # for injecting latent back during decoding

        # -----  Decoder  -----
        # self.decoder_fc = nn.Sequential(
        #     nn.Linear(latent_dim, seq_len * input_dim),
        #     nn.Dropout(dropout)
        # )
        self.decoder_fc = nn.Linear(latent_dim, seq_len * input_dim)

        # Simple weight init
        self.apply(self._init_weights)

    # ---------------------------------------------------------
    #  Helper functions
    # ---------------------------------------------------------
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.)
            nn.init.constant_(module.weight, 1.)

    # ---------------------------------------------------------
    #  Encoder
    # ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape  # Expect B×50×750
        assert T == self.seq_len and D == self.input_dim, f"Input shape must be (B, {self.seq_len}, {self.input_dim})"

        # ---- Time features ----
        time_in = self.time_pos_enc(x)            # (B, 50, 750)
        time_out = self.time_encoder(time_in)     # (B, 50, 750)
        time_repr = time_out
        # time_repr = time_out.mean(dim=1)          # (B, 750)

        # ---- Frequency features ----
        # Split into 3 locations along feature dim (size 250 each)
        loc_slices = torch.chunk(x, self.n_locations, dim=2)     # 3 × (B, 50, 250)
        loc_tokens = []
        for loc in loc_slices:                    # (B, 50, 250)
            loc = loc.permute(0, 2, 1)           # (B, 250, 50) – each freq‑pt is a token of length 50
            # loc = self.freq_token_proj(loc)      # (B, 250, 750)
            loc_in = self.freq_token_pos_enc(loc)
            loc_out = self.freq_local_encoder(loc_in)
            # loc = self.freq_token_proj(loc)  # (B, 250, 750)
            # loc_repr = loc.mean(dim=1)           # (B, 750)
            loc_repr = self.freq_pool(loc_out.view(B, T * self.loc_dim))         # (B, 750)
            loc_tokens.append(loc_repr.unsqueeze(1))
        freq_stack = torch.cat(loc_tokens, dim=1)  # (B, 3, 750)
        freq_stack = self.freq_pos_enc(freq_stack)
        freq_stack = self.freq_global_encoder(freq_stack)  # (B, 3, 750)
        freq_stack = self.freq_token_proj(freq_stack)      # (B, 3, 750)
        freq_repr = freq_stack
        # freq_repr = freq_stack.mean(dim=1)  # (B, 750)

        # ---- Fusion via cross‑attention ----
        # fusion = self.cross_attn(time_repr.unsqueeze(1), freq_repr.unsqueeze(1))  # (B, 1, 750)
        fusion = self.cross_attn(time_repr, freq_repr)
        # fusion = fusion.permute(1, 0, 2)  # (B, 50, 750)
        fusion = self.fusion_pool(fusion.view(B, T * D)) # (B, 750)

        # ---- Latent ----
        z = self.to_latent(fusion)  # (B, latent_dim)
        return z

    # ---------------------------------------------------------
    #  Decoder
    # ---------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        recon = self.decoder_fc(z)               # (B, 50*750)
        recon = recon.view(B, self.seq_len, self.input_dim)  # (B, 50, 750)
        return recon

    # ---------------------------------------------------------
    #  Forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Returns latent z and reconstruction recon"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ================== 训练与推理函数 ==================
