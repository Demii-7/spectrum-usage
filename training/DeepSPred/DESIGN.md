# DeepSPred — Design Document

## Paper
"Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning"
arXiv:2408.06870v3. We implement **3D-SwinSTB** (spectrogram prediction task).

---

## Why This Format Matches the Paper

The paper's model expects: `(T frames, H height, W width, C=3 RGB channels)`

Our adaptation:
| Dimension | Paper | Ours | How |
|---|---|---|---|
| C=3 | RGB color channels | 3 channels from colormap | dBm → [0,1] → jet colormap → (R,G,B) |
| H | 256 STFT time sub-steps | 256 grouped minutes | stack 256 consecutive minute rows into one frame |
| W | 256 frequency bins | 250→256 (padded) | 250 bins per node, zero-padded to 256 |
| T | T input frames | T input frames | T consecutive H-minute spectrogram frames |

**Final per-sample shape: `(T=8, C=3, H=16, W=256)` — nearly identical to the paper.**

---

## Data Pipeline (per node)

```
cc2_smoke_test.csv  (2000 rows × 250 bins)  OR  merged CSV columns for one node
     │
     ▼  compute min/max on train rows only
     
normalize to [0,1]:   (T, 250)  →  (T, 250)   float in [0,1]
apply jet colormap:   (T, 250)  →  (T, 250, 3) float in [0,1]
group H=16 minutes:   (T, 250, 3) → (N_frames, 16, 250, 3)
pad width 250→256:    (N_frames, 16, 250, 3) → (N_frames, 16, 256, 3)

chronological split:  train 80% / val 10% / test 10% of frames
sliding window (stride=1):
   x_padded = frames[i : i+T_in]          → (T_in, 3, 16, 256)  channel-first
   y_orig   = frames[i+T_in : i+2*T_in]   → (T_in, 3, 16, 250)  unpadded target
```

**Three nodes (CC1, CC2, LW1) are treated as independent datasets**
pooled together as separate training samples. The model is node-agnostic.

**Smoke test:** `cc2_smoke_test.csv` — CC2 only, 2000 rows. No downloads needed.

---

## Architecture: 3D-SwinSTB

```
Input (B, T_in, 3, 16, 256)
  → permute to (B, 3, T_in, 16, 256)              [Conv3d convention]

PatchEmbed3D    Conv3d(3, C=96, kernel=(2,2,4), stride=(2,2,4))
  → token grid: (T_in/2, 8, 64),  tokens: (B, L, 96)

Encoder
  Stage 1:  SwinBlocks×2, heads=4     → S1 (B, T/2·8·64, 96)
  Merge:    PatchMerging3D (H×W /4)   → (B, T/2·4·32, 192)
  Stage 2:  SwinBlocks×4, heads=8     → S2 (B, T/2·4·32, 192)
  Merge:    PatchMerging3D (H×W /4)   → (B, T/2·2·16, 384)
  Stage 3:  SwinBlocks×2, heads=16    → S3 (B, T/2·2·16, 384)

Bottleneck: SwinBlocks×2, heads=16    → Xde (B, T/2·2·16, 384)

Decoder  (symmetric, with skip connections + linear after each concat)
  Concat(Xde, S3) → Linear(768,384) → SwinBlocks×2 → PatchExpanding → (B, T/2·4·32, 192)
  Concat(_, S2)   → Linear(384,192) → SwinBlocks×4 → PatchExpanding → (B, T/2·8·64, 96)
  Concat(_, S1)   → Linear(192,96)  → SwinBlocks×2

ProjectionHead
  reshape to (B, 96, T/2, 8, 64)
  ConvTranspose3d(96, 256, kernel=(2,2,4), stride=(2,2,4)) → (B, 256, T, 16, 256)
  GELU
  Conv3d(256, 3, 1) → (B, 3, T, 16, 256)
  Sigmoid → clamp output to [0,1]
  Crop W: 256→250
  Permute → (B, T_in, 3, 16, 250)

Loss: MSELoss on RGB [0,1] predictions vs targets
```

---

## SwinTransformerBlock3D

Each block:
```
x → LN → WindowAttention3D (W-MSA or SW-MSA) → residual → LN → MLP(GELU) → residual
```
- Even-indexed blocks: regular windows (no shift)
- Odd-indexed blocks: shifted windows (cyclic shift by window_size//2)
- 3D relative position bias: learned table `B̂ ∈ R^{(2P-1)×(2M_h-1)×(2M_w-1) × heads}`
- Shifted window mask: computed once per (T,H,W) grid in BasicLayer

## PatchMerging3D
Concatenates 2×2 spatial neighbors → Linear(4C → 2C) + LN. Spatial resolution halved, channels doubled.

## PatchExpanding3D
Linear(C → 2C) → einops.rearrange (2×2 spatial upsample) → LN. Spatial resolution doubled, channels halved.

---

## Key Config Parameters

```yaml
model:
  embed_dim: 96          # C in paper
  depths: [2, 4, 2]     # SwinBlocks per encoder stage
  num_heads: [4, 8, 16]
  patch_size: [2, 2, 4]  # (Tp, Hp, Wp)
  window_size: [2, 2, 4] # (P, M_h, M_w)

frames:
  minutes_per_frame: 16  # H dimension
  w_pad: 256             # pad W from 250 to 256
```

---

## Checkpoint Contents

```python
{
  "epoch": int,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "norm_stats": {
      "CC1": {"vmin": float, "vmax": float},
      "CC2": {"vmin": float, "vmax": float},
      "LW1": {"vmin": float, "vmax": float},
  },
  "config": dict,
  "metrics": {"val_loss": float, "rmse": float, "mae": float, "r2": float},
}
```

---

## Evaluation Metrics

Computed in **RGB space** (like the paper's pixel-level MSE):
- RMSE, MAE, R² on predicted vs true RGB frames

Also computed in **dBm space** via colormap inversion (nearest-neighbor LUT):
- RMSE_dBm, MAE_dBm — physically interpretable, comparable to other models in repo

---

## Files

| File | Purpose |
|---|---|
| `dataset.py` | load CSV, colormap, frame grouping, sliding windows |
| `model.py` | 3D-SwinSTB architecture |
| `train.py` | training loop |
| `evaluate.py` | test metrics + plots |
| `inference.py` | standalone inference |
| `utils.py` | seed, device, metrics, checkpoint I/O |
| `config.yaml` | full 3-node training config |
| `smoke_test/config.yaml` | CC2 only, 3 epochs, fast |
| `requirements.txt` | dependencies |
