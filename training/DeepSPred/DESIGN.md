# DeepSPred — Design Document

## Paper
"Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning"
arXiv:2408.06870v3. We implement **3D-SwinSTB** (spectrogram prediction task).

---

## Why This Format Matches the Paper

The paper's model expects: `(T frames, H height, W width, C=3 RGB channels)`.

Our adaptation:
| Dimension | Paper | Ours | How |
|---|---|---|---|
| C=3 | RGB color channels | 3 channels from colormap | dBm → [0,1] → jet colormap → (R,G,B) |
| H | 256 STFT time sub-steps | 256 grouped rows | stack 256 consecutive rows into one frame |
| W | 256 frequency bins | 250→256 (padded) | 250 bins per node, zero-padded to 256 |
| T | T input frames | configurable lookback `T_in` | consecutive spectrogram frames |
| K | K output frames | configurable horizon `K` | direct one-pass future prediction |

**Default per-sample shape: `(T_in=60, C=3, H=256, W=256)` in, `(K=5, C=3, H=256, W=250)` out.**

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

chronological split:  train 4/6 / val 1/6 / test 1/6 of frames
sliding window (stride=1):
   x_padded = frames[i : i+T_in]          → (T_in, 3, 256, 256)  channel-first
   y_orig   = frames[i+T_in : i+T_in+K]   → (K,    3, 256, 250)  unpadded target
```

**Three nodes (CC1, CC2, LW1) are treated as independent datasets**
pooled together as separate training samples. The model is node-agnostic.

**Smoke test:** `cc2_smoke_test.csv` — CC2 only, 2000 rows. No downloads needed.

---

## Architecture: 3D-SwinSTB

```
Input (B, T_in, 3, 256, 256)
  → permute to (B, 3, T_in, 256, 256)             [Conv3d convention]

PatchEmbed3D    Conv3d(3, C=96, kernel=(2,4,4), stride=(2,4,4))
  → token grid: (T_in/2, 64, 64),  tokens: (B, L, 96)

Encoder
  Stage 1:  SwinBlocks×2, heads=4     → S1 (B, T/2·64·64, 96)
  Merge:    PatchMerging3D (H×W /4)   → (B, T/2·32·32, 192)
  Stage 2:  SwinBlocks×4, heads=8     → S2 (B, T/2·32·32, 192)
  Merge:    PatchMerging3D (H×W /4)   → (B, T/2·16·16, 384)
  Stage 3:  SwinBlocks×2, heads=16    → S3 (B, T/2·16·16, 384)

Bottleneck: SwinBlocks×2, heads=16    → Xde (B, T/2·16·16, 384)

Decoder  (symmetric, with skip connections + linear after each concat)
  Concat(Xde, S3) → Linear(768,384) → SwinBlocks×2 → PatchExpanding → (B, T/2·32·32, 192)
  Concat(_, S2)   → Linear(384,192) → SwinBlocks×4 → PatchExpanding → (B, T/2·64·64, 96)
  Concat(_, S1)   → Linear(192,96)  → SwinBlocks×2

ProjectionHead
  reshape to (B, 96, T/2, 64, 64)
  ConvTranspose3d(96, 384, kernel=(2,4,4), stride=(2,4,4)) → (B, 384, T_in, 256, 256)
  Linear(T_in, K) over the temporal axis                   → (B, 384, K, 256, 256)
  GELU
  Conv3d(384, 3, 1) → (B, 3, K, 256, 256)
  Sigmoid → clamp output to [0,1]
  Crop W: 256→250
  Permute → (B, K, 3, 256, 250)

Loss: MSELoss on RGB [0,1] predictions vs targets
```

---

## SwinTransformerBlock3D

Each paper-style 3DSTB stage is implemented by `BasicLayer`, which stacks alternating regular-window and shifted-window 3D Swin blocks.

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
  patch_size: [2, 4, 4]  # (Tp, Hp, Wp)
  window_size: [2, 7, 7] # (P, M_h, M_w)

windowing:
  input_frames: 60
  output_frames: 5

frames:
  minutes_per_frame: 256 # H dimension
  w_pad: 256             # pad W from 250 to 256

training:
  batch_size: 1
  epochs: 20
  learning_rate: 0.001
  patience: 4
  early_stopping_threshold: 0.01
  early_stopping_epochs: 4
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
  "metrics": {"val_loss": float, "rmse": float, "mae": float, "r2": float, "total_training_time_sec": float},
}
```

---

## Evaluation Metrics

Computed in **RGB space**:
- MSE, RMSE, MAE, R², PSNR, SSIM, LPIPS

Also computed in **dBm space** via colormap inversion (nearest-neighbor LUT):
- MSE, RMSE, MAE, R², PSNR, SSIM

Evaluation saves:
- `predictions.npy`, `targets.npy`
- `predictions_rgb.csv`, `targets_rgb.csv`
- `predictions_dbm.npy`, `targets_dbm.npy`
- `predictions_dbm.csv`, `targets_dbm.csv`
- `metrics.json`
- `sample_prediction.png`

Each evaluation sample keeps its source `node_name`, so dBm inversion and per-node dBm metrics use the correct normalization statistics for pooled multi-node test sets.

---

## Files

| File | Purpose |
|---|---|
| `dataset.py` | load CSV, colormap, frame grouping, sliding windows |
| `model.py` | 3D-SwinSTB architecture |
| `train.py` | training loop |
| `evaluate.py` | test metrics + saved outputs |
| `inference.py` | standalone inference |
| `utils.py` | seed, device, metrics, checkpoint I/O |
| `config.yaml` | full 3-node training config |
| `smoke_test/config.yaml` | CC2 only, 3 epochs, fast |
| `requirements.txt` | dependencies |
