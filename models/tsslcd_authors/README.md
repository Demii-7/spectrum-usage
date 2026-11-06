# TSS-LCD, authors' released model code

Copied from github.com/Xlab2024/TSS-LCD at commit `b156d0a007` (2025-12-03). The model classes are the same in the first
commit `d2d4f29`. The repository ships no LICENSE; the project owner decided (2026-11-05) to vendor it anyway.

| file | status |
|---|---|
| `NoiseNet.py` | unmodified (diffusion model, cosine schedule s=0.0018, their `p_sample`, `inference`) |
| `F2Cnet.py` | unmodified (future autoencoder) |
| `Context2CondNew.py` | model classes only; `ContextTransformerAE` takes `n_locations` instead of the hard-coded 3 x 250 layout (defaults reproduce the original) |

Their data preparation (Savitzky-Golay smoothing, scaler) and test harness (eval-time smoothing) are **not** copied: they are
re-implemented with on/off switches in `training/ray/tsslcd_authors.py`.
