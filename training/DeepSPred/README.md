# DeepSPred

Implementation of a DeepSPred-style 3D pyramid Swin predictor in `training/DeepSPred/`.

## Defaults

- Lookback `T_in = 60`
- Prediction horizon `K = 5`
- Train/val/test split `4:1:1`
- Hidden dimension `C = 96`
- Depths `2,4,2`
- Patch size `(2,4,4)`
- Window size `(2,7,7)`
- Batch size `1`
- Epochs `20`
- Learning rate `0.001`
- Early stopping patience `4`
- Early stopping threshold `0.01` over `4` epochs

## Architecture

- Non-autoregressive one-pass sequence prediction
- 3D patch embedding
- Hierarchical encoder with patch merging
- Bottleneck
- Hierarchical decoder with patch expansion
- Pyramid skip connections
- Horizon-aware projection head that maps `T_in` history to `K` future frames in one forward pass

`BasicLayer` is the repo implementation of a paper-style 3DSTB stage: alternating 3D window attention and shifted-window attention blocks with LayerNorm, MLP, and residual connections.

## Scripts

Training and evaluation are separate:

```bash
python training/DeepSPred/train.py --config training/DeepSPred/config.yaml
python training/DeepSPred/evaluate.py --checkpoint training/DeepSPred/checkpoints/best_model.pt --config training/DeepSPred/config.yaml
```

Smoke test:

```bash
python training/DeepSPred/train.py --config training/DeepSPred/smoke_test/config.yaml
python training/DeepSPred/evaluate.py --checkpoint training/DeepSPred/smoke_test/checkpoints/best_model.pt --config training/DeepSPred/smoke_test/config.yaml
```

Standalone inference:

```bash
python training/DeepSPred/inference.py --checkpoint training/DeepSPred/checkpoints/best_model.pt --csv training/data/merged_power_data_sub6GHz_avg_per_minute.csv --out predictions.npy
```

## Outputs

Training writes:
- `best_model.pt`
- `final_model.pt`
- `training_log.json`

Evaluation writes:
- `predictions.npy`
- `targets.npy`
- `predictions_rgb.csv`
- `targets_rgb.csv`
- `predictions_dbm.npy`
- `targets_dbm.npy`
- `predictions_dbm.csv`
- `targets_dbm.csv`
- `metrics.json`
- `sample_prediction.png`

CSV exports include `sample_idx`, `horizon_idx`, and `node_name` so dBm-space outputs can be traced back to the correct per-node normalization statistics.

## Metrics

RGB-space metrics:
- MSE
- RMSE
- MAE
- R2
- PSNR
- SSIM
- LPIPS

dBm-space metrics:
- MSE
- RMSE
- MAE
- R2
- PSNR
- SSIM

If `lpips` is not installed, evaluation still runs and reports that LPIPS is unavailable.
