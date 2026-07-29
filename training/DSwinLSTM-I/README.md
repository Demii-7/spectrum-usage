# DSwinLSTM-I

DSwinLSTM-I is implemented once in `models/DSwinLSTM_I.py` and trained through
the shared 4D pipeline. The architecture is a reconstruction based on the
DSwinLSTM-I paper, with SwinLSTM mechanics adapted from
<https://github.com/SongTang-x/SwinLSTM>.

## Current Data Mode

The shared loader imputes configured source gaps and rejects remaining
non-finite map values before windowing. It therefore calls the model without an
observation mask, which is equivalent to an all-observed mask. The model keeps
its imputation units for future incomplete-input experiments, but current runs
do not train or evaluate learned imputation.

The model accepts an optional channel-first observation mask:

```python
prediction = model(x, observation_mask)
```

Both tensors have shape `(B, T, F, H, W)`. Mask values are in `[0, 1]`, where
`1` is observed and `0` is missing. Calling `model(x)` treats every value as
observed.

## Shared Configuration

DSwinLSTM-I requires `data.representation: 4d`. Its active configuration uses
z-score preprocessing, so the reconstruction head must remain unbounded:

```yaml
training:
  model_name: dswinlstm_i

dswinlstm_i:
  model:
    input_sequence_length: 60
    prediction_horizon: 60
    patch_shape: [1, 1]
    embed_dim: 32
    hidden_dims: [32, 64]
    encoder_units: 2
    decoder_units: 2
    swin_depths: [1, 2, 2, 1]
    num_heads: [2, 4, 4, 2]
    window_size: 2
    use_patch_merging: true
    use_patch_expanding: true
    use_imputation_unit: true
    mask_as_input_channel: false
    output_activation: none
    decoder_feedback: pixel_feedback
    padding_mode: reflect
    drop_path_rate: 0.0
  train:
    val_fraction: 0.1
    train_stride: 5
    val_stride: 10
    test_stride: 60
    batch_size: 2
    epochs: 50
    learning_rate: 0.0001
    weight_decay: 0.0
    optimizer: adam
    gradient_clip_norm: 5.0
    early_stopping: true
    early_stopping_patience: 10
    seed: 42
```

The smaller patches and attention windows adapt the paper-inspired hierarchy
to the project's `10x10` maps while avoiding unnecessary padding.

## Train And Evaluate

Select `dswinlstm_i` in a 4D shared configuration, then run:

```bash
./.venv/bin/python -m training.common.train_integrated --config <config.yaml>
./.venv/bin/python -m training.common.evaluation_integrated --config <config.yaml> --name <run-name>
```

Checkpoints, training logs, forecasts, and metrics use the standard shared run
layout under `runs/<run-name>/`.
