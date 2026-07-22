# Autoformer-CSA

Autoformer-CSA adapts the Autoformer long-horizon forecasting architecture for
the spectrum data pipeline. It preserves Autoformer's series decomposition,
FFT-based auto-correlation, encoder-decoder layout, and decoder trend
initialization. CSAM replaces the feed-forward sublayer in each encoder and
decoder layer.

The paper-style path uses a 4D spectrum map with shape `(time, height, width,
frequency)`. The runner reshapes it into `height * width` independent streams
with shape `(time, frequency)`, trains one shared model across those streams,
and exports predictions in `(samples, frequency, height, width)` layout. The
shared loader fits per-frequency normalization statistics on the training
portion and supplies those normalized streams to the model.

The 2D path remains available for direct comparison with the existing
residual-family runs. It does not reproduce the paper's spatial data setup.

## Commands

Use the shared configuration. The Autoformer section lives in
`training/common/config.yaml` and `training/common/config.smoke.yaml`.

```bash
./.venv/bin/python training/Autoformer-CSA/train_integrated.py \
  --config training/common/config.yaml \
  --name autoformer_csa_powder

./.venv/bin/python training/Autoformer-CSA/evaluate_integrated.py \
  --config training/common/config.yaml \
  --name autoformer_csa_powder
```

Use the paper-style map configuration for `M=96` and
`K ∈ {60, 120, 240, 300}`:

```bash
./.venv/bin/python training/Autoformer-CSA/train_integrated.py \
  --config training/configs/config_autoformer_csa_paper_map.yaml \
  --name autoformer_csa_paper_map

./.venv/bin/python training/Autoformer-CSA/evaluate_integrated.py \
  --config training/configs/config_autoformer_csa_paper_map.yaml \
  --name autoformer_csa_paper_map
```

The evaluator reads checkpoints from the named training run. Use
`--checkpoint '/path/{chunk_id}_autoformer_csa.pt'` to evaluate another set of
chunk checkpoints. Add `--skip-plots` when you only need metrics and forecast
archives.

## Configuration

The shared `autoformer_csa` section controls architecture and training:

```yaml
autoformer_csa:
  seq_len: 96
  label_len: 48
  pred_len: 60
  batch_size: 32
  epochs: 20
  learning_rate: 0.0001
  optimizer: adam
  loss: rmse
  gradient_clip: 5.0
  early_stopping: true
  patience: 6
  model:
    d_model: 64
    d_ff: 256
    encoder_layers: 2
    decoder_layers: 1
    n_heads: 8
    moving_avg: 25
    dropout: 0.05
    factor: 3
    csam_kernel_size: 7
    output_attention: false
```

`pred_len` must equal the largest value in `windowing.horizons`. The model uses
one direct decoder pass and evaluation records each requested horizon from that
sequence. `label_len` supplies the observed decoder context described in the
Autoformer paper.

The current implementation uses zero-valued time markers because the shared
loader exposes spectrum arrays rather than timestamp features. The value
embedding, decomposition, auto-correlation, and CSAM paths remain active.

## Outputs

Training writes a run under `runs/` containing:

- `config.yaml`
- `checkpoints/{chunk_id}_autoformer_csa.pt`
- `{chunk_id}_training_log.csv`

Evaluation adds:

- `aggregate_metrics.csv`
- `per_frequency_metrics.csv`
- `per_band_metrics.csv`
- `forecasts/*_predictions.npz`
- `forecasts/*_targets.npz`
- `forecasts/*_metadata.json`
- `report.txt`

Each checkpoint stores the model state, model configuration, frequencies,
normalization statistics, and training metadata. Evaluation rejects a
checkpoint when its frequencies or normalization statistics differ from the
loaded data.

## Model Details

Autoformer-CSA follows the paper's decomposition path:

1. The encoder and decoder decompose each series with a centered moving average.
2. Auto-correlation discovers repeated periods with FFT correlation and
   aggregates delayed values.
3. CSAM first applies the paper's `d_model -> d_ff` point-wise projection and
   ReLU. Separate 1D convolutions score max-pooled and average-pooled channels.
   A 1D convolution then scores temporal positions, and a final point-wise
   projection returns the features to `d_model`.
4. The decoder accumulates projected trends from each layer and adds them to the
   projected seasonal output.

The implementation is self-contained in `model.py`. It does not import or
modify an external Autoformer checkout.
