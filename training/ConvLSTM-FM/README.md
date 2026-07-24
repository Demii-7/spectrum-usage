# ConvLSTM-FM

Config name: `convlstmfm`. Trained through the shared integrated pipeline
(`training/common/train_integrated.py`), same as `convlstm` / `residualconvlstm`.

## Scope

Reimplements the core idea of *Self-Supervised Radio Pre-training: Toward
Foundational Models for Spectrogram Learning* (Aboulfotouh, Eshaghbeigi,
Karslidis, Abou-Zeid -- IEEE GLOBECOM 2024): pretrain a ConvLSTM backbone with
masked-reconstruction self-supervision, then reuse it -- optionally frozen --
as the encoder for a downstream forecasting task.

The paper's full pipeline covers a lot this repo doesn't have: real IQ-capture
spectrogram recordings sliced into "sentences" of tokens, a segmentation
downstream task (5G-NR vs. LTE vs. noise, on a separate simulated dataset), and
an RB-occupancy evaluation metric. None of that data exists here, so this
implementation keeps only the piece that maps onto data this repo already
has (the same spectrum maps used by `convlstm` / `residualconvlstm`):

- the **ConvLSTM backbone**, pretrained via **masked-reconstruction
  self-supervision** (mask a random subset of input timesteps with noise
  matching the window's own mean/std, train the model to reconstruct the
  original -- masked MSE loss, i.e. unmasked timesteps contribute zero loss);
- reused as the encoder for this repo's existing **one-step spectrum-map
  forecasting** task, optionally with the backbone frozen so only the head
  keeps training after pretraining.

No segmentation task, IQ/spectrogram tokenization, or RB-occupancy metric was
built -- they'd need datasets this repo doesn't have. See
`models/ConvLSTM_FM.py` for the implementation.

## How it works

- `models/ConvLSTM_FM.py` defines `ConvLSTMFMForecaster`: a multi-layer
  ConvLSTM backbone (paper default: 5 layers, 64 channels, 3x3 kernels, ReLU)
  plus a Conv2d head, standing in for the paper's Conv3D head since the
  backbone already models time recurrently.
- **Stage A (pretraining):** `reconstruct(x)` reconstructs every timestep of a
  masked input window; `pretrain_backbone(...)` runs the masked-reconstruction
  loop directly on the training chunk's own data, once, before the shared
  trainer starts. This is triggered from `model_factory.build_model` when
  `convlstmfm.model.pretrain_epochs > 0`.
- **Stage B (fine-tuning):** `forward(x)` predicts the single next timestep
  (the shared one-step forecasting interface used by
  `training/common/forecasting.py`). Setting
  `freeze_backbone_after_pretrain: true` freezes the pretrained ConvLSTM
  encoder so only the head keeps training, matching the paper's "only the
  final layer is fine-tuned."

## Running it

```bash
python3 training/common/train_integrated.py --config training/configs/config_convlstm_fm.yaml
```

This trains `convlstmfm` (and `lookbackmean4d`, the baseline it's compared against) for
every chunk in `data.chunks`, writing checkpoints and logs to `runs/<name>/`, same
as any other model in the shared pipeline. To evaluate afterwards:

```bash
python3 training/common/evaluation_integrated.py --config training/configs/config_convlstm_fm.yaml --name <name>
```

(`<name>` must match the run name used during training -- see `training/README.md`
"Evaluating a trained model" for the full walkthrough.)

Note: this config's `data.files` paths point at POWDER map recordings
(`data/powder/...`) the same way `config_residual_convlstm.yaml` does. That raw
data isn't checked into this repo -- see the top-level `training/README.md`
download instructions before running against it. The architecture itself has
been smoke-tested directly through `training/common/model_factory.build_model`
and `training/common/train_integrated.train_model` with synthetic map data
(construction, Stage-A pretraining, backbone freezing, and a training epoch all
verified end-to-end).

See `training/configs/config_convlstm_fm.yaml` for the `convlstmfm:` section
(model architecture + pretraining knobs) and the top-level `training/README.md`
config reference for the shared pipeline options.
