# ConvLSTM-FM

Config name: `convlstmfm`. Trained through the shared integrated pipeline
(`training/common/train_integrated.py`), same as `convlstm` / `residualconvlstm`.

## Paper-native pipeline

Reimplements the core idea of *Self-Supervised Radio Pre-training: Toward
Foundational Models for Spectrogram Learning* (Aboulfotouh, Eshaghbeigi,
Karslidis, Abou-Zeid -- IEEE GLOBECOM 2024): pretrain a ConvLSTM backbone with
masked-reconstruction self-supervision, then reuse it -- optionally frozen --
as the encoder for a downstream forecasting task.

`preprocessing.py` implements the IQ path independently of the repository's map
forecasting pipeline: non-overlapping 2 ms recording slices, complex STFT
(FFT 1024, Hann window 512, hop 512 by default), configurable magnitude/power,
log floor and FFT shift, random successive 10-20 ms sentence
concatenation/resizing to 256x256, and 16
width tokens shaped `(B, T, 1, H, Wtoken)`. Sentences are completed within one
recording only; unused tail slices are discarded rather than joined to the next
recording.

- the **ConvLSTM backbone**, pretrained via **masked-reconstruction
  self-supervision** (mask complete radio tokens, or complete map timesteps in
  the shared map adaptation, with noise
  matching the window's own mean/std, train the model to reconstruct the
  original -- masked MSE loss, i.e. unmasked timesteps contribute zero loss);
- reused as the encoder for this repo's existing **one-step spectrum-map
  forecasting** task, optionally with the backbone frozen so only the head
  keeps training after pretraining.

`pretrain.py` accepts complex NumPy recordings and precomputed NumPy/PyTorch
sentence or token tensors. It saves encoder and reconstruction-head states plus
the full preprocessing metadata. See `example_pretrain.json`.

`models/ConvLSTM_FM_Segmentation.py` provides the frozen-backbone downstream
model. It concatenates the encoded token widths while retaining all 64 feature
channels, then applies two Conv2d layers to produce three-class pixel logits.
`segmentation.py` provides cross-entropy training, row-normalized confusion
matrices, and an explicit multiclass-to-signal/noise conversion.

The original paper's NR/LTE simulation data is unavailable in this repository.
The code does **not** generate or infer NR/LTE ground-truth labels. Segmentation
training and evaluation require externally supplied, spatially aligned labeled
inputs; class IDs and the noise-class ID must be chosen to match that dataset.

## How it works

- `models/ConvLSTM_FM.py` defines `ConvLSTMFMForecaster`: a multi-layer
  ConvLSTM backbone (paper default: 5 layers, 64 channels, 3x3 kernels, ReLU)
  plus the paper's temporal-spatial Conv3D reconstruction and forecasting head.
- **Stage A (pretraining):** `reconstruct(x)` reconstructs every timestep of a
  masked input window; `pretrain_backbone(...)` runs the masked-reconstruction
  loop over the shared trainer's segment-safe training loader. Model factory
  construction has no training side effects; Stage A runs after device setup
  and before the forecasting optimizer is created when
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

Paper-native token pretraining is separate:

```bash
python3 training/ConvLSTM-FM/pretrain.py --config training/ConvLSTM-FM/example_pretrain.json
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
