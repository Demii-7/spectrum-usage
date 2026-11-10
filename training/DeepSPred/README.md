# DeepSPred — reproduction of Pan et al. (2025), 3D-SwinSTB spectrogram prediction

Reproduction of *"Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning"*
(IEEE TWC 24(1):509-525) on the authors' released raw I/Q data. Training produces the
self-describing checkpoints in `checkpoints/repro/deepspred/`; evaluation re-scores those
checkpoints and rebuilds `repro/results.json`.

**Read `REPRODUCTION.md`** for what the code does differently from the paper, and why every
result is what it is (noise floor, JPEG rendering hypothesis, metric space, run ledger, verdict
against the paper's anchors). This file only says how to run things.

## Pipeline

```
data/datasets/deepspred/<FM|LTE|Cross-Validation>-Spectrum-Dataset/*.xls    raw release (gitignored)
      |  fm_data.py: I/Q -> STFT (hann, two-sided, hop 126) -> dB -> jet -> uint8 -> optional JPEG
      v
training/DeepSPred/repro/run_paper_faithful.py            TRAIN (resumable by run key)
      |  AdamW lr 1e-3, batch 1, <= 20 epochs, the paper's early-stopping rule
      v
checkpoints/repro/deepspred/<sha1(run_key)[:12]>.pt       13 checkpoints (object store:
                                                          spectrum/checkpoints/repro/deepspred/)
      |
      v
training/DeepSPred/repro/verify_checkpoints.py            EVALUATE (re-scores every checkpoint,
      |                                                  checks the documented cells)
      v
training/DeepSPred/repro/results.json                     13 run records + _probes payloads
```

## Environment (uv)

```bash
uv venv --python 3.10
uv pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -r training/DeepSPred/requirements.txt
```

## Run

```bash
cd training/DeepSPred/repro

# train the headline run (~7-8 h on a 24 GB GPU; runs already in results.json are skipped)
python run_paper_faithful.py --data ../../../data/datasets/deepspred/FM-Spectrum-Dataset \
    --horizon 20 --seed 0 --jpeg-quality 95

# evaluate every checkpoint; expect "documented cells checked: 25, DIFFER: 0"
python verify_checkpoints.py
```

Ablations, grayscale, transfer learning, the metric-space probes and the full reasoning behind
every switch: `REPRODUCTION.md` sections 2-4. Windows and cache notes: `REPRODUCTION.md`
section 7.3 (in short: build frame caches with `load_frames(..., workers=1)` on Windows —
`fm_data` otherwise uses a `fork` pool — and expect ~2-4 h and ~75 GB RAM per full verification).

## Files

| File | Purpose |
|---|---|
| `REPRODUCTION.md` | code vs paper, reasoning, results, verdict — the one doc to read |
| `repro/fm_data.py` | raw I/Q release -> rendered frames (STFT/dB/colormap/JPEG), caching |
| `repro/metrics.py` | MSE (4 candidate spaces), PSNR, SSIM (windowed + global), LPIPS |
| `repro/run_paper_faithful.py` | training; paper regime hardcoded; writes the checkpoints |
| `repro/verify_checkpoints.py` | evaluation; re-scores checkpoints; rebuilds results.json |
| `repro/results.json` | the run records + probe payloads (single results file) |
| `requirements.txt` | dependencies (repro + integrated tracks) |
| `{dataset,model,utils,config_support,train_integrated,evaluate_integrated}.py` | integrated POWDER benchmark track — a separate regime wired into `training/common/` (loaded by `training/common/specialized_models.py`); its numbers are never mixed with the reproduction |

The shared architecture lives at `models/DeepSPred.py` (repo root); `model.py` is a shim
re-exporting it for the historical `from model import SwinSTB3D` import path.

## Retraining on your own data (standalone track)

`train.py` / `evaluate.py` / `inference.py` turn any per-timestamp power matrix into the
paper's image regime and train on it:

```bash
python training/DeepSPred/train.py --config <your-config.yaml>
python training/DeepSPred/evaluate.py --checkpoint <dir>/best_model.pt --config <your-config.yaml>
python training/DeepSPred/inference.py --checkpoint <dir>/best_model.pt --csv <data.csv> --out predictions.npy
```

Where a new data form plugs in, in order of likelihood of change:

1. **`data:` (config)** — `dataset_path` plus one entry per node/site with its column range
   in the CSV. Multiple nodes are pooled as independent training samples (windows never mix
   nodes).
2. **`frames:`** — `minutes_per_frame` = H (how many consecutive rows form one image),
   `w_pad` (model input width; must be >= `w_orig`, divisible by `patch_w x 4`), `w_orig`
   (your real bin count; the model crops back to it).
3. **`windowing:`** — `input_frames` (T) and `output_frames` (K). With the default
   `head: paper`, **K must equal T** (eq. 22); set `head: temporal` to decouple them.
   T must be divisible by `patch_size[0]`.
4. **`dataset.py`** — only if your data is not "one CSV: rows = timestamps, columns = bins".
   `_load_node()` is the single place that reads raw data into a `(T, W)` float matrix;
   everything downstream (train-stats normalisation, colormap, framing, chronological
   split, windowing) is data-form agnostic.
5. **`model:`** — shrink for small datasets (the benchmark used `embed_dim: 32`,
   `depths: [2,2,2]` = 1.6 M params at a 60-row frame).

While changing things, keep two guards green: `pytest training/DeepSPred/test_deepspred.py`
(26 architecture regression tests) and a 1-2 epoch run on a small slice before any long run.
