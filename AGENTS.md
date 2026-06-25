# Environment

## Python virtual environment
```bash
python3 -m venv /tmp/spectrum-env
source /tmp/spectrum-env/bin/activate
```

## Dependencies (CPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib tqdm scikit-learn pyyaml
```

## TimeRAN (momentfm) on Python 3.12
`momentfm==0.1.4` pins old packages without Python 3.12 wheels. Install relaxed:
```bash
pip install momentfm --no-deps
pip install transformers==4.36.0
```
This works because `transformers==4.36.0` has Python 3.12 wheels and is API-compatible with `momentfm==0.1.4`.

## Smoke tests
```bash
source /tmp/spectrum-env/bin/activate

# ConvLSTM
rm -rf training/ConvLSTM/{checkpoints,evaluation}
CUDA_VISIBLE_DEVICES="" python3 training/ConvLSTM/train.py --config training/ConvLSTM/smoke_test/config.yaml
CUDA_VISIBLE_DEVICES="" python3 training/ConvLSTM/evaluate.py --checkpoint training/ConvLSTM/checkpoints/best_model.pt --config training/ConvLSTM/smoke_test/config.yaml

# TimeRAN
rm -rf training/TimeRAN/{checkpoints,evaluation}
CUDA_VISIBLE_DEVICES="" python3 training/TimeRAN/train_head.py --config training/TimeRAN/smoke_test/config.yaml
CUDA_VISIBLE_DEVICES="" python3 training/TimeRAN/evaluate.py --checkpoint training/TimeRAN/checkpoints/best_model.pt --config training/TimeRAN/smoke_test/config.yaml

# STS-PredNet
rm -rf training/STS-PredNet/{checkpoints,evaluation}
CUDA_VISIBLE_DEVICES="" python3 training/STS-PredNet/train.py --config training/STS-PredNet/smoke_test/config.yaml
CUDA_VISIBLE_DEVICES="" python3 training/STS-PredNet/evaluate.py --checkpoint training/STS-PredNet/checkpoints/best_model.pt --config training/STS-PredNet/smoke_test/config.yaml

# TSS-LCD (3-stage training)
rm -rf training/TSS-LCD/{checkpoints,evaluation}
CUDA_VISIBLE_DEVICES="" python3 training/TSS-LCD/train_autoencoder.py --config training/TSS-LCD/smoke_test/config.yaml
CUDA_VISIBLE_DEVICES="" python3 training/TSS-LCD/train_tss_condition.py --config training/TSS-LCD/smoke_test/config.yaml --autoencoder_checkpoint training/TSS-LCD/checkpoints/best_autoencoder.pt
CUDA_VISIBLE_DEVICES="" python3 training/TSS-LCD/train_diffusion.py --config training/TSS-LCD/smoke_test/config.yaml --autoencoder_checkpoint training/TSS-LCD/checkpoints/best_autoencoder.pt --tss_checkpoint training/TSS-LCD/checkpoints/best_tss_condition.pt
CUDA_VISIBLE_DEVICES="" python3 training/TSS-LCD/evaluate.py --config training/TSS-LCD/smoke_test/config.yaml --checkpoint training/TSS-LCD/checkpoints/best_diffusion.pt --autoencoder_checkpoint training/TSS-LCD/checkpoints/best_autoencoder.pt
```
