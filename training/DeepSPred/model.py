"""Backward-compat shim: the 3D-SwinSTB architecture lives in models/DeepSPred.py.

Per training/REPRODUCTION_VS_BENCHMARK.md section 2, models/<Model>.py is the single
architecture module shared by the reproduction (repro/) and the integrated pipeline. This
file re-exports it for the historical ``from model import SwinSTB3D`` import path used by
train_integrated.py / evaluate_integrated.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.DeepSPred import *          # noqa: F401,F403,E402
from models.DeepSPred import (          # noqa: E402
    BasicLayer, DropPath, PaperProjectionHead, PatchEmbed3D, PatchExpanding3D,
    PatchMerging3D, ProjectionHead, SwinSTB3D, SwinTransformerBlock3D,
    TemporalProjectionHead, WindowAttention3D, _channel_reduction_schedule,
    compute_attn_mask, crop_tokens, window_partition, window_reverse,
)
