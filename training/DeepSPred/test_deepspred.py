"""Unit tests for the 3D-SwinSTB architecture (DeepSPred).

Required by training/REPRODUCTION_VS_BENCHMARK.md section 6.2. Every test below is a regression
test for a defect actually found in this file on 2026-10-30 (MODEL_CARD.md section 8), or an
invariant the paper asserts. Each asserts a property rather than a tied value, per section 7.

    pytest tests/test_deepspred.py -q
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

# Portable copy of tests/test_deepspred.py: imports through training/DeepSPred/model.py
# (the shim), which resolves models/DeepSPred.py at the repo root or the vendored
# DeepSPred_arch.py when this folder is used standalone. The canonical copy is
# tests/test_deepspred.py in the repository.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    DropPath,
    PaperProjectionHead,
    PatchMerging3D,
    SwinSTB3D,
    _channel_reduction_schedule,
    compute_attn_mask,
    window_partition,
)


def cfg(*, embed_dim=32, depths=(2, 2, 2), heads=(2, 4, 4), patch=(1, 4, 4), window=(1, 7, 7),
        h=60, w_pad=256, w_orig=200, t_in=1, t_out=1, drop_path=0.0, head="temporal"):
    return {
        "model": {"embed_dim": embed_dim, "depths": list(depths), "num_heads": list(heads),
                  "patch_size": list(patch), "window_size": list(window), "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": drop_path,
                  "head": head},
        "frames": {"minutes_per_frame": h, "w_pad": w_pad, "w_orig": w_orig},
        "windowing": {"input_frames": t_in, "output_frames": t_out},
    }


# --- regression: bug 8.12, mask built on a grid window_partition could not view ---------------

@pytest.mark.parametrize("window", [(1, 7, 7), (2, 7, 7), (1, 2, 2), (2, 4, 4)])
def test_attn_mask_grid_matches_window_partition(window):
    """The mask must live on the same window-aligned grid the blocks partition."""
    P, Mh, Mw = window
    T, H, W = 4, 15, 64                      # 15 and 64 are not multiples of 7
    Tp, Hp, Wp = (-(-T // P) * P, -(-H // Mh) * Mh, -(-W // Mw) * Mw)
    mask = compute_attn_mask(Tp, Hp, Wp, window, (P // 2, Mh // 2, Mw // 2), "cpu")
    n_windows = window_partition(torch.zeros(1, Tp, Hp, Wp, 1), window).shape[0]
    n_tokens = P * Mh * Mw
    assert mask.shape == (n_windows, n_tokens, n_tokens)


def test_window_7_forward_runs():
    """window_size 7 is the paper's own setting; it used to raise RuntimeError on view."""
    m = SwinSTB3D(cfg()).eval()
    with torch.no_grad():
        y = m(torch.rand(2, 1, 3, 60, 256))
    assert y.shape == (2, 1, 3, 60, 200)


# --- regression: bug 8.13, merge reported H//2 after padding an odd grid up --------------------

@pytest.mark.parametrize("h,w", [(15, 64), (7, 7), (5, 33), (16, 32)])
def test_patch_merging_reports_the_grid_it_returns(h, w):
    """The returned token count must equal the product of the reported grid."""
    C, T = 32, 2
    merged, t_out, h_out, w_out = PatchMerging3D(C)(torch.rand(1, T * h * w, C), T, h, w)
    assert merged.shape[1] == t_out * h_out * w_out
    assert (h_out, w_out) == (-(-h // 2), -(-w // 2))


# --- regression: bug 8.14, decoder skip concat on a mismatched grid ----------------------------

@pytest.mark.parametrize("h", [60, 56, 68])   # 60/4=15 odd, 56/4=14 even, 68/4=17 odd
def test_decoder_skips_survive_odd_token_grids(h):
    m = SwinSTB3D(cfg(h=h)).eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 3, h, 256))
    assert y.shape == (1, 1, 3, h, 200)


# --- regression: bug 8.15, stochastic-depth schedule exhausted when depths[-1] > 2 -------------

@pytest.mark.parametrize("depths", [(2, 2, 2), (2, 4, 2), (2, 2, 4), (4, 2, 6)])
def test_builds_for_any_stage_depths(depths):
    m = SwinSTB3D(cfg(depths=depths, drop_path=0.1))
    assert len(m.bottleneck.blocks) == depths[-1], "bottleneck mirrors the third encoder stage"


# --- regression: bug 8.16, DropPath was the identity -------------------------------------------

def test_drop_path_actually_drops():
    torch.manual_seed(0)
    dropped = (DropPath(0.5).train()(torch.ones(4096, 8)).abs().sum(dim=1) == 0).float().mean()
    assert 0.4 < float(dropped) < 0.6


def test_drop_path_is_identity_in_eval_and_at_zero():
    x = torch.rand(64, 8)
    assert torch.equal(DropPath(0.5).eval()(x), x)
    assert torch.equal(DropPath(0.0).train()(x), x)


# --- regression: bug 8.17, evaluate_integrated called load_chunk without val_fraction ----------

def test_load_chunk_call_sites_pass_val_fraction():
    """The shared loader requires val_fraction; DeepSPred was the last caller on the old form.
    Skipped when the repo root (training/common) is not available, i.e. standalone-folder use."""
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    try:
        from training.common.data import load_chunk
    except ImportError:
        pytest.skip("repo root not available (standalone DeepSPred folder)")
    assert "val_fraction" in inspect.signature(load_chunk).parameters
    src = (repo / "training" / "DeepSPred" / "evaluate_integrated.py").read_text()
    for line in src.splitlines():
        if "load_chunk(" in line and "import" not in line and not line.strip().startswith("#"):
            assert "val_fraction" in line, f"call site omits val_fraction: {line.strip()}"


# --- invariants the paper asserts --------------------------------------------------------------

def test_horizon_may_differ_from_lookback():
    """The TEMPORAL head decouples K from T; the paper's eq. (22) ties them (MODEL_CARD 11.1)."""
    m = SwinSTB3D(cfg(t_in=4, t_out=2, head="temporal")).eval()
    with torch.no_grad():
        assert m(torch.rand(1, 4, 3, 60, 256)).shape == (1, 2, 3, 60, 200)


def test_paper_head_is_the_default_and_forces_k_equals_t():
    """models/DeepSPred.py defaults are paper-faithful (REPRODUCTION_VS_BENCHMARK 3): the
    eq. (22) head, for which K = T is structural."""
    m = SwinSTB3D(cfg(t_in=2, t_out=2, head="paper")).eval()
    with torch.no_grad():
        assert m(torch.rand(1, 2, 3, 60, 256)).shape == (1, 2, 3, 60, 200)
    with pytest.raises(ValueError):
        SwinSTB3D(cfg(t_in=4, t_out=2, head="paper"))


def test_paper_head_channel_schedule():
    """eq. (22): n = ceil(log2(C/3)) 1x1x1 inverse convolutions, C -> ... -> 3."""
    assert _channel_reduction_schedule(96) == [(96, 48), (48, 24), (24, 12), (12, 6), (6, 3)]
    assert _channel_reduction_schedule(48) == [(48, 24), (24, 12), (12, 6), (6, 3)]
    head = PaperProjectionHead(96, (2, 4, 4), 256)
    convs = [(c.in_channels, c.out_channels) for c in head.reduce]
    assert convs == _channel_reduction_schedule(96)
    assert (head.deconv.in_channels, head.deconv.out_channels) == (96, 96)
    assert tuple(head.deconv.kernel_size) == (2, 4, 4)
    assert tuple(head.deconv.stride) == (2, 4, 4)


def test_training_shim_reexports_shared_module():
    """training/DeepSPred/model.py must serve the historical import path unchanged."""
    import model as shim  # noqa: E402
    assert shim.SwinSTB3D is SwinSTB3D


def test_output_is_bounded_to_unit_range():
    m = SwinSTB3D(cfg()).eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 3, 60, 256) * 10 - 5)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_decoder_mirrors_encoder_with_reversed_heads():
    m = SwinSTB3D(cfg(depths=(2, 4, 2), heads=(4, 8, 16)))
    assert [len(m.enc1.blocks), len(m.enc2.blocks), len(m.enc3.blocks)] == [2, 4, 2]
    assert [len(m.dec1.blocks), len(m.dec2.blocks), len(m.dec3.blocks)] == [2, 4, 2]
    assert [m.dec1.blocks[0].attn.num_heads,
            m.dec2.blocks[0].attn.num_heads,
            m.dec3.blocks[0].attn.num_heads] == [16, 8, 4]


def test_every_parameter_receives_gradient():
    m = SwinSTB3D(cfg()).train()
    m(torch.rand(1, 1, 3, 60, 256)).square().mean().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not dead, f"no gradient reaches: {dead[:5]}"
