"""The three LLM arms must be runnable through the integrated pipeline, not only their own scripts.

The article benchmarks three: Time-LLM with GPT-2, Time-LLM with BERT (same class, different
backbone) and OFA/GPT4TS (its own class -- no reprogramming, no prompt, ln+wpe partially
fine-tuned). All three must be reachable from training/common/train_integrated.py.

build_model has had a spectrumllm branch since the port, but the name was missing from
SUPPORTED_MODELS, so training/common/train_integrated.py rejected it and the article
reproduction had to run standalone. These tests pin the wiring shut.

The pretrained backbone needs a network download, so the untrained control is used here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from training.common.model_factory import SUPPORTED_MODELS, build_model

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {
    "spectrumllm": REPO / "training/configs/config_spectrumllm.yaml",
    "spectrumllm_bert": REPO / "training/configs/config_spectrumllm_bert.yaml",
    "ofa": REPO / "training/configs/config_ofa.yaml",
}

_SMALL = {
    "input_sequence_length": 16, "prediction_horizon": 4, "patch_length": 8, "patch_stride": 4,
    "d_model": 8, "d_ff": 8, "num_prototypes": 8, "attention_heads": 4, "dropout": 0.1,
    "prompt_mode": "none", "backbone": "random", "random_d_llm": 32, "random_vocab_size": 64,
    "llm_layers": 1, "revin_affine": False, "channel_independent": False,
}


@pytest.mark.parametrize("name", ["spectrumllm", "ofa"])
def test_arm_is_a_supported_model(name: str) -> None:
    assert name in SUPPORTED_MODELS, (
        "train_integrated refuses any model outside SUPPORTED_MODELS"
    )


def test_ofa_builds_and_partially_fine_tunes_the_backbone() -> None:
    """OFA's defining property: ln and wpe trainable, the rest of the backbone frozen."""
    pytest.importorskip("transformers")
    cfg = {k: v for k, v in _SMALL.items()
           if k not in ("d_model", "d_ff", "num_prototypes", "attention_heads",
                        "prompt_mode", "random_d_llm", "random_vocab_size")}
    cfg["backbone"] = "openai-community/gpt2"
    data = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)
    try:
        model = build_model("ofa", {"ofa": {"model": cfg}}, data)
    except Exception as exc:  # the backbone download is the only likely failure here
        pytest.skip(f"backbone unavailable: {exc}")
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.backbone.named_parameters() if not p.requires_grad]
    assert trainable and frozen
    assert all(("ln" in n) or ("wpe" in n) for n in trainable)


def test_build_model_returns_a_working_forecaster() -> None:
    train_data = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)
    model = build_model("spectrumllm", {"spectrumllm": {"model": dict(_SMALL)}}, train_data)
    out = model(torch.randn(2, 16, 3))
    assert out.shape == (2, 4, 3)


def test_build_model_infers_channel_count_from_the_data() -> None:
    for n_ch in (2, 5):
        data = np.random.default_rng(0).normal(size=(64, n_ch)).astype(np.float32)
        model = build_model("spectrumllm", {"spectrumllm": {"model": dict(_SMALL)}}, data)
        assert model(torch.randn(1, 16, n_ch)).shape == (1, 4, n_ch)


def test_build_model_rejects_non_2d_training_data() -> None:
    bad = np.zeros((8, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="time, features"):
        build_model("spectrumllm", {"spectrumllm": {"model": dict(_SMALL)}}, bad)


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_shipped_config_is_well_formed(label: str) -> None:
    cfg = yaml.safe_load(CONFIGS[label].read_text(encoding="utf-8"))
    name = cfg["training"]["models"][0]
    assert name in SUPPORTED_MODELS
    assert cfg["data"]["representation"] == "2d"
    assert cfg["data"]["chunks"], "a chunk id is what names the central checkpoint"
    model_cfg = cfg[name]["model"]
    for key in ("patch_length", "patch_stride", "channel_independent", "backbone", "llm_layers"):
        assert key in model_cfg, f"{label} must pin {key}"
    if name == "spectrumllm":
        # the two squeeze points, never left to model defaults
        for key in ("d_model", "d_ff", "prompt_stats", "prompt_mode"):
            assert key in model_cfg, f"{label} must pin {key}"
    else:
        # OFA has no reprogramming stage or prompt; those keys would be silently ignored
        for key in ("d_model", "d_ff", "num_prototypes", "prompt_mode"):
            assert key not in model_cfg, f"{label} must not pin {key}, OFA ignores it"


def test_the_two_timellm_configs_differ_only_in_backbone() -> None:
    a = yaml.safe_load(CONFIGS["spectrumllm"].read_text(encoding="utf-8"))["spectrumllm"]["model"]
    b = yaml.safe_load(CONFIGS["spectrumllm_bert"].read_text(encoding="utf-8"))["spectrumllm"]["model"]
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"backbone"}, f"expected only the backbone to differ, got {differing}"
