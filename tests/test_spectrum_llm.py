import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.SpectrumLLM import (
    PatchTokenizer,
    RevIN,
    SpectrumLLMForecaster,
    autocorrelation_lags,
    series_statistics,
)
from training.common.model_factory import build_model

# The pretrained backbone needs a network download, so the suite exercises the
# declared untrained control. Faithfulness of the gpt2 path is covered by the
# smoke run documented in training/SpectrumLLM/README.md.
_BACKBONE = {
    "backbone": "random",
    "random_d_llm": 64,
    "random_vocab_size": 256,
    "random_ffn_dim": 128,
    "llm_layers": 1,
    "prompt_mode": "none",
}


def _model_config(**overrides) -> dict:
    config = {
        "input_size": 3,
        "input_sequence_length": 16,
        "prediction_horizon": 4,
        "patch_length": 8,
        "patch_stride": 4,
        "d_model": 16,
        "num_prototypes": 8,
        "attention_heads": 4,
        "dropout": 0.0,
        **_BACKBONE,
    }
    config.update(overrides)
    return {"model": config}


def test_spectrum_llm_shape_and_gradients() -> None:
    model = SpectrumLLMForecaster(_model_config())
    x = torch.rand(2, 16, 3, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 4, 3)
    output.sum().backward()
    assert x.grad is not None


def test_backbone_is_frozen_and_adapters_are_trainable() -> None:
    model = SpectrumLLMForecaster(_model_config())
    assert all(not p.requires_grad for p in model.backbone.parameters())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("patch_embedding") for name in trainable)
    assert any(name.startswith("reprogramming") for name in trainable)
    assert any(name.startswith("output_projection") for name in trainable)
    assert not any(name.startswith("backbone") for name in trainable)


def test_patch_tokenizer_counts_and_shapes() -> None:
    # replication padding of `stride` before unfolding, as Time-LLM does, so the
    # count is (L - P)/S + 2 = 4 for L=16, P=8, S=4  [T:layers/Embed.py:166,180]
    tokenizer = PatchTokenizer(patch_length=8, stride=4)
    assert tokenizer.num_patches(16) == 4
    patches = tokenizer(torch.rand(2, 16, 3))
    assert patches.shape == (2, 4, 8 * 3)


def test_patch_tokenizer_rejects_short_sequence() -> None:
    with pytest.raises(ValueError):
        PatchTokenizer(patch_length=32, stride=4).num_patches(16)


def test_revin_round_trip_restores_scale() -> None:
    revin = RevIN(num_features=3, affine=False)
    x = torch.randn(4, 20, 3) * 7.0 + 55.0
    normalized = revin.normalize(x)
    assert abs(float(normalized.mean())) < 1e-4
    torch.testing.assert_close(revin.denormalize(normalized), x, rtol=1e-4, atol=1e-4)


def test_series_statistics_reports_prompt_fields() -> None:
    stats = series_statistics(torch.arange(12, dtype=torch.float32).reshape(2, 6), num_lags=3)
    assert len(stats) == 2
    assert stats[0]["trend"] == "upward"
    assert stats[0]["min"] == 0.0 and stats[0]["max"] == 5.0
    assert len(stats[0]["lags"]) == 3


def test_lags_are_autocorrelation_not_largest_samples() -> None:
    """Time-LLM ranks FFT autocorrelation lags, not the biggest samples
    [T:models/TimeLLM.py:257-264].

    A signal with an integer number of periods has an exactly tied circular
    autocorrelation at every multiple of the period, so topk's ordering among them is
    arbitrary; assert the returned set, not its order. Lag 0 leads only once the tie
    is broken.
    """
    time = torch.arange(64, dtype=torch.float32)
    periodic = torch.sin(2 * torch.pi * time / 16).reshape(1, 64, 1)
    lags = autocorrelation_lags(periodic, top_k=3)
    assert lags.shape == (1, 3)
    assert all(min(abs(int(v) - k * 16) for k in range(5)) <= 1 for v in lags[0])

    torch.manual_seed(0)
    mixed = (torch.sin(2 * torch.pi * time / 16)
             + 0.02 * time
             + 0.1 * torch.randn(64)).reshape(1, 64, 1)
    assert int(autocorrelation_lags(mixed, top_k=3)[0, 0]) == 0
    assert series_statistics(mixed, num_lags=3)[0]["lags"][0] == 0

    ramp = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1)
    assert series_statistics(ramp, num_lags=3)[0]["lags"] != [63, 62, 61]


def test_rejects_wrong_input_dimensions() -> None:
    model = SpectrumLLMForecaster(_model_config())
    with pytest.raises(ValueError):
        model(torch.rand(2, 10, 3))
    with pytest.raises(ValueError):
        model(torch.rand(2, 16))


def test_build_model_wires_spectrumllm() -> None:
    train_data = np.random.default_rng(0).standard_normal((64, 3)).astype(np.float32)
    config = {"spectrumllm": {"model": {k: v for k, v in _model_config()["model"].items()
                                       if k != "input_size"}}}
    model = build_model("spectrumllm", config, train_data)
    assert isinstance(model, SpectrumLLMForecaster)
    assert model(torch.rand(2, 16, 3)).shape == (2, 4, 3)


def test_build_model_rejects_non_2d_data() -> None:
    config = {"spectrumllm": {"model": {k: v for k, v in _model_config()["model"].items()
                                       if k != "input_size"}}}
    with pytest.raises(ValueError):
        build_model("spectrumllm", config, np.zeros((4, 3, 2, 2), dtype=np.float32))
