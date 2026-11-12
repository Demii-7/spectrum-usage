import pytest

torch = pytest.importorskip("torch")

from models.ITransformer import ITransformerForecaster
from models.RevINMLP import RevINMLPForecaster

BASE = {"input_sequence_length": 96, "prediction_horizon": 12, "input_size": 7}


def test_channel_mixing_off_leaves_model_unchanged():
    torch.manual_seed(0)
    plain = RevINMLPForecaster({"model": {**BASE}})
    torch.manual_seed(0)
    off = RevINMLPForecaster({"model": {**BASE, "channel_mixing": False}})
    assert list(plain.state_dict()) == list(off.state_dict())
    assert not any("mixer" in k for k in plain.state_dict())
    x = torch.randn(3, 96, 7)
    assert torch.equal(plain.eval()(x), off.eval()(x))


def test_channel_mixing_starts_identical_then_mixes():
    torch.manual_seed(0)
    plain = RevINMLPForecaster({"model": {**BASE}}).eval()
    torch.manual_seed(0)
    mix = RevINMLPForecaster({"model": {**BASE, "channel_mixing": True}}).eval()
    x = torch.randn(3, 96, 7)
    assert mix(x).shape == (3, 12, 7)
    assert torch.allclose(plain(x), mix(x), atol=1e-6)                       # zero-initialised residual
    loss = mix.train()(x).pow(2).mean()
    loss.backward()
    assert mix.mixer.weight.grad.abs().sum() > 0
    with torch.no_grad():
        mix.mixer.weight.normal_()
    mix.eval()
    assert not torch.allclose(plain(x), mix(x))


def test_itransformer_shapes_level_and_training_step():
    model = ITransformerForecaster({"model": {**BASE, "dropout": 0.0, "d_model": 32, "n_heads": 4, "d_ff": 64}})
    x = torch.randn(3, 96, 7)
    out = model.eval()(x)
    assert out.shape == (3, 12, 7)
    assert torch.allclose(model(x + 5.0), out + 5.0, atol=1e-3)                # normalization removes the level
    assert model(torch.randn(2, 96, 11)).shape == (2, 12, 11)                  # any number of channels
    model.train()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    loss = (model(x) - torch.randn(3, 12, 7)).pow(2).mean()
    loss.backward()
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    names = set(model.state_dict())
    assert {"enc_embedding.value_embedding.weight", "encoder.attn_layers.0.attention.query_projection.weight",
            "encoder.attn_layers.0.conv1.weight", "encoder.norm.weight", "projection.weight"} <= names
