import pytest

torch = pytest.importorskip("torch")

from models.DLinear import DLinearForecaster
from models.NLinear import NLinearForecaster
from models.PatchTST import PatchTSTForecaster
from models.RevINMLP import RevINMLPForecaster
from models.ResidualNonlinearAutoregressive import ResidualNonlinearAutoregressiveForecaster

BASE = {"input_sequence_length": 96, "prediction_horizon": 12, "input_size": 7}


def test_shapes():
    x = torch.randn(3, 96, 7)
    assert PatchTSTForecaster({"model": {**BASE}})(x).shape == (3, 12, 7)
    assert PatchTSTForecaster({"model": {**BASE, "patch_length": 8, "patch_stride": 4, "d_model": 64, "n_heads": 8, "e_layers": 1}})(x).shape == (3, 12, 7)
    assert DLinearForecaster({"model": {**BASE}})(x).shape == (3, 12, 7)
    assert ResidualNonlinearAutoregressiveForecaster({"model": {**BASE}})(x).shape == (3, 12, 7)


def test_patchtst_is_level_equivariant_and_channel_independent():
    model = PatchTSTForecaster({"model": {**BASE, "dropout": 0.0}}).eval()
    x = torch.randn(2, 96, 7)
    out = model(x)
    assert torch.allclose(model(x + 5.0), out + 5.0, atol=1e-3)                  # instance normalization removes the level
    assert torch.allclose(model(x[:, :, :3])[:, :, 1], out[:, :, 1], atol=1e-5)    # a channel's forecast ignores the others


def test_nonlinear_head_starts_as_the_linear_model():
    config = {"model": {**BASE, "ridge_alpha": 0.01}}
    model = ResidualNonlinearAutoregressiveForecaster(config).eval()
    x = torch.randn(4, 96, 7)
    assert torch.allclose(model(x), model.linear(x))
    assert model.ridge_penalty().ndim == 0


def test_new_models_shapes_level_behaviour_and_patchtst_ablations():
    x = torch.randn(3, 96, 7)
    nl = NLinearForecaster({"model": {**BASE}}).eval()
    assert nl(x).shape == (3, 12, 7) and torch.allclose(nl(x), x[:, -1:, :].expand(-1, 12, -1))          # starts as persistence
    assert torch.allclose(nl(x + 4.0), nl(x) + 4.0, atol=1e-5)
    mlp = RevINMLPForecaster({"model": {**BASE, "mlp_dropout": 0.0}}).eval()
    assert mlp(x).shape == (3, 12, 7) and torch.allclose(mlp(x + 5.0), mlp(x) + 5.0, atol=1e-3)           # instance norm removes the level
    assert torch.allclose(mlp(x[:, :, :3])[:, :, 1], mlp(x)[:, :, 1], atol=1e-5)                          # channel independent
    for extra in ({"instance_norm": False}, {"channel_mode": "joint"}, {"channel_mode": "joint", "instance_norm": False},
                  {"patch_length": 1, "patch_stride": 1}):
        model = PatchTSTForecaster({"model": {**BASE, **extra, "dropout": 0.0}}).eval()
        assert model(x).shape == (3, 12, 7)
    joint = PatchTSTForecaster({"model": {**BASE, "channel_mode": "joint", "dropout": 0.0}}).eval()
    assert not torch.allclose(joint(x)[:, :, 0], joint(torch.cat([x[:, :, :1], torch.randn(3, 96, 6)], dim=2))[:, :, 0], atol=1e-4)  # joint mixes channels
    plain = PatchTSTForecaster({"model": {**BASE, "dropout": 0.0}}).eval()
    assert torch.allclose(plain(x)[:, :, 0], plain(torch.cat([x[:, :, :1], torch.randn(3, 96, 6)], dim=2))[:, :, 0], atol=1e-4)      # independent does not
