import pytest

torch = pytest.importorskip("torch")

from models.DLinear import DLinearForecaster
from models.PatchTST import PatchTSTForecaster
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
