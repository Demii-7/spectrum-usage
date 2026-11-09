import pytest
import torch

from models.LookbackMean import LookbackMeanForecaster
from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster


def _config(**extra):
    return {"model": {"input_sequence_length": 6, "prediction_horizon": 1, "input_size": 2, **extra}}


def test_lookback_mean_defaults_unchanged_and_median_window():
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]]])
    assert LookbackMeanForecaster(_config())(x).item() == pytest.approx(x.mean().item())
    assert LookbackMeanForecaster(_config(statistic="median", window=5))(x).item() == 4.0     # newest five: 2, 3, 4, 5, 100
    assert LookbackMeanForecaster(_config(statistic="median", window=3))(x).item() == 5.0     # 4, 5, 100
    assert LookbackMeanForecaster(_config(statistic="median"))(x).item() == 3.0               # lower median of six values
    with pytest.raises(ValueError):
        LookbackMeanForecaster(_config(window=7))


def test_residual_median_starts_as_median_and_is_level_equivariant():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 2)
    model = ResidualLinearAutoregressiveForecaster(_config(baseline="median", baseline_window=4, ridge_alpha=0.0))
    out = model(x)                                              # head starts at zero: exactly the median of the newest four
    assert torch.allclose(out[:, 0], x[:, -4:].median(dim=1).values)
    with torch.no_grad():
        model.residual.weight.normal_()
    shifted = model(x + 7.0)
    assert torch.allclose(shifted, model(x) + 7.0, atol=1e-4)    # shifting the window shifts the forecast by the same amount


def test_default_residual_model_is_unchanged():
    x = torch.randn(3, 6, 2)
    model = ResidualLinearAutoregressiveForecaster(_config(ridge_alpha=0.0))
    assert torch.allclose(model(x)[:, 0], x.mean(dim=1))
