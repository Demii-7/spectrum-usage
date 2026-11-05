import pytest

torch = pytest.importorskip("torch")

from models.LinearAutoregressive import LinearAutoregressiveForecaster  # noqa: E402
from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster  # noqa: E402


def _cfg(horizon, features=4, lags=12, alpha=0.5):
    return {"model": {"input_sequence_length": lags, "prediction_horizon": horizon, "input_size": features,
                      "ridge_alpha": alpha}}


def test_one_step_model_keeps_its_original_parameter_shapes():
    m = LinearAutoregressiveForecaster(_cfg(1))
    assert tuple(m.weight.shape) == (4, 12) and tuple(m.bias.shape) == (4,)
    assert m(torch.randn(3, 12, 4)).shape == (3, 1, 4)


def test_direct_model_has_one_weight_vector_per_feature_and_output_step():
    m = LinearAutoregressiveForecaster(_cfg(5))
    assert tuple(m.weight.shape) == (4, 12, 5) and tuple(m.bias.shape) == (4, 5)
    x = torch.randn(3, 12, 4)
    out = m(x)
    assert out.shape == (3, 5, 4)
    manual = torch.stack([(x * m.weight[:, :, h].T.unsqueeze(0)).sum(dim=1) + m.bias[:, h] for h in range(5)], dim=1)
    assert torch.allclose(out, manual, atol=1e-5)


def test_ridge_penalty_has_the_same_scale_for_one_step_and_direct_models():
    one = LinearAutoregressiveForecaster(_cfg(1))
    direct = LinearAutoregressiveForecaster(_cfg(3))
    with torch.no_grad():
        direct.weight.copy_(one.weight.unsqueeze(-1).repeat(1, 1, 3))
    assert torch.allclose(one.ridge_penalty(), direct.ridge_penalty(), atol=1e-6)


def test_residual_direct_model_starts_as_the_lookback_mean_for_every_step():
    m = ResidualLinearAutoregressiveForecaster(_cfg(6))
    x = torch.randn(2, 12, 4)
    out = m(x)
    assert out.shape == (2, 6, 4)
    assert torch.allclose(out, x.mean(dim=1, keepdim=True).expand(2, 6, 4), atol=1e-6)


def test_direct_model_handles_map_inputs():
    cfg = {"model": {"input_sequence_length": 8, "prediction_horizon": 3, "input_size": 2 * 3 * 3, "ridge_alpha": 0.0}}
    m = ResidualLinearAutoregressiveForecaster(cfg)
    assert m(torch.randn(2, 8, 2, 3, 3)).shape == (2, 3, 2, 3, 3)


def test_residual_direct_model_learns_a_persistent_series_better_than_the_lookback_mean():
    torch.manual_seed(0)
    n, lags, horizon, features = 600, 12, 4, 2
    x = torch.zeros(n, features)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + torch.randn(features)
    starts = range(0, n - lags - horizon)
    xs = torch.stack([x[s:s + lags] for s in starts])
    ys = torch.stack([x[s + lags:s + lags + horizon] for s in starts])
    m = ResidualLinearAutoregressiveForecaster(_cfg(horizon, features, lags, alpha=0.0))
    baseline = ((xs.mean(dim=1, keepdim=True) - ys) ** 2).mean().item()
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        loss = ((m(xs) - ys) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 0.6 * baseline
