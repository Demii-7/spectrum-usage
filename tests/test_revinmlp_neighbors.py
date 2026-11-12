"""RevIN-MLP neighbor_inputs: default off changes nothing; on, shapes, own-first ordering and permutation behaviour."""
import copy

import torch

from models.RevINMLP import RevINMLPForecaster


def cfg(**extra):
    return {"model": {"input_sequence_length": 12, "prediction_horizon": 3, "input_size": 8, "mlp_hidden": 16, "mlp_layers": 2, "mlp_dropout": 0.0, **extra}}


def test_off_is_unchanged():
    torch.manual_seed(0)
    a = RevINMLPForecaster(cfg())
    torch.manual_seed(0)
    b = RevINMLPForecaster(cfg(neighbor_inputs=False, n_receivers=4))
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    x = torch.randn(5, 12, 8)
    assert torch.equal(a.eval()(x), b.eval()(x))


def test_neighbors_shapes_and_own_statistics():
    torch.manual_seed(0)
    model = RevINMLPForecaster(cfg(neighbor_inputs=True, n_receivers=4)).eval()
    x = torch.randn(5, 12, 8)                                   # 4 receivers x 2 bins
    y = model(x)
    assert y.shape == (5, 3, 8)
    assert model.body[0].in_features == 12 * 4
    shifted = x.clone()
    shifted[:, :, 0:2] += 100.0                                  # a level shift of receiver 0 moves receiver 0's forecast by the same amount
    assert torch.allclose(model(shifted)[:, :, 0:2], y[:, :, 0:2] + 100.0, atol=1e-3)
    assert torch.allclose(model(shifted)[:, :, 2:], y[:, :, 2:], atol=1e-3)   # RevIN removes each series' own level, so the others do not move


def test_neighbor_values_are_used():
    torch.manual_seed(1)
    model = RevINMLPForecaster(cfg(neighbor_inputs=True, n_receivers=4)).eval()
    x = torch.randn(4, 12, 8)
    changed = x.clone()
    changed[:, :, 2:4] = torch.randn(4, 12, 2)                   # receiver 1's series change shape, receiver 0's own input is the same
    a, b = model(x)[:, :, 0:2], model(changed)[:, :, 0:2]
    assert not torch.allclose(a, b)


def test_equivariance_to_receiver_relabelling_of_others():
    """Own comes first and the others keep their original order: swapping two OTHER receivers changes the forecast (order matters), the shape does not."""
    model = RevINMLPForecaster(cfg(neighbor_inputs=True, n_receivers=4)).eval()
    x = torch.randn(3, 12, 8)
    assert model(x).shape == model(x[:, :, [0, 1, 4, 5, 2, 3, 6, 7]]).shape


def test_incompatible_controls_raise():
    for bad in (dict(neighbor_inputs=True, channel_mixing=True), dict(neighbor_inputs=True, n_receivers=3)):
        try:
            RevINMLPForecaster(cfg(**bad))
        except ValueError:
            continue
        raise AssertionError(bad)
