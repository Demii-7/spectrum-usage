import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.TemporalConvNet import CausalConv1d, TemporalConvNetForecaster
from training.common.model_factory import build_model


def _config() -> dict:
    return {
        "temporalconvnet": {
            "model": {
                "input_sequence_length": 8,
                "prediction_horizon": 1,
                "hidden_channels": [4, 4, 4],
                "kernel_size": 2,
                "dropout": 0.0,
                "leveled_init": True,
            }
        }
    }


def test_temporal_conv_net_shape_and_gradients() -> None:
    model = TemporalConvNetForecaster(
        {"model": {**_config()["temporalconvnet"]["model"], "input_size": 3}}
    )
    x = torch.rand(2, 8, 3, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 1, 3)
    output.sum().backward()
    assert x.grad is not None


def test_joint_temporal_conv_net_uses_other_features() -> None:
    model = TemporalConvNetForecaster(
        {
            "model": {
                **_config()["temporalconvnet"]["model"],
                "input_size": 3,
                "feature_mode": "joint",
                "leveled_init": False,
            }
        }
    )
    x = torch.rand(2, 8, 3, requires_grad=True)
    target = model(x)[:, :, 0].sum()
    target.backward()

    assert model(x).shape == (2, 1, 3)
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, :, 1:]) > 0


def test_independent_temporal_conv_net_does_not_use_other_features() -> None:
    model = TemporalConvNetForecaster(
        {"model": {**_config()["temporalconvnet"]["model"], "input_size": 3}}
    )
    x = torch.rand(2, 8, 3, requires_grad=True)
    model(x)[:, :, 0].sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, :, 1:]) == 0


def test_causal_convolution_prefix_does_not_depend_on_future() -> None:
    layer = CausalConv1d(1, 2, kernel_size=3, dilation=2)
    original = torch.randn(1, 1, 10)
    changed = original.clone()
    changed[:, :, 6:] += 100
    torch.testing.assert_close(layer(original)[:, :, :6], layer(changed)[:, :, :6])


def test_temporal_conv_net_factory_registration() -> None:
    model = build_model("temporalconvnet", _config(), np.zeros((20, 3), dtype=np.float32))
    assert isinstance(model, TemporalConvNetForecaster)
