import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.LSTMAttn import LSTMAttnForecaster
from training.common.model_factory import build_model


def _config() -> dict:
    return {
        "lstmattn": {
            "model": {
                "input_sequence_length": 6,
                "prediction_horizon": 4,
                "hidden_size": 8,
                "attention_size": 5,
                "num_layers": 1,
                "dropout": 0.0,
            }
        }
    }


def test_lstm_attention_shape_weights_and_gradients() -> None:
    model = LSTMAttnForecaster(
        {"model": {**_config()["lstmattn"]["model"], "input_size": 3}}
    )
    output = model(torch.randn(2, 6, 3))
    assert output.shape == (2, 4, 3)
    assert model.last_attention_weights is not None
    assert model.last_attention_weights.shape == (2, 4, 6)
    torch.testing.assert_close(
        model.last_attention_weights.sum(dim=-1), torch.ones(2, 4)
    )
    output.sum().backward()
    assert model.attention.score.weight.grad is not None


def test_lstm_attention_factory_registration() -> None:
    model = build_model("lstmattn", _config(), np.zeros((20, 3), dtype=np.float32))
    assert isinstance(model, LSTMAttnForecaster)
