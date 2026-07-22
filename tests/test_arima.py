import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.ARIMA import ARIMAForecaster
from training.common.model_factory import build_model


def _config() -> dict:
    return {
        "arima": {
            "model": {
                "input_sequence_length": 12,
                "prediction_horizon": 3,
                "p": 1,
                "d": 1,
                "q": 0,
                "ridge": 1e-8,
            }
        }
    }


def test_arima_extrapolates_linear_series() -> None:
    model = ARIMAForecaster({"model": {**_config()["arima"]["model"], "input_size": 2}})
    time = torch.arange(12, dtype=torch.float32)
    x = torch.stack((time, 2 * time), dim=-1).unsqueeze(0)
    expected = torch.tensor([[[12.0, 24.0], [13.0, 26.0], [14.0, 28.0]]])
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
    assert list(model.parameters()) == []


def test_arima_rejects_unsupported_moving_average_order() -> None:
    config = {"model": {**_config()["arima"]["model"], "input_size": 1, "q": 1}}
    with pytest.raises(ValueError, match="q=0"):
        ARIMAForecaster(config)


def test_arima_factory_registration() -> None:
    model = build_model("arima", _config(), np.zeros((20, 2), dtype=np.float32))
    assert isinstance(model, ARIMAForecaster)
