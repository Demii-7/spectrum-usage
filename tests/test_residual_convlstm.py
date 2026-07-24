import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    torch = None


def _model_config(prediction_horizon: int = 60) -> dict:
    return {
        "model": {
            "input_channels": 2,
            "grid_height": 2,
            "grid_width": 3,
            "input_sequence_length": 4,
            "prediction_horizon": prediction_horizon,
            "hidden_channels": [2],
            "kernel_size": [[1, 1]],
            "num_encoder_layers": 1,
            "decoder_hidden_channels": 2,
            "decoder_kernel_size": [1, 1],
            "decoder_lstm_hidden": 3,
            "dropout": 0.0,
            "use_batch_norm": False,
            "fc_hidden_channels": 0,
            "cell_activation": "tanh",
            "use_channel_projection": False,
        }
    }


@unittest.skipIf(torch is None, "torch is not available")
class ResidualConvLSTMTests(unittest.TestCase):
    def setUp(self) -> None:
        from models.ResidualConvLSTM import ResidualConvLSTMForecaster

        self.forecaster_type = ResidualConvLSTMForecaster
        self.model = ResidualConvLSTMForecaster(_model_config()).cpu()
        self.x = torch.randn(2, 4, 2, 2, 3)

    def test_direct_output_has_full_horizon_shape(self) -> None:
        self.assertEqual(self.model(self.x).shape, (2, 60, 2, 2, 3))

    def test_zero_initialized_head_exactly_repeats_lookback_mean(self) -> None:
        expected = self.x.mean(dim=1, keepdim=True).expand(-1, 60, -1, -1, -1)
        torch.testing.assert_close(self.model(self.x), expected, rtol=0.0, atol=0.0)

    def test_residual_head_receives_gradients_and_trains(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        target = self.x.mean(dim=1, keepdim=True).expand(-1, 60, -1, -1, -1) + 1.0
        initial = self.model(self.x).detach().clone()

        torch.nn.functional.mse_loss(self.model(self.x), target).backward()

        self.assertGreater(self.model.residual.output_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.model.residual.output_head.bias.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(self.model(self.x).detach(), initial))

    def test_model_factory_constructs_direct_forecaster(self) -> None:
        from training.common.model_factory import build_model

        config = {"residualconvlstm": _model_config()}
        model = build_model(
            "residualconvlstm",
            config,
            np.zeros((8, 2, 3, 2), dtype=np.float32),
        )

        self.assertIsInstance(model, self.forecaster_type)
        self.assertEqual(model.residual.prediction_horizon, 60)
        self.assertEqual(model(torch.randn(1, 4, 2, 2, 3)).shape, (1, 60, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
