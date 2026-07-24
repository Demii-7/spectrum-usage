import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    torch = None


@unittest.skipIf(torch is None, "torch is not available")
class ResidualVanillaLSTMTests(unittest.TestCase):
    def setUp(self) -> None:
        from models.ResidualVanillaLSTM import ResidualVanillaLSTMForecaster

        self.model = ResidualVanillaLSTMForecaster(
            {
                "model": {
                    "input_size": 4,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "bidirectional": False,
                    "output_strategy": "final_hidden",
                    "input_sequence_length": 6,
                    "prediction_horizon": 60,
                    "dropout": 0.0,
                }
            }
        ).cpu()
        self.x = torch.randn(3, 6, 4)

    def test_direct_output_matches_vanilla_lstm_shape(self) -> None:
        self.assertEqual(self.model(self.x).shape, (3, 60, 4))

    def test_zero_initialized_head_repeats_current_lookback_mean(self) -> None:
        expected = self.x.mean(dim=1, keepdim=True).expand(-1, 60, -1)

        torch.testing.assert_close(self.model(self.x), expected, rtol=0.0, atol=0.0)

    def test_residual_head_receives_gradients_and_can_train(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        target = self.x.mean(dim=1, keepdim=True).expand(-1, 60, -1) + 1.0
        initial = self.model(self.x).detach().clone()

        loss = torch.nn.functional.mse_loss(self.model(self.x), target)
        loss.backward()

        self.assertGreater(self.model.residual.output_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.model.residual.output_head.bias.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(self.model(self.x).detach(), initial))


if __name__ == "__main__":
    unittest.main()
