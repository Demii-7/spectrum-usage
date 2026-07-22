import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


MODEL_PATH = Path(__file__).parents[1] / "training" / "Autoformer-CSA" / "model.py"


def _load_model_module():
    spec = importlib.util.spec_from_file_location("autoformer_csa_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not available")
class AutoformerCSATests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_model_module()
        self.config = self.module.DotConfig(
            seq_len=12,
            label_len=6,
            pred_len=5,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            e_layers=2,
            d_layers=1,
            n_heads=4,
            moving_avg=5,
            dropout=0.0,
            factor=2,
            output_attention=False,
            csam_kernel_size=3,
        )

    def test_direct_forecast_shape(self) -> None:
        model = self.module.AutoformerCSAForecaster(self.config)
        output = model(torch.randn(2, 12, 7))
        self.assertEqual(tuple(output.shape), (2, 5, 7))

    def test_shared_forecast_contract(self) -> None:
        from training.common.forecasting import forecast

        model = self.module.AutoformerCSAForecaster(self.config)
        inputs = torch.randn(2, 12, 7)
        output = forecast(
            model=model,
            x=inputs,
            prediction_horizon=5,
            rollout_horizon=5,
        )
        self.assertEqual(tuple(output.shape), (2, 5, 7))

    def test_csam_preserves_embedded_output_width(self) -> None:
        csam = self.module.CSAM(d_model=16, d_ff=32, kernel_size=3, dropout=0.0)
        output = csam(torch.randn(2, 16, 12))
        self.assertEqual(tuple(output.shape), (2, 16, 12))

    def test_model_has_no_external_autoformer_requirement(self) -> None:
        model = self.module.AutoformerCSAForecaster(self.config)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
