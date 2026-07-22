import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).parents[1] / "training" / "common" / "validation_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("validation_diagnostics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ConvLSTMValidationGuardTests(unittest.TestCase):
    def test_diagnostics_are_named_by_one_based_horizon(self) -> None:
        self.assertEqual(
            MODULE.prediction_diagnostic_log([1.5, 2.25]),
            {"val_max_abs_prediction_t1": 1.5, "val_max_abs_prediction_t2": 2.25},
        )

    def test_guard_rejects_nonfinite_and_excessive_predictions(self) -> None:
        self.assertEqual(MODULE.prediction_guard_status([float("nan")], 20.0), (False, "nonfinite_predictions"))
        self.assertEqual(
            MODULE.prediction_guard_status([1.0, 20.01], 20.0),
            (False, "prediction_magnitude_exceeded"),
        )
        self.assertEqual(MODULE.prediction_guard_status([20.0], 20.0), (True, "valid"))

    def test_threshold_is_required_and_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires validation_prediction"):
            MODULE.prediction_guard_threshold({})
        with self.assertRaisesRegex(ValueError, "finite and > 0"):
            MODULE.prediction_guard_threshold({"validation_prediction_magnitude_threshold": 0})


if __name__ == "__main__":
    unittest.main()
