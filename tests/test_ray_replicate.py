import unittest

from training.ray.replicate import _is_test_split, _source_validation_metrics


class RayReplicationTests(unittest.TestCase):
    def test_loader_test_split_is_selected(self):
        self.assertTrue(_is_test_split("2d_test"))
        self.assertTrue(_is_test_split("test"))
        self.assertFalse(_is_test_split("2d_validation"))

    def test_source_validation_metadata_keeps_strings_and_numeric_values(self):
        metrics = _source_validation_metrics({
            "objective": 1.2,
            "objective_name": "val_mean_horizon_mae_db",
            "val_loss": 0.3,
            "forecast_epoch": 4,
            "completed": True,
        })
        self.assertEqual(metrics["objective_name"], "val_mean_horizon_mae_db")
        self.assertEqual(metrics["objective"], 1.2)
        self.assertEqual(metrics["forecast_epoch"], 4.0)
        self.assertNotIn("completed", metrics)


if __name__ == "__main__":
    unittest.main()
