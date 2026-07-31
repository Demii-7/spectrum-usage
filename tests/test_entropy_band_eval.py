import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.analysis.ray_entropy_band_eval import (
    error_metrics,
    load_regions,
    parse_args,
    summarize_forecasts,
)


ROOT = Path(__file__).resolve().parents[1]


class EntropyBandEvaluationTests(unittest.TestCase):
    def test_plan_regions_cover_the_full_band(self):
        regions = load_regions(ROOT / "evaluation/analysis/plan_regions_600_800.csv")
        self.assertEqual(len(regions), 17)
        self.assertEqual(int(regions["bin_count"].sum()), 200)
        self.assertEqual(float(regions.iloc[0]["start_mhz"]), 600.5)
        self.assertEqual(float(regions.iloc[-1]["end_mhz"]), 799.5)

    def test_error_metrics(self):
        values = error_metrics(np.array([[1.0, 4.0]]), np.array([[0.0, 2.0]]))
        self.assertEqual(values["n_values"], 2)
        self.assertAlmostEqual(values["mae_db"], 1.5)
        self.assertAlmostEqual(values["mse_db2"], 2.5)
        self.assertAlmostEqual(values["rmse_db"], np.sqrt(2.5))
        self.assertAlmostEqual(values["bias_db"], 1.5)

    def test_forecast_summary_is_wide_by_horizon(self):
        regions = pd.DataFrame([{
            "band_id": "R1",
            "start_mhz": 600.5,
            "end_mhz": 601.5,
            "bin_count": 2,
            "behavior_category": "mixed_activity",
            "is_noise_floor": False,
        }])
        entropy = {"R1": {
            **regions.iloc[0].to_dict(),
            "entropy": 0.75,
            "diff_E_actual_normalized": 0.75,
        }}
        predictions = {1: np.array([[1.0, 3.0]]), 15: np.array([[2.0, 4.0]])}
        targets = {1: np.array([[0.0, 1.0]]), 15: np.array([[0.0, 0.0]])}
        rows = summarize_forecasts(
            predictions,
            targets,
            np.array([600.5, 601.5]),
            regions,
            entropy,
            model="example",
            seed=40,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["t1_mae_db"], 1.5)
        self.assertAlmostEqual(rows[0]["t15_mae_db"], 3.0)
        self.assertAlmostEqual(rows[0]["mean_horizon_mae_db"], 2.25)
        self.assertEqual(rows[0]["seed"], 40)

    def test_gpu_request_cannot_exceed_half_a_gpu(self):
        self.assertEqual(parse_args([]).gpus_per_task, 0.5)
        with self.assertRaises(SystemExit):
            parse_args(["--gpus-per-task", "0.51"])


if __name__ == "__main__":
    unittest.main()
