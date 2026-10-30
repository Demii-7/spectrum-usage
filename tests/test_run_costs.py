import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from training.common.run_costs import (
    collect_run_costs,
    parameter_counts,
    read_run_costs,
    write_run_costs,
)
from training.ray.trainable import TuneReporter


class TinyForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)

    def forward(self, x):
        return self.linear(x[:, -1:, :])


class RunCostsTest(unittest.TestCase):
    def test_parameter_counts_split_trainable(self):
        counts = parameter_counts(TinyForecaster())
        self.assertEqual(counts["cost_parameters_total"], 20 + 6)
        self.assertEqual(counts["cost_parameters_trainable"], 20)

    def test_collect_records_cost_fields_and_restores_mode(self):
        model = TinyForecaster()
        model.train()
        costs = collect_run_costs(
            model, torch.zeros(5, 6, 4), 1, 3, torch.device("cpu"),
            model_config={"hidden": 4}, train_config={"lr": 0.1},
        )
        self.assertNotIn("cost_error", costs)
        for key in ("cost_parameters_total", "cost_gpu_name", "cost_torch_version",
                    "cost_inference_ms_per_batch", "cost_inference_ms_per_sample",
                    "cost_model_config_json", "cost_train_config_json"):
            self.assertIn(key, costs)
        self.assertEqual(costs["cost_inference_batch_size"], 5)
        self.assertTrue(model.training)

    def test_failures_are_recorded_not_raised(self):
        costs = collect_run_costs(
            TinyForecaster(), torch.zeros(2, 6, 99), 1, 3, torch.device("cpu"), {}, {},
        )
        self.assertIn("cost_error", costs)
        self.assertIn("cost_parameters_total", costs)

    def test_costs_round_trip_and_reach_the_ray_report(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run_costs(Path(directory), "powder_600_800", {"cost_parameters_total": 7})
            costs = read_run_costs(Path(directory))
            self.assertEqual(read_run_costs(Path(directory) / "missing"), {})
        reports = []
        reporter = TuneReporter(report=lambda metrics, checkpoint=None: reports.append(metrics))
        reporter({"prunable": True, "epoch": 1, "metrics": {"val_mean_horizon_mae_db": 1.0}})
        reporter.run_costs = costs
        reporter.complete("checkpoint")
        self.assertEqual(reports[-1]["cost_parameters_total"], 7)
        self.assertEqual(reporter.selected_metrics["cost_parameters_total"], 7)


if __name__ == "__main__":
    unittest.main()
