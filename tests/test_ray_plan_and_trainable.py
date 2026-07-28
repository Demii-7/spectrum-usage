import inspect
from pathlib import Path
import tempfile
import unittest

from training.ray.run import (
    DEFAULT_CANDIDATE_BUDGET,
    DEFAULT_MAX_CONCURRENT_TRIALS,
    RERANK_SEEDS,
    SEARCH_SEED,
    anchor_points,
    build_plan,
    build_rerank_candidates,
    candidate_budget,
    ray_anchor_points,
)
from training.ray.trainable import TuneReporter, run_integrated_trial


def valid_config():
    return {
        "training": {"model_name": "vanillalstm"},
        "data": {"representation": "2d", "split": {"ranges": {
            "train": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
            "validation": {"start": "2024-01-03T00:00:00Z", "end": "2024-01-04T00:00:00Z"},
            "test": {"start": "2024-01-05T00:00:00Z", "end": "2024-01-06T00:00:00Z"},
        }}},
        "vanillalstm": {"model": {}, "train": {"epochs": 2}},
    }


class RayPlanAndTrainableTests(unittest.TestCase):
    def test_default_budget_is_exactly_three_anchors_plus_nine_random(self):
        self.assertEqual(DEFAULT_CANDIDATE_BUDGET, 12)
        self.assertEqual(candidate_budget(), {"total": 12, "anchors": 3, "random": 9})
        plan = build_plan(valid_config(), ["vanillalstm"])
        entry = plan["models"][0]
        self.assertEqual(entry["search"]["seed"], SEARCH_SEED)
        self.assertEqual(entry["search"]["candidate_budget"]["total"], 12)
        self.assertEqual(entry["search"]["candidate_budget"]["historical"], 1)
        self.assertEqual(entry["search"]["candidate_budget"]["random"], 8)
        self.assertEqual(entry["search"]["max_concurrent_trials"],
                         DEFAULT_MAX_CONCURRENT_TRIALS)
        self.assertEqual(entry["search"]["grace_period"], 5)
        self.assertEqual(len(anchor_points(entry)), 3)
        self.assertEqual({point["capacity"] for point in anchor_points(entry)},
                         {"tiny", "small", "reference"})
        self.assertEqual({point["seed"] for point in anchor_points(entry)}, {42})

    def test_temporal_conv_net_gets_longer_asha_grace_period(self):
        config = valid_config()
        config["training"]["model_name"] = "temporalconvnet"
        config["temporalconvnet"] = {"model": {}, "train": {"epochs": 20}}
        entry = build_plan(config, ["temporalconvnet"])["models"][0]
        self.assertEqual(entry["search"]["grace_period"], 12)

    def test_rerank_is_separate_and_uses_three_seeds(self):
        entry = build_plan(valid_config(), ["vanillalstm"])["models"][0]
        self.assertEqual(tuple(entry["rerank"]["seeds"]), RERANK_SEEDS)
        self.assertEqual(entry["rerank"]["total_runs"], 9)
        self.assertNotIn("seeds", entry["search"])
        selections = {
            "best": {"architecture": {"model.hidden_size": 32}},
            "best-simple": {"architecture": {"model.hidden_size": 8}},
            "reference": {"architecture": {"model.hidden_size": 128}},
        }
        runs = build_rerank_candidates("vanillalstm", selections)
        self.assertEqual(len(runs), 9)
        self.assertEqual({run["seed"] for run in runs}, {41, 42, 43})
        self.assertEqual({run["selection"] for run in runs}, set(selections))

    def test_ray_anchor_tokens_preserve_coupled_architecture_bundles(self):
        entry = build_plan(valid_config(), ["vanillalstm"])["models"][0]
        points, architectures = ray_anchor_points(entry)
        self.assertEqual([point["architecture"] for point in points],
                         ["tiny", "small", "reference",
                          "historical:summary70_residual_family"])
        self.assertEqual(
            architectures["reference"],
            entry["anchors"]["reference"]["architecture"],
        )
        historical = entry["historical_candidates"]["summary70_residual_family"]
        self.assertEqual(
            architectures["historical:summary70_residual_family"],
            historical["architecture"],
        )

    def test_asha_trial_calls_trainer_in_process_and_attaches_checkpoint(self):
        reports = []
        calls = []

        def trainer(config, model_name, run_directory, callback=None):
            calls.append((config, model_name, run_directory))
            checkpoints = run_directory / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "chunk_vanillalstm.pt").write_bytes(b"integrated")
            callback({"stage": "pretrain", "epoch": 1, "prunable": False,
                      "metrics": {"loss": 1.0}})
            callback({"stage": "train", "epoch": 1, "prunable": True,
                      "metrics": {"val_loss": 0.4}})

        reporter = TuneReporter(
            lambda metrics, checkpoint=None: reports.append((metrics, checkpoint))
        )
        parameters = {
            "architecture": {"model.hidden_size": 8, "model.num_layers": 1,
                             "model.dropout": 0.0},
            "train.learning_rate": 0.001,
            "train.weight_decay": 0.0,
            "train.batch_size": 32,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_integrated_trial(
                valid_config(), "vanillalstm", parameters, 42, Path(directory),
                trainer=trainer, reporter=reporter,
                checkpoint_factory=lambda path: ("checkpoint", path),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(reporter.events), 2)
        self.assertEqual(len(reports), 2)  # Forecast epoch plus completion checkpoint.
        self.assertIsNone(reports[0][1])
        self.assertEqual(reports[1][1][0], "checkpoint")
        self.assertEqual(result["prunable_epochs_reported"], 1)
        self.assertTrue(result["event_log"].endswith("training_events.jsonl"))

    def test_completion_attaches_checkpoint_to_best_epoch_metrics(self):
        reports = []
        reporter = TuneReporter(
            lambda metrics, checkpoint=None: reports.append((metrics, checkpoint))
        )
        reporter({"stage": "train", "epoch": 1, "prunable": True, "is_best": True,
                  "metrics": {"val_loss": 0.3, "val_mean_horizon_mae_db": 1.0}})
        reporter({"stage": "train", "epoch": 2, "prunable": True, "is_best": False,
                  "metrics": {"val_loss": 0.5, "val_mean_horizon_mae_db": 1.4}})
        reporter.complete("best-checkpoint")

        final, checkpoint = reports[-1]
        self.assertEqual(checkpoint, "best-checkpoint")
        self.assertEqual(final["objective"], 1.0)
        self.assertEqual(final["val_loss"], 0.3)
        self.assertEqual(final["forecast_epoch"], 1)
        self.assertEqual(final["final_objective"], 1.4)
        self.assertEqual(final["final_val_loss"], 0.5)
        self.assertEqual(final["final_forecast_epoch"], 2)
        self.assertEqual(final["training_iteration"], 3)

    def test_guard_ineligible_epochs_are_logged_but_not_reported(self):
        reports = []
        reporter = TuneReporter(
            lambda metrics, checkpoint=None: reports.append((metrics, checkpoint))
        )
        reporter({"stage": "train", "epoch": 1, "prunable": True,
                  "selection_eligible": False,
                  "metrics": {"val_loss": 0.1, "val_mean_horizon_mae_db": 0.2}})
        reporter({"stage": "train", "epoch": 2, "prunable": True,
                  "selection_eligible": True, "is_best": True,
                  "metrics": {"val_loss": 0.3, "val_mean_horizon_mae_db": 1.0}})

        self.assertEqual(len(reporter.events), 2)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reporter.report_count, 1)
        self.assertEqual(reporter.selected_metrics["forecast_epoch"], 2)

    def test_asha_path_contains_no_subprocess_call(self):
        source = inspect.getsource(run_integrated_trial)
        self.assertNotIn("subprocess", source)
        self.assertIn("trainer(config, model_name, run_directory, callback=callback)", source)


if __name__ == "__main__":
    unittest.main()
