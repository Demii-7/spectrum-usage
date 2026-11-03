import json
import unittest
from pathlib import Path

import yaml

from training.ray.final_eval import (
    build_eval_config, config_file_for, parse_checkpoint_key, summarize_metrics,
)
from training.ray.fixed_train import band_limits, pick_best_prefix, swap_band

COSTS = {"cost_model_config_json": json.dumps({"hidden_size": 64, "num_layers": 1, "input_sequence_length": 60,
                                              "prediction_horizon": 1}),
         "cost_train_config_json": json.dumps({"batch_size": 32, "epochs": 20, "seed": 7, "test_stride": 1})}


def _base(model="vanillalstm", rep="2d"):
    return yaml.safe_load(Path(config_file_for(model, rep)).read_text())


class CheckpointKeyTests(unittest.TestCase):
    def test_parses_exported_keys(self):
        entry = parse_checkpoint_key("checkpoints-oct2026-r2/2d/vanillalstm/seed-42/powder_600_800_t4t6_vanillalstm.pt")
        self.assertEqual((entry["prefix"], entry["rep"], entry["model"], entry["seed"]),
                         ("checkpoints-oct2026-r2", "2d", "vanillalstm", 42))

    def test_ignores_non_checkpoints(self):
        self.assertIsNone(parse_checkpoint_key("checkpoints-oct2026/2d/vanillalstm/MANIFEST.json"))
        self.assertIsNone(parse_checkpoint_key("checkpoints-oct2026/2d/vanillalstm/seed-42/costs.json"))


class BuildEvalConfigTests(unittest.TestCase):
    def test_rebuilds_the_checkpoint_architecture_and_range(self):
        config = build_eval_config(_base(), "vanillalstm", COSTS, test_start="2026-10-27T20:00:00Z",
                                   test_end="2026-11-03T20:00:00Z", test_stride=1)
        self.assertEqual(config["vanillalstm"]["model"]["hidden_size"], 64)
        self.assertEqual(config["vanillalstm"]["train"]["test_stride"], 1)
        self.assertEqual(config["data"]["split"]["ranges"]["test"]["end"], "2026-11-03T20:00:00Z")
        self.assertEqual(config["training"]["models"], ["vanillalstm"])

    def test_data_dir_swap_only_changes_paths(self):
        config = build_eval_config(_base(), "vanillalstm", COSTS, test_start="a", test_end="b",
                                   data_dir_swap=("data/powder/", "data/powder_final/"))
        self.assertTrue(all(f["path"].startswith("data/powder_final/") for f in config["data"]["files"]))

    def test_does_not_mutate_the_base_config(self):
        base = _base()
        before = base["vanillalstm"]["model"]["hidden_size"]
        build_eval_config(base, "vanillalstm", COSTS, test_start="a", test_end="b")
        self.assertEqual(base["vanillalstm"]["model"]["hidden_size"], before)
        self.assertNotEqual(before, 64)


class SummaryTests(unittest.TestCase):
    def test_means_per_split_and_horizon(self):
        rows = [{"split": "2d_test", "horizon": 1, "mae_db": 1.0}, {"split": "2d_test", "horizon": 1, "mae_db": 3.0},
                {"split": "2d_test", "horizon": 15, "mae_db": 2.0}, {"split": "2d_validation", "horizon": 1, "mae_db": 5.0}]
        summary = summarize_metrics(rows)
        self.assertEqual(summary["test_mae_db_t1"], 2.0)
        self.assertEqual(summary["test_mean_horizon_mae_db"], 2.0)
        self.assertEqual(summary["validation_mean_horizon_mae_db"], 5.0)


class SwapBandTests(unittest.TestCase):
    def test_every_core_config_swaps_cleanly(self):
        for model, rep, _ in __import__("training.ray.campaign_all", fromlist=["PLAN"]).PLAN:
            config = swap_band(_base(model, rep), "2400_2600")
            self.assertTrue(all("/2400_2600/" in f["path"] and "/600_800/" not in f["path"]
                                for f in config["data"]["files"]), (model, rep))
            chunk = config["data"]["chunks"][0]
            self.assertEqual((chunk["start_mhz"], chunk["end_mhz"]), (2400.0, 2600.0))
            if config["data"].get("map"):
                self.assertEqual(config["data"]["map"]["name"], "oct2026_core4_2400_2600")

    def test_splits_and_nodes_are_unchanged(self):
        before, after = _base("convlstm", "4d"), swap_band(_base("convlstm", "4d"), "2400_2600")
        self.assertEqual(before["data"]["split"], after["data"]["split"])
        self.assertEqual([f["partition"] for f in before["data"]["files"]],
                         [f["partition"] for f in after["data"]["files"]])

    def test_band_limits_and_best_prefix(self):
        self.assertEqual(band_limits("2400_2600"), (2400.0, 2600.0))
        self.assertEqual(pick_best_prefix({"r1": 1.07, "r2": 1.10}), "r1")
        self.assertEqual(pick_best_prefix({"r1": 1.07, "r2": 0.97}), "r2")


if __name__ == "__main__":
    unittest.main()
