import json
import unittest
from pathlib import Path

import yaml

from training.ray.final_eval import (
    build_eval_config, config_file_for, parse_checkpoint_key, restrict_files_to_site, summarize_metrics,
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


class PerSiteTests(unittest.TestCase):
    def test_per_receiver_config_keeps_one_receiver_in_every_partition(self):
        config = restrict_files_to_site(_base(), "moran")
        self.assertEqual(sorted({f["partition"] for f in config["data"]["files"]}), ["test", "train", "validation"])
        self.assertTrue(all("/moran-nuc1/" in f["path"] for f in config["data"]["files"]))
        self.assertEqual(len(config["data"]["files"]), 3)

    def test_pooled_scoring_restricts_only_the_test_partition(self):
        config = restrict_files_to_site(_base(), "madsen", ("test",))
        by = {p: [f["path"] for f in config["data"]["files"] if f["partition"] == p] for p in ("train", "validation", "test")}
        self.assertEqual(len(by["train"]), 4)       # normalization still fitted on all four receivers
        self.assertEqual(len(by["validation"]), 4)
        self.assertEqual(len(by["test"]), 1)
        self.assertIn("/madsen-nuc1/", by["test"][0])

    def test_unknown_receiver_is_rejected_and_base_is_untouched(self):
        base = _base()
        n = len(base["data"]["files"])
        with self.assertRaises(ValueError):
            restrict_files_to_site(base, "web")
        self.assertEqual(len(base["data"]["files"]), n)


if __name__ == "__main__":
    unittest.main()


def test_ensure_site_files_adds_held_out_receiver():
    from training.ray.final_eval import ensure_site_files, restrict_files_to_site

    config = {"data": {"files": [
        {"path": "data/powder/humanities-nuc1/RUN1/600_800/p.csv", "partition": "train"},
        {"path": "data/powder/madsen-nuc1/RUN1/600_800/p.csv", "partition": "train"},
        {"path": "data/powder/humanities-nuc1/RUN2/600_800/p.csv", "partition": "validation"},
        {"path": "data/powder/humanities-nuc1/RUN2/600_800/p.csv", "partition": "test"},
    ]}}
    extended = ensure_site_files(config, "law73")
    law = restrict_files_to_site(extended, "law73")["data"]["files"]
    assert [(f["path"], f["partition"]) for f in law] == [
        ("data/powder/law73-nuc1/RUN1/600_800/p.csv", "train"),
        ("data/powder/law73-nuc1/RUN2/600_800/p.csv", "validation"),
        ("data/powder/law73-nuc1/RUN2/600_800/p.csv", "test")]
    assert len(config["data"]["files"]) == 4  # input untouched
    assert ensure_site_files(config, "madsen") == config  # already listed: no change


def test_pool_sites_keeps_exactly_the_requested_receivers():
    from training.ray.final_eval import pool_sites

    config = {"data": {"files": [
        {"path": "data/powder/humanities-nuc1/R1/b/p.csv", "partition": "train"},
        {"path": "data/powder/madsen-nuc1/R1/b/p.csv", "partition": "train"},
        {"path": "data/powder/humanities-nuc1/R2/b/p.csv", "partition": "test"},
    ]}}
    pooled = pool_sites(config, ["humanities", "law73"])["data"]["files"]
    assert sorted(f["path"].split("/")[2] for f in pooled) == [
        "humanities-nuc1", "humanities-nuc1", "law73-nuc1", "law73-nuc1"]


def test_summarize_metrics_pools_rmse_and_mape_by_target_count():
    from training.ray.final_eval import summarize_metrics

    rows = [
        {"split": "a_test", "horizon": 1, "mae_db": 1.0, "rmse_db": 1.0, "n_targets": 100,
         "mape_linear_pct": 10.0, "mape_dbm_pct": 1.0},
        {"split": "b_test", "horizon": 1, "mae_db": 3.0, "rmse_db": 3.0, "n_targets": 300,
         "mape_linear_pct": 30.0, "mape_dbm_pct": 3.0},
    ]
    out = summarize_metrics(rows)
    assert out["test_mae_db_t1"] == 2.0  # unchanged: plain mean over rows
    assert abs(out["test_mse_db2_t1"] - (100 * 1 + 300 * 9) / 400) < 1e-9  # pooled by n_targets
    assert abs(out["test_rmse_db_t1"] - out["test_mse_db2_t1"] ** 0.5) < 1e-9
    assert abs(out["test_mape_linear_pct_t1"] - 25.0) < 1e-9
    # old-format rows (no new columns) still work
    legacy = summarize_metrics([{"split": "a_test", "horizon": 1, "mae_db": 1.0}])
    assert legacy == {"test_mae_db_t1": 1.0, "test_mean_horizon_mae_db": 1.0}


def test_per_bin_test_rows_keeps_only_test_split_and_tags_context():
    import pandas as pd

    from training.ray.final_eval import per_bin_test_rows

    frame = pd.DataFrame({"split": ["humanities_validation", "humanities_test", "madsen_test"],
                          "frequency_mhz": [600.5, 600.5, 601.5], "mae_db": [1.0, 2.0, 3.0]})
    out = per_bin_test_rows(frame, {"model": "m", "scored_site": "all"})
    assert list(out["mae_db"]) == [2.0, 3.0]
    assert set(out["model"]) == {"m"} and "scored_site" in out.columns


def test_per_bin_test_rows_context_overrides_existing_columns():
    import pandas as pd

    from training.ray.final_eval import per_bin_test_rows

    frame = pd.DataFrame({"split": ["a_test"], "model": ["from_evaluator"], "mae_db": [1.0]})
    out = per_bin_test_rows(frame, {"model": "from_context", "seed": 42})
    assert list(out["model"]) == ["from_context"] and list(out.columns).count("model") == 1


def test_horizon_jobs_skip_one_step_and_already_direct_models():
    from training.ray.fixed_train import horizon_jobs

    entries = [{"rep": "2d", "model": m} for m in
               ("vanillalstm", "linearar2d", "lookbackmean2d", "autoformer_csa", "temporalconvnet")]
    jobs = horizon_jobs(entries, [15, 60])
    assert [(j["model"], j["horizon"]) for j in jobs] == [
        ("vanillalstm", 15), ("vanillalstm", 60), ("autoformer_csa", 15)]


def test_set_direct_horizon_sets_model_and_scoring_horizons():
    from training.ray.fixed_train import set_direct_horizon

    base = {"windowing": {"lookback": 60, "horizons": [1, 15, 60]},
            "vanillalstm": {"model": {"prediction_horizon": 1}, "train": {}}}
    c15 = set_direct_horizon(base, "vanillalstm", 15)
    assert c15["windowing"]["horizons"] == [1, 15] and c15["vanillalstm"]["model"]["prediction_horizon"] == 15
    c60 = set_direct_horizon(base, "vanillalstm", 60)
    assert c60["windowing"]["horizons"] == [1, 15, 60] and c60["vanillalstm"]["model"]["prediction_horizon"] == 60
    assert base["windowing"]["horizons"] == [1, 15, 60]  # input untouched


def test_spectral_context_jobs_cover_every_cell_and_skip_non_2d():
    from training.ray.spectral_context_fixed import make_jobs, record_key

    entries = [{"rep": "2d", "model": "vanillalstm", "costs_key": "k"},
               {"rep": "1d", "model": "linearar1d", "costs_key": "k"}]
    regions = [{"band_id": "R1", "start_mhz": 600.5, "end_mhz": 607.5},
               {"band_id": "R3", "start_mhz": 622.5, "end_mhz": 641.5}]
    jobs = make_jobs(entries, regions, [40, 41])
    assert len(jobs) == 1 * 2 * 2 * 2 and {j["model"] for j in jobs} == {"vanillalstm"}
    assert {j["condition"] for j in jobs} == {"full_context", "region_only"}
    assert record_key("p/", jobs[0]).startswith("p/vanillalstm/R1-")
    assert len({record_key("p", j) for j in jobs}) == len(jobs)  # unique per cell
