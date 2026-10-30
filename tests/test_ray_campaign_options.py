import unittest

from training.ray.config import validate_tuning_config
from training.ray.run import build_plan


def _config(model, representation="2d"):
    return {
        "training": {"model_name": model},
        "data": {"representation": representation, "split": {"ranges": {
            "train": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
            "validation": {"start": "2024-01-03T00:00:00Z", "end": "2024-01-04T00:00:00Z"},
            "test": {"start": "2024-01-05T00:00:00Z", "end": "2024-01-06T00:00:00Z"},
        }}},
        model: {"model": {}, "train": {"epochs": 20}},
    }


class CampaignPlanOptionTests(unittest.TestCase):
    def test_overrides_reach_the_plan(self):
        entry = build_plan(
            _config("vanillalstm"), ["vanillalstm"], candidates=16, grace_period=8,
            reduction_factor=2, brackets=2, max_concurrent=12, gpu_fraction=0.25,
        )["models"][0]
        search = entry["search"]
        self.assertEqual(search["candidate_budget"]["total"], 16)
        self.assertEqual(
            (search["grace_period"], search["reduction_factor"], search["brackets"],
             search["max_concurrent_trials"]),
            (8, 2, 2, 12),
        )
        self.assertEqual(entry["resources"]["gpu"], 0.25)

    def test_defaults_are_unchanged_without_overrides(self):
        search = build_plan(_config("vanillalstm"), ["vanillalstm"])["models"][0]["search"]
        self.assertEqual((search["grace_period"], search["reduction_factor"], search["brackets"]),
                         (5, 3, 1))

    def test_gpu_fraction_never_gives_cpu_only_models_a_gpu(self):
        entry = build_plan(_config("linearar2d"), ["linearar2d"], gpu_fraction=0.25)["models"][0]
        self.assertEqual(entry["resources"]["gpu"], 0.0)

    def test_temporal_conv_net_keeps_its_longer_grace_period(self):
        entry = build_plan(_config("temporalconvnet"), ["temporalconvnet"], grace_period=8)["models"][0]
        self.assertEqual(entry["search"]["grace_period"], 12)

    def test_residual_lstm_validates_for_both_1d_and_2d_data(self):
        for representation in ("1d", "2d"):
            validate_tuning_config(_config("residualvanillalstm", representation), "residualvanillalstm")


if __name__ == "__main__":
    unittest.main()
