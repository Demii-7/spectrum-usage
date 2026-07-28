import unittest

from training.ray.config import ConfigurationError, inject_parameters, validate_tuning_config


def base_config():
    return {
        "training": {"model_name": "vanillalstm"},
        "data": {
            "representation": "2d",
            "split": {"ranges": {
                "train": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
                "validation": {"start": "2024-01-03T00:00:00Z", "end": "2024-01-04T00:00:00Z"},
                "test": {"start": "2024-01-05T00:00:00Z", "end": "2024-01-06T00:00:00Z"},
            }},
        },
        "vanillalstm": {"model": {"hidden_size": 128}, "train": {"seed": 42}},
    }


class RayConfigTests(unittest.TestCase):
    def test_injection_is_nested_validated_and_non_mutating(self):
        source = base_config()
        architecture = {
            "model.hidden_size": 32,
            "model.num_layers": 1,
            "model.dropout": 0.0,
        }
        result = inject_parameters(source, "vanillalstm", {
            "architecture": architecture,
            "train.learning_rate": 0.001,
            "train.batch_size": 32,
        }, seed=7)
        self.assertEqual(result["vanillalstm"]["model"]["hidden_size"], 32)
        self.assertEqual(result["vanillalstm"]["train"]["seed"], 7)
        self.assertEqual(result["training"]["models"], ["vanillalstm"])
        self.assertEqual(source["vanillalstm"]["model"]["hidden_size"], 128)

    def test_rejects_unknown_or_out_of_bounds_values(self):
        with self.assertRaisesRegex(ConfigurationError, "outside"):
            inject_parameters(base_config(), "vanillalstm", {"model.input_size": 10}, seed=1)
        with self.assertRaisesRegex(ConfigurationError, "bounded"):
            inject_parameters(base_config(), "vanillalstm", {"architecture": {
                "model.hidden_size": 999, "model.num_layers": 1, "model.dropout": 0.0,
            }}, seed=1)

    def test_execution_requires_explicit_validation_range(self):
        config = base_config()
        del config["data"]["split"]["ranges"]["validation"]
        with self.assertRaisesRegex(ConfigurationError, "explicit"):
            validate_tuning_config(config, "vanillalstm")


if __name__ == "__main__":
    unittest.main()
