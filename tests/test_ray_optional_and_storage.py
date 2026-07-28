import tempfile
import unittest
from pathlib import Path

from training.ray.manifests import result_manifest, write_manifest
from training.ray.scheduler import ASHAConfig
from training.ray.storage import MinIOConfig
from training.ray.trainable import TuneReporter


class RayOptionalAndStorageTests(unittest.TestCase):
    def test_asha_configuration_is_ray_independent(self):
        config = ASHAConfig(max_t=20, grace_period=3)
        self.assertEqual(config.to_dict()["metric"], "objective")
        self.assertEqual(config.to_dict()["max_t"], 20)

    def test_callback_reports_only_prunable_events_and_prefers_physical_metric(self):
        reports = []
        reporter = TuneReporter(lambda metrics, checkpoint=None: reports.append((metrics, checkpoint)))
        reporter({"stage": "pretrain", "epoch": 1, "prunable": False,
                  "metrics": {"masked_reconstruction_loss": 0.9}})
        reporter({"stage": "train", "epoch": 2, "prunable": True,
                  "metrics": {"val_loss": 0.5, "val_mean_horizon_mae_db": 0.25}})
        self.assertEqual(len(reporter.events), 2)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0][0]["training_iteration"], 1)
        self.assertEqual(reports[0][0]["objective"], 0.25)
        self.assertEqual(reports[0][0]["objective_name"], "val_mean_horizon_mae_db")
        self.assertEqual(reports[0][0]["stage"], "train")

    def test_callback_fallback_has_manifest_warning(self):
        reporter = TuneReporter(lambda metrics, checkpoint=None: None)
        reporter({"stage": "train", "epoch": 1, "prunable": True,
                  "metrics": {"val_loss": 0.5}})
        self.assertEqual(reporter.last_metrics["objective_name"], "val_loss")
        self.assertIn("not directly comparable", reporter.warnings[0])

    def test_minio_uses_environment_without_exposing_secrets_in_uri(self):
        config = MinIOConfig("experiments", "project/run", "http://minio:9000")
        environment = config.environment({
            "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"
        })
        self.assertEqual(config.storage_path, "s3://experiments/project/run")
        self.assertEqual(environment["AWS_ENDPOINT_URL"], "http://minio:9000")
        self.assertNotIn("secret", config.storage_path)

    def test_manifest_write_is_json_and_atomic(self):
        manifest = result_manifest(experiment_id="exp", selections={"best": "a"}, candidates=[])
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(Path(directory) / "result.json", manifest)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertIn('"schema_version": 1', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
