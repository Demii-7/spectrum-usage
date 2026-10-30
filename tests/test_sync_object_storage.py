import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "sync_object_storage", Path(__file__).resolve().parents[1] / "data/sync_object_storage.py"
)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class SyncObjectStorageTests(unittest.TestCase):
    def test_checkpoint_listing_is_paginated_and_confined_to_checkpoint_prefix(self):
        calls = []

        def call_s3(operation, **request):
            calls.append(request.copy())
            self.assertEqual(request["Prefix"], "checkpoints/")
            if "ContinuationToken" not in request:
                return {"Contents": [{"Key": "checkpoints/"},
                                     {"Key": "checkpoints/1d/model/seed-42/weights.pt"}],
                        "IsTruncated": True, "NextContinuationToken": "next"}
            return {"Contents": [{"Key": "checkpoints/MANIFEST.tsv"}]}

        entries = sync.list_checkpoint_objects(SimpleNamespace(call_s3=call_s3), "spectrum")
        self.assertEqual(set(entries), {"checkpoints/1d/model/seed-42/weights.pt", "checkpoints/MANIFEST.tsv"})
        self.assertEqual(calls[1]["ContinuationToken"], "next")
        for key in entries:
            sync.validate_key(key, checkpoints=True)
        for key in ("powder/weights.pt", "checkpoints/../../weights.pt", "/checkpoints/weights.pt"):
            with self.assertRaises(ValueError):
                sync.validate_key(key, checkpoints=True)

    def test_paginated_all_sites_listing_skips_raw_subtrees(self):
        calls = []

        def call_s3(operation, **request):
            self.assertEqual(operation, "list_objects_v2")
            calls.append((request["Prefix"], request.get("ContinuationToken")))
            if request["Prefix"] == "":
                if not request.get("ContinuationToken"):
                    return {"CommonPrefixes": [{"Prefix": "powder/"}, {"Prefix": "raw/"}],
                            "IsTruncated": True, "NextContinuationToken": "page2"}
                return {"CommonPrefixes": [{"Prefix": "ara/"}]}
            prefix = request["Prefix"]
            return {"Contents": [{"Key": prefix + "power_1mhz_avg_per_minute.csv"},
                                 {"Key": prefix + "ignored.csv"}],
                    "CommonPrefixes": [{"Prefix": prefix + "raw/"}]}

        objects = sync.list_processed_objects(SimpleNamespace(call_s3=call_s3), "spectrum", "")
        self.assertEqual(set(objects), {"powder/power_1mhz_avg_per_minute.csv",
                                        "ara/power_1mhz_avg_per_minute.csv"})
        self.assertIn(("", "page2"), calls)
        self.assertFalse(any("raw" in prefix for prefix, _ in calls))

    def test_spectrum_credentials_do_not_use_default_minio_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            access = Path(directory) / "access"
            secret = Path(directory) / "secret"
            access.write_text("source-access\n")
            secret.write_text("source-secret\n")
            environment = {
                "AWS_ACCESS_KEY_ID": "minio-access", "AWS_SECRET_ACCESS_KEY": "minio-secret",
                "S3_BUCKET": "outputs", "AWS_ENDPOINT_URL": "http://minio:9000",
                "SPECTRUM_AWS_ACCESS_KEY_ID_FILE": str(access),
                "SPECTRUM_AWS_SECRET_ACCESS_KEY_FILE": str(secret),
                "SPECTRUM_S3_ENDPOINT_URL": "https://source.example",
                "SPECTRUM_S3_BUCKET": "spectrum",
            }
            with patch.dict(sync.os.environ, environment, clear=True):
                bucket, endpoint, credentials = sync.storage_settings(
                    SimpleNamespace(spectrum_storage=True, bucket=None)
                )
            self.assertEqual((bucket, endpoint), ("spectrum", "https://source.example"))
            self.assertEqual(credentials, {"key": "source-access", "secret": "source-secret"})

    def test_saved_inventory_sync_is_incremental_and_generates_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "powder/node/run/power_1mhz_avg_per_minute.csv"
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"bucket": "spectrum", "endpoint": "https://source.example",
                                             "objects": {key: {"Size": 7, "LastModified": 1700000000}}}))
            downloads = []

            def get(remote, local, **kwargs):
                downloads.append(remote)
                Path(local).write_bytes(b"a,b\n1,2")

            fs = SimpleNamespace(get=get)
            argv = ["sync", "--inventory-in", str(inventory), "--output-dir", str(root / "data"),
                    "--checksums-out", str(root / "checksums.json"), "--env-file", str(root / "absent"),
                    "--summary-only"]
            with patch.dict(sync.os.environ, {"S3_BUCKET": "spectrum", "S3_ENDPOINT_URL": "https://source.example"}, clear=True), \
                 patch.dict("sys.modules", {"fsspec": SimpleNamespace(filesystem=lambda *args, **kwargs: fs)}), \
                 patch("sys.argv", argv):
                self.assertEqual(sync.main(), 0)
                self.assertEqual(sync.main(), 0)
            self.assertEqual(downloads, [f"spectrum/{key}"])
            checksums = json.loads((root / "checksums.json").read_text())
            self.assertEqual(checksums["objects"][key]["size"], 7)
            self.assertEqual(len(checksums["objects"][key]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
