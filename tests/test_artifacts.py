import json
from pathlib import Path

from training.ray.artifacts import artifact_keys, save_artifact, sha256_of


class FakeClient:
    def __init__(self):
        self.uploaded, self.objects = {}, {}

    def upload_file(self, path, bucket, key):
        self.uploaded[key] = Path(path).read_bytes()

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body


def test_save_artifact_writes_weights_and_sidecar(tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"abc")
    client = FakeClient()
    out = save_artifact(client, "b", "p/q/", weights, {"seed": 3, "config": {"lr": 1}})
    assert client.uploaded["p/q/m.pth"] == b"abc"
    meta = json.loads(client.objects["p/q/m.pth.meta.json"])
    assert meta["seed"] == 3 and meta["bytes"] == 3 and meta["sha256"] == sha256_of(weights) == out["sha256"]
    assert "code_revision" in meta and meta["config"] == {"lr": 1}


def test_save_artifact_reports_failure_without_raising(tmp_path):
    out = save_artifact(FakeClient(), "b", "p", tmp_path / "missing.pth")
    assert "artifact_error" in out
    assert artifact_keys("p/", "x") == ("p/x", "p/x.meta.json")
