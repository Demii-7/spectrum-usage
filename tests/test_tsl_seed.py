import json

import pytest

from training.ray.tsl_runs import SEED_PATCH, patch_seed
from training.ray.tune_tsl import load_winners

RUN_PY = """import os
import torch

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
"""


def test_patch_makes_the_seed_come_from_the_environment_and_defaults_to_2021():
    patched, changed = patch_seed(RUN_PY)
    assert changed and SEED_PATCH in patched and "fix_seed = 2021" not in patched
    assert patched.count("fix_seed") == RUN_PY.count("fix_seed")          # only the one assignment changed
    namespace = {}
    exec("import os\n" + SEED_PATCH, namespace)                           # default without the variable
    assert namespace["fix_seed"] == 2021


def test_patch_reads_tsl_seed(monkeypatch):
    monkeypatch.setenv("TSL_SEED", "43")
    namespace = {}
    exec("import os\n" + SEED_PATCH, namespace)
    assert namespace["fix_seed"] == 43


def test_patch_is_idempotent_and_refuses_an_unknown_library():
    once, _ = patch_seed(RUN_PY)
    twice, changed = patch_seed(once)
    assert twice == once and not changed
    with pytest.raises(ValueError):
        patch_seed("fix_seed = 1\n")


class FakeClient:
    def __init__(self, tables):
        self.tables = tables

    def get_object(self, Bucket, Key):
        if Key not in self.tables:
            raise KeyError(Key)
        body = json.dumps(self.tables[Key]).encode()
        return {"Body": type("B", (), {"read": lambda self: body})()}


def test_load_winners_filters_models_and_datasets_and_merges_prefixes():
    w1 = {"spec4/DLinear": {"index": 3, "params": {"learning_rate": 0.001}}, "spec151/DLinear": {"index": 1, "params": {}}}
    w2 = {"spec25/Informer": {"index": 5, "params": {"d_model": 64}}}
    client = FakeClient({"a/winners.json": w1, "b/winners.json": w2})
    out = load_winners(client, "bucket", ["a/", "b"], ["DLinear", "Informer"], ["spec4", "spec25"])
    assert set(out) == {("spec4", "DLinear"), ("spec25", "Informer")}
    assert out[("spec25", "Informer")]["params"] == {"d_model": 64}
    with pytest.raises(SystemExit):
        load_winners(client, "bucket", ["missing"], ["DLinear"], ["spec4"])
