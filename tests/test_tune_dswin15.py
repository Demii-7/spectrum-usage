import json
import random
from pathlib import Path

from training.ray import tune_dswin15 as t

INHERITED = json.loads(Path("training/configs/fixed/dswinlstm_i_inherited_1min_r2.json").read_text())


def test_candidate_two_is_the_inherited_configuration():
    from training.ray.registry import get_model_spec

    spec = get_model_spec("dswinlstm_i", require_executable=True)
    cands = t.candidates_for(spec, "600_800", 8)
    assert len(cands) == 8
    fixed = t.fixed_from_candidate(cands[2], INHERITED, "x")
    assert fixed["model"] == INHERITED["model"]
    for k in ("batch_size", "epochs", "learning_rate", "weight_decay", "optimizer", "gradient_clip_norm"):
        assert fixed["train"][k] == INHERITED["train"][k]


def test_candidates_are_deterministic_and_overlay_only_search_keys():
    from training.ray.registry import get_model_spec

    spec = get_model_spec("dswinlstm_i", require_executable=True)
    a = t.candidates_for(spec, "2400_2600", 8)
    b = t.candidates_for(spec, "2400_2600", 8)
    assert a == b
    for cand in a:
        fixed = t.fixed_from_candidate(cand, INHERITED, "c", epochs=2)
        assert fixed["train"]["epochs"] == 2
        assert fixed["model"]["prediction_horizon"] == INHERITED["model"]["prediction_horizon"]      # keeps the direct multi-step setting
        assert set(fixed["model"]) == set(INHERITED["model"])
        assert fixed["model"]["embed_dim"] == cand["architecture"]["model.embed_dim"]


def test_inherited_json_not_mutated():
    before = json.dumps(INHERITED, sort_keys=True)
    t.fixed_from_candidate({"architecture": {"model.embed_dim": 8}, "train.batch_size": 1}, INHERITED, "c")
    assert json.dumps(INHERITED, sort_keys=True) == before
