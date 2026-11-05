import random
from pathlib import Path

import yaml

from training.ray.final_eval import config_file_for
from training.ray.powder15 import HORIZONS, LOOKBACK, apply_powder15
from training.ray.registry import get_model_spec
from training.ray.spaces import choice, loguniform, randint, uniform
from training.ray.tune_fixed import (
    CAPACITIES, candidate_list, draw, group_prefix, parse_models, pick_winner, prepared,
)


def test_draw_respects_each_domain_kind():
    rng = random.Random(0)
    assert draw(choice("a", "b"), rng) in ("a", "b")
    assert 1.0 <= draw(uniform(1.0, 2.0), rng) <= 2.0
    assert 1e-4 <= draw(loguniform(1e-4, 1e-2), rng) <= 1e-2
    assert 3 <= draw(randint(3, 9), rng) < 9


def test_candidate_list_starts_with_anchors_is_unique_and_deterministic():
    spec = get_model_spec("vanillalstm", require_executable=True)
    a = candidate_list(spec, 16, random.Random("x"))
    b = candidate_list(spec, 16, random.Random("x"))
    assert a == b and len(a) == 16
    assert a[:3] == [dict(spec.anchor(c)) for c in CAPACITIES]
    from training.ray.candidate import canonical_json
    assert len({canonical_json(c) for c in a}) == 16
    for sample in a[3:]:
        assert set(sample) == set(spec.space)      # random draws cover every searched parameter


def test_pick_winner_takes_lowest_objective_and_ignores_failures():
    records = [{"ok": True, "objective": 0.5, "candidate_id": "b"}, {"ok": False, "objective": 0.1, "candidate_id": "a"},
               {"ok": True, "objective": 0.4, "candidate_id": "c"}, {"ok": True, "objective": float("nan"), "candidate_id": "d"}]
    assert pick_winner(records)["candidate_id"] == "c"


def test_parse_models_and_prefix():
    assert parse_models(["vanillalstm:2d", "linearar1d:1d"]) == [("vanillalstm", "2d"), ("linearar1d", "1d")]
    try:
        parse_models(["vanillalstm"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert group_prefix("tuning/", "powder15", "600_800", "2d", "lstmattn") == "tuning/powder15/600_800/2d/lstmattn"


def test_prepared_config_keeps_the_new_lookback_after_injection():
    base = yaml.safe_load(Path(config_file_for("vanillalstm", "2d")).read_text())
    config = apply_powder15(base, "600_800", "/repo", list(HORIZONS))
    spec = get_model_spec("vanillalstm", require_executable=True)
    params = dict(spec.anchor("reference"))
    out = prepared(config, "powder15", "vanillalstm", params, 42)
    assert out["vanillalstm"]["model"]["input_sequence_length"] == LOOKBACK
    assert out["vanillalstm"]["model"]["prediction_horizon"] == 1
    assert out["vanillalstm"]["train"]["seed"] == 42 and out["vanillalstm"]["train"]["test_stride"] == 1
    assert out["windowing"]["lookback"] == LOOKBACK
