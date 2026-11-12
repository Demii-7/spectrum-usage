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


def test_finals_from_an_older_winner_are_not_counted_as_done():
    import inspect

    from training.ray import tune_fixed

    source = inspect.getsource(tune_fixed.launch)
    assert 'r.get("candidate_id") == winner["candidate_id"]' in source


def test_fixed_config_overlays_architecture_and_sets_the_15_minute_output_length():
    from training.ray.powder15 import CORE_NODES, apply_powder15_4d
    from training.ray.tune_fixed import apply_fixed_config, fixed_final_calls, parse_args

    base = yaml.safe_load(Path(config_file_for("dswinlstm_i", "4d")).read_text())
    config = apply_powder15_4d(base, "600_800", list(HORIZONS), CORE_NODES, {"height": 10, "width": 10}, "p15_600_800_core4-p2")
    fixed = {"label": "inherited:1min-r2",
             "model": {"embed_dim": 48, "hidden_dims": [48, 96], "input_sequence_length": 60, "prediction_horizon": 60},
             "train": {"batch_size": 3, "learning_rate": 5e-4, "train_stride": 5, "seed": 1, "early_stopping": True}}
    out = apply_fixed_config(config, "dswinlstm_i", fixed, 42)
    section = out["dswinlstm_i"]
    assert section["model"]["embed_dim"] == 48 and section["model"]["hidden_dims"] == [48, 96]
    assert section["model"]["input_sequence_length"] == LOOKBACK              # the dataset's lookback, not the 1-minute 60
    assert section["model"]["prediction_horizon"] == max(HORIZONS) == 48      # direct output covers the longest horizon
    assert section["train"]["batch_size"] == 3 and section["train"]["learning_rate"] == 5e-4
    assert section["train"]["seed"] == 42 and section["train"]["train_stride"] == 1      # dataset strides and seed win
    assert section["train"]["early_stopping"] is False                                   # selection as in the searches
    # a one-step model stays one-step
    one = apply_fixed_config(config, "dswinlstm_i", {"model": {"embed_dim": 16}}, 7)
    assert one["dswinlstm_i"]["model"]["prediction_horizon"] == 1

    class Client:       # no finished records: every seed is launched; a finished one is skipped
        records: list = []
    import training.ray.tune_fixed as tf
    args = parse_args(["--source", "powder15_4d", "--models", "dswinlstm_i:4d", "--fixed-config", "x.json"])
    assert args.fixed_config == "x.json" and args.fixed_seeds == [42] and args.datasets == ["600_800", "2400_2600"]
    tf_read = tf.read_records
    try:
        tf.read_records = lambda client, bucket, prefix: Client.records
        calls = fixed_final_calls(args, fixed, None, "b", lambda model: {})
        assert [(c[2]["dataset"], c[2]["seed"], c[2]["candidate_id"]) for c in calls] == [
            ("600_800", 42, "inherited:1min-r2"), ("2400_2600", 42, "inherited:1min-r2")]
        Client.records = [{"ok": True, "seed": 42, "candidate_id": "inherited:1min-r2"}]
        assert fixed_final_calls(args, fixed, None, "b", lambda model: {}) == []
    finally:
        tf.read_records = tf_read
