import json

from training.ray.tsl_runs import parse_best_validation, run_command
from training.ray.tune_tsl import BUDGET, DEFAULTS, SPACES, candidate_list, flags, pick_winner


def test_candidates_start_with_defaults_and_are_distinct_and_reproducible():
    for model in DEFAULTS:
        a, b = candidate_list(model), candidate_list(model)
        assert a == b and len(a) == BUDGET
        assert a[0] == DEFAULTS[model]
        assert len({json.dumps(c, sort_keys=True) for c in a}) == BUDGET


def test_candidates_stay_inside_the_space_and_heads_divide_width():
    for model, space in SPACES.items():
        for c in candidate_list(model)[1:]:
            assert set(c) == set(space)
            for name, domain in space.items():
                if domain[0] == "choice":
                    assert c[name] in domain[1]
                else:
                    assert domain[1] <= c[name] <= domain[2]
            if "n_heads" in c:
                assert c["d_model"] % c["n_heads"] == 0


def test_flags_and_command_extra_and_des():
    assert flags({"learning_rate": 0.001, "d_model": 64}) == ["--learning_rate", "0.001", "--d_model", "64"]
    cmd = run_command("PatchTST", "spec25", 12, 42, "/d/", "800", training=True, inverse=False, has_seed_flag=True,
                      extra=["--d_model", "64"], des="tune-PatchTST-c3")
    assert cmd[cmd.index("--des") + 1] == "tune-PatchTST-c3" and cmd[cmd.index("--d_model") + 1] == "64"


def test_pick_winner_uses_only_complete_candidates():
    scores = {0: {12: 1.0, 192: 3.0}, 1: {12: 0.1}, 2: {12: 1.0, 192: 1.0}, 3: {12: float("nan"), 192: 0.0}}
    assert pick_winner(scores) == (2, 1.0)
    assert pick_winner({1: {12: 0.1}}) is None


def test_parse_best_validation_takes_the_minimum_epoch():
    text = "Epoch: 1, Steps: 9 | Train Loss: 0.9 Vali Loss: 0.800 Test Loss: 0.7\nEpoch: 2 | Vali Loss: 0.6123 Test Loss: 0.5\nVali Loss: 0.7"
    assert parse_best_validation(text) == 0.6123
    assert parse_best_validation("nothing") is None
