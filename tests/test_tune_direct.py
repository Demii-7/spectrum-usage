import pytest

from training.ray.tune_direct import (BUDGET, DEFAULTS, FINAL_HORIZONS, PIPELINE_NAME, TUNE_HORIZONS, build_config, candidate_list,
                                      default_train_stride, final_key, group, pick_winner, trial_key)

ROOT = "."


@pytest.mark.parametrize("model", sorted(PIPELINE_NAME))
def test_candidates_start_with_default_and_are_distinct(model):
    c = candidate_list(model)
    assert len(c) == BUDGET and c[0] == DEFAULTS[model] and c == candidate_list(model)
    assert len({tuple(sorted(p.items())) for p in c}) == BUDGET


def test_controls_use_exactly_ofas_candidates_and_flags():
    assert candidate_list("ofa_rand") == candidate_list("ofa") == candidate_list("ofa_scratch")
    for model, flags in (("ofa", {}), ("ofa_rand", {"backbone_init": "random", "backbone_trainable": "ofa"}),
                         ("ofa_scratch", {"backbone_init": "random", "backbone_trainable": "all"})):
        config, name = build_config(model, "15min", "600_800", 48, 42, candidate_list("ofa")[1], ROOT, train_stride=1, test_stride=1)
        assert name == "ofa" and config["ofa"]["model"]["channel_independent"] is True and config["ofa"]["train"]["batch_size"] == 4
        for key, value in flags.items():
            assert config["ofa"]["model"][key] == value
        assert ("backbone_init" in config["ofa"]["model"]) == bool(flags)


def test_ofa_does_not_vary_unused_dims_and_timellm_does():
    assert all("d_model" not in p for p in candidate_list("ofa"))
    assert len({p["d_model"] for p in candidate_list("timellm")}) > 1


def test_patch_stride_never_exceeds_the_patch():
    for model in ("ofa", "timellm", "patchtst"):
        for p in candidate_list(model):
            assert 0 < p["patch_stride"] <= p["patch_length"]


@pytest.mark.parametrize("model", sorted(PIPELINE_NAME))
@pytest.mark.parametrize("resolution", ["15min", "1min"])
def test_build_config(model, resolution):
    horizon = TUNE_HORIZONS[resolution][1]
    params = candidate_list(model)[3]
    config, name = build_config(model, resolution, "2400_2600", horizon, 42, params, ROOT,
                                train_stride=default_train_stride(model, resolution), test_stride=1)
    section = config[name]
    assert section["model"]["prediction_horizon"] == horizon
    assert section["model"]["input_sequence_length"] == (96 if resolution == "15min" else 60)
    assert section["train"]["train_stride"] == default_train_stride(model, resolution) == section["train"]["val_stride"]
    assert section["train"]["seed"] == 42 and config["training"]["models"] == [name]
    assert config["data"]["split"]["ranges"]["test"]["end"] == "2026-11-03T20:00:00Z"      # the full test range, not the preliminary one
    assert all("/2400_2600/" in f["path"] or "2400_2600" in f["path"] for f in config["data"]["files"])
    for key, value in params.items():
        assert value in (section["model"].get(key), section["train"].get(key), section["train"].get({"learning_rate": "learning_rate"}.get(key, key)))
    assert config["windowing"]["horizons"] == [horizon]
    assert (section["model"].get("channel_independent") is True) == (model in ("ofa", "timellm"))


def test_strides():
    assert default_train_stride("timellm", "15min") == 2 and default_train_stride("ofa", "15min") == 1
    assert {default_train_stride(m, "1min") for m in PIPELINE_NAME} == {15}


def test_pick_winner_and_keys():
    s = {0: {4: 0.5, 48: 0.9}, 1: {4: 0.1}, 2: {4: 0.4, 48: 0.8}, 3: {4: float("nan"), 48: 0.1}}
    assert pick_winner(s, (4, 48))[0] == 2 and pick_winner({1: {4: 0.1}}, (4, 48)) is None
    b = group("p/", "15min", "600_800", "ofa")
    assert trial_key(b, 3, 4) == "p/15min/600_800/ofa/trials/c03_h4.json" and final_key(b, 48, 40) == "p/15min/600_800/ofa/finals/h48/seed-40.json"
    assert FINAL_HORIZONS["15min"] == (4, 12, 24, 48)
