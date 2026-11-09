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
    assert (section["model"].get("channel_independent") is True) == (model in ("ofa", "ofa_rand", "ofa_scratch", "timellm"))


def test_strides():
    assert default_train_stride("timellm", "15min") == 2 and default_train_stride("ofa", "15min") == 1
    assert {default_train_stride(m, "1min") for m in PIPELINE_NAME} == {15}


def test_pick_winner_and_keys():
    s = {0: {4: 0.5, 48: 0.9}, 1: {4: 0.1}, 2: {4: 0.4, 48: 0.8}, 3: {4: float("nan"), 48: 0.1}}
    assert pick_winner(s, (4, 48))[0] == 2 and pick_winner({1: {4: 0.1}}, (4, 48)) is None
    b = group("p/", "15min", "600_800", "ofa")
    assert trial_key(b, 3, 4) == "p/15min/600_800/ofa/trials/c03_h4.json" and final_key(b, 48, 40) == "p/15min/600_800/ofa/finals/h48/seed-40.json"
    assert FINAL_HORIZONS["15min"] == (4, 12, 24, 48)


def test_patchtst_ablations_share_patchtsts_candidates_except_the_removed_ingredient():
    base = candidate_list("patchtst")
    for model, flag in (("patchtst_noinorm", {"instance_norm": False}), ("patchtst_joint", {"channel_mode": "joint"})):
        assert candidate_list(model) == base
        config, name = build_config(model, "15min", "600_800", 48, 42, base[2], ROOT, train_stride=1, test_stride=1)
        assert name == "patchtst" and all(config["patchtst"]["model"][k] == v for k, v in flag.items())
    nopatch = candidate_list("patchtst_nopatch")
    assert all(p["patch_length"] == 1 and p["patch_stride"] == 1 for p in nopatch) and len(nopatch) == BUDGET
    assert [{k: v for k, v in p.items() if not k.startswith("patch_")} for p in nopatch] == \
           [{k: v for k, v in p.items() if not k.startswith("patch_")} for p in base] or len({tuple(sorted(p.items())) for p in nopatch}) == BUDGET
    config, _ = build_config("patchtst_nopatch", "1min", "2400_2600", 60, 42, nopatch[3], ROOT, train_stride=15, test_stride=15)
    assert config["patchtst"]["model"]["patch_length"] == 1 and "instance_norm" not in config["patchtst"]["model"]


def test_linear_models_use_the_registry_space():
    from training.ray.registry import get_model_spec  # the campaign's own space, for reference

    for model in ("residuallinearar2d", "linearar2d"):
        cands = candidate_list(model)
        assert cands[0] == {"learning_rate": 1e-3, "ridge_alpha": 0.01, "batch_size": 64}
        assert {c["ridge_alpha"] for c in cands} <= {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}
        assert all(1e-5 <= c["learning_rate"] <= 1e-2 and c["batch_size"] in (32, 64, 128) for c in cands[1:])
        config, name = build_config(model, "15min", "600_800", 4, 42, cands[1], ROOT, train_stride=1, test_stride=1)
        assert name == model and config[model]["model"]["ridge_alpha"] == cands[1]["ridge_alpha"] and config[model]["train"]["weight_decay"] == 0.0


def test_electrosense_horizons_and_strides():
    from training.ray.tune_direct import TUNE_HORIZONS

    assert TUNE_HORIZONS["electrosense"] == (12, 192) and FINAL_HORIZONS["electrosense"] == (12, 24, 48, 96, 192)
    assert default_train_stride("patchtst", "electrosense") == 1 and default_train_stride("timellm", "electrosense") == 1


class TimeRANVariantTests:
    def test_variants_share_candidates_and_set_backbone_flags(self):
        from training.ray import tune_direct as td

        first = td.candidate_list("timeran_lp")
        assert len(first) == td.BUDGET
        for variant in td.TIMERAN_FAMILY:
            assert td.candidate_list(variant) == first
            assert td.PIPELINE_NAME[variant] == "timeran"
        assert td.CONTROLS["moment_rand"] == {"backbone_init": "random", "backbone_trainable": "all"}
        assert td.CONTROLS["timeran_lp"]["backbone_trainable"] == "head"

    def test_params_land_in_the_right_sections(self):
        from training.ray import tune_direct as td

        model, train = td.template("timeran_ff", ".")
        assert model["context_length"] == 512
        config = td.apply_params({"timeran": {"model": model, "train": train}}, "timeran",
                                 {"learning_rate": 1e-4, "batch_size": 8, "head_dropout": 0.2, "weight_decay": 1e-5})
        assert config["timeran"]["model"]["head_dropout"] == 0.2
        assert config["timeran"]["train"]["batch_size"] == 8
