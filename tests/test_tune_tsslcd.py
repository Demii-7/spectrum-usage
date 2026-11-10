from training.ray.tune_tsslcd import DEFAULTS, candidate_id, candidate_list


def test_budget_eight_is_a_prefix_of_sixteen():
    eight, sixteen = candidate_list(8), candidate_list(16)
    assert len(eight) == 8 and len(sixteen) == 16
    assert [candidate_id(c) for c in eight] == [candidate_id(c) for c in sixteen[:8]]


def test_first_candidate_is_the_authors_defaults_and_all_are_unique():
    candidates = candidate_list(16)
    assert candidates[0] == DEFAULTS
    assert len({candidate_id(c) for c in candidates}) == 16


def test_draws_stay_in_the_search_space():
    for c in candidate_list(16)[1:]:
        assert c["latent_dim"] in (16, 32, 48, 64) and c["batch_size"] in (64, 128, 256)
        assert 3e-5 <= c["learning_rate"] <= 3e-4 and c["train_mask_ratio"] in (0.0, 0.1, 0.2)


def test_spatial15_sts_branches_are_copied_for_the_model_factory():
    from training.ray.spatial15 import model_config_for

    config = {"windowing": {"lookback": 96}, "stsprednet": {"use_closeness": True, "lc": 96, "lp": 1, "model": {}}}
    out = model_config_for(config, "stsprednet")
    assert out["stsprednet"]["branches"] == {"use_closeness": True, "lc": 96, "lp": 1}
    assert "branches" not in config["stsprednet"]                       # the input config is not modified
    assert "branches" not in model_config_for({"windowing": {}, "linearar4d": {"model": {}}}, "linearar4d")["linearar4d"]
