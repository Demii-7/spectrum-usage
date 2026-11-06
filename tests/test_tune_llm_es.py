from training.ray.tune_llm_es import DEFAULT_PARAMS, candidate_list, final_key, flags, pick_winner, trial_key


def test_candidates_start_with_default_and_are_distinct():
    for arm in ("ofa", "timellm"):
        c = candidate_list(arm)
        assert len(c) == 16 and c[0] == DEFAULT_PARAMS
        assert len({tuple(sorted(p.items())) for p in c}) == 16
        assert c == candidate_list(arm)                      # deterministic


def test_ofa_does_not_vary_unused_dims():
    assert {(p["d_model"], p["d_ff"]) for p in candidate_list("ofa")} == {(32, 32)}
    assert len({p["d_model"] for p in candidate_list("timellm")}) > 1


def test_pick_winner_needs_both_horizons():
    scores = {0: {12: 0.5, 192: 0.9}, 1: {12: 0.1}, 2: {12: 0.4, 192: 0.8}, 3: {12: float("nan"), 192: 0.1}}
    assert pick_winner(scores) == (2, 0.6000000000000001) or pick_winner(scores)[0] == 2
    assert pick_winner({1: {12: 0.1}}) is None


def test_flags_and_keys():
    assert flags(DEFAULT_PARAMS)[:2] == ["--lr", "0.0001"]
    assert trial_key("p/", 4, "ofa", 3, 12) == "p/trials/spec4/ofa/c03_h12.json"
    assert final_key("p", 25, "timellm", 192, 40) == "p/finals/spec25/timellm/h192/seed-40.json"
