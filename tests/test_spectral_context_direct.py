from training.ray.spectral_context_direct import conditions_for, make_tasks, record_key

REGIONS = {"600_800": [{"band_id": "R3", "start_mhz": 1.0, "end_mhz": 2.0}, {"band_id": "R4", "start_mhz": 3.0, "end_mhz": 4.0}],
           "2400_2600": [{"band_id": "R2", "start_mhz": 5.0, "end_mhz": 6.0}]}


def test_input_only_is_for_the_matched_pair_only():
    assert conditions_for("revinmlp") == ["full_context", "region_only", "input_only"]
    assert conditions_for("revinmlp_mix") == ["full_context", "region_only", "input_only"]
    assert conditions_for("patchtst_joint") == ["full_context", "region_only"]


def test_tasks_cover_every_cell_once_and_start_with_the_first_seed():
    seeds = {"revinmlp": (40, 41), "itransformer": (42,)}
    tasks = make_tasks(["revinmlp", "itransformer"], ["600_800", "2400_2600"], REGIONS, seeds)
    assert len(tasks) == 3 * 2 * 3 + 2 * 3 * 1 * 1 + 0 and len({(t["model"], t["band"], t["region"]["band_id"], t["condition"], t["seed"]) for t in tasks}) == len(tasks)
    assert [t["seed"] for t in tasks] == sorted(t["seed"] for t in tasks)
    assert all(t["horizons"] == [4, 12, 24, 48] for t in tasks)


def test_record_keys_carry_the_band_and_failures_are_separate():
    ok = record_key("p/", "2400_2600", "revinmlp", "R2", "region_only", 12, 41)
    assert ok == "p/2400_2600/revinmlp/R2-region_only-h12-seed-41.json"
    assert record_key("p", "2400_2600", "revinmlp", "R2", "region_only", 12, 41, failed=True).endswith(".error.json")
