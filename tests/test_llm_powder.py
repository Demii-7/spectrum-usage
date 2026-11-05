from training.ray.llm_powder import RESOLUTIONS, build_config, result_key

ROOT = "."


def test_one_minute_config_is_channel_independent_direct_and_final_range():
    config, name = build_config("ofa", "1min", "600_800", 15, 41, ROOT, train_stride=7, epochs=3)
    assert name == "ofa" and config["training"]["models"] == ["ofa"]
    model, train = config["ofa"]["model"], config["ofa"]["train"]
    assert model["channel_independent"] is True and model["prediction_horizon"] == 15
    assert model["input_sequence_length"] == RESOLUTIONS["1min"]["lookback"] == 60
    assert train["train_stride"] == 7 and train["epochs"] == 3 and train["seed"] == 41 and train["test_stride"] == 1
    assert config["data"]["split"]["ranges"]["test"]["end"] == "2026-11-03T20:00:00Z"
    assert config["windowing"] == {"lookback": 60, "horizons": [15]}


def test_fifteen_minute_config_points_at_blocked_files_with_lookback_96():
    config, name = build_config("timellm", "15min", "2400_2600", 24, 40, ROOT)
    assert name == "spectrumllm"
    assert all("data/powder15/" in e["path"] and "/2400_2600/" in e["path"] and e["path"].endswith("power_15min.csv")
               for e in config["data"]["files"])
    assert config["spectrumllm"]["model"]["input_sequence_length"] == 96
    assert config["spectrumllm"]["model"]["prediction_horizon"] == 24
    assert config["windowing"]["lookback"] == 96


def test_result_key_layout():
    assert result_key("p/", "15min", "600_800", "ofa", 4, 40) == "p/15min/600_800/ofa/h4/seed-40.json"
