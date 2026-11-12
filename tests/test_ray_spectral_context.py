from pathlib import Path

from training.ray.spectral_context import non_noise_regions, parse_args, regional_config


ROOT = Path(__file__).resolve().parents[1]


def test_regions_exclude_four_noise_floor_bands():
    regions = non_noise_regions(ROOT / "evaluation/analysis/plan_regions_600_800.csv")
    assert len(regions) == 13
    assert {region["band_id"] for region in regions}.isdisjoint({"R2", "R7", "R15", "R17"})


def test_regional_conditions_share_loss_target_but_only_one_masks_inputs():
    base = {"data": {"value": 1}}
    region = {"band_id": "R3", "start_mhz": 622.5, "end_mhz": 641.5}
    full = regional_config(base, region, "full_context", 40)
    masked = regional_config(base, region, "region_only", 40)
    assert full["data"]["loss_frequency_ranges"] == [[622.5, 641.5]]
    assert masked["data"]["loss_frequency_ranges"] == [[622.5, 641.5]]
    assert "mask" not in full["data"]
    assert masked["data"]["mask"]["frequency_ranges"] == [[622.5, 641.5]]
    assert masked["data"]["mask"]["seed"] == 40


def test_gpu_request_is_capped_at_half():
    required = [
        "--config", "config.yaml", "--model", "model", "--search-path", "search",
        "--output-dir", "out", "--table", "table.csv", "--prefix", "prefix",
    ]
    assert parse_args(required).gpus_per_task == 0.5


def test_input_only_masks_inputs_keeps_real_targets_and_full_loss_and_noise_ranges_are_overridable():
    base = {"data": {"value": 1}}
    region = {"band_id": "R3", "start_mhz": 622.5, "end_mhz": 641.5}
    only = regional_config(base, region, "input_only", 41, noise_ranges=[[2400.5, 2496.5]])
    assert "loss_frequency_ranges" not in only["data"]
    assert only["data"]["mask"]["real_targets"] is True
    assert only["data"]["mask"]["calibration_frequency_ranges"] == [[2400.5, 2496.5]]
    assert regional_config(base, region, "region_only", 41)["data"]["mask"]["real_targets"] is False


def test_region_only_fullloss_masks_inputs_and_targets_with_the_loss_over_all_bins_and_leaves_the_other_conditions_unchanged():
    base = {"data": {"value": 1}}
    region = {"band_id": "R3", "start_mhz": 622.5, "end_mhz": 641.5}
    v2 = regional_config(base, region, "region_only_fullloss", 42, noise_ranges=[[2400.5, 2496.5]])
    assert "loss_frequency_ranges" not in v2["data"]
    assert v2["data"]["mask"]["real_targets"] is False                      # targets are the masked (noise-floor) series
    assert v2["data"]["mask"]["frequency_ranges"] == [[622.5, 641.5]]
    assert v2["data"]["mask"]["calibration_frequency_ranges"] == [[2400.5, 2496.5]]
    assert v2["data"]["mask"]["seed"] == 42
    # the existing conditions keep their meaning
    ro = regional_config(base, region, "region_only", 42)
    assert ro["data"]["loss_frequency_ranges"] == [[622.5, 641.5]] and ro["data"]["mask"]["real_targets"] is False
    assert regional_config(base, region, "input_only", 42)["data"]["mask"]["real_targets"] is True
    assert "mask" not in regional_config(base, region, "full_context", 42)["data"]
