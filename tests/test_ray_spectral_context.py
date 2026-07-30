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
