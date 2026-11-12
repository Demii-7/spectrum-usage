"""Check accounting boundaries that could overstate the appendix compute bill."""

import pytest

from evaluation.analysis.appendix_costs import BACKUP, DIRECT, core_summary, direct_summary


def test_core_reused_search_seed_is_not_charged_for_a_second_fit():
    root = "checkpoints-oct2026/"
    prefix = root + "1d/vanillalstm1d/"
    records = [{"key": prefix + "MANIFEST.json", "data": {"entries": [
        {"seed": seed, "metrics": {"cost_from_hpo": seed == 42, "objective": 1.0}} for seed in range(40, 45)
    ]}}]
    for seed in range(40, 45):
        records.append({"key": prefix + f"seed-{seed}/chunk_run_costs.json",
                        "data": {"cost_training_duration_sec": 3600}})
    records.append({"key": BACKUP + "spectrum-usage/ray/oct2026-core4/1d/vanillalstm1d-search/t0/result.json",
                    "max_seconds": 3600, "data": {"done": True, "cost_ray_gpu_assigned": .5}})
    records.append({"key": BACKUP + "spectrum-usage/ray/oct2026-core4/1d/vanillalstm1d-replicate/t0/result.json",
                    "max_seconds": 60, "data": {"config": {"seed": 42}}})
    result = core_summary("1d", "vanillalstm1d", records)
    # Search paid for seed 42; its replicate task only evaluated the checkpoint.
    assert result["allocated_hours_lower_bound"] == pytest.approx((3600 + 4 * 3600 + 60) * .5 / 3600)
    assert result["fit_minutes"] == 60
    assert result["fit_n"] == 5
    assert len(result["reused_hpo"]) == 1


def test_direct_cost_includes_all_horizons_and_failed_recorded_attempts():
    records = []
    for index in range(16):
        for horizon in (1, 60):
            records.append({"key": DIRECT + f"patchtst/trials/c{index}_h{horizon}.json",
                            "data": {"kind": "trial", "index": index, "horizon": horizon,
                                     "wall_seconds": 10, "ok": True}})
    for horizon in (1, 15, 60):
        for seed in range(40, 45):
            records.append({"key": DIRECT + f"patchtst/finals/h{horizon}/seed-{seed}.json",
                            "data": {"kind": "final", "seed": seed, "horizon": horizon,
                                     "wall_seconds": 20, "ok": True,
                                     "run_costs": {"cost_training_duration_sec": 10, "cost_ray_gpu_assigned": .25}}})
    result = direct_summary("patchtst", records)
    assert result["allocated_hours_lower_bound"] == pytest.approx((32 * 10 + 15 * 20) * .25 / 3600)
    assert result["grid_complete"]
    assert result["fit_n"] == 5
    records[-1]["data"]["ok"] = False
    failed = direct_summary("patchtst", records)
    assert not failed["grid_complete"]
    assert failed["fit_n"] == 4
    assert failed["allocated_hours_lower_bound"] == result["allocated_hours_lower_bound"]


def test_consolidated_winner_uses_validation_not_latest_round():
    records = []
    for root, objective, duration in (("checkpoints-oct2026/", 1.0, 60),
                                       ("checkpoints-oct2026-r2/", 2.0, 120)):
        prefix = root + "2d/vanillalstm/"
        records.append({"key": prefix + "MANIFEST.json", "data": {"entries": [
            {"seed": 42, "metrics": {"objective": objective}}
        ]}})
        for seed in range(40, 45):
            records.append({"key": prefix + f"seed-{seed}/chunk_run_costs.json",
                            "data": {"cost_training_duration_sec": duration}})
    result = core_summary("2d", "vanillalstm", records)
    assert result["winner_prefix"] == "checkpoints-oct2026/"
    assert result["fit_minutes"] == 1
    # Both rounds' five final fits still count towards compute spent.
    assert result["allocated_hours_lower_bound"] == pytest.approx(5 * (60 + 120) * .5 / 3600)
