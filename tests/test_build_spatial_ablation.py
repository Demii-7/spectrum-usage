from evaluation.analysis.build_spatial_ablation import paired_deltas


def _row(condition, mae, permutation_seed=""):
    return {
        "model": "m",
        "seed": "42",
        "horizon": "15",
        "receiver": "web",
        "region_id": "R5",
        "condition": condition,
        "permutation_seed": permutation_seed,
        "is_noise_floor": "False",
        "n_origins": "10",
        "mae_db": str(mae),
    }


def test_paired_deltas_match_same_model_seed_horizon_receiver_and_region():
    rows = [_row("geometry_correct", 1.25), _row("geometry_permuted", 1.5, "10000")]
    result = paired_deltas(rows, "geometry_permuted", "geometry_correct")
    assert len(result) == 1
    assert result[0]["delta_mae_db"] == 0.25
    assert result[0]["permutation_seed"] == "10000"
