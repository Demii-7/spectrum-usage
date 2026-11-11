"""Pure-function checks for the 1-minute offset test (the forward pass itself is tested on the cluster)."""
from training.ray.offset_response_1min import STRIDE, out_key, resources_for


def test_out_key_separates_rounds_and_horizons():
    base = {"band": "600_800", "rep": "2d", "model": "lstmattn", "seed": 40, "prefix": "checkpoints-oct2026"}
    assert out_key("p/", base) == "p/600_800/2d/lstmattn/seed-40"
    assert out_key("p", {**base, "prefix": "checkpoints-oct2026-r2"}) == "p/600_800/2d/lstmattn__r2/seed-40"
    direct = {"band": "2400_2600", "rep": "2d", "model": "revinmlp", "seed": 41, "source": "direct", "horizon": 15}
    assert out_key("p", direct) == "p/2400_2600/2d/revinmlp/h15/seed-41"


def test_resources_small_neural_models_get_a_small_gpu_share_everything_else_cpu():
    assert resources_for("lstmattn", False) == 0.125
    assert resources_for("linearar2d", False) == 0.0
    assert resources_for("revinmlp", True) == 0.0
    assert STRIDE["1d"] > STRIDE["2d"]
