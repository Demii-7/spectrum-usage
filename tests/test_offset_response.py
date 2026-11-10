import numpy as np

from training.ray.offset_response import HORIZONS, response_rows


def _forecasts(shape, fn):
    rng = np.random.default_rng(0)
    base = {h: rng.normal(size=shape) for h in HORIZONS}
    return base, {h: fn(base[h]) for h in HORIZONS}


def test_a_model_that_carries_the_level_has_gain_one():
    base, shifted = _forecasts((50, 8), lambda x: x + 10.0)
    for row in response_rows(base, shifted, 10.0):
        assert abs(row["gain"] - 1.0) < 1e-9 and row["gain_sd"] < 1e-9 and row["share_within_0p5_db"] == 1.0


def test_a_model_that_ignores_the_level_has_gain_zero():
    base, shifted = _forecasts((50, 8), lambda x: x.copy())
    for row in response_rows(base, shifted, -20.0):
        assert row["gain"] == 0.0 and row["share_within_0p5_db"] == 0.0


def test_a_saturating_model_follows_small_shifts_and_flattens_for_large_ones():
    clip = lambda c: (lambda x: x + np.clip(c, -5.0, 5.0))
    for offset, expected in ((5.0, 1.0), (20.0, 0.25), (-20.0, 0.25)):
        base, shifted = _forecasts((30, 4), clip(offset))
        assert all(abs(r["gain"] - expected) < 1e-9 for r in response_rows(base, shifted, offset))


def test_negative_offsets_keep_their_sign():
    base, shifted = _forecasts((10, 3), lambda x: x - 5.0)
    row = response_rows(base, shifted, -5.0)[0]
    assert abs(row["gain"] - 1.0) < 1e-9 and abs(row["mean_shift_db"] + 5.0) < 1e-9
