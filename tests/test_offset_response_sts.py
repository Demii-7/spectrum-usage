import dataclasses

import numpy as np

from training.ray.offset_response_sts import make_shift, receiver_arrays, to_dbm


@dataclasses.dataclass
class _Series:
    values_dbm: np.ndarray
    names: tuple = ("a", "b")


def test_shift_adds_a_constant_to_every_receiver_without_mutating_the_input():
    series = _Series(np.zeros((5, 2, 3)))
    out = make_shift(7.5)(series)
    assert np.all(out.values_dbm == 7.5) and np.all(series.values_dbm == 0.0) and out.names == series.names


def test_to_dbm_applies_the_checkpoint_statistics_per_bin_and_is_the_identity_without_them():
    pred = np.ones((4, 2, 3, 5))
    norm = {"mean_dbm": np.arange(5.0).reshape(1, 5), "std_dbm": np.full((1, 5), 2.0)}
    out = to_dbm(pred, norm)
    assert out.shape == pred.shape and np.allclose(out[0, 0, 0], 2.0 + np.arange(5.0))
    assert np.array_equal(to_dbm(pred, None), pred)


def test_a_level_carrying_model_has_gain_one_through_the_pipeline_helpers():
    from training.ray.offset_response import response_rows

    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 2, 3, 5))                         # (N, H, R, F)
    horizons = [4, 12]
    rows = response_rows(receiver_arrays(base, horizons, 1), receiver_arrays(base + 10.0, horizons, 1), 10.0, horizons)
    assert {r["horizon"] for r in rows} == {4, 12} and all(abs(r["gain"] - 1.0) < 1e-12 for r in rows)
