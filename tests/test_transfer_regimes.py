import numpy as np
import pandas as pd
import pytest

from training.common import transfer_regimes as R


def test_intrinsic_classes_and_regime_flags():
    assert R.intrinsic_class("revinmlp") == "full" and R.intrinsic_class("timellm") == "full"
    assert R.intrinsic_class("patchtst_noinorm") == "none" and R.intrinsic_class("nlinear") == "none"
    assert R.intrinsic_class("residuallinearar2d") == "partial"
    assert R.regime_flags("linearar2d", "A") == (False, None)
    assert R.regime_flags("linearar2d", "B") == (True, None)
    assert R.regime_flags("revinmlp", "B") == (False, None)          # RevIN is not centered a second time
    assert R.regime_flags("linearar2d", "C") == (False, "mean_std")
    assert R.regime_flags("linearar2d", "Cm") == (False, "mean")
    with pytest.raises(ValueError):
        R.regime_flags("linearar2d", "D")


def test_calibration_mask_is_strictly_before_the_first_target():
    stamps = pd.date_range("2026-10-26T20:00:00Z", periods=200, freq="15min")
    start = stamps[96]
    mask = R.calibration_mask(stamps, start)
    assert mask.sum() == 96 and mask[:96].all() and not mask[96:].any()


def test_refit_uses_only_calibration_rows_and_floors_the_std():
    rng = np.random.default_rng(0)
    data = rng.normal(-80, 2, size=(200, 5))
    data[:96, 0] = -75.0                                              # a constant (zero-variance) bin in the calibration interval
    data[96:, :] += 100.0                                             # anything after the calibration interval must not matter
    mask = np.arange(200) < 96
    base = {"mean_dbm": np.full((1, 5), -80.0), "std_dbm": np.full((1, 5), 2.0), "other": "kept"}
    full = R.refit_normalization(base, data, mask, "mean_std")
    assert full["other"] == "kept" and full["mean_dbm"].shape == (1, 5)
    assert np.isclose(full["mean_dbm"][0, 0], -75.0)
    assert np.isclose(full["std_dbm"][0, 0], 0.25 * 2.0)              # floor
    assert np.all(np.abs(full["mean_dbm"][0, 1:] + 80) < 2)                  # the +100 dB after the interval was not used
    mean_only = R.refit_normalization(base, data, mask, "mean")
    assert np.array_equal(mean_only["std_dbm"], base["std_dbm"]) and not np.array_equal(mean_only["mean_dbm"], base["mean_dbm"])


def test_refit_rejects_short_or_nonfinite_calibration():
    base = {"mean_dbm": np.zeros(3), "std_dbm": np.ones(3)}
    with pytest.raises(ValueError):
        R.refit_normalization(base, np.zeros((10, 3)), np.ones(10, dtype=bool), "mean")
    data = np.zeros((100, 3))
    data[5, 1] = np.nan
    with pytest.raises(ValueError):
        R.refit_normalization(base, data, np.ones(100, dtype=bool), "mean")
    with pytest.raises(ValueError):
        R.refit_normalization(base, np.zeros((100, 4)), np.ones(100, dtype=bool), "mean")
