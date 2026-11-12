import numpy as np
import pytest

from training.common.frequency_shift import (band_frequencies, region_errors, region_mask, shift_other_bins,
                                              shift_other_bins_multi_receiver)


def make(t=300, f=20):
    rng = np.random.default_rng(0)
    return rng.normal(size=(t, f)).astype(np.float32)


def test_region_mask_inclusive_and_band_frequencies():
    freq = band_frequencies("600_800")
    assert freq[0] == 600.5 and freq[-1] == 799.5 and len(freq) == 200
    mask = region_mask(freq, 600.5, 607.5)
    assert mask.sum() == 8 and mask[:8].all()
    with pytest.raises(ValueError):
        region_mask(freq, 900.0, 901.0)


def test_shift_zero_is_identity_copy():
    x = make(); m = np.zeros(20, bool); m[3:7] = True
    y = shift_other_bins(x, m, 0)
    assert np.array_equal(x, y) and y is not x


def test_region_bins_untouched_and_other_bins_rolled():
    x = make(); m = np.zeros(20, bool); m[3:7] = True
    y = shift_other_bins(x, m, 96)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert np.array_equal(y[:, m], x[:, m])
    assert np.array_equal(y[96:, ~m], x[:-96, ~m])           # value at t is the value of t - 96
    assert not np.array_equal(y[:, ~m], x[:, ~m])


def test_wrap_around():
    x = make(); m = np.zeros(20, bool); m[0] = True
    y = shift_other_bins(x, m, 210)
    assert np.array_equal(y[:210, ~m], x[-210:, ~m])         # the first rows come from the end of the series


def test_input_not_modified():
    x = make(); before = x.copy(); m = np.zeros(20, bool); m[1] = True
    shift_other_bins(x, m, 96)
    assert np.array_equal(x, before)


def test_multi_receiver_same_bins_at_every_receiver():
    x = make(t=300, f=3 * 10); m = np.zeros(10, bool); m[2:4] = True
    y = shift_other_bins_multi_receiver(x, m, 96)
    for r in range(3):
        block_x, block_y = x[:, r * 10:(r + 1) * 10], y[:, r * 10:(r + 1) * 10]
        assert np.array_equal(block_y[:, m], block_x[:, m])
        assert np.array_equal(block_y[96:, ~m], block_x[:-96, ~m])
    with pytest.raises(ValueError):
        shift_other_bins_multi_receiver(make(t=10, f=25), m, 96)


def test_mask_shape_check():
    with pytest.raises(ValueError):
        shift_other_bins(make(), np.zeros(5, bool), 96)


def test_region_errors_only_region_bins():
    p = np.zeros((4, 6)); t = np.zeros((4, 6)); t[:, 4:] = 10.0
    m = np.array([1, 1, 0, 0, 0, 0], bool)
    assert np.allclose(region_errors(p, t, m), 0.0)
    m2 = np.array([0, 0, 0, 0, 1, 1], bool)
    assert np.allclose(region_errors(p, t, m2), 10.0)
