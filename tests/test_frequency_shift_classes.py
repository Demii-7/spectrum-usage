import numpy as np

from training.common.frequency_shift import (class_bin_sums, disturb_series, disturbed_mask, level_windows, shift_other_bins)


def test_disturbed_masks():
    in_region = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    classes = np.array(["P", "N", "P", "N", "noise", "middle"])
    assert disturbed_mask(in_region, classes, "all").tolist() == [True, True, False, False, True, True]
    assert disturbed_mask(in_region, classes, "P_only").tolist() == [True, False, False, False, False, False]
    assert disturbed_mask(in_region, classes, "N_only").tolist() == [False, True, False, False, False, False]
    assert disturbed_mask(in_region, classes, "noise_only").tolist() == [False, False, False, False, True, False]


def test_region_bins_never_disturbed_and_all_matches_existing_shift():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(300, 6))
    in_region = np.array([0, 1, 1, 0, 0, 0], dtype=bool)
    classes = np.array(["P", "P", "N", "N", "noise", "middle"])
    mask = disturbed_mask(in_region, classes, "all")
    new = disturb_series(data, mask, "shift96")
    assert np.array_equal(new, shift_other_bins(data, in_region, 96))          # `all` reproduces the existing ablation
    assert np.array_equal(new[:, in_region], data[:, in_region])
    only_p = disturb_series(data, disturbed_mask(in_region, classes, "P_only"), "shift144")
    assert np.array_equal(only_p[:, 1:], data[:, 1:]) and not np.array_equal(only_p[:, 0], data[:, 0])
    assert np.array_equal(only_p[:, 0], np.roll(data[:, 0], 144))


def test_level_windows():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(5, 96, 6))
    disturbed = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    out = level_windows(windows, disturbed)
    assert np.allclose(out[:, :, 0], windows[:, :, 0].mean(axis=1, keepdims=True)) and np.allclose(out[:, :, 0].std(axis=1), 0)
    assert np.array_equal(out[:, :, ~disturbed], windows[:, :, ~disturbed])
    assert level_windows(windows, np.zeros(6, dtype=bool)).tolist() == windows.tolist()


def test_class_bin_sums():
    err = np.abs(np.arange(12, dtype=float).reshape(2, 6))
    in_region = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    classes = np.array(["P", "N", "P", "middle", "P", "N"])
    s = class_bin_sums(err, in_region, classes)
    assert s["P"].tolist() == [0 + 2, 6 + 8] and s["N"].tolist() == [1, 7] and s["middle"].tolist() == [3, 9]
