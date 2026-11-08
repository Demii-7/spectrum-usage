import numpy as np

from training.ray import tsslcd_firstcommit as F


def test_window_ranges_reproduce_the_documented_split_and_overlap():
    train, test = F.window_ranges(6838)               # the recovered series
    assert len(test) == 3017                          # n_test in the collaborator's metrics.json
    assert test.stop == 6838 - 60 and train.stop == int((6838 - 60) * 0.8)
    overlap = len(set(train) & set(test)) / len(test)
    assert abs(overlap - 0.55) < 0.01                 # "55% of test windows are also training windows"


def test_authors_average_is_np_convolve_same_and_identity_for_window_one():
    x = np.random.default_rng(0).normal(size=(20, 3))
    assert F.authors_average(x, 1) is x
    out = F.authors_average(x, 6)
    np.testing.assert_allclose(out[:, 1], np.convolve(x[:, 1], np.ones(6) / 6, mode="same"))
    assert abs(out[0, 0]) < abs(x[:3, 0]).sum()       # zero padding at the edge, as in their code


def test_prepare_off_returns_the_unfiltered_normalized_series_and_on_filters_it():
    rng = np.random.default_rng(1)
    source = (rng.normal(size=(300, 4)) + np.arange(300)[:, None] * 0.01).astype(np.float32)
    scaler, norm, y_off = F.prepare(source, "off")
    np.testing.assert_allclose(y_off, norm)
    assert norm.min() >= -1e-6 and norm.max() <= 1 + 1e-6      # min-max over the whole series
    _, _, y_on = F.prepare(source, "on")
    assert not np.allclose(y_on, norm) and y_on.std() < norm.std()


def test_score_grid_has_both_targets_and_both_averaging_settings_and_zero_error_for_a_perfect_raw_match():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(200, 3)).astype(np.float32)
    scaler, norm, y = F.prepare(source, "on")
    raw_gt = norm[60:160]
    grid = F.score_grid(raw_gt, y[60:160], raw_gt, scaler, persistence_inputs=(norm[59:159], norm[60:160]))
    assert set(grid) == {"authors_protocol__model_target__averaging_on", "model_target__averaging_off",
                         "raw_target__averaging_on", "raw_target__averaging_off", "persistence__raw_target"}
    assert grid["raw_target__averaging_off"]["mse"] < 1e-9           # predictions equal the raw ground truth
    assert grid["model_target__averaging_off"]["mse"] > 1e-6         # the filtered target differs from the raw one


def test_score_steps_alignment_and_persistence():
    import numpy as np

    from training.ray.tsslcd_authors import MinMax
    from training.ray.tsslcd_firstcommit import C, FH, score_steps

    rng = np.random.default_rng(0)
    norm = np.cumsum(rng.normal(size=(300, 5)), axis=0).astype(np.float32)
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    scaler = MinMax().fit(norm)
    starts = np.arange(20, 120)
    perfect = np.stack([norm[starts + C + k] for k in range(FH)], axis=1)
    steps = score_steps(perfect, norm, norm, starts, scaler)
    assert list(steps) == [str(k) for k in range(1, FH + 1)]
    assert all(steps[k]["raw_target__averaging_off"]["mse"] < 1e-9 for k in steps)         # prediction = the right future row at every step
    persistence = [steps[str(k)]["persistence__raw_target"]["mse"] for k in range(1, FH + 1)]
    assert persistence[-1] > persistence[0] > 0                                           # holding the last value gets worse further out
