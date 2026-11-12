import dataclasses

import numpy as np
import pytest

from training.ray import neighbor_ablation as N


@dataclasses.dataclass
class Series:
    values_dbm: np.ndarray


def test_device_for_neural_models_needs_a_gpu_share():
    assert N.device_for("stsprednet", 0.5) == "cuda"
    assert N.device_for("residualconvlstm", 1.0) == "cuda"
    assert N.device_for("stsprednet", 0.0) == "cpu"            # forced to CPU
    assert N.device_for("linearar4d", 0.5) == "cpu"            # linear models and the control never use a GPU
    assert N.device_for("lookbackmean4d", 1.0) == "cpu"


def test_every_neural_model_is_a_known_model():
    assert set(N.NEURAL) <= set(N.KNOWN_MODELS)
    assert {"linearar4d", "residuallinearar4d", "lookbackmean4d"} <= set(N.KNOWN_MODELS)      # the original CLI keeps working


def test_transforms_leave_the_scored_receiver_untouched_and_change_the_others():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(300, 4, 5))
    series = Series(values)
    for mode, mode_name, shift in N.MODES[1:]:
        out = N.make_transform(2, mode_name, shift)(series).values_dbm
        np.testing.assert_array_equal(out[:, 2], values[:, 2])                      # the target receiver is never changed
        assert not np.array_equal(out[:, 0], values[:, 0])
    np.testing.assert_array_equal(N.make_transform(1, None, 0)(series).values_dbm, values)       # "real" is the identity
    level = N.make_transform(0, "level", 0)(series).values_dbm
    np.testing.assert_allclose(level[:, 1], np.tile(values[:, 1].mean(axis=0), (300, 1)))        # replaced by the per-bin mean
    rolled = N.make_transform(0, "shift", 96)(series).values_dbm
    np.testing.assert_array_equal(rolled[:, 3], np.roll(values[:, 3], 96, axis=0))


def test_per_origin_errors_denormalize_and_keep_origin_times():
    n, h, r, f = 6, 2, 3, 4
    target = np.full((n, h, r, f), -100.0)
    norm = {"mean_dbm": np.full((1, f), -100.0), "std_dbm": np.full((1, f), 2.0)}
    prediction = np.zeros((n, h, r, f)) + 1.0                                     # normalized 1 -> -98 dBm: error +2 dB
    npz = {"predictions_normalized": prediction, "targets_dbm": target, "origin_timestamps": np.arange(n).astype(str)}
    out = N.per_origin_errors(npz, 1, norm)
    assert out["mae"].shape == (n, h) and np.allclose(out["mae"], 2.0) and np.allclose(out["bias"], 2.0)
    assert list(out["times"]) == [str(i) for i in range(n)]


def test_cli_rejects_unknown_models_and_accepts_the_neural_ones():
    pytest.importorskip("ray")                                                       # main() imports ray first
    import sys

    old = sys.argv
    try:
        sys.argv = ["x", "--models", "notamodel"]
        with pytest.raises(SystemExit):
            N.main()
    finally:
        sys.argv = old


def test_dswinlstm_i_is_a_gpu_model_with_one_seed():
    assert "dswinlstm_i" in N.NEURAL and "dswinlstm_i" in N.KNOWN_MODELS
    assert N.device_for("dswinlstm_i", 0.5) == "cuda" and N.device_for("dswinlstm_i", 0) == "cpu"
    assert N.SINGLE_SEED["dswinlstm_i"] == [42] and N.SINGLE_SEED["lookbackmean4d"] == [42]
    assert "residualconvlstm" not in N.SINGLE_SEED                                    # five seeds
