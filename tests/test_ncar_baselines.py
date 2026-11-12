"""The item-1 rolled-out median equals the pipeline's one-step lookbackmean2d (median) rolled out; same-time-yesterday indexing."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_rolled_median_matches_pipeline():
    from models.LookbackMean import LookbackMeanForecaster
    from training.common.forecasting import autoregressive_rollout
    from training.ray.ncar_baselines import baseline_forecasts, rolled_median

    rng = np.random.default_rng(1)
    data = rng.normal(size=(300, 3)).astype(np.float32)
    starts = np.arange(0, 100, 7)
    w = torch.from_numpy(data[starts[:, None] + np.arange(96)])
    for k in (2, 3, 6, 96):
        m = LookbackMeanForecaster({"model": {"prediction_horizon": 1, "input_sequence_length": 96, "statistic": "median", "window": k}})
        ref = autoregressive_rollout(m, w, 48).numpy()
        assert np.abs(ref - rolled_median(w.numpy(), k, 48)).max() == 0
    p = baseline_forecasts(data, starts, {"tuned_median": {1: 3, 4: 3, 12: 2, 24: 96, 48: 96}})
    for h in (1, 4, 12, 24, 48):
        assert np.array_equal(p["same_time_yesterday"][h], data[starts + h - 1])
        assert np.array_equal(p["persistence"][h], data[starts + 95])
