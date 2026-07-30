import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


STS = Path(__file__).resolve().parents[1] / "training" / "STS-PredNet"
sys.path.insert(0, str(STS))

from dataset import (  # noqa: E402
    STSPredNetDataset,
    generate_target_indices,
    required_history,
    resolve_branch_config,
)
from stsprednet import STSPredNet  # noqa: E402
from train_integrated import to_sts_layout, validation_with_training_context  # noqa: E402
from evaluate_integrated import rollout_required_history  # noqa: E402
from linear_ar_baseline import LagMatchedLinearAR, predict_recursive  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "evaluate_daily_history", STS / "evaluate_daily_history.py",
)
_daily_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_daily_evaluation)


def branches(enabled):
    return {
        "use_closeness": "closeness" in enabled,
        "use_period": "period" in enabled,
        "use_trend": "trend" in enabled,
        "lc": 3, "lp": 2, "lq": 2,
        "period_interval": 5, "trend_interval": 7,
        "prediction_offset": 2,
    }


@pytest.mark.parametrize("enabled", [
    set(combo)
    for size in range(1, 4)
    for combo in itertools.combinations(("closeness", "period", "trend"), size)
])
def test_dataset_supports_every_nonempty_branch_combination(enabled):
    b = branches(enabled)
    target = required_history(b)
    data = np.arange(30, dtype=np.float32).reshape(30, 1, 1)
    ds = STSPredNetDataset(data, [target], b["use_closeness"], b["use_period"],
                           b["use_trend"], b["lc"], b["lp"], b["lq"],
                           b["period_interval"], b["trend_interval"],
                           b["prediction_offset"])
    sample = ds[0]
    assert enabled == ({"closeness", "period", "trend"} & sample.keys())
    if "period" in enabled:
        assert sample["period"][:, 0, 0, 0].tolist() == [target - 10, target - 5]
    if "trend" in enabled:
        assert sample["trend"][:, 0, 0, 0].tolist() == [target - 14, target - 7]
    assert generate_target_indices(30, 2, b["use_closeness"], b["use_period"],
                                   b["use_trend"], 3, 2, 2, 5, 7)[0] == target


def test_resolution_rejects_or_explicitly_drops_unavailable_branches():
    b = branches({"closeness", "trend"})
    with pytest.raises(ValueError, match="trend"):
        resolve_branch_config(b, available_length=10)
    b["allow_unavailable_branches"] = True
    resolved = resolve_branch_config(b, available_length=10)
    assert resolved["enabled_branches"] == ["closeness"]
    assert resolved["unavailable_branches"] == ["trend"]


def test_model_requires_exactly_the_configured_inputs():
    config = {
        "model": {"input_channels": 1, "map_height": 1, "map_width": 2,
                  "num_layers": 1, "hidden_dim": 2, "kernel_size": [1, 1],
                  "output_activation": "linear", "fusion_weight_shape": "per_branch"},
        "branches": {"use_closeness": False, "use_period": True,
                     "use_trend": False, "share_branch_weights": False},
    }
    model = STSPredNet(config)
    period = torch.zeros(1, 2, 1, 1, 2)
    assert model(None, period, None).shape == (1, 1, 1, 2)
    with pytest.raises(ValueError, match="period"):
        model(None, None, None)
    with pytest.raises(ValueError, match="closeness"):
        model(period, period, None)


def test_validation_can_use_training_history_without_training_targets():
    train = np.arange(8, dtype=np.float32).reshape(8, 1)
    validation = np.arange(8, 11, dtype=np.float32).reshape(3, 1)
    combined, targets = validation_with_training_context(
        train, validation, (), history=4,
    )
    assert combined[:, 0].tolist() == [4, 5, 6, 7, 8, 9, 10]
    assert targets.tolist() == [4, 5, 6]


def test_common_map_layout_is_converted_to_frequency_first():
    common = np.zeros((2, 3, 4, 5), dtype=np.float32)
    assert to_sts_layout(common).shape == (2, 5, 3, 4)


def test_recursive_daily_branches_include_horizon_in_required_history():
    b = branches({"closeness", "period"})
    b["lc"] = 60
    b["lp"] = 2
    b["period_interval"] = 1440
    assert rollout_required_history(b, 1) == 2880
    assert rollout_required_history(b, 480) == 3359


def test_lag_matched_ar_recursion_uses_generated_recent_frames():
    model = LagMatchedLinearAR((1, 1, 1), n_lags=2)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[1].fill_(1.0)
        model.bias.zero_()
    data = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
    prediction = predict_recursive(
        model, data, np.array([8]), horizon=3, lags=[2, 1],
        device=torch.device("cpu"),
    )
    assert prediction[:, 0, 0, 0].tolist() == [5.0]


def test_daily_evaluator_keeps_recursive_state_per_origin():
    model = LagMatchedLinearAR((1, 1, 1), n_lags=2)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[1].fill_(1.0)
        model.bias.zero_()
    start = np.datetime64("2026-07-03T00:00")
    history = {
        start + np.timedelta64(index, "m"): np.full((1, 1, 1), index, dtype=np.float32)
        for index in range(12)
    }
    prediction = _daily_evaluation.ar_predict(
        model, torch.device("cpu"), history, [start + np.timedelta64(8, "m"), start + np.timedelta64(9, "m")],
        horizon=3, lags=[2, 1],
    )
    assert prediction[:, 0, 0, 0].tolist() == [5.0, 6.0]


def test_daily_evaluator_preflight_requires_observed_recent_t6_context():
    target = np.datetime64("2026-07-03T18:00")
    history = {
        target + np.timedelta64(index, "m"): np.zeros((1, 1, 1), dtype=np.float32)
        for index in range(-3000, 1)
    }
    recent_times = {target + np.timedelta64(index, "m") for index in range(-119, 1)}
    valid = _daily_evaluation.valid_origins(
        [target], history, horizon=60, lags=[60, 2, 1, 2880, 1440],
        recent_lags={1, 2, 60}, recent_times=recent_times,
    )
    assert valid == [target]
    recent_times.remove(target - np.timedelta64(119, "m"))
    assert not _daily_evaluation.valid_origins(
        [target], history, horizon=60, lags=[60, 2, 1, 2880, 1440],
        recent_lags={1, 2, 60}, recent_times=recent_times,
    )
