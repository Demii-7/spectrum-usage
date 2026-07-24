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
