from training.ray.registry import MODEL_REGISTRY
from training.ray.run import build_plan


def test_tcn_search_is_independent_only_with_twelve_epoch_grace_period():
    config = {
        "data": {
            "representation": "2d",
            "split": {"ranges": {
                "train": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"},
                "validation": {"start": "2026-01-03T00:00:00Z", "end": "2026-01-04T00:00:00Z"},
                "test": {"start": "2026-01-05T00:00:00Z", "end": "2026-01-06T00:00:00Z"},
            }},
        },
        "training": {"models": ["temporalconvnet"]},
        "temporalconvnet": {"model": {}, "train": {"epochs": 20}},
    }
    entry = build_plan(config, ["temporalconvnet"])["models"][0]
    assert entry["search"]["grace_period"] == 12

    spec = MODEL_REGISTRY["temporalconvnet"]
    modes = {
        bundle["model.feature_mode"]
        for bundle in spec.space["architecture"].choices
    }
    assert modes == {"independent"}
    assert all(
        parameters["architecture"]["model.feature_mode"] == "independent"
        for parameters in spec.historical.values()
    )
