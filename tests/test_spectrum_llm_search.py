"""The three LLM arms must be searchable through training/grid_search, not a parallel script.

grid_search.py dispatches on model_name at ten sites and raises for anything it does not
know, so "the model is registered" is not the same as "the search can run it". These tests
pin the wiring and, more importantly, check the shipped search configs against the keys the
injector actually honours -- a mistyped key would otherwise produce a results.csv full of
trials that all silently ran the default architecture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {
    "spectrumllm": REPO / "training/grid_search/config_spectrumllm.yaml",
    "ofa": REPO / "training/grid_search/config_ofa.yaml",
}

gsearch = pytest.importorskip("training.grid_search.grid_search")


@pytest.mark.parametrize("name", ["spectrumllm", "ofa"])
def test_every_search_stage_is_wired(name: str) -> None:
    hp = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))["hyperparameter_search"]
    smoke = gsearch._build_smoke_grid(name)
    arch = gsearch._build_architecture_combos(hp, name)
    opt = gsearch._build_optimization_combos(hp, name, arch[:2])
    assert smoke and arch and opt
    assert len(opt) == 2 * len(hp["optimization_search"]["candidates"])
    # every stage must be describable, which is the tenth dispatch site
    assert gsearch._smoke_trial_description(smoke[0], name)


@pytest.mark.parametrize("name", ["spectrumllm", "ofa"])
def test_shipped_search_config_uses_only_honoured_keys(name: str) -> None:
    cfg = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))
    assert cfg["training"]["model_name"] == name
    allowed = set(gsearch._LLM_MODEL_KEYS[name]) | gsearch._LLM_TRAIN_KEYS
    hp = cfg["hyperparameter_search"]
    sources = (
        [hp["architecture_search"]["fixed"]]
        + list(hp["architecture_search"]["candidates"])
        + list(hp["optimization_search"]["candidates"])
        + gsearch._build_smoke_grid(name)
    )
    for block in sources:
        unknown = set(block) - allowed
        assert not unknown, f"{name} search config carries unhonoured key(s) {sorted(unknown)}"


@pytest.mark.parametrize("name", ["spectrumllm", "ofa"])
def test_injection_reaches_the_model_and_train_sections(name: str) -> None:
    cfg = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))
    params = dict(gsearch._build_smoke_grid(name)[0])
    params.update({"epochs": 3, "seed": 7})
    gsearch._inject_spectrumllm_params(cfg, params, name)
    assert cfg[name]["train"]["epochs"] == 3
    assert cfg[name]["train"]["seed"] == 7
    assert cfg[name]["train"]["learning_rate"] == params["learning_rate"]
    assert cfg[name]["model"]["patch_length"] == params["patch_length"]
    # lookback drives the lookback, never the stale value in the config body
    assert cfg[name]["model"]["input_sequence_length"] == cfg["windowing"]["lookback"]


def test_ofa_rejects_reprogramming_keys() -> None:
    """OFAForecaster would ignore d_model, so a search over it must fail loudly."""
    cfg = yaml.safe_load(CONFIGS["ofa"].read_text(encoding="utf-8"))
    params = dict(gsearch._build_smoke_grid("ofa")[0])
    params["d_model"] = 32
    with pytest.raises(ValueError, match="d_model"):
        gsearch._inject_spectrumllm_params(cfg, params, "ofa")


def test_a_mistyped_key_is_not_silently_dropped() -> None:
    cfg = yaml.safe_load(CONFIGS["spectrumllm"].read_text(encoding="utf-8"))
    params = dict(gsearch._build_smoke_grid("spectrumllm")[0])
    params["d_moddel"] = 64
    with pytest.raises(ValueError, match="d_moddel"):
        gsearch._inject_spectrumllm_params(cfg, params, "spectrumllm")
