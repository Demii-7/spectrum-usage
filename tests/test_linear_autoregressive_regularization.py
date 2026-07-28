from pathlib import Path

import pytest
import torch
import yaml

from models.LinearAutoregressive import LinearAutoregressiveForecaster
from models.ResidualLinearAutoregressive import (
    ResidualLinearAutoregressiveForecaster,
)


ROOT = Path(__file__).resolve().parents[1]
LINEAR_NAMES = {
    "linearar1d",
    "linearar2d",
    "linearar4d",
    "residuallinearar1d",
    "residuallinearar2d",
    "residuallinearar4d",
}


def model_config(input_size: int, ridge_alpha: float = 1.0) -> dict:
    return {"model": {
        "input_sequence_length": 3,
        "prediction_horizon": 1,
        "input_size": input_size,
        "ridge_alpha": ridge_alpha,
    }}


@pytest.mark.parametrize("shape", [(2, 3, 1), (2, 3, 4), (2, 3, 2, 2, 3)])
@pytest.mark.parametrize(
    "model_type",
    [LinearAutoregressiveForecaster, ResidualLinearAutoregressiveForecaster],
)
def test_linear_ar_variants_support_all_input_representations(shape, model_type):
    input_size = int(torch.tensor(shape[2:]).prod().item())
    model = model_type(model_config(input_size))
    prediction = model(torch.randn(*shape))
    assert prediction.shape == (shape[0], 1, *shape[2:])


@pytest.mark.parametrize("residualized", [False, True])
def test_ridge_penalty_regularizes_weights_but_not_bias(residualized):
    model = (
        ResidualLinearAutoregressiveForecaster(model_config(2, ridge_alpha=2.0))
        if residualized
        else LinearAutoregressiveForecaster(model_config(2, ridge_alpha=2.0))
    )
    head = model.residual if residualized else model
    with torch.no_grad():
        head.weight.fill_(2.0)
        head.bias.fill_(3.0)

    penalty = model.ridge_penalty()
    penalty.backward()

    assert penalty.item() == pytest.approx(24.0)
    assert torch.all(head.weight.grad != 0)
    assert head.bias.grad is None


def test_ridge_alpha_must_be_non_negative():
    with pytest.raises(ValueError, match="ridge_alpha must be non-negative"):
        LinearAutoregressiveForecaster(model_config(1, ridge_alpha=-1.0))


def test_integrated_linear_ar_configs_have_learning_rate_and_ridge_alpha():
    for path in (ROOT / "training" / "configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for model_name in LINEAR_NAMES & config.keys():
            section = config[model_name]
            assert section["model"]["ridge_alpha"] >= 0, (path, model_name)
            assert section["train"]["learning_rate"] > 0, (path, model_name)
            assert section["train"].get("weight_decay", 0.0) == 0.0, (
                path,
                model_name,
            )

    standalone = yaml.safe_load(
        (ROOT / "training" / "LinearAutoregressive" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert standalone["model"]["ridge_alpha"] >= 0
    assert standalone["train"]["learning_rate"] > 0
