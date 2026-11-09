import pytest
import torch

from models.LookbackMean import LookbackMeanForecaster
from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster


def _config(**extra):
    return {"model": {"input_sequence_length": 6, "prediction_horizon": 1, "input_size": 2, **extra}}


def test_lookback_mean_defaults_unchanged_and_median_window():
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]]])
    assert LookbackMeanForecaster(_config())(x).item() == pytest.approx(x.mean().item())
    assert LookbackMeanForecaster(_config(statistic="median", window=5))(x).item() == 4.0     # newest five: 2, 3, 4, 5, 100
    assert LookbackMeanForecaster(_config(statistic="median", window=3))(x).item() == 5.0     # 4, 5, 100
    assert LookbackMeanForecaster(_config(statistic="median"))(x).item() == 3.0               # lower median of six values
    with pytest.raises(ValueError):
        LookbackMeanForecaster(_config(window=7))


def test_residual_median_starts_as_median_and_is_level_equivariant():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 2)
    model = ResidualLinearAutoregressiveForecaster(_config(baseline="median", baseline_window=4, ridge_alpha=0.0))
    out = model(x)                                              # head starts at zero: exactly the median of the newest four
    assert torch.allclose(out[:, 0], x[:, -4:].median(dim=1).values)
    with torch.no_grad():
        model.residual.weight.normal_()
    shifted = model(x + 7.0)
    assert torch.allclose(shifted, model(x) + 7.0, atol=1e-4)    # shifting the window shifts the forecast by the same amount


def test_default_residual_model_is_unchanged():
    x = torch.randn(3, 6, 2)
    model = ResidualLinearAutoregressiveForecaster(_config(ridge_alpha=0.0))
    assert torch.allclose(model(x)[:, 0], x.mean(dim=1))


def test_pick_k_selects_on_validation_only():
    from training.ray.median_study import pick_k

    rows = [{"statistic": "median", "k": 5, "range": "validation", "mae_t1": 1.0, "mae_t15": 2.0},
            {"statistic": "median", "k": 30, "range": "validation", "mae_t1": 1.2, "mae_t15": 1.4},
            {"statistic": "median", "k": 5, "range": "test", "mae_t1": 0.1, "mae_t15": 0.1},       # test rows never influence the pick
            {"statistic": "mean", "k": 60, "range": "validation", "mae_t1": 0.0, "mae_t15": 0.0}]
    assert pick_k(rows, "median", (1, 15)) == 30
    assert pick_k(rows, "median", (1, 15), per_horizon=True) == {1: 5, 15: 30}


def test_train_mean_ewma_and_persistence():
    x = torch.tensor([[[1.0, 10.0], [2.0, 10.0], [3.0, 10.0], [4.0, 10.0]]])          # (1, 4, 2)
    assert torch.all(LookbackMeanForecaster(_config(statistic="train_mean"))(x) == 0)
    assert torch.allclose(LookbackMeanForecaster(_config(statistic="mean", window=1))(x)[0, 0], x[0, -1])    # persistence is K = 1
    cfg = {"model": {"input_sequence_length": 4, "prediction_horizon": 1, "statistic": "ewma", "gain": 1.0}}
    assert torch.allclose(LookbackMeanForecaster(cfg)(x)[0, 0], x[0, -1])                                   # gain 1 is persistence
    cfg["model"]["gain"] = 1e-6                                                                              # gain -> 0 is the plain mean
    assert torch.allclose(LookbackMeanForecaster(cfg)(x)[0, 0], x.mean(dim=1)[0], atol=1e-3)
    cfg["model"]["gain"] = [1.0, 1e-6]                                                                       # per-bin gains
    out = LookbackMeanForecaster(cfg)(x)[0, 0]
    assert abs(out[0].item() - 4.0) < 1e-4 and abs(out[1].item() - 10.0) < 1e-3
    with pytest.raises(ValueError):
        LookbackMeanForecaster({"model": {"input_sequence_length": 4, "prediction_horizon": 1, "statistic": "ewma"}})


def test_baseline_table_selects_on_validation_per_horizon():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
    from baseline_table import table

    def row(rng, k, m1, m4):
        return {"statistic": "median", "k": k, "range": rng, "mae_t1": m1, "mae_t4": m4,
                "metrics": {"test_mae_db_t1": m1, "test_mse_db2_t1": 2 * m1, "test_rmse_db_t1": 1.0, "test_mape_linear_pct_t1": 3.0,
                            "test_mape_dbm_pct_t1": 4.0, "test_mae_db_t4": m4, "test_mse_db2_t4": 2 * m4, "test_rmse_db_t4": 1.0,
                            "test_mape_linear_pct_t4": 3.0, "test_mape_dbm_pct_t4": 4.0}}
    rows = [row("validation", 5, 1.0, 2.0), row("validation", 30, 1.5, 1.2), row("test", 5, 0.9, 2.1), row("test", 30, 1.4, 1.1)]
    frame = table({("15min", "600_800", "median"): {"rows": rows}})
    per = frame[(frame.selection == "per_horizon") & (frame.family == "median")].set_index("horizon")
    assert per.loc[1, "parameter_key"] == 5 and per.loc[4, "parameter_key"] == 30          # each horizon picks its own validation winner
    assert per.loc[1, "mae_db"] == 0.9 and per.loc[4, "mae_db"] == 1.1                        # test numbers at the chosen parameter
