import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("boto3")
spatial_paired_all = pytest.importorskip("spatial_paired_all")


def fake_table(scale: float) -> pd.DataFrame:
    return pd.DataFrame({"band": ["600_800"], "model": ["stsprednet"], "versus": ["persistence"], "receiver": ["pooled"], "n_origins": [100],
                         "mae_model": [1.0 * scale], "mae_other": [2.0 * scale], "delta_model_minus_other": [-1.0 * scale],
                         "ci_low": [-1.5 * scale], "ci_high": [-0.5 * scale]})


def test_add_extra_adds_h3_and_mse_columns_and_keeps_the_existing_ones(monkeypatch):
    calls = []

    def fake_compare(client, metric="mae", horizons=(4, 12, 24, 48)):
        calls.append((metric, tuple(horizons)))
        return fake_table({"mae": 1.0, "mse": 10.0}[metric] * (0.5 if len(horizons) == 3 else 1.0))

    monkeypatch.setattr(spatial_paired_all, "compare", fake_compare)
    table = fake_table(1.0)
    out = spatial_paired_all.add_extra(table, client=None)
    assert out["mae_model"].iloc[0] == 1.0 and out["delta_model_minus_other"].iloc[0] == -1.0     # existing columns keep their meaning
    for column in ("mae_model_h3", "mae_other_h3", "delta_model_minus_other_h3", "ci_low_h3", "n_origins_h3",
                   "mse_model", "mse_other", "delta_mse", "ci_low_mse", "ci_high_mse", "n_origins_mse",
                   "mse_model_h3", "delta_mse_h3", "ci_low_mse_h3"):
        assert column in out.columns, column
    assert out["mae_model_h3"].iloc[0] == 0.5 and out["mse_model"].iloc[0] == 10.0 and out["mse_model_h3"].iloc[0] == 5.0
    assert calls == [("mae", (4, 12, 24)), ("mse", (4, 12, 24, 48)), ("mse", (4, 12, 24))]
