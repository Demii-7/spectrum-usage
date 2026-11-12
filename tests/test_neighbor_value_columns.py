import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("boto3")
neighbor_value = pytest.importorskip("neighbor_value")


def test_add_extra_keeps_existing_columns_and_adds_suffixed_ones():
    table = pd.DataFrame({"band": ["a", "a"], "model": ["m", "m"], "mode": ["x", "x"], "receiver": ["r1", "pooled"], "mae_real": [1.0, 2.0]})
    extra = pd.DataFrame({"band": ["a", "a"], "model": ["m", "m"], "mode": ["x", "x"], "receiver": ["r1", "pooled"], "mae_real": [1.5, 2.5],
                          "ci_low": [0.1, 0.2], "ci_high": [0.3, 0.4], "n_origins": [10, 20]})
    out = neighbor_value.add_extra(table, extra, ["band", "model", "mode"], {"mae_real": "mae_real"}, "mae", "_h3")
    assert out["mae_real"].tolist() == [1.0, 2.0]                                   # the existing column keeps its meaning
    assert out["mae_real_h3"].tolist() == [1.5, 2.5] and out["ci_low_h3"].tolist() == [0.1, 0.2] and out["n_origins_h3"].tolist() == [10, 20]
    mse = neighbor_value.add_extra(table, extra.rename(columns={"mae_real": "mse_real"}), ["band", "model", "mode"], {"mse_real": "mse_real"}, "mse", "")
    assert mse["mse_real"].tolist() == [1.5, 2.5] and "ci_low_mse" in mse.columns


def test_horizon_sets():
    assert neighbor_value.H3 == (4, 12, 24) and neighbor_value.HORIZONS == (4, 12, 24, 48)
