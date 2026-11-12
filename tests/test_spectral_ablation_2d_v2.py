import numpy as np
import pandas as pd
import pytest

v2 = pytest.importorskip("spectral_ablation_2d_v2", reason="analysis module needs boto3 via training.ray.storage")


def test_region_mae_is_the_bin_mean_averaged_over_the_four_horizons():
    rows = [{"frequency_mhz": f, "horizon": h, "mae_db": float(h) + (0.0 if f < 602 else 10.0)} for f in (600.5, 601.5, 602.5) for h in v2.HORIZONS]
    table = pd.DataFrame(rows)
    region = type("R", (), {"start_mhz": 600.0, "end_mhz": 601.9})()
    assert v2.region_mae(table, region) == pytest.approx(np.mean(v2.HORIZONS))
    missing = table[table.horizon != 48]
    assert np.isnan(v2.region_mae(missing, region))


def test_summary_counts_regions_and_uses_the_agreed_source_value():
    by = pd.DataFrame({"band": "600_800", "source": v2.SOURCE, "model": "vanillalstm", "region": ["R1", "R3", "R4"], "width_mhz": [8.0, 20.0, 10.0],
                       "full_context": [1.0, 1.2, 1.1], "region_only": [1.1, 1.1, 1.3], "delta": [0.1, -0.1, 0.2], "n_seeds": 1,
                       "ref_revinmlp_full": 0.9, "ref_1d_final": np.nan, "note": "n"})
    s = v2.summarize(by).iloc[0]
    assert s.source == "2d_final_vs_region" and s.regions == 3 and s.regions_context_helps == 2 and s.regions_context_hurts == 1
    assert s.mean_delta == pytest.approx(0.2 / 3)
