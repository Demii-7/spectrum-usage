import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "evaluation/analysis/spatial15_table.py"
spec = importlib.util.spec_from_file_location("spatial15_table", path)
T = importlib.util.module_from_spec(spec)
sys.modules["spatial15_table"] = T
spec.loader.exec_module(T)


def test_condition_labels_map_to_plain_names():
    assert T.parse_condition("geometry_correct") == ("insample", None)
    assert T.parse_condition("six_to_guesthouse-law73") == ("absent", "law73")
    assert T.parse_condition("seven_to_guesthouse-guesthouse") == ("added", "guesthouse")
    with pytest.raises(ValueError):
        T.parse_condition("six_to_new")


def test_keys_are_parsed_and_eval_files_ignored():
    P = T.PREFIX
    assert T.parse_key(P + "600_800/loo-moran/linearar4d/seed-41/result.json") == {"band": "600_800", "variant": "loo-moran", "model": "linearar4d",
                                                                                "seed": 41, "perm": None}
    assert T.parse_key(P + "600_800/permuted/linearar4d/seed-40/p10000/result.json")["perm"] == 10000
    assert T.parse_key(P + "600_800/core4-p2/linearar4d/seed-40/eval/geometry_correct/result.json") is None
    assert T.parse_key(P + "600_800/core4-p2/linearar4d/seed-40/weights/x.pt") is None


def test_flatten_and_tables_on_a_tiny_synthetic_result_set():
    P = T.PREFIX

    def cell(band, variant, model, seed, base, absent, added):
        return {"ok": True, "map__test_mean_horizon_mae_db": base / 2,
                "receivers": {"geometry_correct": {"mean_horizon_mae_db": base, "n_origins": 10},
                              "six_to_guesthouse-guesthouse": {"mean_horizon_mae_db": absent, "n_origins": 10},
                              "seven_to_guesthouse-guesthouse": {"mean_horizon_mae_db": added, "n_origins": 10},
                              "six_to_guesthouse-law73": {"mean_horizon_mae_db": absent + 1, "n_origins": 10},
                              "seven_to_guesthouse-law73": {"mean_horizon_mae_db": added + 1, "n_origins": 10}}}

    raw = {f"{P}600_800/core4-p2/linearar4d/seed-{s}/result.json": cell("600_800", "core4-p2", "linearar4d", s, 1.0 + 0.1 * (s - 40), 4.0, 2.0)
           for s in (40, 41, 42)}
    raw[P + "600_800/permuted/linearar4d/seed-40/p10000/result.json"] = {"ok": True, "mean_horizon_mae_db": 2.5, "n_origins": 10}
    cells, rec = T.flatten(raw)
    assert len(cells) == 3 and set(rec.condition) == {"insample", "absent", "added", "permuted"}
    s2 = T.table_s2(cells, rec)
    row = s2[(s2.model == "linearar4d") & (s2.variant == "core4-p2")].iloc[0]
    assert row.n_seeds == 3 and np.isclose(row.insample_mae, 1.1) and np.isclose(row.absent_mae, 4.5) and np.isclose(row.map_mae, 0.55)
    s4 = T.table_s4(rec)
    assert np.isclose(s4.iloc[0].permuted_mae, 2.5) and np.isclose(s4.iloc[0].delta, 1.4)
    cov = T.coverage(cells, rec[rec.condition == "permuted"])
    done = cov[(cov.band == "600_800") & (cov.model == "linearar4d")].iloc[0]
    assert done.cells_done == 3 and done.cells_expected == 60 and done.permuted_cells == 1
    assert cov[(cov.band == "600_800") & (cov.model == "residualconvlstm")].iloc[0].cells_done == 0


def test_mean_sd_and_format():
    assert T.fmt(*T.mean_sd(pd.Series([1.0, 3.0]))) == "2.000±1.414"
    assert T.fmt(*T.mean_sd(pd.Series([2.0]))) == "2.000"
    assert T.fmt(*T.mean_sd(pd.Series([], dtype=float))) == "-"


def test_dswinlstm_i_is_single_seed_base_cell_and_marked():
    P = T.PREFIX

    raw = {f"{P}600_800/core4-p2/dswinlstm_i/seed-42/result.json": {
        "ok": True, "map__test_mean_horizon_mae_db": 1.0,
        "receivers": {"geometry_correct": {"mean_horizon_mae_db": 2.0, "n_origins": 5}}}}
    cells, rec = T.flatten(raw)
    cov = T.coverage(cells, rec[rec.condition == "permuted"])
    row = cov[(cov.band == "600_800") & (cov.model == "dswinlstm_i")].iloc[0]
    assert row.cells_done == 1 and row.cells_expected == 1          # one seed, the base map only
    s2 = T.annotate(T.table_s2(cells, rec))
    assert s2.iloc[0].n_seeds == 1 and "inherited:1min-r2" in s2.iloc[0].note
