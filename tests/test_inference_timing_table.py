import pandas as pd

from evaluation.analysis.inference_timing_table import KEEP, combine


def test_combine_fills_columns_and_orders():
    a = pd.DataFrame({"run": ["h100"], "model": ["b"], "rep": ["2d"], "band": ["600_800"], "horizon": [None], "requested_batch": [1], "ok": [True]})
    b = pd.DataFrame({"run": ["h100three"], "model": ["a"], "rep": ["2d"], "band": ["600_800"], "horizon": [4], "requested_batch": [32], "ok": [False], "error": ["x"]})
    t = combine([a, b])
    assert list(t.columns) == KEEP and list(t.model) == ["a", "b"] and t.ok.tolist() == [False, True]
