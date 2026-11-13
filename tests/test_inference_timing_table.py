import pandas as pd

from evaluation.analysis.inference_timing_table import KEEP, combine


def test_combine_fills_columns_and_orders():
    a = pd.DataFrame({"run": ["h100"], "model": ["b"], "rep": ["2d"], "band": ["600_800"], "horizon": [None], "requested_batch": [1], "ok": [True]})
    b = pd.DataFrame({"run": ["h100three"], "model": ["a"], "rep": ["2d"], "band": ["600_800"], "horizon": [4], "requested_batch": [32], "ok": [False], "error": ["x"]})
    t = combine([a, b])
    assert list(t.columns) == KEEP and list(t.model) == ["a", "b"] and t.ok.tolist() == [False, True]


def test_tsslcd_rows_from_record():
    from evaluation.analysis.inference_timing_table import tsslcd_rows

    rec = {"ok": True, "band": "600_800", "resolution": "15min", "seed": 42, "wall_seconds": 5.0,
           "timing": [{"batch": 1, "median_s_per_call": 2.0, "ms_per_window": 2000.0, "sampling_steps": 1000, "params": 7, "peak_gpu_mb": 1.0, "gpu": "g"}]}
    r = tsslcd_rows(rec, "h100three-tsslcd-tuned")[0]
    assert r["model"] == "tsslcd_tuned" and r["forward_passes_per_call"] == 1000 and r["ms_per_spectrum"] == 500.0
