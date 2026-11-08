from training.ray.rescore import LINEAR, record_key, stride_for


def e(source, rep, model, **kw):
    return {"source": source, "prefix": "p", "band": "600_800", "rep": rep, "model": model, "seed": 40, "horizon": None, **kw}


def test_stride_policy():
    assert stride_for(e("band", "1d", "vanillalstm1d")) == 10 and stride_for(e("oct", "1d", "residualvanillalstm"), 5) == 5
    assert stride_for(e("band", "1d", "linearar1d")) == 1 and stride_for(e("oct", "4d", "convlstm")) == 1
    assert stride_for(e("p15tuned", "1d", "vanillalstm1d")) == 1                      # 15-minute 1D is small enough at stride 1


def test_record_keys_and_linear_set():
    assert record_key("rescored/", e("band", "2d", "lstmattn")) == "rescored/band/600_800/2d/lstmattn/seed-40.json"
    assert record_key("r", e("oct", "4d", "convlstm", prefix="checkpoints-oct2026-r2")) == "r/checkpoints-oct2026-r2/600_800/4d/convlstm/seed-40.json"
    assert "residuallinearar4d" in LINEAR and "convlstm" not in LINEAR
