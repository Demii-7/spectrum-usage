import numpy as np

from training.ray.spatial15 import VARIOGRAM_FILE, grid_for, load_variogram, variants
from training.ray.spatial15_origins import per_origin_errors


def test_per_origin_errors_denormalizes_and_averages_over_bins():
    n, h, r, f = 3, 2, 2, 4
    prediction = np.zeros((n, h, r, f))
    target = np.full((n, h, r, f), 10.0)
    target[:, :, 1, :] += 2.0
    npz = {"predictions_normalized": prediction, "targets_dbm": target,
           "origin_timestamps": np.array(["2026-10-27T20:00:00Z", "2026-10-27T20:15:00Z", "2026-10-27T20:30:00Z"])}
    normalization = {"mean_dbm": np.full((1, f), 10.0), "std_dbm": np.full((1, f), 1.0)}
    first = per_origin_errors(npz, 0, normalization)
    second = per_origin_errors(npz, 1, normalization)
    assert first["mae"].shape == (n, h) and np.allclose(first["mae"], 0.0) and np.allclose(second["mae"], 2.0)
    assert list(first["times"]) == list(npz["origin_timestamps"])


def test_krige_variant_is_additive_and_carries_the_frozen_variogram():
    names = [v["name"] for v in variants(["s2"])]
    assert names[:7] == ["core4-p2", "core4-p1", "core4-p4", "core4-p16", "core4-p2-linear", "core4-p2-g5", "core4-p2-g20"]
    assert names[-1] == "core4-krige"
    krige = next(v for v in variants(["s2"]) if v["name"] == "core4-krige")
    grid = grid_for(krige, {"origin_lon": 0.0, "origin_lat": 0.0, "width_meters": 1.0, "height_meters": 1.0})
    assert grid["method"] == "kriging" and grid["power"] == 2.0 and grid["power_domain"] == "db"    # IDW power 2 still samples the forecast
    import pathlib

    root = str(pathlib.Path(__file__).resolve().parents[1])
    assert (pathlib.Path(root) / VARIOGRAM_FILE).exists()
    for band in ("600_800", "2400_2600"):
        v = load_variogram(root, band)
        assert v["model"] == "exponential" and v["sill"] >= 0 and v["nugget"] >= 0 and v["range"] > 0
