"""Region-only masking for the TSS-LCD spectral-context ablation (training/ray/tsslcd_authors.py, tsslcd_spectral_ablation.py)."""
import numpy as np
import pandas as pd
import pytest

from training.ray import tsslcd_authors as T
from training.ray import tsslcd_spectral_ablation as A

N_SITES, N_BINS = 4, 200
FREQS = 600.5 + np.arange(N_BINS)


def synthetic(rows=500, seed=0):
    rng = np.random.default_rng(seed)
    data = np.empty((rows, N_SITES * N_BINS))
    for s in range(N_SITES):
        data[:, s * N_BINS:(s + 1) * N_BINS] = -90 + 3 * s + rng.normal(scale=2.0, size=(rows, N_BINS))
    for s in range(N_SITES):
        data[:, s * N_BINS + 8:s * N_BINS + 22] -= 25.0     # the noise-floor region (608.5-621.5 MHz) of every receiver is much quieter than the rest
    return data


CALIBRATION = [[608.5, 621.5]]
KEEP = (FREQS >= 622.5) & (FREQS <= 641.5)


def test_mask_keeps_the_region_and_replaces_the_rest_at_every_receiver():
    data = synthetic()
    params = T.region_mask_params(data, N_SITES, N_BINS, FREQS, CALIBRATION)
    masked = T.apply_region_mask(data, KEEP, params, N_BINS, seed=7)
    assert masked.shape == data.shape and masked is not data
    for s in range(N_SITES):
        block, original = masked[:, s * N_BINS:(s + 1) * N_BINS], data[:, s * N_BINS:(s + 1) * N_BINS]
        np.testing.assert_array_equal(block[:, KEEP], original[:, KEEP])                 # in-region columns untouched
        outside = block[:, ~KEEP]
        centre, std = params[s]
        assert abs(outside.mean() - centre) < 0.2 and abs(outside.std() - std) < 0.2     # noise-floor statistics
        assert outside.mean() < original[:, ~KEEP].mean() - 10                            # far below the original out-of-region level
    # the same bins at all receivers, receiver-specific noise levels
    assert params[0] != params[1]


def test_mask_is_reproducible_and_depends_on_the_seed():
    data = synthetic()
    params = T.region_mask_params(data, N_SITES, N_BINS, FREQS, CALIBRATION)
    a, b, c = (T.apply_region_mask(data, KEEP, params, N_BINS, seed=s) for s in (3, 3, 4))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_params_follow_the_calibration_bins():
    data = synthetic()
    quiet = T.region_mask_params(data, N_SITES, N_BINS, FREQS, CALIBRATION)
    loud = T.region_mask_params(data, N_SITES, N_BINS, FREQS, [[700.5, 720.5]])
    assert all(q[0] < l[0] - 10 for q, l in zip(quiet, loud))
    with pytest.raises(ValueError):
        T.region_mask_params(data, N_SITES, N_BINS, FREQS, [[10.0, 11.0]])


def test_matches_the_2d_pipeline_mask_when_torch_is_available():
    pytest.importorskip("torch")
    from training.common.data import apply_2d_spectral_mask
    data = synthetic()[:, :N_BINS]
    params = T.region_mask_params(data, 1, N_BINS, FREQS, CALIBRATION)
    cfg = {"frequency_ranges": [[622.5, 641.5]], "replacement": "low_tail_gaussian", "calibration_frequency_ranges": CALIBRATION,
           "low_tail_quantile": 0.05, "minimum_noise_std_db": 0.05, "seed": 7}
    reference = apply_2d_spectral_mask(data, FREQS, cfg, training_data=data, split_seed_offset=0)
    ours = T.apply_region_mask(data, KEEP, params, N_BINS, seed=7)
    np.testing.assert_array_equal(ours[:, KEEP], reference[:, KEEP])
    assert abs(ours[:, ~KEEP].mean() - reference[:, ~KEEP].mean()) < 0.1 and abs(ours[:, ~KEEP].std() - reference[:, ~KEEP].std()) < 0.1


def _patch_loader(monkeypatch, rows=400):
    data = synthetic(rows)
    monkeypatch.setattr(T, "load_partition", lambda root, partition, band="600_800", sites=T.SITES, max_gap=T.MAX_GAP, profile="1min":
                        (data.copy(), pd.date_range("2026-10-27", periods=rows, freq="15min"), np.ones(rows, dtype=bool)))
    return data


def test_build_partition_default_is_unchanged_and_the_mask_acts_before_the_scaler(monkeypatch):
    data = _patch_loader(monkeypatch)
    raw0, model0, bounds0, scaler0 = T.build_partition("root", "train", None, "off", band="600_800", profile="15min")
    raw1, model1, bounds1, scaler1 = T.build_partition("root", "train", None, "off", band="600_800", profile="15min", spectral_mask=None)
    np.testing.assert_array_equal(raw0, raw1)
    np.testing.assert_array_equal(scaler0.min, scaler1.min)
    mask = {"keep": KEEP, "params": T.region_mask_params(data, N_SITES, N_BINS, FREQS, CALIBRATION), "n_bins": N_BINS, "seed": 5,
            "offsets": {"train": 0, "validation": 1, "test": 2}}
    raw_m, model_m, bounds_m, scaler_m = T.build_partition("root", "train", None, "off", band="600_800", profile="15min", spectral_mask=mask)
    in_cols = np.concatenate([s * N_BINS + np.flatnonzero(KEEP) for s in range(N_SITES)])
    out_cols = np.concatenate([s * N_BINS + np.flatnonzero(~KEEP) for s in range(N_SITES)])
    np.testing.assert_allclose(scaler_m.inverse(raw_m)[:, in_cols], data[:, in_cols], atol=1e-9)           # the region's columns are the real series
    assert scaler_m.inverse(raw_m)[:, out_cols].mean() < data[:, out_cols].mean() - 10                        # the rest is noise-floor noise
    assert not np.allclose(scaler_m.min[out_cols], scaler0.min[out_cols])                                      # the scaler was fit on the masked series
    np.testing.assert_array_equal(raw_m, model_m)                                                              # savgol off: model target = raw


def test_partitions_use_distinct_noise_streams(monkeypatch):
    data = _patch_loader(monkeypatch)
    mask = {"keep": KEEP, "params": T.region_mask_params(data, N_SITES, N_BINS, FREQS, CALIBRATION), "n_bins": N_BINS, "seed": 5,
            "offsets": {"train": 0, "validation": 1, "test": 2}}
    a = T.build_partition("root", "train", None, "off", "600_800", "15min", spectral_mask=mask)[0]
    b = T.build_partition("root", "validation", None, "off", "600_800", "15min", spectral_mask=mask)[0]
    assert not np.array_equal(a[:, ~np.tile(KEEP, N_SITES)], b[:, ~np.tile(KEEP, N_SITES)])


def test_plan_reading_skips_noise_regions_and_returns_calibration_ranges(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text("band_id,start_mhz,end_mhz,bin_count,behavior_category,is_noise_floor\nR1,600.5,607.5,8,x,false\nR2,608.5,621.5,14,noise_floor,true\n"
                    "R3,622.5,641.5,20,y,false\n")
    regions, calibration = A.read_plan(plan)
    assert [r["band_id"] for r in regions] == ["R1", "R3"] and calibration == [[608.5, 621.5]]
    real = A.read_plan(A.ROOT / A.PLANS["600_800"])
    assert len(real[0]) == 13 and len(real[1]) == 4
    assert len(A.read_plan(A.ROOT / A.PLANS["2400_2600"])[0]) == 5
