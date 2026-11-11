import numpy as np
import pytest

from training.common.kriging import (empirical_semivariogram, fit_exponential_variogram, kriging_weights, variogram_gamma)
from training.common.map_builder import _map_from_sites
from training.common.spatial_checkpoint_evaluation import build_idw_maps, derive_frozen_grid

SITES_X = np.array([0.0, 800.0, 100.0, 650.0])
SITES_Y = np.array([0.0, 120.0, 700.0, 900.0])
VARIOGRAM = {"model": "exponential", "nugget": 0.4, "sill": 3.0, "range": 500.0}
NO_NUGGET = {"model": "exponential", "nugget": 0.0, "sill": 3.0, "range": 500.0}


def _grid(n=7, extent=1000.0):
    gx, gy = np.meshgrid(np.linspace(-100.0, extent, n), np.linspace(-100.0, extent, n))
    return gx, gy


def test_weights_sum_to_one_everywhere():
    gx, gy = _grid()
    for variogram in (VARIOGRAM, NO_NUGGET):
        weights = kriging_weights(SITES_X, SITES_Y, gx, gy, variogram)
        assert weights.shape == gx.shape + (4,)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-9)


@pytest.mark.parametrize("variogram", [VARIOGRAM, NO_NUGGET])
def test_exact_at_the_receivers_with_zero_variance(variogram):
    weights, variance = kriging_weights(SITES_X, SITES_Y, SITES_X, SITES_Y, variogram, return_variance=True)
    # target i is site i: the weight vector is the unit vector and the variance zero (also with a nugget: gamma jumps at h > 0)
    np.testing.assert_allclose(weights, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(variance, 0.0, atol=1e-12)


def test_variance_is_positive_away_from_receivers_and_grows_far_out():
    near = kriging_weights(SITES_X, SITES_Y, np.array([350.0]), np.array([400.0]), VARIOGRAM, return_variance=True)[1]
    far = kriging_weights(SITES_X, SITES_Y, np.array([5000.0]), np.array([5000.0]), VARIOGRAM, return_variance=True)[1]
    assert near[0] > 0 and far[0] > near[0]


def test_symmetric_geometry_gives_equal_weights_at_the_centroid():
    x = np.array([-100.0, 100.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, -100.0, 100.0])
    weights = kriging_weights(x, y, np.array([0.0]), np.array([0.0]), VARIOGRAM)
    np.testing.assert_allclose(weights[0], 0.25, atol=1e-9)


def test_single_site_takes_weight_one():
    weights = kriging_weights(np.array([5.0]), np.array([5.0]), np.array([[0.0, 9.0]]), np.array([[0.0, 3.0]]), VARIOGRAM)
    np.testing.assert_allclose(weights, 1.0)


def test_variogram_validation():
    with pytest.raises(ValueError):
        variogram_gamma(np.array([1.0]), {"model": "spherical", "nugget": 0, "sill": 1, "range": 1})
    with pytest.raises(ValueError):
        variogram_gamma(np.array([1.0]), {"nugget": -1.0, "sill": 1, "range": 1})
    assert variogram_gamma(np.array([0.0, 100.0]), VARIOGRAM)[0] == 0.0


def test_empirical_semivariogram_known_case():
    z = np.zeros((10, 3, 2))
    z[:, 1, :] = 2.0                      # site 1 is 2 dB above sites 0 and 2
    distances, gammas = empirical_semivariogram(z, np.array([0.0, 10.0, 30.0]), np.zeros(3))
    np.testing.assert_allclose(distances, [10.0, 30.0, 20.0])
    np.testing.assert_allclose(gammas, [2.0, 0.0, 2.0])


@pytest.mark.parametrize("n_lags", [6, 10])
def test_fit_recovers_exponential_parameters(n_lags):
    h = np.linspace(100.0, 1500.0, n_lags)
    truth = {"model": "exponential", "nugget": 0.5, "sill": 4.0, "range": 600.0}
    fit = fit_exponential_variogram(h, variogram_gamma(h, truth))
    for key in ("nugget", "sill", "range"):
        assert fit[key] == pytest.approx(truth[key], rel=0.02, abs=0.02)


def test_map_builder_idw_unchanged_and_kriging_reproduces_constants():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 5, 3)).astype(np.float32)             # (site, time, frequency)
    lons = np.array([-111.840, -111.836, -111.838, -111.834])
    lats = np.array([40.760, 40.762, 40.766, 40.764])
    grid = {"height": 4, "width": 5, "power": 2.0}
    idw, gx, gy = _map_from_sites(data, lons, lats, grid)
    # the pre-existing formula, written out
    from training.common.map_builder import _local_xy, grid_origin

    lon0, lat0 = grid_origin(lons, lats, grid)
    x, y = _local_xy(lons, lats, lon0, lat0)
    d = np.hypot(gx[..., None] - x, gy[..., None] - y)
    w = 1.0 / np.maximum(d, 1e-6) ** 2.0
    expected = np.einsum("stf,hws->thwf", data.astype(np.float64), w) / w.sum(-1)[None, :, :, None]
    np.testing.assert_allclose(idw, expected, rtol=1e-5, atol=1e-5)
    constant = np.full_like(data, 7.5)
    mapped, _, _ = _map_from_sites(constant, lons, lats, {**grid, "method": "kriging", "variogram": VARIOGRAM})
    np.testing.assert_allclose(mapped, 7.5, atol=1e-4)
    with pytest.raises(ValueError):
        _map_from_sites(data, lons, lats, {**grid, "method": "kriging", "variogram": VARIOGRAM, "power_domain": "linear"})


def test_frozen_grid_kriging_matches_the_training_map_builder():
    rng = np.random.default_rng(1)
    lons = np.array([-111.840, -111.836, -111.838, -111.834])
    lats = np.array([40.760, 40.762, 40.766, 40.764])
    values = rng.normal(size=(6, 4, 3))                              # (time, receiver, frequency)
    grid_cfg = {"height": 4, "width": 5, "method": "kriging", "variogram": VARIOGRAM}
    training_map, _, _ = _map_from_sites(np.moveaxis(values, 1, 0).astype(np.float32), lons, lats, grid_cfg)
    frozen = derive_frozen_grid(lons, lats, height=4, width=5, method="kriging", variogram=VARIOGRAM)
    evaluation_map = build_idw_maps(values, lons, lats, frozen)       # (time, H, W, F)
    np.testing.assert_allclose(evaluation_map, training_map, atol=2e-4)
    with pytest.raises(ValueError):
        derive_frozen_grid(lons, lats, method="kriging")
