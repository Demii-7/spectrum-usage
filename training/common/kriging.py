"""Ordinary kriging weights for the spatial maps (an alternative to inverse-distance weighting).

The benchmark's 4D maps interpolate the receivers' series onto a grid. ``map_builder`` does it with inverse-distance weights (IDW). This
module gives the same kind of weights from ordinary kriging with one frozen variogram, so the interpolation step can be swapped without
touching anything else: for a fixed variogram the ordinary-kriging weights depend only on the receiver and grid-cell coordinates, not on
the measurements, so they form a ``(H, W, S)`` array exactly like the IDW weights (each cell's weights sum to one; they may be negative
outside the receivers' hull). Interpolation stays in dB.

Variogram: exponential with a nugget, ``gamma(0) = 0`` and ``gamma(h) = nugget + sill * (1 - exp(-h / range))`` for ``h > 0`` (practical
range about ``3 * range``). It is fitted by relative least squares to the empirical semivariogram of the TRAINING-range receiver series
(``training/ray/fit_variogram.py``), once per band, and frozen in ``evaluation/analysis/kriging_variograms.json`` before any scoring; the
fitted parameters travel inside the grid configuration (``grid["variogram"]``), so a map cache is keyed on them.

How rough the variogram is: four receivers give 6 pairwise distances and the exponential model has 3 parameters, so the fit has 3
degrees of freedom; six receivers give 15 pairs. The fit script reports a bootstrap over time blocks for the parameter spread.

Kriging with a nugget as defined here is still an exact interpolator at the receivers (the discontinuity is at ``h > 0``): the weight of
a receiver at its own location is one and the prediction variance is zero.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

VARIOGRAM_MODELS = ("exponential",)


def variogram_gamma(distance: np.ndarray, variogram: Mapping[str, Any]) -> np.ndarray:
    """Semivariance at ``distance`` (same units as the coordinates, metres); zero at distance zero."""
    model = str(variogram.get("model", "exponential"))
    if model not in VARIOGRAM_MODELS:
        raise ValueError(f"unsupported variogram model {model!r}; use one of {VARIOGRAM_MODELS}")
    nugget, sill, scale = float(variogram["nugget"]), float(variogram["sill"]), float(variogram["range"])
    if nugget < 0 or sill < 0 or scale <= 0:
        raise ValueError("variogram nugget and sill must be >= 0 and range > 0")
    h = np.asarray(distance, dtype=np.float64)
    gamma = nugget + sill * (1.0 - np.exp(-h / scale))
    return np.where(h > 0.0, gamma, 0.0)


def kriging_weights(site_x: np.ndarray, site_y: np.ndarray, target_x: np.ndarray, target_y: np.ndarray,
                    variogram: Mapping[str, Any], *, return_variance: bool = False):
    """Ordinary-kriging weights of ``S`` sites for every target point.

    ``target_x``/``target_y`` may have any (equal) shape ``T``; the result has shape ``T + (S,)``. With ``return_variance`` the
    kriging variance (the prediction variance in the variogram's units, dB^2) with shape ``T`` is returned as well.
    """
    sx, sy = np.asarray(site_x, dtype=np.float64).ravel(), np.asarray(site_y, dtype=np.float64).ravel()
    tx, ty = np.asarray(target_x, dtype=np.float64), np.asarray(target_y, dtype=np.float64)
    if sx.shape != sy.shape or tx.shape != ty.shape or not sx.size:
        raise ValueError("site and target coordinates must be non-empty matching arrays")
    n = sx.size
    site_distance = np.hypot(sx[:, None] - sx[None, :], sy[:, None] - sy[None, :])
    system = np.zeros((n + 1, n + 1))
    system[:n, :n] = variogram_gamma(site_distance, variogram)
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    target_distance = np.hypot(tx[..., None] - sx, ty[..., None] - sy)               # T + (S,)
    rhs = np.concatenate([variogram_gamma(target_distance, variogram), np.ones(target_distance.shape[:-1] + (1,))], axis=-1)
    flat = rhs.reshape(-1, n + 1)
    solution = np.linalg.pinv(system) @ flat.T                                          # (S+1, N); pinv guards coincident sites
    weights = solution[:n].T.reshape(target_distance.shape)
    exact = target_distance == 0.0                                                      # a target at a site takes that site's value
    if exact.any():
        rows = exact.any(axis=-1)
        weights[rows] = exact[rows].astype(np.float64)
    if not return_variance:
        return weights
    variance = (solution[:n].T * flat[:, :n]).sum(axis=-1) + solution[n]
    variance = variance.reshape(target_distance.shape[:-1])
    variance = np.where(exact.any(axis=-1), 0.0, np.maximum(variance, 0.0))
    return weights, variance


def empirical_semivariogram(values: np.ndarray, site_x: np.ndarray, site_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise distances and semivariances of ``values`` shaped (time, sites, bins), pooled over time and bins.

    ``gamma_ij = 0.5 * mean((z_i - z_j)^2)`` over every finite pair of values. Returns (distances, gammas), one entry per site pair.
    """
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError("values must be (time, sites, bins)")
    n = z.shape[1]
    sx, sy = np.asarray(site_x, dtype=np.float64), np.asarray(site_y, dtype=np.float64)
    distances, gammas = [], []
    for i in range(n):
        for j in range(i + 1, n):
            difference = z[:, i, :] - z[:, j, :]
            finite = np.isfinite(difference)
            distances.append(float(np.hypot(sx[i] - sx[j], sy[i] - sy[j])))
            gammas.append(float(0.5 * np.mean(difference[finite] ** 2)))
    return np.asarray(distances), np.asarray(gammas)


def fit_exponential_variogram(distances: np.ndarray, gammas: np.ndarray) -> dict[str, float]:
    """Relative least squares fit of ``nugget + sill * (1 - exp(-h / range))`` to (distance, semivariance) pairs.

    Minimises ``sum(((model - gamma) / gamma)^2)`` (the Cressie weighting when every lag has the same number of pairs) with
    ``nugget >= 0``, ``sill >= 0`` and ``range`` between 1% and 100x the largest distance. Several starting points are tried; the best
    is kept. Returns the parameters and the fit's relative residual.
    """
    from scipy.optimize import least_squares

    h, g = np.asarray(distances, dtype=np.float64), np.asarray(gammas, dtype=np.float64)
    if h.size < 3 or np.any(g <= 0):
        raise ValueError("need at least three lags with positive semivariance")
    top = float(h.max())

    def residual(p):
        nugget, sill, scale = p
        return ((nugget + sill * (1.0 - np.exp(-h / scale))) - g) / g

    best = None
    for range_fraction in (0.1, 0.3, 1.0, 3.0):
        for nugget_fraction in (0.0, 0.3, 0.7):
            start = [nugget_fraction * g.max(), (1.0 - nugget_fraction) * g.max(), range_fraction * top]
            fit = least_squares(residual, start, bounds=([0.0, 0.0, 0.01 * top], [10.0 * g.max(), 10.0 * g.max(), 100.0 * top]))
            if best is None or fit.cost < best.cost:
                best = fit
    nugget, sill, scale = (float(v) for v in best.x)
    return {"model": "exponential", "nugget": nugget, "sill": sill, "range": scale, "relative_rss": float(2.0 * best.cost)}
