"""Synthetic end-to-end check of training/ray/level_diagnostics.py (CPU, a few bins, no Ray)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from training.ray import level_diagnostics as ld  # noqa: E402

SIG_EPS, SIG_ETA = 1.0, 0.05


def _write_run(root: Path, node: str, run: str, band: str, start: str, end: str, rng, nbins: int = 4) -> None:
    index = pd.date_range(start, end, freq="min")
    level = -100 + np.cumsum(rng.normal(0, SIG_ETA, (len(index), nbins)), axis=0)
    values = level + rng.normal(0, SIG_EPS, (len(index), nbins))
    frame = pd.DataFrame(values, columns=[f"{600.5 + i}" for i in range(nbins)])
    frame.insert(0, "timestamp_utc", (index + pd.Timedelta(seconds=15)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = root / ld.RAW_ROOT / node / run / band / ld.RAW_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    rng = np.random.default_rng(0)
    node, band = "humanities-nuc1", "600_800"
    _write_run(root, node, ld.TRAIN_RUN, band, "2026-10-15T14:34Z", "2026-10-22T13:13Z", rng)
    _write_run(root, node, ld.TEST_RUN, band, "2026-10-25T13:55Z", "2026-11-03T20:30Z", rng)
    return ld.cell_job({"node": node, "band": band}, {"repo_root": str(root), "gbm": False})


def test_job_succeeds_and_fills_every_table(result):
    assert result["meta"]["ok"], result["meta"].get("trace")
    for name, rows in result["tables"].items():
        assert rows, name
    experiments = {r["experiment"] for r in result["rows"]}
    assert {"baselines", "seasonal", "probe_per_bin", "probe_pooled", "aggregation"} <= experiments


def test_state_space_moments_recover_the_simulated_variances(result):
    train = next(r for r in result["tables"]["statespace"] if r.get("split") == "train" and "horizon" not in r and "fit" not in r)
    assert train["sigma_eps2_median"] == pytest.approx(SIG_EPS ** 2, rel=0.1)
    assert train["sigma_eta2_median"] == pytest.approx(SIG_ETA ** 2, abs=0.01)


def test_prediction_error_fit_recovers_the_simulated_variances(result):
    fit = next(r for r in result["tables"]["statespace"] if r.get("fit") == "prediction_error")
    assert fit["sigma_eps2_median"] == pytest.approx(SIG_EPS ** 2, rel=0.1)
    assert fit["sigma_eta2_median"] == pytest.approx(SIG_ETA ** 2, rel=0.6)


def test_mean_and_last_anchors_span_the_same_unregularized_class(result):
    rows = [r for r in result["tables"]["equivalence"] if r["pair"] == "mean60-last"]
    assert rows and max(r["max_abs_diff_db"] for r in rows) < 1e-6


def test_lookback_mean_beats_persistence_on_a_noisy_local_level(result):
    test = {r["predictor"]: r["mae"] for r in result["rows"]
            if r["experiment"] == "baselines" and r["horizon"] == 1 and r["split"] == "test"}
    assert test["mean60"] < test["persistence"]
    assert test["kalman_local_level"] <= test["mean60"] * 1.02


def test_corrections_find_little_beyond_the_level_on_a_pure_local_level(result):
    probe = [r for r in result["rows"] if r["experiment"] == "probe_per_bin" and r["split"] == "test"
             and r["predictor"] == "mean60:val_scaled" and r["horizon"] == 1]
    assert probe and probe[0]["ci_lo"] < 0.1


def test_block_bootstrap_skill_is_zero_for_identical_errors():
    rng = np.random.default_rng(1)
    e = rng.random(3000)
    skill, lo, hi = ld.block_bootstrap_skill(e, e, rng)
    assert skill == 0 and lo == 0 and hi == 0


def test_kalman_gain_limits():
    assert ld.kalman_gain(np.array([1e6]))[0] == pytest.approx(1.0, abs=1e-3)
    assert ld.kalman_gain(np.array([1e-8]))[0] < 1e-3


def test_random_walk_mean_var_matches_simulation():
    rng = np.random.default_rng(3)
    steps = rng.normal(size=(20000, 60))
    mu = np.cumsum(steps, axis=1)                       # mu_{t+1..t+60} - mu_t
    sim = mu[:, 45:60].mean(axis=1).var()
    assert ld.random_walk_mean_var(60, 15) == pytest.approx(sim, rel=0.05)
    assert ld.random_walk_mean_var(60, 1) == 60


def test_sampling_part_runs_and_reports_floors(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo2")
    rng = np.random.default_rng(5)
    node, band = "madsen-nuc1", "2400_2600"
    _write_run(root, node, ld.TRAIN_RUN, band, "2026-10-15T14:34Z", "2026-10-22T13:13Z", rng, nbins=3)
    _write_run(root, node, ld.TEST_RUN, band, "2026-10-25T13:55Z", "2026-11-03T20:30Z", rng, nbins=3)
    out = ld.sampling_job({"node": node, "band": band}, {"repo_root": str(root)})
    assert out["meta"]["ok"], out["meta"].get("trace")
    rows = pd.DataFrame(out["rows"])
    floor = rows[(rows.part == "floor") & (rows.horizon == 15)].set_index("target")
    # averaging the target removes most of the white noise: the avg15 floor is far below the point floor
    assert floor.loc["avg15", "floor_mse"] < 0.5 * floor.loc["point", "floor_mse"]
    test = rows[(rows.part == "sampling") & (rows.split == "test")]
    assert {"1min", "block", "decim0", "decim14"} <= set(test.rep)
    # on a pure local level, no predictor should get much below the floor
    pt = test[(test.target == "point") & (test.horizon == 15) & (test.lookback_min == 240)]
    assert pt.mse.min() > 0.9 * floor.loc["point", "floor_mse"]
