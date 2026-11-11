"""Pure helpers of the cross-band positional transfer runner."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.ray import transfer_crossband as X  # noqa: E402


def _touch(root: Path, node: str, run: str, band: str) -> None:
    f = root / "data" / "powder" / node / run / band / "power_1mhz_avg_per_minute.csv"
    f.parent.mkdir(parents=True)
    f.write_text("timestamp_utc\n")


def test_raw_files_pick_only_the_july_run_and_new_bands(tmp_path):
    _touch(tmp_path, "cpg-nuc1", "20260703T1839Z", "800_1000")
    _touch(tmp_path, "cpg-nuc1", "20260703T1839Z", "5725_5925")
    _touch(tmp_path, "cpg-nuc1", "20260703T1839Z", "600_800")                    # a trained band: not a cross-band target
    _touch(tmp_path, "cpg-nuc1", "20260628T0437Z", "800_1000")                   # other run: excluded
    _touch(tmp_path, "guesthouse-nuc1", "20260703T1839Z-backup-before-merge", "800_1000")   # backup copy: excluded
    _touch(tmp_path, "law73-nuc1", "20260703T1839Z", "800_1000")                 # not in the receiver list
    got = X.raw_files(str(tmp_path))
    assert sorted((g["site"], g["band"]) for g in got) == [("cpg-nuc1", "5725_5925"), ("cpg-nuc1", "800_1000")]
    assert all(g["platform"] == "powder" and g["run"] == "20260703T1839Z" for g in got)


def test_pairs_and_band_registration():
    assert len(X.source_target_pairs()) == 4 and ("600_800", "5725_5925") in X.source_target_pairs()
    from training.ray import transfer_external as te

    assert te.BANDS["800_1000"] == (800.0, 1000.0) and te.BANDS["5725_5925"] == (5725.0, 5925.0)       # conversion knows the new bands
    pytest.importorskip("torch")                                                                       # task_for builds a TransferTask (torch module)
    task = te.task_for({"platform": "powder", "site": "cpg-nuc1", "run": "r", "band": "800_1000", "raw_start": "2026-07-03T18:45:00+00:00",
                        "raw_end": "2026-07-06T09:45:00+00:00", "target_start": "2026-07-04T18:45:00+00:00", "target_end": "2026-07-06T09:45:00+00:00"})
    assert task.frequency_start_mhz == 800.0 and task.frequency_end_mhz == 1000.0


def test_default_regimes():
    assert X.default_regimes("residuallinearar2d") == ["A", "B"]
    assert X.default_regimes("linearar2d", regime_c=True) == ["A", "B", "C", "Cm"]
    assert X.default_regimes("revinmlp", regime_c=True) == ["A"]               # full intrinsic normalization: B and C identical to A


def test_level_mismatch_db():
    data = np.full((60, 200), -128.0)
    base = {"mean_dbm": np.full((1, 200), -122.0)}
    mask = np.ones(60, dtype=bool)
    assert X.level_mismatch_db(base, data, mask) == pytest.approx(-6.0)
    mask[:30] = False
    data[30:] = -130.0
    assert X.level_mismatch_db(base, data, mask) == pytest.approx(-8.0)           # only calibration rows count
    with pytest.raises(ValueError):
        X.level_mismatch_db({"mean_dbm": np.zeros(10)}, data, mask)
