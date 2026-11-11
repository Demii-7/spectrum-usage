"""June-July targets in transfer_all: target selection, the core/external dispatch and the frozen table."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.ray import transfer_all as ta  # noqa: E402


def test_core_target_dispatch():
    assert ta.is_core_target({"platform": "powder", "run": "20261025T1355Z"})
    assert not ta.is_core_target({"platform": "powder", "run": "20260628T0437Z"})      # June-July POWDER run: converted like an external site
    assert not ta.is_core_target({"platform": "ara", "run": "20261025T1355Z"})


def _table(tmp_path):
    rows = [{"platform": "powder", "site": "cpg-nuc1", "run": "20260628T0437Z", "band": "600_800", "use": True, "n_origins": 327},
            {"platform": "powder", "site": "guesthouse-nuc1", "run": "20260703T1839Z", "band": "600_800", "use": False, "n_origins": 0},
            {"platform": "powder", "site": "ebc-nuc1", "run": "20260703T1839Z", "band": "2400_2600", "use": True, "n_origins": 105}]
    path = tmp_path / "july.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_july_targets_exclude_unused_and_core(tmp_path):
    path = _table(tmp_path)
    got = ta.targets(str(tmp_path), ["powder"], ["600_800", "2400_2600"], str(path.name), core=False)
    assert [(t["site"], t["band"]) for t in got] == [("cpg-nuc1", "600_800"), ("ebc-nuc1", "2400_2600")]
    assert all(not ta.is_core_target(t) for t in got)


def test_july_targets_band_filter_and_core_flag(tmp_path):
    path = _table(tmp_path)
    got = ta.targets(str(tmp_path), ["powder"], ["600_800"], str(path.name), core=True)
    sites = {t["site"] for t in got}
    assert "cpg-nuc1" in sites and {"guesthouse", "law73"} <= sites                    # core receivers added when asked
    assert ta.target_key(got[0]).count("/") == 3


def test_july_table_key_parsing_and_groups():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
    import transfer_july_all_table as T

    a = T.parse_key("transfer-all-july-2026/C/powder/cpg-nuc1/20260628T0437Z/600_800/2d/linearar2d/seed-40/summary.json")
    assert a["regime"] == "C" and a["site"] == "cpg-nuc1" and a["rep"] == "2d" and a["horizon_tag"] is None
    b = T.parse_key("transfer-all-july-2026/A/powder/ebc-nuc1/20260703T1839Z/2400_2600/2d/revinmlp/h12/seed-41/summary.json")
    assert b["model"] == "revinmlp" and b["horizon_tag"] == 12
    c = T.parse_key("transfer15-july-2026/baseline/powder/madsen-nuc1/20260628T0437Z/600_800/persistence/summary.json")
    assert c["regime"] == "baseline" and c["model"] == "persistence"
    assert T.parse_key("transfer15-july-2026/A/powder/x/summary.json") is None
    assert T.site_group("moran-nuc1") == "core4" and T.site_group("cpg-nuc1") == "other"


def test_july_table_aggregation_marks_incomplete_groups():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
    import transfer_july_all_table as T

    rows = []
    for site in ("humanities-nuc1", "madsen-nuc1"):
        for h in (4, 12, 24, 48):
            rows.append({"regime": "baseline", "site": site, "run": "20260628T0437Z", "band": "600_800", "rep": "baseline", "model": "persistence",
                         "horizon_tag": None, "target": f"{site}/r/600_800", "seed": 0, "horizon": h, "mae": 1.0})
    for site in ("humanities-nuc1",):                      # model scored on one of the two targets only
        for h in (4, 12, 24, 48):
            for seed in (40, 41):
                rows.append({"regime": "C", "site": site, "run": "20260628T0437Z", "band": "600_800", "rep": "2d", "model": "linearar2d",
                             "horizon_tag": None, "target": f"{site}/r/600_800", "seed": seed, "horizon": h, "mae": 0.5 + 0.1 * (seed - 40)})
    out = T.aggregate(pd.DataFrame(rows))
    row = out[(out.model == "linearar2d")].iloc[0]
    assert abs(row.mae_t12 - 0.55) < 1e-9 and not row.complete                           # seeds averaged, group incomplete
    assert out[out.model == "persistence"].iloc[0].complete
