import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
from tsslcd_per_bin_tables import per_bin_table  # noqa: E402


def test_per_bin_table_means_over_receivers():
    n_sites, n_bins, horizons = 4, 3, [1, 15]
    mae = np.zeros((2, n_sites * n_bins))
    for h in range(2):
        for s in range(n_sites):
            mae[h, s * n_bins:(s + 1) * n_bins] = 10 * h + s + np.arange(n_bins) * 0.1       # receiver-major columns
    record = {"seed": 42, "per_column_mae": {"horizons": horizons, "n_sites": n_sites, "n_bins": n_bins,
                                              "frequencies_mhz": [600.5, 601.5, 602.5], "mae_db": mae.tolist()}}
    table = per_bin_table(record, "600_800", "1min")
    assert len(table) == 6 and set(table.horizon) == {1, 15}
    first = table[(table.horizon == 1) & (table.frequency_mhz == 600.5)].mae_db.iloc[0]
    assert abs(first - 1.5) < 1e-9                                                         # mean of 0, 1, 2, 3
    last = table[(table.horizon == 15) & (table.frequency_mhz == 602.5)].mae_db.iloc[0]
    assert abs(last - (10 + 1.5 + 0.2)) < 1e-9
