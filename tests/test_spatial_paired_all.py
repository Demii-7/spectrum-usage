import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "analysis"))
import spatial_paired_all as P  # noqa: E402


class FakeClient:
    def __init__(self, blobs):
        self.blobs = blobs

    def get_object(self, Bucket, Key):
        if Key not in self.blobs:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.blobs[Key])}


def make_npz(receiver, mode):
    buf = io.BytesIO()
    n = 5
    mae = np.tile(np.arange(n, dtype=float)[:, None], (1, 4))            # (origins, horizons)
    mid = f"|{mode}" if mode else ""
    np.savez(buf, horizons=np.array([4, 12, 24, 48]), **{f"{receiver}{mid}|mae": mae, f"{receiver}{mid}|times": np.arange(n).astype(str)})
    return buf.getvalue()


def test_status_marks_single_and_partial_seed_rows():
    assert P.status("dswinlstm_i", 1) == "single seed by design"
    assert P.status("stsprednet", 3) == "partial seeds (3 of 5)"
    assert P.status("revinmlp_nbr", 3) == "all seeds"


def test_npz_series_reads_plain_and_mode_layouts_and_returns_none_when_missing():
    client = FakeClient({"a.npz": make_npz("madsen", None), "b.npz": make_npz("madsen", "real")})
    assert P.npz_series(client, "a.npz", "madsen", None).tolist() == [0, 1, 2, 3, 4]
    assert P.npz_series(client, "b.npz", "madsen", "real").tolist() == [0, 1, 2, 3, 4]
    assert P.npz_series(client, "c.npz", "madsen", "real") is None


def test_series_4d_falls_back_to_the_neighbour_ablation_real_mode():
    key = f"{P.NBR_PREFIX}/600_800/stsprednet/seed-40/per_origin.npz"
    s, n = P.series_4d(FakeClient({key: make_npz("madsen", "real")}), "600_800", "stsprednet", "madsen")
    assert n == 1 and s["mae"].tolist() == [0, 1, 2, 3, 4]
    assert P.series_4d(FakeClient({}), "600_800", "stsprednet", "madsen") == (None, 0)
