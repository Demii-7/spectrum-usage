"""Rescore transfer cells of the NCAR batch on exactly the intervals listed in the paper's evaluation-set table.

The batch scores every usable window of a target. The paper's table lists one interval per testbed, so for those targets this script
restricts every stored forecast (``forecasts.npz`` of the model cells and of the item-1 baselines) to target times inside the listed
interval and recomputes the error. Nothing is re-run; reads ``transfer-ncar-2026-11/cells/`` and ``item1/``, writes a new table only.

    python -m evaluation.analysis.transfer_paper_scope --stamp v1 --workers 32

Output: ``s3://spectrum/transfer-ncar-2026-11/tables/paper-scope/<stamp>/transfer_paper_scope.csv`` (one row per cell and horizon).
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# target-key prefix -> (first target time, last target time), as listed in Table "Transferability and generalization evaluation sets"
SCOPE = {
    "aerpaw/CC1/": ("2022-02-09T18:00Z", "2022-02-13T11:45Z"),
    "aerpaw/CC2/": ("2022-02-09T18:00Z", "2022-02-13T11:45Z"),
    "aerpaw/LW1/": ("2022-02-09T18:00Z", "2022-02-13T11:45Z"),
    "ara/ames-ue-000/20260702T1920Z/600_800": ("2026-07-02T19:20Z", "2026-07-04T21:20Z"),
    "ara/horticulture-ue-004/20260702T1920Z/600_800": ("2026-07-02T19:20Z", "2026-07-04T21:20Z"),
}
HORIZONS = (4, 12, 24, 48)
REGIMES = ("A", "C")


def scope_of(target: str):
    for prefix, (a, b) in SCOPE.items():
        if target.startswith(prefix):
            return pd.Timestamp(a), pd.Timestamp(b)
    return None


def main(argv=None) -> None:
    from evaluation.analysis import ncar_transfer_tables as nt
    from training.ray.ncar_cells import load_npz, per_bin_table
    from training.ray.storage import spectrum_s3_client

    p = argparse.ArgumentParser()
    p.add_argument("--stamp", required=True)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args(argv)
    client = spectrum_s3_client()
    metas = []
    for platform in ("aerpaw", "ara"):
        for regime in REGIMES:
            for k in nt.list_keys(client, f"transfer-ncar-2026-11/cells/{regime}/{platform}/"):
                m = nt.parse_cell(k)
                if m and scope_of(m["target"]):
                    metas.append(m)
        for k in nt.list_keys(client, f"transfer-ncar-2026-11/item1/{platform}/"):
            m = nt.parse_baseline(k)
            if m and scope_of(m["target"]):
                metas.append({**m, "model": m["baseline"], "rep": "baseline", "source": "baseline", "condition": "baseline",
                              "source_band": m["target_band"], "seed": 0, "trained_horizon": None})
    print(f"[scope] {len(metas)} cells", flush=True)

    def one(m):
        first, last = scope_of(m["target"])
        mats = load_npz(client.get_object(Bucket=nt.BUCKET, Key=m["key"].replace("per_bin.csv", "forecasts.npz"))["Body"].read())
        rows = []
        for h in mats:
            if h not in HORIZONS or (m.get("trained_horizon") and h != m["trained_horizon"]):
                continue
            t = pd.to_datetime(mats[h]["origin"], unit="s", utc=True) + pd.Timedelta(minutes=15 * h)
            keep = np.asarray((t >= first) & (t <= last))
            if not keep.any():
                continue
            pb = per_bin_table({h: mats[h]}, {h: keep})
            rows.append({k: m[k] for k in ("model", "rep", "source", "condition", "source_band", "target_band", "group", "target", "receiver", "seed")}
                        | {"horizon": h, "n_windows": int(keep.sum()), "n_windows_full": int(len(keep)), "mae_db": pb.mae_db.mean(),
                           "mse_db2": pb.mse_db2.mean(), "first_target": str(t[keep].min()), "last_target": str(t[keep].max())})
        return rows

    with ThreadPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(one, metas) for r in rs]
    out = pd.DataFrame(rows)
    key = f"transfer-ncar-2026-11/tables/paper-scope/{a.stamp}/transfer_paper_scope.csv"
    client.put_object(Bucket=nt.BUCKET, Key=key, Body=out.to_csv(index=False).encode())
    print(f"[scope] wrote s3://{nt.BUCKET}/{key} ({len(out)} rows)", flush=True)
    pd.set_option("display.width", 250)
    print(out.groupby(["group", "target", "horizon"]).n_windows.agg(["min", "max"]).to_string())
    g = out[out.horizon.isin([4, 12, 24]) & out.model.isin(["revinmlp", "tuned_median", "same_time_yesterday", "residuallinearar1d"])]
    print(g.groupby(["group", "target_band", "model", "condition"]).mae_db.mean().round(3).to_string())


if __name__ == "__main__":
    main()
