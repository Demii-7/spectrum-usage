"""Checks of the TSS-LCD spectral-context ablation (reads the records of ``training.ray.tsslcd_spectral_ablation``).

* consistency: the full-band error recomputed from the per-column output of the full-context re-scoring (mean over all 800 columns) against the stored whole-band
  value of the original run (``raw_target__off__mae_db_t<h>`` of ``tsslcd-authors-15min-default[-2400]``), per horizon, plus the eval-only run's own
  zero-difference check (``zero_offset_max_abs_diff_db``);
* mask: for every region-only run whether the in-region columns were unchanged and how the replaced columns' statistics compare with the noise floor;
* run: wall seconds and peak GPU memory of every run.

    python evaluation/analysis/tsslcd_ablation_check.py [--out evaluation/results/tsslcd_spectral_ablation_checks.csv] [--prefix ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

FULL = {"600_800": "tsslcd-authors-15min-default", "2400_2600": "tsslcd-authors-15min-default-2400"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/results/tsslcd_spectral_ablation_checks.csv")
    p.add_argument("--prefix", default="oct2026-spectral-context-15min-tsslcd")
    a = p.parse_args()
    client = spectrum_s3_client()
    rows = []
    for band in FULL:
        keys = [o["Key"] for pg in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=f"{a.prefix}/{band}/")
                for o in pg.get("Contents", []) if o["Key"].endswith(".json")]
        with ThreadPoolExecutor(8) as pool:
            records = list(pool.map(lambda k: (k, json.loads(client.get_object(Bucket="spectrum", Key=k)["Body"].read())), keys))
        for key, r in sorted(records):
            seed = r.get("seed")
            region = "full_context" if "/full_context/" in key else key.split("/region_only/")[1].split("/")[0]
            rows.append({"band": band, "run": region, "seed": seed, "kind": "run", "name": "wall_seconds", "value": r.get("wall_seconds"), "ok": r.get("ok")})
            rows.append({"band": band, "run": region, "seed": seed, "kind": "run", "name": "peak_gpu_mb", "value": r.get("peak_gpu_mb"), "ok": r.get("ok")})
            if not r.get("ok"):
                rows.append({"band": band, "run": region, "seed": seed, "kind": "run", "name": "error", "value": str(r.get("error"))[:200], "ok": False})
                continue
            if region == "full_context" and r.get("per_column_mae"):
                src = json.loads(client.get_object(Bucket="spectrum", Key=f"{FULL[band]}/savgol-off/seed-{seed}.json")["Body"].read())
                for j, h in enumerate(r["per_column_mae"]["horizons"]):
                    recomputed = float(np.mean(r["per_column_mae"]["mae_db"][j]))
                    stored = src["metrics"].get(f"raw_target__off__mae_db_t{h}")
                    rows.append({"band": band, "run": region, "seed": seed, "kind": "consistency", "name": f"t{h}_mean_over_all_columns_vs_stored_whole_band",
                                 "value": recomputed, "stored": stored, "diff": None if stored is None else recomputed - stored, "ok": True})
                rows.append({"band": band, "run": region, "seed": seed, "kind": "consistency", "name": "eval_only_max_abs_diff_vs_stored_db",
                             "value": r.get("zero_offset_max_abs_diff_db"), "ok": True})
            if r.get("mask_region"):
                m = r["mask_region"]["mask_check"]
                for k, v in m.items():
                    rows.append({"band": band, "run": region, "seed": seed, "kind": "mask", "name": k, "value": v, "ok": True})
                rows.append({"band": band, "run": region, "seed": seed, "kind": "mask", "name": "noise_params_db_receiver0",
                             "value": json.dumps(r["mask_region"]["noise_params_db"][0]), "ok": True})
    frame = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(frame[frame.kind == "consistency"].to_string(index=False))
    mk = frame[frame.kind == "mask"]
    print("mask checks: in_region_unchanged all true:", bool(mk[mk.name == "in_region_unchanged"].value.astype(bool).all()) if len(mk) else None)
    runs = frame[(frame.kind == "run") & frame.name.isin(["wall_seconds", "peak_gpu_mb"])]
    print(runs.assign(value=pd.to_numeric(runs.value, errors="coerce")).groupby(["band", "name"]).value.describe()[["count", "mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
