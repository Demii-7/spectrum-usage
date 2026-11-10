"""Power-difference entropy of the October POWDER traces, per frequency bin and per annotated region (cluster-side; needs data/powder*).

Same measure as the paper's entropy section (Ding et al.): the series is quantized into Q = 8 states by common dB thresholds on its
first differences and the entropy rate is estimated by LZ78 phrase counting, normalized by log2 Q (``compute_region_entropy.metrics``,
the ``E_actual_normalized`` of the differences). It is computed on the October TRAINING run (20261015T1434Z, 2026-10-15 to 10-22), so the
band's character is measured on data that no test score depends on, for each of the four core receivers, then averaged over receivers.

* per bin: every 1 MHz bin's own series (both bands, 1-minute and 15-minute traces);
* per region (600-800 only, the annotated plan ``plan_regions_600_800.csv``): the series is the median over the region's bins, as in
  the paper.

    python evaluation/analysis/october_entropy.py --root /workspace/spectrum-usage --out /tmp/october_entropy.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.analysis.compute_region_entropy import DEFAULT_DIFF_EDGES, metrics  # noqa: E402

NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1")
RUN = "20261015T1434Z"
EDGES = np.asarray(DEFAULT_DIFF_EDGES, dtype=float)
TRACES = {"1min": ("data/powder", "power_1mhz_avg_per_minute.csv"), "15min": ("data/powder15", "power_15min.csv")}


def diff_entropy(series: np.ndarray) -> tuple[float, int]:
    values = np.diff(series[np.isfinite(series)])
    if len(values) < 50:
        return float("nan"), int(len(values))
    return float(metrics(values, EDGES, include_diff=False)["E_actual_normalized"]), int(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/workspace/spectrum-usage")
    parser.add_argument("--out", default="october_entropy.csv")
    args = parser.parse_args()
    root = Path(args.root)
    regions = pd.read_csv(root / "evaluation/analysis/plan_regions_600_800.csv")
    rows = []
    for resolution, (folder, name) in TRACES.items():
        for band in ("600_800", "2400_2600"):
            per_node = {}
            for node in NODES:
                frame = pd.read_csv(root / folder / node / RUN / band / name)
                frame = frame.drop(columns=["timestamp_utc"])
                per_node[node] = frame
            frequencies = [float(c) for c in per_node[NODES[0]].columns]
            for index, frequency in enumerate(frequencies):
                vals = [diff_entropy(per_node[n].iloc[:, index].to_numpy(dtype=float)) for n in NODES]
                rows.append({"resolution": resolution, "band": band, "level": "bin", "id": f"{frequency:.1f}", "start_mhz": frequency,
                             "end_mhz": frequency, "entropy": float(np.nanmean([v[0] for v in vals])),
                             "entropy_by_node": ";".join(f"{v[0]:.4f}" for v in vals), "n_diffs": int(min(v[1] for v in vals)),
                             "behavior": ""})
            if band == "600_800":
                for region in regions.itertuples():
                    keep = [i for i, f in enumerate(frequencies) if region.start_mhz - 1e-6 <= f <= region.end_mhz + 1e-6]
                    vals = [diff_entropy(np.nanmedian(per_node[n].iloc[:, keep].to_numpy(dtype=float), axis=1)) for n in NODES]
                    rows.append({"resolution": resolution, "band": band, "level": "region", "id": region.band_id,
                                 "start_mhz": region.start_mhz, "end_mhz": region.end_mhz, "entropy": float(np.nanmean([v[0] for v in vals])),
                                 "entropy_by_node": ";".join(f"{v[0]:.4f}" for v in vals), "n_diffs": int(min(v[1] for v in vals)),
                                 "behavior": f"{region.behavior_category}{'|noise_floor' if region.is_noise_floor else ''}"})
            print(f"[entropy] {resolution} {band} done", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[entropy] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
