"""Frozen zero-shot transfer (stage A) and per-window centering (stage B) of the 15-minute 1D/2D checkpoints to the POWDER June-July 2026 collection.

Table "Transferability and generalization evaluation sets", row "Separate collection": the two June-July runs (collection start 2026-06-28
04:37 UTC, running to 07-03 about 02:50; and 2026-07-03 18:39 to 07-06 about 09:50) at the receivers humanities, madsen, moran, sagepoint,
guesthouse, cpg and ebc, in 600-800 and 2400-2600 MHz, same-band transfer, identical protocol to ``training.ray.transfer_external`` (which this
module reuses unchanged: target freezing and conversion, the stage A/B jobs, the persistence and lookback-mean baselines).

Differences from the external-site runs: the platform is ``powder``, a target is one (receiver, run, band) file, and the frozen table is
``evaluation/analysis/transfer_july_intervals.csv``; results go under ``transfer15-july-2026/``. The two runs are separated by a gap of about 16 h
(07-03 03:00 to 18:39), so they stay two targets per receiver and band (they are never merged). Nothing in the selection looks at model performance.

    python -m training.ray.transfer_july freeze
    python -m training.ray.transfer_july run --skip-done
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

INTERVALS_CSV = "evaluation/analysis/transfer_july_intervals.csv"
PREFIX = "transfer15-july-2026"
NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1", "guesthouse-nuc1", "cpg-nuc1", "ebc-nuc1")
RUN_DAYS = ("20260628", "20260703")                 # run directories that start on these days (the exact minute differs per receiver)


def raw_files(repo_root: str) -> list[dict]:
    from training.ray import transfer_external as te

    found = []
    for node in NODES:
        for path in sorted(glob.glob(f"{repo_root}/data/powder/{node}/*/*/power_1mhz_avg_per_minute.csv")):
            parts = path.split("/")
            run, band = parts[-3], parts[-2]
            if run[:8] in RUN_DAYS and "backup" not in run and band in te.BANDS:
                found.append({"platform": "powder", "site": node, "run": run, "band": band, "raw": path})
    return found


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="convert every target and write the frozen interval table")
    f.add_argument("--repo-root", default="/workspace/spectrum-usage")
    f.add_argument("--out", default=INTERVALS_CSV)
    r = sub.add_parser("run", help="score every checkpoint on every frozen target (stages A and B) plus the baselines")
    r.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    r.add_argument("--platforms", nargs="+", default=["powder"])
    r.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    r.add_argument("--models", nargs="+")
    r.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    r.add_argument("--intervals", default=INTERVALS_CSV)
    r.add_argument("--gpu-fraction", type=float, default=0.0)
    r.add_argument("--memory-gb", type=float, default=12.0)
    r.add_argument("--max-parallel", type=int, default=24)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--bucket", default=None)
    r.add_argument("--prefix", default=PREFIX)
    r.add_argument("--code-revision", default=None)
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    from training.ray import transfer_external as te

    if args.command == "freeze":
        te.raw_files = raw_files                    # the same freeze(), applied to the June-July POWDER files
        te.INTERVALS_CSV = INTERVALS_CSV
        te.cmd_freeze(argparse.Namespace(repo_root=args.repo_root, out=args.out))
    else:
        te.cmd_run(args)


if __name__ == "__main__":
    main()
