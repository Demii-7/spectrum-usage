"""Cross-band positional transfer of the October-trained 15-minute models to the POWDER July 2026 "New frequency" bands. READY, NOT RUN.

Table "Transferability and generalization evaluation sets", row "New frequency": the 07-03 2026 run (2026-07-03 18:39 to about 07-06 09:50 UTC)
at humanities, madsen, moran, sagepoint, guesthouse, cpg and ebc in 800-1000 MHz and 5725-5925 MHz (200 one-MHz bins each, columns at x.5 MHz).
The user has NOT chosen this experiment yet (options in training/CAMPAIGN_STATUS.md 36.5: this positional transfer with centering, the same with
a refitted normalizer, or a footnote). Nothing here has been run beyond a one-cell smoke test.

What is measured: a checkpoint trained on 600-800 MHz or 2400-2600 MHz (200 bins) is applied bin-by-bin, by position, to the 200 bins of a new
band (all four source/target pairs). The model never sees a frequency axis, so this scores only whether temporal structure learned on one band
carries to another, not any knowledge of the new band's spectrum. Regimes (``training.common.transfer_regimes``):
  A   frozen weights and the checkpoint's own per-bin source statistics (a diagnostic here: the statistics belong to other bins);
  B   per-window centering (the lookback mean of each window and bin removed before prediction, added back); the planned headline regime;
  C   (``--regime-c``) causal normalizer refit from the 24 h before the first scored origin, mean and std (Cm: mean only).
Full-intrinsic-normalization models (RevIN-MLP, PatchTST and variants) are run in A only (B and C are identical by construction).

Caveats that go with any result:
  * 5725-5925 MHz is almost entirely noise floor (mean level about -131 dBm; 800-1000 about -128 dBm; the training bands about -122 dBm at 600-800),
    so errors are small in absolute terms and persistence/lookback mean are hard to beat; a ratio to the lookback mean is the meaningful number;
  * the level mismatch between a source band's per-bin statistics and the target band is 6-9 dB (``level_mismatch_db`` measures it per target),
    which is why regime A is expected to be poor and B or C is needed;
  * the per-bin source statistics (mean and std in dBm per bin position) are meaningless for a different band; they act only as a fixed affine
    map per position, and regime C replaces them with target statistics from an earlier interval;
  * only the 07-03 run is usable (the June 28 run did not cover these bands), 61-63 hours per receiver (about 104-110 origins); guesthouse has two
    gaps above 10 minutes in that run (longest segment 21 h, no origin) and is excluded by the freeze rules;
  * combined site-and-band shifts are absent: the site is POWDER throughout, so this isolates the band change.

Targets are frozen before any model is scored (``freeze``), by the rules of ``training.ray.transfer_external``: longest segment with no raw gap above
10 minutes, core 15-minute conversion, scored interval starting 24 h after the first block, at least 96 origins, one shared set of origins per
target. Converted files go to ``data/ext15/powder/<site>/<run>/<band>/`` (distinct from the same-band July conversions).

Outputs: ``<prefix>/from-<source band>/<regime>/powder/<site>/<run>/<target band>/<rep>/<model>[/h<h>]/seed-N/{summary.json, per_origin.csv}``
(identical layout to ``transfer_all`` under a per-source-band prefix) and baselines ``<prefix>/baseline/powder/<site>/<run>/<target band>/<name>/``.

    python -m training.ray.transfer_crossband freeze
    python -m training.ray.transfer_crossband run --skip-done                      # regimes A and B
    python -m training.ray.transfer_crossband run --skip-done --regime-c           # also C and Cm
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CROSS_BANDS = {"800_1000": (800.0, 1000.0), "5725_5925": (5725.0, 5925.0)}
SOURCE_BANDS = ("600_800", "2400_2600")
INTERVALS_CSV = "evaluation/analysis/transfer_crossband_intervals.csv"
PREFIX = "transfer-crossband-july-2026"
RUN_DAY = "20260703"
NODES = ("humanities-nuc1", "madsen-nuc1", "moran-nuc1", "sagepoint-nuc1", "guesthouse-nuc1", "cpg-nuc1", "ebc-nuc1")
LOOKBACK = 96

from training.ray import transfer_external as te  # noqa: E402

te.BANDS.update(CROSS_BANDS)           # conversion and task construction of transfer_external know the new bands (also in every worker: module import)


# ------------------------------------------------------------------------------------------ pure helpers
def raw_files(repo_root: str) -> list[dict[str, Any]]:
    """The 07-03 run's raw per-minute files for the two new bands (backup copies excluded)."""
    found = []
    for node in NODES:
        for band in CROSS_BANDS:
            for path in sorted(glob.glob(f"{repo_root}/data/powder/{node}/*/{band}/power_1mhz_avg_per_minute.csv")):
                run = path.split("/")[-3]
                if run[:8] == RUN_DAY and "backup" not in run:
                    found.append({"platform": "powder", "site": node, "run": run, "band": band, "raw": path})
    return found


def source_target_pairs(source_bands=SOURCE_BANDS, target_bands=tuple(CROSS_BANDS)) -> list[tuple[str, str]]:
    """Every (source band, target band) pair: each checkpoint band is applied to each new band."""
    return [(s, t) for s in source_bands for t in target_bands]


def default_regimes(model: str, regime_c: bool = False) -> list[str]:
    """Regimes to run: A and B (plus C and Cm with ``regime_c``); full-intrinsic-normalization models only A (B, C identical by construction)."""
    from training.common.transfer_regimes import intrinsic_class

    if intrinsic_class(model) == "full":
        return ["A"]
    return ["A", "B"] + (["C", "Cm"] if regime_c else [])


def level_mismatch_db(base_normalization: dict[str, Any], data: np.ndarray, mask: np.ndarray) -> float:
    """Mean over bins of (target calibration-interval mean level minus the checkpoint's per-bin source mean), dB: how far the source statistics are
    from the new band's level. ``data`` is (T, 200) dB, ``mask`` the calibration rows."""
    rows = np.asarray(data, dtype=np.float64)[np.asarray(mask)]
    source = np.asarray(base_normalization["mean_dbm"], dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[1] != source.size:
        raise ValueError("bin counts differ")
    return float((rows.mean(axis=0) - source).mean())


def targets(repo_root: str, bands: list[str], intervals: str = INTERVALS_CSV) -> list[dict[str, Any]]:
    table = pd.read_csv(Path(repo_root) / intervals)
    return [{**r} for r in table.to_dict("records") if r["use"] and r["band"] in bands]


# ---------------------------------------------------------------------------------------------- workers
def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """One model group on one frozen target, regimes on identical windows (``transfer_all.job`` with a per-source-band output prefix)."""
    from training.ray import transfer_all as ta

    source_band = spec["entries"][0]["band"]
    return ta.job(spec, {**args, "prefix": f"{args['prefix'].strip('/')}/from-{source_band}"})


def baseline_job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Persistence and the lookback mean on one frozen target (independent of the source band)."""
    return te.baseline_job(spec, args)


# ---------------------------------------------------------------------------------------------- drivers
def cmd_freeze(a: argparse.Namespace) -> None:
    te.raw_files = raw_files                        # the same freeze(), applied to the 07-03 run's new-band files
    te.INTERVALS_CSV = INTERVALS_CSV
    te.cmd_freeze(argparse.Namespace(repo_root=a.repo_root, out=a.out))


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    sys.path.insert(0, a.repo_root)
    from training.ray import transfer_all as ta
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    try:
        revision = a.code_revision or subprocess.run(["git", "-C", a.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                                     check=True).stdout.strip()
    except Exception:
        revision = "unknown"
    tgts = targets(a.repo_root, a.target_bands, a.intervals)
    if a.targets:
        tgts = [t for t in tgts if any(s in ta.target_key(t) for s in a.targets)]
    groups: list[list[dict[str, Any]]] = []
    if "rolled" in a.kinds:
        groups += [[e] for e in discover_sources(client, bucket, ["p15tuned"], a.models)
                   if e["band"] in a.sources and e["rep"] in a.reps and int(e["seed"]) in a.seeds]
    if "direct" in a.kinds:
        by: dict[tuple, list] = {}
        for e in discover_sources(client, bucket, ["direct"], a.direct_models or [m for m in ta.CHEAP_DIRECT]):
            if e["resolution"] == "15min" and e["band"] in a.sources and int(e["seed"]) in a.seeds and int(e["horizon"]) in a.horizons:
                by.setdefault((e["model"], e["band"], int(e["seed"])), []).append(e)
        groups += [sorted(v, key=lambda e: int(e["horizon"])) for v in by.values()]
    jobs = [{"target": t, "entries": g, "regimes": a.regimes or default_regimes(g[0]["model"], a.regime_c)} for g in groups for t in tgts]
    base_jobs = [{"target": t} for t in tgts]

    def done(j: dict[str, Any]) -> bool:
        e = j["entries"][-1]
        key = ta.output_base(f"{a.prefix.strip('/')}/from-{e['band']}", j["regimes"][-1], j["target"], e["rep"], e["model"], int(e["seed"]),
                             int(e["horizon"]) if e.get("source") == "direct" else None)
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/summary.json")
            return True
        except Exception:
            return False

    if a.skip_done:
        jobs = [j for j in jobs if not done(j)]
        base_jobs = [j for j in base_jobs if not _baseline_done(client, bucket, a.prefix, j["target"])]
    print(f"[crossband] {len(tgts)} targets, {len(groups)} model groups, {len(jobs)} jobs + {len(base_jobs)} baseline jobs, revision {revision[:12]}",
          flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "code_revision": revision, "threads": a.threads}
    ray.init(address=a.address, ignore_reinit_error=True)
    memory = int(a.memory_gb * 2 ** 30)
    work = [("baseline", j) for j in base_jobs] + [("model", j) for j in jobs]
    pending, ok, failed = {}, 0, 0
    queue = list(work)
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            kind, j = queue.pop(0)
            fn = baseline_job if kind == "baseline" else job
            pending[ray.remote(num_cpus=2, num_gpus=0, memory=memory, max_retries=1)(fn).remote(j, shared)] = (kind, j)
        finished, _ = ray.wait(list(pending), num_returns=1)
        kind, j = pending.pop(finished[0])
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        name = "baselines" if kind == "baseline" else f"{j['entries'][0]['model']} from {j['entries'][0]['band']}"
        shown = {r: m for r, m in (row.get("mae") or {}).items()}
        print(f"[crossband] {ta.target_key(j['target'])} {name} ok={row.get('ok')} mae={shown} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[crossband] done: {ok} ok, {failed} failed", flush=True)


def _baseline_done(client, bucket: str, prefix: str, t: dict[str, Any]) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=f"{te.baseline_dir(prefix, t, 'lookbackmean')}/summary.json")
        return True
    except Exception:
        return False


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="convert the 07-03 new-band files and write the frozen interval table")
    f.add_argument("--repo-root", default="/workspace/spectrum-usage")
    f.add_argument("--out", default=INTERVALS_CSV)
    f.set_defaults(func=cmd_freeze)
    r = sub.add_parser("run", help="score checkpoints positionally on every frozen target")
    r.add_argument("--kinds", nargs="+", default=["rolled", "direct"], choices=["rolled", "direct"])
    r.add_argument("--sources", nargs="+", default=list(SOURCE_BANDS), choices=list(SOURCE_BANDS), help="checkpoint bands")
    r.add_argument("--target-bands", nargs="+", default=list(CROSS_BANDS), choices=list(CROSS_BANDS))
    r.add_argument("--targets", nargs="+", help="only targets whose key contains one of these strings (smoke tests)")
    r.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    r.add_argument("--models", nargs="+", help="rolled-out models (default all)")
    r.add_argument("--direct-models", nargs="+", help="direct models (default: the cheap set, no isolated libraries)")
    r.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    r.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 12, 24, 48], help="trained horizons (direct models)")
    r.add_argument("--regimes", nargs="+", choices=["A", "B", "C", "Cm"])
    r.add_argument("--regime-c", action="store_true", help="also run C and Cm (normalizer refit on the 24 h before the first origin)")
    r.add_argument("--intervals", default=INTERVALS_CSV)
    r.add_argument("--memory-gb", type=float, default=6.0)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--max-parallel", type=int, default=24)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--bucket", default=None)
    r.add_argument("--prefix", default=PREFIX)
    r.add_argument("--code-revision", default=None)
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo_root)
    args.func(args)


if __name__ == "__main__":
    main()
