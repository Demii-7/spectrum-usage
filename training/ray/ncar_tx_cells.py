"""NCAR transfer batch: score every model cell once, with per-window arrays and per-bin metrics, into ``transfer-ncar-2026-11/cells/``.

One cell = (model, seed, condition, target), scored by the unchanged ``transfer_all.job`` (``save_arrays`` on, so ``forecasts.npz`` and
``per_bin.csv`` are written next to ``summary.json``/``per_origin.csv``). Items 3, 4, 4b, 9 and 10 are computed from these arrays on CPU.

Target groups (``--groups``): ``external`` (AERPAW/ARA/COSMOS, ``transfer_external_intervals.csv``), ``heldout`` (guesthouse, law73, October
test week), ``july`` (``transfer_july_intervals.csv``), ``core4_oct`` (the core four, in distribution; frozen only, the reference),
``core4_week2`` (the core four on the second test week; frozen only).
Conditions: A frozen, B centered, C refit, Cm mean-only refit (``training/common/transfer_regimes.py``); full-intrinsic-normalization models
(RevIN family, PatchTST, OFA, Time-LLM, MOMENT, TimeRAN) only A (identical by construction). Only A on core4_oct and core4_week2.
Stages (``--stage``): ``gpu-full`` (OFA, Time-LLM, MOMENT, TimeRAN), ``core`` (the 1D/2D core models and the cheap direct models; GPU share
``--gpu-fraction``, 0 = CPU). Order: level-keeping models' external cells first, then the rest of the external cells, then held-out, July,
core4 October, week 2. Seeds: those of the existing transfer runs (``transfer_all_regimes.csv``): 40-44, TCN 40/41/43/44, Time-LLM 40-42.

    python -m training.ray.ncar_tx_cells --stage gpu-full --skip-done
    python -m training.ray.ncar_tx_cells --stage core --gpu-fraction 0.1 --skip-done
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

PREFIX = "transfer-ncar-2026-11/cells"
ROLLED = ("linearar1d", "residuallinearar1d", "vanillalstm1d", "residualvanillalstm", "linearar2d", "residuallinearar2d", "vanillalstm",
          "lstmattn", "temporalconvnet", "autoformer_csa")
DIRECT_CORE = ("revinmlp", "revinmlp_mix", "patchtst", "patchtst_joint", "nlinear", "dlinear", "linearar2d", "residuallinearar2d",
               # added 11-12 22:30 UTC: every other 15-minute direct model with finals in S3 (Part 1 ran the first three, and
               # DeepSPred had no transfer run; iTransformer left out on request). Separate jobs (--models), so running jobs are not duplicated.
               "patchtst_noinorm", "patchtst_nopatch", "residualnonlinear2d", "deepspred2d")
DIRECT_LLM = ("timellm", "ofa", "moment_lp", "timeran_lp", "ofa_rand", "ofa_scratch")
SEEDS = {"temporalconvnet": (40, 41, 43, 44), "timellm": (40, 41, 42)}
DEFAULT_SEEDS = (40, 41, 42, 43, 44)
GROUP_ORDER = ("external", "heldout", "core4_oct", "july", "core4_week2", "heldout_week2")    # deadline order: the reference week before July, week 2 last
# brief item 9: the June-July gaps for the large models are OFA and MOMENT at 600-800, Time-LLM everywhere, TimeRAN in both bands
JULY_ITEM9 = {"ofa": ("600_800",), "moment_lp": ("600_800",), "timellm": ("600_800", "2400_2600"), "timeran_lp": ("600_800", "2400_2600")}


def group_targets(root: str, group: str, bands: list[str]) -> list[dict[str, Any]]:
    from training.ray import transfer_all as ta
    from training.ray.ncar_cells import core_targets

    if group == "external":
        return ta.targets(root, ["cosmos", "aerpaw", "ara"], bands, ta.EXTERNAL_INTERVALS, core=False)
    if group == "heldout":
        return [t for t in ta.targets(root, ["powder"], bands, ta.EXTERNAL_INTERVALS, core=True) if t["platform"] == "powder"]
    if group == "july":
        return ta.targets(root, ["powder"], bands, ta.JULY_INTERVALS, core=False)
    if group in ("core4_oct", "core4_week2"):
        return core_targets(bands, group.split("_")[1])
    if group == "heldout_week2":                       # guesthouse, law73 on the second test week ("Later interval" row of the paper's table)
        return core_targets(bands, "week2", receivers=("guesthouse", "law73"))
    raise ValueError(group)


def model_groups(client, bucket: str, stage: str, bands: list[str], models: list[str] | None, horizons: list[int]) -> list[list[dict]]:
    from training.ray.inference_benchmark import discover_sources

    out: list[list[dict]] = []
    if stage == "core":
        rolled = [m for m in ROLLED if not models or m in models]
        for e in (discover_sources(client, bucket, ["p15tuned"], rolled) if rolled else []):   # empty would mean "every model"
            if e["band"] in bands and int(e["seed"]) in SEEDS.get(e["model"], DEFAULT_SEEDS):
                out.append([{**e, "source_kind": "rolled"}])
    direct = [m for m in (DIRECT_CORE if stage == "core" else DIRECT_LLM) if not models or m in models]
    by: dict[tuple, dict[int, dict]] = {}
    if not direct:                                     # an empty list would mean "every model" to discover_sources
        return out
    for e in discover_sources(client, bucket, ["direct"], direct):
        if e["resolution"] == "15min" and e["band"] in bands and int(e["seed"]) in SEEDS.get(e["model"], DEFAULT_SEEDS) \
                and int(e["horizon"]) in horizons:
            by.setdefault((e["model"], e["band"], int(e["seed"])), {})[int(e["horizon"])] = {**e, "source_kind": "direct"}
    out += [[v[h] for h in sorted(v)] for v in by.values()]
    return out


def build_jobs(root: str, groups: list[str], model_sets: list[list[dict]], targets_filter: list[str] | None,
               july_item9: bool = False) -> list[dict[str, Any]]:
    from training.common.transfer_regimes import intrinsic_class
    from training.ray import transfer_all as ta
    from training.ray.ncar_cells import LEVEL_KEEPING

    bands = sorted({g[0]["band"] for g in model_sets})
    jobs = []
    for group in groups:
        for t in group_targets(root, group, bands):
            if targets_filter and not any(s in ta.target_key(t) for s in targets_filter):
                continue
            for g in model_sets:
                if g[0]["band"] != t["band"]:
                    continue
                model = g[0]["model"]
                if july_item9 and group == "july" and model in JULY_ITEM9 and t["band"] not in JULY_ITEM9[model]:
                    continue
                full = intrinsic_class(model) == "full"
                regimes = ["A"] if (full or group in ("core4_oct", "core4_week2")) else ["A", "B", "C", "Cm"]
                priority = (0 if (group == "external" and model in LEVEL_KEEPING) else 1, GROUP_ORDER.index(group))
                jobs.append({"target": t, "entries": g, "regimes": regimes, "group": group, "priority": priority})
    jobs.sort(key=lambda j: j["priority"])
    return jobs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("gpu-full", "core"), required=True)
    p.add_argument("--groups", nargs="+", default=list(GROUP_ORDER), choices=list(GROUP_ORDER))
    p.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--targets", nargs="+", help="only targets whose key contains one of these strings")
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48], help="trained horizons of the direct models")
    p.add_argument("--gpu-fraction", type=float, default=None, help="GPU share per job (default: transfer_all.GPU_FRACTION, 0 for core)")
    p.add_argument("--gpu-share", nargs="+", default=[], help="per-model override, e.g. timellm=0.5")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="stop after this many jobs (smoke tests)")
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--july-item9", action="store_true", help="June-July large-model cells only where brief item 9 asks for them")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--code-revision", default=None, help="the commit synced to the cluster (recorded in every summary)")
    p.add_argument("--address", default="auto")
    p.add_argument("--repo-root", default="/workspace/spectrum-usage")
    a = p.parse_args(argv)
    sys.path.insert(0, a.repo_root)
    os.chdir(a.repo_root)
    import ray

    from training.ray import transfer_all as ta
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    revision = a.code_revision or "unknown"            # the NCAR snapshot is not a git clone: pass the synced commit
    sets = model_groups(client, bucket, a.stage, a.bands, a.models, a.horizons)
    jobs = build_jobs(a.repo_root, a.groups, sets, a.targets, a.july_item9)
    overrides = {k: float(v) for k, v in (s.split("=") for s in a.gpu_share)}

    def done(j: dict[str, Any]) -> bool:
        e = j["entries"][-1]
        key = ta.output_base(a.prefix, j["regimes"][-1], j["target"], e["rep"], e["model"], int(e["seed"]),
                             int(e["horizon"]) if e.get("source") == "direct" else None)
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/per_bin.csv")
            return True
        except Exception:
            return False

    total = len(jobs)
    if a.skip_done:
        jobs = [j for j in jobs if not done(j)]
    if a.limit:
        jobs = jobs[:a.limit]
    from collections import Counter

    print(f"[ncar-cells] stage {a.stage}: {len(sets)} model groups, {total} jobs, {len(jobs)} to run, revision {revision[:12]}", flush=True)
    print(f"[ncar-cells] by model: {dict(Counter(j['entries'][0]['model'] for j in jobs))}", flush=True)
    print(f"[ncar-cells] by group: {dict(Counter(j['group'] for j in jobs))}", flush=True)
    if a.dry_run:
        return
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "code_revision": revision, "threads": a.threads,
              "save_arrays": True}
    ray.init(address=a.address, ignore_reinit_error=True)
    queue, pending, ok, failed = list(jobs), {}, 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            j = queue.pop(0)
            m = j["entries"][0]["model"]
            gpu = overrides.get(m, a.gpu_fraction if a.gpu_fraction is not None else ta.GPU_FRACTION.get(m, 0.0))
            memory = int((24.0 if m in ta.GPU_FRACTION else a.memory_gb) * 2 ** 30)
            spec = {k: j[k] for k in ("target", "entries", "regimes")}
            pending[ray.remote(num_cpus=a.cpus, num_gpus=gpu, memory=memory, max_retries=1)(ta.job).remote(spec, shared)] = j
        finished, _ = ray.wait(list(pending), num_returns=1)
        j = pending.pop(finished[0])
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}", "model": j["entries"][0]["model"],
                   "target": ta.target_key(j["target"])}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        a_mae = (row.get("mae") or {}).get("A", {})
        print(f"[ncar-cells] {j['group']} {row.get('target')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} "
              f"A t+12={a_mae.get('12')} {row.get('error', '')} ({row.get('wall_seconds')}s, gpu peak {row.get('gpu_peak_gb')} GB) "
              f"[{len(queue) + len(pending)} left]", flush=True)
    print(f"[ncar-cells] done: {ok} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
