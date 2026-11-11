"""Frozen transfer in regimes A, B and C for the final POWDER 15-minute models, on the held-out POWDER receivers and the external sites.

Regimes (definitions in ``training.common.transfer_regimes``):
  A  frozen; weights and the checkpoint's source normalization.
  B  per-window centering (not applied to models that normalize each window themselves).
  C  causal normalizer refit: per-bin mean and standard deviation refitted on the calibration interval (the loaded rows before the first
     scored forecast origin: the 96 blocks = 24 h of history of the first origin; floor on the standard deviation of 0.25 x the source's),
     frozen weights, identical origins. Cm: mean only.
Models: the tuned one-step 1D/2D checkpoints (``p15tuned``, scored on all horizons 1/4/12/24/48 by rollout) and the horizon-specific direct
finals (``direct``: one checkpoint per trained horizon, scored at that horizon only). Targets: POWDER guesthouse and law73 (October test
week) and the usable COSMOS/AERPAW/ARA targets of ``evaluation/analysis/transfer_external_intervals.csv``; same-band transfer only.
Full-intrinsic-normalization models (RevIN family, OFA, Time-LLM, MOMENT, TimeRAN) are run in A only by default: B and C are identical to A
by construction (see transfer_regimes); ``--regimes A B C Cm`` forces all four to verify this on a subset.

Outputs: ``<prefix>/<regime>/<platform>/<site>/<run>/<band>/<rep>/<model>[/h<h>]/seed-N/{summary.json, per_origin.csv}``.

    python -m training.ray.transfer_all run --source direct --models revinmlp patchtst --skip-done
    python -m training.ray.transfer_all run --source rolled --regimes C Cm --skip-done
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREFIX = "transfer-all-oct2026"
LOOKBACK = 96
ROLLED_HORIZONS = (1, 4, 12, 24, 48)
POWDER_ORIGINS = 625                       # per receiver and band (transfer15)
CHEAP_DIRECT = ("revinmlp", "patchtst", "patchtst_noinorm", "patchtst_joint", "patchtst_nopatch", "dlinear", "nlinear", "linearar2d",
                "residuallinearar2d", "residualnonlinear2d")
LLM_DIRECT = ("ofa", "ofa_rand", "ofa_scratch", "timellm", "timeran_lp", "moment_lp")
GPU_FRACTION = {"timellm": 1.0, "ofa": 0.5, "ofa_rand": 0.5, "ofa_scratch": 0.5, "timeran_lp": 0.25, "moment_lp": 0.25}
REGIME_TEXT = {
    "A": "frozen: checkpoint weights and checkpoint source normalization",
    "B": "per-window centering (lookback mean per bin removed in source normalization, added back); not applied to full-intrinsic models",
    "C": "causal normalizer refit: per-bin mean and std from the calibration interval (loaded rows before the first scored origin), std floored "
         "at 0.25 x the source std; frozen weights",
    "Cm": "as C with the mean refitted only (source std kept)"}


# ---------------------------------------------------------------------------------------------- targets
def targets(repo_root: str, groups: list[str], bands: list[str], intervals: str = "evaluation/analysis/transfer_external_intervals.csv") -> list[dict]:
    from training.ray import transfer15 as t15
    from training.ray.powder15 import TEST_RUN

    out: list[dict[str, Any]] = []
    if "powder" in groups:
        out += [{"platform": "powder", "site": r, "run": TEST_RUN, "band": b, "n_origins": POWDER_ORIGINS} for b in bands for r in t15.HELD_OUT]
    table = pd.read_csv(Path(repo_root) / intervals)
    for row in table.to_dict("records"):
        if row["use"] and row["platform"] in groups and row["band"] in bands:
            out.append({**row})
    return out


def target_key(t: dict[str, Any]) -> str:
    return f"{t['platform']}/{t['site']}/{t['run']}/{t['band']}"


def prepare_target(root: str, t: dict[str, Any]):
    """(task, converted 15-minute file) for a target, converting on this node if needed."""
    from training.ray import transfer15 as t15
    from training.ray import transfer_external as te

    if t["platform"] == "powder":
        from training.ray.powder15 import SPATIAL_NODES, ensure_powder15

        ensure_powder15(t["band"], root, SPATIAL_NODES)
        return t15.target_task(t["band"], t["site"]), t15.target_file(root, t["band"], t["site"])
    te.convert(root, t)
    return te.task_for(t), te.converted_path(root, t)


def output_base(prefix: str, regime: str, t: dict[str, Any], rep: str, model: str, seed: int, horizon: int | None) -> str:
    return f"{prefix.strip('/')}/{regime}/{target_key(t)}/{rep}/{model}" + (f"/h{horizon}" if horizon else "") + f"/seed-{seed}"


# --------------------------------------------------------------------------------------------------- job
def job(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one model (all its trained horizons for a direct model) on one target, every requested regime on identical windows."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    t, entries, regimes = spec["target"], spec["entries"], spec["regimes"]
    label, seed = entries[0]["model"], int(entries[0]["seed"])
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record: dict[str, Any] = {"model": label, "seed": seed, "target": target_key(t), "regimes": regimes, "mae": {}}
    started = time.time()
    try:
        import torch

        torch.set_num_threads(max(1, int(args.get("threads", 2))))        # 40+ concurrent jobs on 48-core hosts thrash with torch's default thread count
        from training.common import transfer_1d, transfer_2d
        from training.common import transfer_regimes as R
        from training.ray import transfer15 as t15

        if label in ("ofa", "ofa_rand", "ofa_scratch", "timellm", "timeran_lp", "moment_lp"):
            from training.ray.llm_powder import use_isolated_libs

            use_isolated_libs()
            if label.startswith(("timeran", "moment")):
                from training.ray.tune_direct import ensure_timeran_weights

                ensure_timeran_weights(client, bucket)
        task, file = prepare_target(root, t)
        source2d = transfer_2d.load_target(task, [file], impute=False, max_missing_gap=0)
        mask = R.calibration_mask(source2d.timestamps, task.target_start)
        facts = R.calibration_facts(source2d.timestamps, mask)
        for entry in entries:
            rep, direct = entry["rep"], entry.get("source") == "direct"
            name = entry.get("pipeline", label)
            horizons = [int(entry["horizon"])] if direct else list(ROLLED_HORIZONS)
            horizon_tag = int(entry["horizon"]) if direct else None
            meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
            config = meta["config"]
            config["windowing"] = {"lookback": LOOKBACK, "horizons": horizons}
            config[name].setdefault("train", {})["test_stride"] = 1
            module = transfer_2d if rep == "2d" else transfer_1d
            with tempfile.TemporaryDirectory(prefix="transfer-all-") as scratch_dir:
                checkpoint = Path(scratch_dir) / Path(entry["pt_key"]).name
                client.download_file(bucket, entry["pt_key"], str(checkpoint))
                base_normalization = torch.load(checkpoint, map_location="cpu", weights_only=False)["normalization"]
                source = module.load_target(task, [file], impute=False, max_missing_gap=0)
                reference_rows = None
                for regime in regimes:
                    center, refit = R.regime_flags(label, regime)
                    override = R.refit_normalization(base_normalization, source2d.data, mask, refit) if refit else None
                    result = t15.evaluate_frozen(module, rep, task, source, name, checkpoint, config, seed, center, override,
                                                 origin_horizon=max(ROLLED_HORIZONS))
                    if reference_rows is None:
                        reference_rows = result["rows"]
                    elif not all(np.array_equal(reference_rows[h], result["rows"][h]) for h in horizons):
                        raise RuntimeError("origins differ between regimes")
                    n_rows = len(result["rows"][horizons[0]])
                    n_origins = n_rows // 200 if rep == "1d" else n_rows
                    if t.get("n_origins") and n_origins != int(t["n_origins"]):
                        raise RuntimeError(f"origin count {n_origins} differs from the frozen {t['n_origins']}")
                    scored = t15.score_all(result["predictions"], result["targets"], result["bins"], result["times"])
                    summary = {"kind": "transfer_all", "regime": regime, "regime_definition": REGIME_TEXT[regime], "model": label,
                               "pipeline": name, "rep": rep, "seed": seed, "trained_horizon": horizon_tag, "target": target_key(t),
                               "platform": t["platform"], "site": t["site"], "run": t["run"], "target_band": t["band"],
                               "source_band": entry["band"], "checkpoint_key": entry["pt_key"], "checkpoint_sha256": meta.get("sha256"),
                               "intrinsic_window_normalization": R.intrinsic_class(label), "centering_applied": center,
                               "normalizer_refit": refit, **(facts if refit else {}), "n_origins": n_origins,
                               "code_revision": args["code_revision"], "metrics": scored["metrics"]}
                    base = output_base(args["prefix"], regime, t, rep, label, seed, horizon_tag)
                    client.put_object(Bucket=bucket, Key=f"{base}/summary.json", Body=json.dumps(summary, default=str, indent=1).encode())
                    client.put_object(Bucket=bucket, Key=f"{base}/per_origin.csv", Body=scored["per_origin"].to_csv(index=False).encode())
                    record["mae"].setdefault(regime, {}).update({str(h): scored["metrics"][str(h)]["mae_db"] for h in horizons})
        record["ok"] = True
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        try:
            client.put_object(Bucket=bucket, Key=f"{args['prefix']}/errors/{target_key(t)}/{label}-seed{seed}.json",
                              Body=json.dumps(record, default=str).encode())
        except Exception:
            pass
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


# ------------------------------------------------------------------------------------------------ driver
def default_regimes(source: str, model: str) -> list[str]:
    from training.common.transfer_regimes import intrinsic_class

    if source == "rolled":
        return ["C", "Cm"]                                  # A and B of the rolled-out models exist (transfer15-oct2026*, transfer15-external-oct2026)
    return ["A"] if intrinsic_class(model) == "full" else ["A", "B", "C", "Cm"]


def cmd_run(a: argparse.Namespace) -> None:
    import ray

    sys.path.insert(0, a.repo_root)
    from training.ray.inference_benchmark import discover_sources
    from training.ray.storage import spectrum_s3_client

    bucket = a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    client = spectrum_s3_client()
    revision = a.code_revision
    if revision is None:
        try:
            revision = subprocess.run(["git", "-C", a.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            revision = "unknown"
    tgts = targets(a.repo_root, a.groups, a.bands)
    if a.targets:
        tgts = [t for t in tgts if any(s in target_key(t) for s in a.targets)]
    if a.source == "rolled":
        entries = [e for e in discover_sources(client, bucket, ["p15tuned"], a.models)
                   if e["band"] in a.bands and e["rep"] in a.reps and int(e["seed"]) in a.seeds]
        groups = [[e] for e in entries]
    else:
        models = a.models or list(CHEAP_DIRECT)
        entries = [e for e in discover_sources(client, bucket, ["direct"], models)
                   if e["resolution"] == "15min" and e["band"] in a.bands and int(e["seed"]) in a.seeds and int(e["horizon"]) in a.horizons]
        by: dict[tuple, list] = {}
        for e in entries:
            by.setdefault((e["model"], e["band"], int(e["seed"])), []).append(e)
        groups = [sorted(v, key=lambda e: int(e["horizon"])) for v in by.values()]
    jobs = []
    for g in groups:
        regimes = a.regimes or default_regimes(a.source, g[0]["model"])
        for t in tgts:
            if t["band"] != g[0]["band"]:
                continue
            jobs.append({"target": t, "entries": g, "regimes": regimes})

    def done(j: dict[str, Any]) -> bool:
        e = j["entries"][-1]
        key = output_base(a.prefix, j["regimes"][-1], j["target"], e["rep"], e["model"], int(e["seed"]),
                          int(e["horizon"]) if e.get("source") == "direct" else None)
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/summary.json")
            return True
        except Exception:
            return False

    if a.skip_done:
        jobs = [j for j in jobs if not done(j)]
    print(f"[transfer-all] {len(tgts)} targets, {len(groups)} model groups, {len(jobs)} jobs, revision {revision[:12]}", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "code_revision": revision, "threads": a.threads}
    ray.init(address=a.address, ignore_reinit_error=True)
    queue, pending, ok, failed = list(jobs), {}, 0, 0

    def submit(j):
        m = j["entries"][0]["model"]
        gpu = a.gpu_fraction if a.gpu_fraction is not None else GPU_FRACTION.get(m, 0.0)
        memory = int((24.0 if m in GPU_FRACTION else a.memory_gb) * 2 ** 30)
        ref = ray.remote(num_cpus=2, num_gpus=gpu, memory=memory, max_retries=1)(job).remote(j, shared)
        pending[ref] = j

    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            submit(queue.pop(0))
        finished, _ = ray.wait(list(pending), num_returns=1)
        j = pending.pop(finished[0])
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}", "model": j["entries"][0]["model"],
                   "target": target_key(j["target"])}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        shown = {r: next(iter(m.values())) if len(m) == 1 else m.get("12") for r, m in (row.get("mae") or {}).items()}
        print(f"[transfer-all] {row.get('target')} {row.get('model')} seed{row.get('seed')} ok={row.get('ok')} "
              f"mae={{{', '.join(f'{r}:{v:.3f}' for r, v in shown.items() if v is not None)}}} {row.get('error', '')} "
              f"({row.get('wall_seconds')}s) [{len(queue) + len(pending)} left]", flush=True)
    print(f"[transfer-all] done: {ok} ok, {failed} failed", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", choices=("rolled", "direct"), required=True)
    r.add_argument("--models", nargs="+")
    r.add_argument("--bands", nargs="+", default=["600_800", "2400_2600"])
    r.add_argument("--groups", nargs="+", default=["powder", "cosmos", "aerpaw", "ara"])
    r.add_argument("--targets", nargs="+", help="only targets whose key contains one of these strings (smoke tests)")
    r.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    r.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    r.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 12, 24, 48], help="trained horizons (direct source)")
    r.add_argument("--regimes", nargs="+", choices=["A", "B", "C", "Cm"])
    r.add_argument("--gpu-fraction", type=float, default=None, help="override (default: 0 for the cheap models, per-model for the LLM ones)")
    r.add_argument("--memory-gb", type=float, default=8.0)
    r.add_argument("--threads", type=int, default=2, help="torch threads per job")
    r.add_argument("--max-parallel", type=int, default=24)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--bucket", default=None)
    r.add_argument("--prefix", default=PREFIX)
    r.add_argument("--code-revision", default=None)
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
