"""Reduced 15-minute search for DSwinLSTM-I (both bands), then automatic finals for a winner that beats the inherited configuration.

Why: the only 15-minute DSwinLSTM-I result (seed 42) uses the 1-minute round-2 configuration (``inherited:1min-r2``, no 15-minute search); it is last
or near last among the 4D models. STS-PredNet and ResidualConvLSTM already have 16-trial 15-minute searches, so only this model is searched here.

Search space: the registry's (``training/ray/registry.py``: five architecture presets x learning rate x weight decay x batch size), the same space the
1-minute search drew from; 8 candidates per band = the three anchors (candidate 2 IS the inherited configuration: embed 32, lr 1e-4, batch 2, weight
decay 0, which gives a paired replicate of the existing seed-42 final) plus five seeded random draws (``tune_fixed.candidate_list``, identical to what
``tune_fixed --budget 8`` would draw). Every candidate is trained with seed 42 exactly like the inherited final (``apply_fixed_config``: output length
48, strides as the finals, selection by validation MAE ``val_mean_horizon_mae_db``), so objectives are directly comparable. Nothing is written to the
existing ``tuning/`` prefix.

    python -m training.ray.tune_dswin15 run [--bands 600_800 2400_2600] [--only-index N] [--epochs E] [--prefix tuning-dswin-search]
    python -m training.ray.tune_dswin15 decide          # winner per band, verdict JSON + trials CSV, finals for a winner >= 3% better
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

MODEL, REP, SOURCE = "dswinlstm_i", "4d", "powder15_4d"
BANDS = ("600_800", "2400_2600")
INHERITED = "training/configs/fixed/dswinlstm_i_inherited_1min_r2.json"
BASELINE_PREFIX = "tuning"                       # the existing seed-42 finals live here (read only)
DEFAULT_PREFIX = "tuning-dswin-search"
SEARCH_SEED = 42
FINAL_SEEDS = (40, 41, 42, 43, 44)
IMPROVEMENT = 0.03                               # a winner must beat the inherited validation objective by this fraction to get finals


def candidates_for(spec, band: str, budget: int = 8) -> list[dict[str, Any]]:
    from .tune_fixed import candidate_list

    return candidate_list(spec, budget, random.Random(f"{SOURCE}|{band}|{REP}|{MODEL}"))


def fixed_from_candidate(candidate: dict[str, Any], inherited: dict[str, Any], label: str, epochs: int | None = None) -> dict[str, Any]:
    """A ``tune_fixed`` fixed-config dict ({"label", "model", "train"}) from a registry candidate, overlaid on the inherited configuration."""
    model = copy.deepcopy(inherited["model"])
    train = copy.deepcopy(inherited["train"])
    for key, value in candidate.items():
        if key == "architecture":
            for name, v in value.items():
                model[name.removeprefix("model.")] = copy.deepcopy(v)
        elif key.startswith("train."):
            train[key.removeprefix("train.")] = value
        elif key.startswith("model."):
            model[key.removeprefix("model.")] = copy.deepcopy(value)
        else:
            raise KeyError(f"unexpected candidate key {key!r}")
    if epochs:
        train["epochs"] = int(epochs)
    return {"label": label, "model": model, "train": train}


def group_prefix(prefix: str, band: str) -> str:
    return f"{prefix.strip('/')}/{SOURCE}/{band}/{REP}/{MODEL}"


def trial_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    import sys

    sys.path.insert(0, args["repo_root"])
    os.chdir(args["repo_root"])
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_fixed import apply_fixed_config, base_config, train_prepared

    client, bucket = spectrum_s3_client(), args["bucket"]
    record = {"kind": "trial", "source": SOURCE, "dataset": job["band"], "model": MODEL, "rep": REP, "candidate_id": job["cid"],
              "parameters": job["parameters"], "seed": SEARCH_SEED, "index": job["index"], "epochs": job["fixed"]["train"]["epochs"],
              "budget": job.get("budget")}
    started = time.time()
    try:
        config, _ = base_config(SOURCE, job["band"], MODEL, REP, args["repo_root"], client, bucket)
        config = apply_fixed_config(config, MODEL, job["fixed"], SEARCH_SEED)
        with tempfile.TemporaryDirectory(prefix="dswin-trial-") as scratch:
            result = train_prepared(config, MODEL, Path(scratch) / "trial")
        record.update(ok=True, objective=result["objective"], objective_name=result["objective_name"])
        costs = result.get("run_costs") or {}
        record["run_costs"] = {k: v for k, v in costs.items() if isinstance(v, (int, float, str, bool))}
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-700:])
    try:
        import torch

        if torch.cuda.is_available():
            record["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    record["wall_seconds"] = round(time.time() - started, 1)
    key = f"{job['prefix']}/trial-{job['cid']}.json"
    try:
        if job.get("keep_first"):                     # hedge copy of a trial another driver also runs: never replace a finished (ok) record
            try:
                existing = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
                if existing.get("ok"):
                    record["note"] = "duplicate not written: an ok record already existed"
                    return record
            except Exception:
                pass
        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, default=str).encode())
    except Exception:
        pass
    return record


def run_calls(calls: list, max_parallel: int, address: str, worker_args: dict[str, Any], label: str) -> list:
    """Run (function, ray options, job) calls with at most ``max_parallel`` in flight (queued tasks fill whatever GPU fractions are free)."""
    import ray

    ray.init(address=address, ignore_reinit_error=True)
    queue, pending, results = list(calls), [], []
    while queue or pending:
        while queue and len(pending) < max_parallel:
            fn, opts, job = queue.pop(0)
            pending.append(ray.remote(**opts)(fn).remote(job, worker_args))
        done, pending = ray.wait(pending, num_returns=1)
        row = ray.get(done[0])
        results.append(row)
        print(f"[dswin15] {label} {row.get('kind')} {row.get('dataset')} cand={row.get('candidate_id')} ok={row.get('ok')} "
              f"obj={row.get('objective')} ({row.get('wall_seconds')}s, peak {row.get('peak_gpu_mb')} MB) [{len(queue) + len(pending)} left]", flush=True)
    return results


def ray_options(spec, gpu_fraction: float, memory_gb: float) -> dict[str, Any]:
    resources = dict(spec.resources.as_ray())
    opts = {"num_cpus": resources["cpu"], "num_gpus": gpu_fraction, "max_retries": 1}
    if memory_gb:
        opts["memory"] = int(memory_gb * 2 ** 30)
    return opts


def read_trials(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    out = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/trial-"):
        for obj in page.get("Contents", []):
            out.append(json.loads(client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()))
    return out


def cmd_run(a: argparse.Namespace) -> None:
    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    inherited = json.loads(Path(a.repo_root, INHERITED).read_text())
    spec = get_model_spec(MODEL, require_executable=True)
    opts = ray_options(spec, a.gpu_fraction, a.memory_gb)
    calls = []
    from .candidate import candidate_id

    for band in a.bands:
        prefix = group_prefix(a.prefix, band)
        done = {t.get("candidate_id") for t in read_trials(client, bucket, prefix) if t.get("ok")}
        for index, cand in enumerate(candidates_for(spec, band, a.budget)):
            if a.only_index and index not in a.only_index:
                continue
            cid = candidate_id(MODEL, cand)
            if cid in done:
                continue
            fixed = fixed_from_candidate(cand, inherited, cid, a.epochs)
            calls.append((trial_job, opts, {"band": band, "prefix": prefix, "index": index, "cid": cid, "parameters": cand, "fixed": fixed, "budget": a.budget, "keep_first": a.keep_first}))
    print(f"[dswin15] {len(calls)} trials to run (prefix {a.prefix})", flush=True)
    run_calls(calls, a.max_parallel, a.address, {"repo_root": a.repo_root, "bucket": bucket}, "search")


def cmd_decide(a: argparse.Namespace) -> None:
    import pandas as pd

    from .registry import get_model_spec
    from .storage import spectrum_s3_client
    from .tune_fixed import final_job, read_records

    client, bucket = spectrum_s3_client(), a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    inherited = json.loads(Path(a.repo_root, INHERITED).read_text())
    spec = get_model_spec(MODEL, require_executable=True)
    opts = ray_options(spec, a.gpu_fraction, a.memory_gb)
    if a.await_trials:
        deadline = time.time() + a.await_timeout_h * 3600
        while True:
            missing = {band: sorted(set(range(a.budget)) - {t.get("index") for t in read_trials(client, bucket, group_prefix(a.prefix, band)) if t.get("ok")})
                       for band in a.bands}
            print(f"[dswin15] waiting for trials, missing indices: {missing}", flush=True)
            if not any(missing.values()) or time.time() > deadline:
                break
            time.sleep(120)
    rows, verdict, final_calls = [], {"improvement_threshold": IMPROVEMENT, "budget": a.budget, "bands": {}}, []
    for band in a.bands:
        prefix = group_prefix(a.prefix, band)
        trials = [t for t in read_trials(client, bucket, prefix) if t.get("index") is not None and t["index"] < a.budget]   # --budget 8 = the first 8 of the 16 (the draws are sequential)
        base = json.loads(client.get_object(Bucket=bucket, Key=f"{group_prefix(BASELINE_PREFIX, band)}/final/seed-42.json")["Body"].read())
        baseline = float(base["objective"])
        ok = [t for t in trials if t.get("ok") and t.get("objective") == t.get("objective")]
        for t in trials:
            rows.append({"band": band, "index": t.get("index"), "candidate_id": t.get("candidate_id"), "ok": t.get("ok"),
                         "objective": t.get("objective"), "wall_seconds": t.get("wall_seconds"), "peak_gpu_mb": t.get("peak_gpu_mb"),
                         "parameters": json.dumps(t.get("parameters"), default=str), "error": t.get("error")})
        if not ok:
            verdict["bands"][band] = {"verdict": "no completed trials", "baseline_objective": baseline, "trials": len(trials)}
            continue
        win = min(ok, key=lambda t: (t["objective"], t["candidate_id"]))
        gain = (baseline - win["objective"]) / baseline
        entry = {"baseline_objective": baseline, "baseline_label": base.get("candidate_id"), "winner": win["candidate_id"], "winner_index": win.get("index"),
                 "winner_objective": win["objective"], "improvement": gain, "trials": len(trials), "trials_ok": len(ok),
                 "winner_parameters": win["parameters"], "verdict": "improves" if gain >= IMPROVEMENT else "does not improve"}
        verdict["bands"][band] = entry
        if gain >= IMPROVEMENT and not a.no_finals:
            fixed = fixed_from_candidate(win["parameters"], inherited, win["candidate_id"])
            done = {r["seed"] for r in read_records(client, bucket, group_prefix(a.prefix, band) + "/final")
                    if r.get("ok") and r.get("candidate_id") == win["candidate_id"]}
            for seed in a.final_seeds:
                if seed not in done:
                    final_calls.append((final_job, opts, {"source": SOURCE, "dataset": band, "model": MODEL, "rep": REP, "prefix": group_prefix(a.prefix, band),
                                                          "parameters": {}, "seed": int(seed), "candidate_id": win["candidate_id"], "search_objective": win["objective"],
                                                          "budget": 0, "fixed": fixed}))
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    tag = "_".join(a.bands) if len(a.bands) < len(BANDS) else ""
    suffix = f"_{tag}" if tag else ""
    for key, body in ((f"analysis-outputs/dswin_search_trials{suffix}.csv", buffer.getvalue().encode()),
                      (f"analysis-outputs/dswin_search_verdict{suffix}.json", json.dumps(verdict, indent=1, default=str).encode())):
        client.put_object(Bucket=bucket, Key=key, Body=body)
    merge_verdicts(client, bucket)
    print("[dswin15] verdict " + json.dumps(verdict, default=str)[:1500], flush=True)
    if final_calls:
        print(f"[dswin15] {len(final_calls)} final runs for improving winners", flush=True)
        results = run_calls(final_calls, a.max_parallel, a.address, {"repo_root": a.repo_root, "bucket": bucket}, "finals")
        verdict["finals"] = [{"dataset": r.get("dataset"), "seed": r.get("seed"), "ok": r.get("ok"), "objective": r.get("objective"),
                              "mae_t12": r.get("test_mae_db_t12")} for r in results]
        client.put_object(Bucket=bucket, Key=f"analysis-outputs/dswin_search_verdict{suffix}.json", Body=json.dumps(verdict, indent=1, default=str).encode())
        merge_verdicts(client, bucket)
    improved = [b for b, e in verdict["bands"].items() if e.get("verdict") == "improves"]
    if improved and not a.no_chain:
        score_chain(improved, a, bucket)


def merge_verdicts(client, bucket: str) -> None:
    """analysis-outputs/dswin_search_verdict.json = the per-band verdict files (written by one driver per band) merged."""
    merged = {"improvement_threshold": IMPROVEMENT, "bands": {}}
    for band in BANDS:
        try:
            part = json.loads(client.get_object(Bucket=bucket, Key=f"analysis-outputs/dswin_search_verdict_{band}.json")["Body"].read())
        except Exception:
            continue
        merged["bands"][band] = part.get("bands", {}).get(band)
        merged["budget"] = part.get("budget")
        if part.get("finals"):
            merged.setdefault("finals", []).extend(part["finals"])
    client.put_object(Bucket=bucket, Key="analysis-outputs/dswin_search_verdict.json", Body=json.dumps(merged, indent=1, default=str).encode())


def score_chain(bands: list[str], a: argparse.Namespace, bucket: str) -> None:
    """The existing scoring chain for the new finals (receivers, left-out/added, permuted x5, per-origin errors, neighbour ablation), all under NEW prefixes."""
    import subprocess

    seeds = [str(s) for s in a.final_seeds]
    common = ["--tuning-prefix", a.prefix]
    steps = []
    for band in bands:
        base = ["python", "-u", "-m", "training.ray.spatial15", "--tuned", "--experiments", "s2", "--variants", "core4-p2", "--models", MODEL, "--seeds", *seeds,
                "--prefix", "spatial15-tuned-dswinsearch", "--memory-gb", "12", "--max-parallel", "5", "--gpu-fraction", "0.25", "--skip-done", "--band", band, *common]
        steps.append(base + ["--eval-only", "--eval-from-tuning"])
        steps.append(base + ["--permuted", "5"])
    steps.append(["python", "-u", "-m", "training.ray.spatial15_origins", "--models", MODEL, "--variants", "core4-p2", "--bands", *bands, "--seeds", *seeds, "--skip-done",
                  "--memory-gb", "12", "--max-parallel", "5", "--gpu-fraction", "0.25", "--prefix", "spatial15-tuned-dswinsearch",
                  "--output-prefix", "spatial15-origin-errors-dswinsearch", *common])
    steps.append(["python", "-u", "-m", "training.ray.neighbor_ablation", "--models", MODEL, "--bands", *bands, "--seeds", *seeds, "--skip-done", "--gpu-fraction", "0.5",
                  "--max-parallel", "10", "--prefix", "neighbor-ablation-dswinsearch", *common])
    for step in steps:
        print("[dswin15] chain:", " ".join(step), flush=True)
        code = subprocess.call(step, cwd=a.repo_root)
        print(f"[dswin15] chain step exit {code}", flush=True)


def cmd_bench(a: argparse.Namespace) -> None:
    """Sharing benchmark: N copies of the inherited candidate (2-3 epochs) concurrently on ONE node's card, N from ``--groups ip:N ...``.

    Each copy gets a 1/N GPU share and is pinned to the node by its custom resource ``node:<ip>``; reports the average epoch time of each copy,
    so the slowdown caused by sharing a card can be read before the real launch."""
    import ray

    from .candidate import candidate_id
    from .registry import get_model_spec
    from .storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), a.bucket or os.environ.get("SPECTRUM_S3_BUCKET", "spectrum")
    inherited = json.loads(Path(a.repo_root, INHERITED).read_text())
    spec = get_model_spec(MODEL, require_executable=True)
    cand = candidates_for(spec, a.bands[0], 8)[2]                      # candidate 2 = the inherited configuration
    ray.init(address=a.address, ignore_reinit_error=True)
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    refs = []
    for group in a.groups:
        ip, n = group.split(":")           # "ip" is only a label here: a STRICT_PACK placement group puts the N copies on one node that has a free card
        n = int(n)
        base = ray_options(spec, 1.0 / n, a.memory_gb)
        pg = placement_group([{"CPU": base["num_cpus"], "GPU": base["num_gpus"], "memory": base["memory"]}] * n, strategy="STRICT_PACK")
        ray.get(pg.ready(), timeout=a.pg_timeout)
        for i in range(n):
            cid = f"{candidate_id(MODEL, cand)}-bench{n}-{ip}-{i}"
            job = {"band": a.bands[0], "prefix": group_prefix(a.prefix, a.bands[0]) + f"/n{n}-{ip}", "index": 2, "cid": cid, "parameters": cand,
                   "fixed": fixed_from_candidate(cand, inherited, cid, a.epochs)}
            opts = {**base, "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)}
            refs.append((ip, n, ray.remote(**opts)(trial_job).remote(job, {"repo_root": a.repo_root, "bucket": bucket})))
    rows = []
    for ip, n, ref in refs:
        r = ray.get(ref)
        costs = r.get("run_costs") or {}
        rows.append((ip, n, r.get("ok"), costs.get("cost_average_epoch_time_sec"), costs.get("cost_gpu_name"), r.get("wall_seconds"), r.get("peak_gpu_mb"), costs.get("cost_hostname")))
        print(f"[dswin15-bench] node {ip} n={n} ok={r.get('ok')} epoch_s={costs.get('cost_average_epoch_time_sec')} gpu={costs.get('cost_gpu_name')} "
              f"host={costs.get('cost_hostname')} wall={r.get('wall_seconds')} peak={r.get('peak_gpu_mb')} err={r.get('error')}", flush=True)
    import statistics as st

    for group in a.groups:
        ip, n = group.split(":")
        v = [r[3] for r in rows if r[0] == ip and r[3]]
        if v:
            print(f"[dswin15-bench] SUMMARY node {ip} n={n}: mean epoch {st.mean(v):.1f}s (min {min(v):.1f}, max {max(v):.1f}) -> 50 epochs {st.mean(v) * 50 / 3600:.2f} h", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "decide", "all", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--bands", nargs="+", default=list(BANDS), choices=BANDS)
        s.add_argument("--prefix", default=DEFAULT_PREFIX)
        s.add_argument("--budget", type=int, default=16, help="candidates per band (16 = the standard search; 8 = the time-boxed fallback: the first 8 of the same list)")
        s.add_argument("--only-index", type=int, nargs="+")
        s.add_argument("--epochs", type=int, default=None, help="override the epochs (smoke tests only; use another --prefix)")
        s.add_argument("--gpu-fraction", type=float, default=0.5)
        s.add_argument("--memory-gb", type=float, default=12.0)
        s.add_argument("--max-parallel", type=int, default=12)
        s.add_argument("--no-finals", action="store_true")
        s.add_argument("--no-chain", action="store_true", help="skip the scoring chain after the finals")
        s.add_argument("--keep-first", action="store_true", help="run: never overwrite a finished (ok) trial record (hedge copies of trials another driver runs)")
        s.add_argument("--await-trials", action="store_true", help="decide: wait until every candidate index below --budget has an ok record (from any driver)")
        s.add_argument("--await-timeout-h", type=float, default=10.0)
        s.add_argument("--final-seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
        s.add_argument("--bucket", default=None)
        s.add_argument("--address", default="auto")
        s.add_argument("--repo-root", default="/workspace/spectrum-usage")
        s.add_argument("--groups", nargs="+", default=[], help="bench only: label:N pairs (N copies on one card)")
        s.add_argument("--pg-timeout", type=float, default=900.0, help="bench only: seconds to wait for a node with a free card")
    a = p.parse_args(argv)
    if a.cmd == "bench":
        a.epochs = a.epochs or 3
        a.prefix = a.prefix if a.prefix != DEFAULT_PREFIX else "tuning-dswin-search-bench"
        return cmd_bench(a)
    if a.cmd in ("run", "all"):
        cmd_run(a)
    if a.cmd in ("decide", "all"):
        cmd_decide(a)


if __name__ == "__main__":
    main()
