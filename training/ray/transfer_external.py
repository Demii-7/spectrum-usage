"""Frozen zero-shot transfer (stage A) and per-window centering (stage B) of the 15-minute 1D/2D checkpoints to COSMOS, AERPAW and ARA.

Same protocol as ``training.ray.transfer15`` (weights and source normalization frozen, lookback 96 x 15 minutes, endpoint error at
steps 1, 4, 12, 24, 48, one shared set of origins per target, full 96-step history and 48-step future in one gap-free segment, per-site
results, identical code for A and B). Same-band transfer only: a 600-800 MHz checkpoint is scored on the 600-800 MHz slice of the target
and a 2400-2600 MHz checkpoint on the 2400-2600 MHz slice; combined site-and-band transfer is not attempted.

Targets are frozen BEFORE any model is scored (``freeze`` writes ``evaluation/analysis/transfer_external_intervals.csv``; the run reads
that file and never recomputes it):
* one target per (platform, site, run, band) whose files exist for a band we have checkpoints for (600_800, 2400_2600; 3500_3700 has no
  checkpoint and is skipped);
* the longest segment of the raw 1-minute trace with no gap longer than 10 minutes, where a row with every bin NaN counts as a missing
  minute (the 1-minute trace is first cut to that segment, so no
  15-minute interpolation ever has to bridge a long gap);
* converted with the core experiments' 15-minute preprocessing (UTC-aligned 15-minute blocks, mean in linear power converted back to dB,
  a block needs 12 of its 15 minutes, interpolation in dB over at most 16 blocks); the share of blocks interpolated is recorded;
* AERPAW wideband files (87-6020 MHz, 5,933 columns at x.5 MHz) are sliced to the 200 bins of each band;
* a target is used only if it has at least ``MIN_ORIGINS`` (96, one day of 15-minute origins) forecast origins, i.e. at least 60 hours of
  contiguous data; shorter runs are listed with the reason and are not scored;
* the scored interval starts 24 hours (96 blocks) after the first block, so every origin has a full history, and ends at the last block.
Nothing in this selection looks at model performance.

Outputs: ``<prefix>/<stage>/<platform>/<site>/<run>/<band>/<rep>/<model>/seed-N/{summary.json, per_bin.csv, per_origin.csv, forecasts/}``
with stage A or B, and ``<prefix>/baseline/<platform>/<site>/<run>/<band>/<name>/`` for persistence and the 96-step lookback mean
(both invariant to window centering). Provenance per record: source and target band, checkpoint key and sha256, source config, target
raw file and converted file (sha256), frozen interval, preprocessing and imputation share, normalization policy, code revision.

    python -m training.ray.transfer_external freeze          # once, on a node that has data/{cosmos,aerpaw,ara}
    python -m training.ray.transfer_external run --skip-done
"""

from __future__ import annotations

import argparse
import glob
import io
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

BANDS = {"600_800": (600.0, 800.0), "2400_2600": (2400.0, 2600.0)}
PLATFORMS = ("cosmos", "aerpaw", "ara")
MAX_RAW_GAP = pd.Timedelta(minutes=10)
STEP = pd.Timedelta(minutes=15)
LOOKBACK, MAX_HORIZON, MIN_ORIGINS = 96, 48, 96
INTERVALS_CSV = "evaluation/analysis/transfer_external_intervals.csv"
CONVERTED_ROOT = "data/ext15"
HORIZONS = (1, 4, 12, 24, 48)


# ------------------------------------------------------------------------------------------ data
def raw_files(repo_root: str) -> list[dict[str, Any]]:
    """Every raw per-minute file for a band we have checkpoints for (wideband AERPAW files count for both bands)."""
    found = []
    for platform in PLATFORMS:
        for path in sorted(glob.glob(f"{repo_root}/data/{platform}/*/*/*/power_1mhz_avg_per_minute.csv")):
            parts = path.split("/")
            site, run, folder = parts[-4], parts[-3], parts[-2]
            bands = list(BANDS) if folder == "87_6020" else ([folder] if folder in BANDS else [])
            for band in bands:
                found.append({"platform": platform, "site": site, "run": run, "band": band, "raw": path})
    return found


def longest_segment(stamps: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last minute of the longest run of raw timestamps with no gap above ``MAX_RAW_GAP``."""
    gaps = stamps.to_series().diff()
    breaks = np.nonzero((gaps > MAX_RAW_GAP).to_numpy())[0]
    edges = np.r_[0, breaks, len(stamps)]
    spans = [(stamps[a], stamps[b - 1]) for a, b in zip(edges[:-1], edges[1:])]
    return max(spans, key=lambda s: s[1] - s[0])


def converted_path(repo_root: str, target: dict[str, Any]) -> Path:
    return Path(repo_root) / CONVERTED_ROOT / target["platform"] / target["site"] / target["run"] / target["band"] / "power_15min.csv"


def convert(repo_root: str, target: dict[str, Any], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict[str, Any]:
    """Slice the band, cut to the segment, convert to 15-minute blocks (core preprocessing) and write the converted file atomically.

    Returns the sidecar facts (also written next to the CSV). Reuses an existing converted file and its sidecar.
    """
    from training.ray.powder15 import block_average_db, fill_gaps

    out = converted_path(repo_root, target)
    sidecar = out.with_suffix(".json")
    if out.exists() and sidecar.exists():
        return json.loads(sidecar.read_text())
    low, high = BANDS[target["band"]]
    header = pd.read_csv(target["raw"], nrows=0).columns
    keep = [c for c in header if c != "timestamp_utc" and low < float(c) < high]
    if len(keep) != 200:
        raise ValueError(f"{target['raw']}: {len(keep)} columns in {low}-{high} MHz, expected 200")
    frame = pd.read_csv(target["raw"], usecols=["timestamp_utc", *keep])
    stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
    frame.index = stamps
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.dropna(how="all")                 # a row with no valid bin is a missing minute, not a sample
    if frame.empty:
        raise ValueError("no valid rows in the band")
    seg_start, seg_end = (start, end) if start is not None else longest_segment(frame.index)
    frame = frame.loc[seg_start:seg_end].reindex(pd.date_range(seg_start, seg_end, freq="min"))
    blocks = block_average_db(frame)
    first, last = blocks.first_valid_index(), blocks.last_valid_index()
    blocks, missing = fill_gaps(blocks.loc[first:last])
    table = blocks.copy()
    table.insert(0, "timestamp_utc", table.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out.parent, delete=False, suffix=".tmp") as tmp:
        pass
    table.to_csv(tmp.name, index=False)
    os.replace(tmp.name, out)
    facts = {"raw_segment_start": seg_start.isoformat(), "raw_segment_end": seg_end.isoformat(), "blocks": int(len(table)),
             "first_block": first.isoformat(), "last_block": last.isoformat(), "missing_share_before_interpolation": float(missing),
             "bins": 200, "columns_kept": [keep[0], keep[-1]]}
    sidecar.write_text(json.dumps(facts))
    return facts


def diagnose(target: dict[str, Any]) -> str:
    """Why a conversion failed: minutes present in the longest raw segment and the share of 15-minute blocks that are valid."""
    try:
        from training.ray.powder15 import MIN_VALID_FRACTION

        header = pd.read_csv(target["raw"], nrows=0).columns
        first_col = next(c for c in header if c != "timestamp_utc")
        frame = pd.read_csv(target["raw"], usecols=["timestamp_utc", first_col])
        stamps = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.floor("min")
        frame.index = stamps
        frame = frame[~frame.index.duplicated(keep="first")].sort_index()
        start, end = longest_segment(frame.index)
        seg = frame.loc[start:end].iloc[:, 0]
        minutes = int((end - start).total_seconds() // 60) + 1
        counts = seg.notna().groupby(seg.index.floor("15min")).sum()
        need = int(np.ceil(MIN_VALID_FRACTION * 15))
        return (f"longest segment {minutes} min with {len(seg)} samples ({len(seg) / minutes:.2f} per minute); "
                f"{float((counts >= need).mean()):.2f} of 15-minute blocks have >= {need} valid minutes")
    except Exception as error:
        return f"diagnosis failed: {error}"


def freeze(repo_root: str, out_csv: str) -> pd.DataFrame:
    """Write the frozen target table (every candidate, with the reason when it is excluded)."""
    rows = []
    for target in raw_files(repo_root):
        try:
            facts = convert(repo_root, target)
        except Exception as error:                      # recorded with the reason, never scored
            rows.append({**{k: target[k] for k in ("platform", "site", "run", "band")},
                         "raw": str(Path(target["raw"]).relative_to(repo_root)), "blocks": 0, "n_origins": 0, "use": False,
                         "reason": f"15-minute conversion failed: {type(error).__name__}: {error}"[:300] + " | " + diagnose(target)})
            continue
        first, last = pd.Timestamp(facts["first_block"]), pd.Timestamp(facts["last_block"])
        n_origins = facts["blocks"] - LOOKBACK - MAX_HORIZON + 1
        use = n_origins >= MIN_ORIGINS
        rows.append({**{k: target[k] for k in ("platform", "site", "run", "band")},
                     "raw": str(Path(target["raw"]).relative_to(repo_root)), "blocks": facts["blocks"],
                     "first_block": first.isoformat(), "last_block": last.isoformat(),
                     "raw_start": first.isoformat(), "raw_end": (last + STEP).isoformat(),
                     "target_start": (first + LOOKBACK * STEP).isoformat(), "target_end": (last + STEP).isoformat(),
                     "n_origins": max(n_origins, 0), "missing_share_before_interpolation": facts["missing_share_before_interpolation"],
                     "use": bool(use), "reason": "" if use else f"fewer than {MIN_ORIGINS} origins ({max(n_origins, 0)}), under 60 hours of contiguous data"})
    table = pd.DataFrame(rows).sort_values(["platform", "site", "run", "band"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table


# ------------------------------------------------------------------------------------------ jobs
def out_base(prefix: str, stage: str, t: dict[str, Any], rep: str, model: str, seed: int) -> str:
    return f"{prefix.strip('/')}/{stage}/{t['platform']}/{t['site']}/{t['run']}/{t['band']}/{rep}/{model}/seed-{seed}"


def baseline_dir(prefix: str, t: dict[str, Any], name: str) -> str:
    return f"{prefix.strip('/')}/baseline/{t['platform']}/{t['site']}/{t['run']}/{t['band']}/{name}"


def task_for(t: dict[str, Any]):
    from training.common.transfer_2d import TransferTask

    low, high = BANDS[t["band"]]
    return TransferTask(task_id=f"{t['platform']}-{t['site']}-{t['run']}-{t['band']}", evaluation="external site, same band",
                        platform=t["platform"], raw_start=pd.Timestamp(t["raw_start"]), raw_end=pd.Timestamp(t["raw_end"]),
                        target_start=pd.Timestamp(t["target_start"]), target_end=pd.Timestamp(t["target_end"]),
                        frequency_start_mhz=low, frequency_end_mhz=high, sites=(t["site"],))


def provenance(repo_root: str, t: dict[str, Any], source_band: str, code_revision: str) -> dict[str, Any]:
    import hashlib

    path = converted_path(repo_root, t)
    facts = json.loads(path.with_suffix(".json").read_text())
    return {"source_band": source_band, "target_band": t["band"], "target_platform": t["platform"], "target_site": t["site"],
            "target_run": t["run"], "target_raw_file": t["raw"], "target_file": str(path.relative_to(repo_root)),
            "target_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frozen_intervals_file": INTERVALS_CSV,
            "raw_window": [t["raw_start"], t["raw_end"]], "target_interval": [t["target_start"], t["target_end"]],
            "target_interval_end_exclusive": True, "lookback_steps": LOOKBACK, "step_minutes": 15, "horizons": list(HORIZONS),
            "n_origins_frozen": int(t["n_origins"]), "code_revision": code_revision,
            "normalization_policy": "source_checkpoint_statistics_by_bin_position; never fitted on target; no post-hoc centering in stage A",
            "normalization_fitted_on_target": False,
            "preprocessing": "15-minute UTC-aligned blocks, mean in linear power converted back to dB, blocks need >= 12 of 15 valid minutes, "
                             "gaps up to 16 blocks interpolated in dB; raw trace cut to its longest segment with no gap above 10 minutes; "
                             "wideband files sliced to the band's 200 bins",
            "missing_share_before_interpolation": facts["missing_share_before_interpolation"], "converted_facts": facts}


def model_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: one checkpoint on one frozen target, stage A and stage B on identical windows."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    entry, t = job["entry"], job["target"]
    from training.ray import transfer15 as t15
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    name, seed = entry["model"], int(entry["seed"])
    base_a, base_b = (out_base(args["prefix"], s, t, entry["rep"], name, seed) for s in ("A", "B"))
    intrinsic = t15.INTRINSIC.get(name, "none")
    record = {"kind": "transfer_external", "source_band": entry["band"], "target_band": t["band"], "platform": t["platform"],
              "site": t["site"], "run": t["run"], "rep": entry["rep"], "model": name, "seed": seed, "checkpoint_key": entry["pt_key"],
              "intrinsic_window_normalization": intrinsic}
    started = time.time()
    try:
        from training.common import transfer_1d, transfer_2d

        if entry["band"] != t["band"]:
            raise ValueError("combined site-and-band transfer is outside this stage")
        convert(root, t)
        meta = json.loads(client.get_object(Bucket=bucket, Key=entry["meta_key"])["Body"].read())
        config = meta["config"]
        config["windowing"] = {"lookback": LOOKBACK, "horizons": list(HORIZONS)}
        config[name].setdefault("train", {})["test_stride"] = 1
        module = transfer_2d if entry["rep"] == "2d" else transfer_1d
        task = task_for(t)
        with tempfile.TemporaryDirectory(prefix="transfer-ext-") as scratch_dir:
            checkpoint = Path(scratch_dir) / Path(entry["pt_key"]).name
            client.download_file(bucket, entry["pt_key"], str(checkpoint))
            source = module.load_target(task, [converted_path(root, t)], impute=False, max_missing_gap=0)
            plain = t15.evaluate_frozen(module, entry["rep"], task, source, name, checkpoint, config, seed, center=False)
            center = intrinsic != "full"
            centered = t15.evaluate_frozen(module, entry["rep"], task, source, name, checkpoint, config, seed, center=center)
            if not all(np.array_equal(plain["rows"][h], centered["rows"][h]) for h in HORIZONS):
                raise RuntimeError("stage A and B origins differ")
            n_rows = len(plain["rows"][HORIZONS[0]])
            n_origins = n_rows // 200 if entry["rep"] == "1d" else n_rows          # flattened 1D rows are origins x 200 bins
            if n_origins != int(t["n_origins"]):
                raise RuntimeError(f"origin count {n_origins} differs from the frozen {t['n_origins']}")
            facts = {**record, **provenance(root, t, entry["band"], args["code_revision"]),
                     "checkpoint_sha256": meta.get("sha256"), "training_code_revision": meta.get("code_revision"),
                     "source_config_candidate_id": meta.get("candidate_id"), "source_parameters": meta.get("parameters"),
                     "source_train_files": [f["path"] for f in meta["config"]["data"]["files"] if f["partition"] == "train"],
                     "n_origins": n_origins}
            scored_a = t15.score_all(plain["predictions"], plain["targets"], plain["bins"], plain["times"])
            scored_b = t15.score_all(centered["predictions"], centered["targets"], centered["bins"], centered["times"])
            t15.upload(client, bucket, base_a, {**facts, "stage": "A", "metrics": scored_a["metrics"]}, scored_a)
            t15.save_forecasts(client, bucket, base_a, plain)
            t15.upload(client, bucket, base_b, {**facts, "stage": "B", "centering_applied": center, "metrics": scored_b["metrics"],
                                                "centering": "per window and frequency bin: lookback mean removed (source normalization), "
                                                             "added back to the forecast; weights and source normalization frozen"
                                                if center else "not applied: the model normalizes each window itself (RevIN)"}, scored_b)
            t15.save_forecasts(client, bucket, base_b, centered)
        record.update(ok=True, mae_a_t12=scored_a["metrics"]["12"]["mae_db"], mae_b_t12=scored_b["metrics"]["12"]["mae_db"])
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        client.put_object(Bucket=bucket, Key=f"{base_a}/error.json", Body=json.dumps(record, default=str).encode())
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


def baseline_job(job: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Remote worker: persistence and the lookback mean on one frozen target (the origins every model uses)."""
    root = args["repo_root"]
    os.chdir(root)
    sys.path.insert(0, root)
    t = job["target"]
    from training.ray import transfer15 as t15
    from training.ray.storage import spectrum_s3_client

    client, bucket = spectrum_s3_client(), args["bucket"]
    record = {"kind": "transfer_external_baseline", "platform": t["platform"], "site": t["site"], "run": t["run"], "target_band": t["band"]}
    started = time.time()
    try:
        from training.common import transfer_2d

        convert(root, t)
        task = task_for(t)
        source = transfer_2d.load_target(task, [converted_path(root, t)], impute=False, max_missing_gap=0)
        starts = transfer_2d.target_window_starts(source.timestamps, source.segments, lookback=LOOKBACK, max_horizon=MAX_HORIZON, stride=1,
                                                  target_start=task.target_start, target_end=task.target_end)
        if len(starts) != int(t["n_origins"]):
            raise RuntimeError(f"origin count {len(starts)} differs from the frozen {t['n_origins']}")
        data = np.asarray(source.data, dtype=np.float32)
        forecasts = {"persistence": data[starts + LOOKBACK - 1],
                     "lookbackmean": np.stack([data[s:s + LOOKBACK].mean(axis=0) for s in starts])}
        for name, pred in forecasts.items():
            predictions, targets, bins, times = {}, {}, {}, {}
            for h in HORIZONS:
                rows = starts + LOOKBACK + h - 1
                predictions[h], targets[h], bins[h] = pred, data[rows], np.empty(0)
                times[h] = t15.origin_times(source, rows, h)
            scored = t15.score_all(predictions, targets, bins, times)
            t15.upload(client, bucket, baseline_dir(args["prefix"], t, name),
                       {**record, "baseline": name, **provenance(root, t, t["band"], args["code_revision"]),
                        "metrics": scored["metrics"], "n_origins": int(len(starts))}, scored)
        record.update(ok=True, n_origins=int(len(starts)))
    except Exception as error:
        import traceback

        record.update(ok=False, error=f"{type(error).__name__}: {error}"[:400], trace=traceback.format_exc()[-1200:])
        client.put_object(Bucket=bucket, Key=f"{baseline_dir(args['prefix'], t, 'error')}/error.json", Body=json.dumps(record, default=str).encode())
    record["wall_seconds"] = round(time.time() - started, 1)
    return record


# ----------------------------------------------------------------------------------------- driver
def cmd_freeze(a: argparse.Namespace) -> None:
    table = freeze(a.repo_root, a.out)
    print(table[["platform", "site", "run", "band", "blocks", "n_origins", "use", "missing_share_before_interpolation"]].to_string())
    print(f"{int(table.use.sum())} of {len(table)} targets usable -> {a.out}")


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
    table = pd.read_csv(Path(a.repo_root) / a.intervals)
    targets = [r for r in table.to_dict("records") if r["use"] and r["band"] in a.bands and r["platform"] in a.platforms]
    entries = [e for e in discover_sources(client, bucket, ["p15tuned"], a.models)
               if e["band"] in a.bands and e["rep"] in a.reps and int(e["seed"]) in a.seeds]

    def done(key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=f"{key}/summary.json")
            return True
        except Exception:
            return False

    jobs = [("baseline", {"target": t}) for t in targets] + \
           [("model", {"entry": e, "target": t}) for t in targets for e in entries if e["band"] == t["band"]]
    if a.skip_done:
        jobs = [(k, j) for k, j in jobs
                if not done(baseline_dir(a.prefix, j["target"], "lookbackmean") if k == "baseline"
                            else out_base(a.prefix, "B", j["target"], j["entry"]["rep"], j["entry"]["model"], int(j["entry"]["seed"])))]
    print(f"[transfer-ext] {len(targets)} targets, {len(entries)} checkpoints, {len(jobs)} jobs to run, revision {revision[:12]}", flush=True)
    shared = {"repo_root": a.repo_root, "bucket": bucket, "prefix": a.prefix, "code_revision": revision}
    ray.init(address=a.address, ignore_reinit_error=True)
    memory = int(a.memory_gb * 2 ** 30)
    model_task = ray.remote(num_cpus=2, num_gpus=a.gpu_fraction, memory=memory, max_retries=1)(model_job)
    base_task = ray.remote(num_cpus=2, num_gpus=0, memory=memory, max_retries=1)(baseline_job)
    queue, pending, ok, failed = list(jobs), [], 0, 0
    while queue or pending:
        while queue and len(pending) < a.max_parallel:
            kind, job = queue.pop(0)
            pending.append((base_task if kind == "baseline" else model_task).remote(job, shared))
        finished, pending = ray.wait(pending, num_returns=1)
        try:
            row = ray.get(finished[0])
        except Exception as error:
            row = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        ok += bool(row.get("ok"))
        failed += not row.get("ok")
        print(f"[transfer-ext] {row.get('platform')} {row.get('site')} {row.get('run')} {row.get('target_band', row.get('source_band'))} "
              f"{row.get('rep', 'baseline')} {row.get('model', '')} seed{row.get('seed', '')} ok={row.get('ok')} "
              f"mae_t12 A={row.get('mae_a_t12')} B={row.get('mae_b_t12')} {row.get('error', '')} ({row.get('wall_seconds')}s) "
              f"[{len(queue) + len(pending)} left]", flush=True)
    print(f"[transfer-ext] done: {ok} ok, {failed} failed", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="convert every candidate target and write the frozen interval table")
    f.add_argument("--repo-root", default="/workspace/spectrum-usage")
    f.add_argument("--out", default=INTERVALS_CSV)
    f.set_defaults(func=cmd_freeze)
    r = sub.add_parser("run", help="score every checkpoint on every frozen target (stages A and B) plus the baselines")
    r.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    r.add_argument("--platforms", nargs="+", default=list(PLATFORMS), choices=list(PLATFORMS))
    r.add_argument("--reps", nargs="+", default=["1d", "2d"], choices=["1d", "2d"])
    r.add_argument("--models", nargs="+")
    r.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    r.add_argument("--intervals", default=INTERVALS_CSV)
    r.add_argument("--gpu-fraction", type=float, default=0.1)
    r.add_argument("--memory-gb", type=float, default=12.0)
    r.add_argument("--max-parallel", type=int, default=8)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--bucket", default=None)
    r.add_argument("--prefix", default="transfer15-external-oct2026")
    r.add_argument("--code-revision", default=None)
    r.add_argument("--address", default="auto")
    r.add_argument("--repo-root", default="/workspace/spectrum-usage")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
