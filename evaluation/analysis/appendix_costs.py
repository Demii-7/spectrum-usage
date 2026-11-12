"""Read-only S3 accounting for the 600--800 MHz, one-minute appendix table.

No jobs are submitted. Totals are lower bounds on allocated compute: archived
Ray reports can precede task termination, and overwritten/lost retries are not
recoverable. Include all archived search rounds and their recorded replicate
tasks. Backfill missing final training time from exported cost files without
counting a reused search-seed checkpoint as a second fit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import shlex
import statistics


CORE_ROOTS = ("checkpoints-oct2026/", "checkpoints-oct2026-r2/")
BACKUP = "oct2026-minio-backup-newcluster/"
DIRECT = "direct-tuning/1min/600_800/"
# Allocations in the campaign runners; completed records must agree with them.
DIRECT_GPU = {"dlinear": .125, "ofa": .25, "patchtst": .25,
              "residuallinearar2d": .125, "residualnonlinear2d": .125, "timellm": .6}
SEEDS = {40, 41, 42, 43, 44}


def s3_client(env_file: Path):
    import boto3

    env = {}
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            words = shlex.split(value)
            env[name.strip()] = words[0] if words else ""
    return boto3.client("s3", endpoint_url=env["S3_ENDPOINT_URL"],
                        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                        region_name=env.get("AWS_DEFAULT_REGION", "us-east-1"))


def collect(client, bucket):
    objects = [o for prefix in (*CORE_ROOTS, BACKUP, DIRECT)
               for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
               for o in page.get("Contents", [])]
    keys = [o["Key"] for o in objects if o["Key"].endswith(("/result.json", "_run_costs.json", "/MANIFEST.json"))
            or (o["Key"].startswith(DIRECT) and o["Key"].endswith(".json")
                and ("/trials/" in o["Key"] or "/finals/" in o["Key"]))]

    def fetch(key):
        data = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if key.endswith("/result.json"):
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            return {"key": key, "data": rows[-1] if rows else {},
                    "max_seconds": max((r.get("time_total_s", 0) for r in rows), default=0)}
        return {"key": key, "data": json.loads(data)}

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(fetch, keys))


def core_summary(rep, model, records):
    cpu = model.startswith(("linearar", "residuallinearar"))
    gpu_share = 0.0 if cpu else .5
    # CPU-only registry trials reserve two cores. GPU rows report GPU hours only.
    multiplier = 2.0 if cpu else gpu_share
    manifests, exported, search, replicas = {}, {}, [], defaultdict(list)
    for record in records:
        key, data = record["key"], record["data"]
        for root in CORE_ROOTS:
            if key.startswith(f"{root}{rep}/{model}/"):
                if key.endswith("/MANIFEST.json"):
                    manifests[root] = data
                elif key.endswith("_run_costs.json"):
                    seed = int(key.split("/seed-")[1].split("/")[0])
                    exported[root, seed] = record
        if key.startswith(BACKUP) and key.endswith("/result.json"):
            parts = key.split("/")
            if parts[4] != rep or parts[5] not in (f"{model}-search", f"{model}-replicate"):
                continue
            share = data.get("cost_ray_gpu_assigned")
            if share is not None and float(share) != gpu_share:
                raise ValueError(f"Unexpected GPU allocation in {key}: {share}")
            if parts[5].endswith("-search"):
                search.append(record)
            else:
                seed = data.get("config", {}).get("seed", data.get("seed"))
                if seed is None:
                    match = re.search(r"seed=(\d+)", key)
                    seed = int(match.group(1)) if match else None
                if seed is not None:
                    root = CORE_ROOTS[1] if parts[3].endswith("-r2") else CORE_ROOTS[0]
                    identity = root, int(seed)
                    replicas[identity].append(record)

    objectives = {}
    for root, manifest in manifests.items():
        candidates = [e["metrics"].get("objective") for e in manifest.get("entries", [])
                      if e.get("seed") == 42 and "metrics" in e]
        if len(candidates) != 1 or candidates[0] is None:
            raise ValueError(f"Missing validation objective: {root}{rep}/{model}")
        objectives[root] = float(candidates[0])
    if not objectives:
        raise ValueError(f"Missing winner manifests: {rep}/{model}")
    selected_root = min(objectives, key=objectives.get)
    final_fits = {seed: r["data"].get("cost_training_duration_sec")
                  for (root, seed), r in exported.items() if root == selected_root}
    if not final_fits or any(t is None for t in final_fits.values()):
        raise ValueError(f"Missing winner fit times: {rep}/{model}")
    seconds = sum(r["max_seconds"] for r in search)
    sources = [{"key": r["key"], "seconds": r["max_seconds"], "phase": "search"} for r in search]
    reused, backfilled = [], []
    for root in CORE_ROOTS:
        metrics = {int(e["seed"]): e["metrics"] for e in manifests.get(root, {}).get("entries", []) if "metrics" in e}
        seeds = {s for r, s in exported if r == root} | {s for r, s in replicas if r == root}
        for seed in sorted(seeds):
            archived = replicas.get((root, seed), [])
            elapsed = sum(r["max_seconds"] for r in archived)
            fit = exported.get((root, seed))
            from_search = metrics.get(seed, {}).get("cost_from_hpo", False)
            # The seed-42 source checkpoint came from search. Only its separate
            # replicate evaluation consumes new time; do not add its fit twice.
            if from_search:
                reused.append(f"{root}seed-{seed}")
            elif fit:
                duration = fit["data"].get("cost_training_duration_sec")
                if duration is None:
                    raise ValueError(f"Missing duration: {fit['key']}")
                if duration > elapsed:
                    backfilled.append(fit["key"])
                elapsed = max(elapsed, duration)
            seconds += elapsed
            sources.append({"keys": [r["key"] for r in archived] + ([fit["key"]] if fit else []),
                            "seconds": elapsed, "phase": "replicate", "seed": seed, "reused_hpo": from_search})
    return {"model": model, "rep": rep, "source": "core", "winner_prefix": selected_root,
            "fit_minutes": statistics.median(final_fits.values()) / 60, "fit_n": len(final_fits),
            "round_objectives": objectives,
            "allocated_hours_lower_bound": seconds * multiplier / 3600,
            "unit": "CPU-core h" if cpu else "H100 h", "allocation": multiplier,
            "search_records": len(search), "search_terminal_records": sum(bool(r['data'].get('done')) for r in search),
            "reused_hpo": reused, "backfilled": backfilled, "sources": sources}


def direct_summary(model, records):
    rows = [r for r in records if r["key"].startswith(f"{DIRECT}{model}/")
            and r["data"].get("kind") in ("trial", "final")]
    share = DIRECT_GPU[model]
    for row in rows:
        recorded = row["data"].get("run_costs", {}).get("cost_ray_gpu_assigned")
        if recorded is not None and float(recorded) != share:
            raise ValueError(f"Unexpected allocation: {row['key']}")
    finals = [r["data"] for r in rows if r["data"]["kind"] == "final" and r["data"].get("ok")]
    fits = {int(r["seed"]): r["run_costs"]["cost_training_duration_sec"] for r in finals
            if r["horizon"] == 60 and r.get("run_costs", {}).get("cost_training_duration_sec") is not None}
    trial_ids = {(r["data"]["index"], r["data"]["horizon"]) for r in rows if r["data"]["kind"] == "trial"}
    final_ids = {(r["horizon"], r["seed"]) for r in finals}
    expected_trials = {(i, h) for i in range(16) for h in (1, 60)}
    expected_finals = {(h, s) for h in (1, 15, 60) for s in SEEDS}
    for row in rows:
        if row["data"].get("wall_seconds") is None:
            raise ValueError(f"Missing task duration: {row['key']}")
    return {"model": model, "rep": "2d", "source": "direct", "fit_minutes": statistics.median(fits.values())/60 if fits else None,
            "fit_n": len(fits), "allocated_hours_lower_bound": sum(r["data"]["wall_seconds"] for r in rows) * share / 3600,
            "unit": "H100 h", "allocation": share, "search_records": len(trial_ids), "final_records": len(final_ids),
            "grid_complete": expected_trials == trial_ids and expected_finals == final_ids,
            "sources": [{"key": r["key"], "seconds": r["data"]["wall_seconds"], "phase": r["data"]["kind"]} for r in rows]}


def summarize(records):
    identities = sorted({(r["key"].split("/")[1], r["key"].split("/")[2]) for r in records
                         if r["key"].startswith(CORE_ROOTS) and r["key"].endswith("_run_costs.json")})
    rows = [core_summary(rep, model, records) for rep, model in identities]
    return rows + [direct_summary(model, records) for model in DIRECT_GPU]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("evaluation/.env"))
    parser.add_argument("--cache", type=Path, help="Previously downloaded raw records (read-only; avoids S3 calls)")
    parser.add_argument("--selection-records", type=Path, help="Archived cross-band/resolution winner records to retain as provenance")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    records = json.loads(args.cache.read_text()) if args.cache else collect(s3_client(args.env_file), "spectrum")
    rows = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = {"scope": "600-800 MHz, one-minute POWDER", "totals": "Recorded allocated-compute lower bounds; no queue time. Includes task setup/validation/evaluation where recorded. Lost retries and unreported tails excluded; missing replicate tails backfilled with training duration. All search rounds and recorded prior replicate sets included; reused HPO seed not fitted twice.", "rows": rows}
    if args.selection_records:
        document["cross_setting_winner_records"] = json.loads(args.selection_records.read_text())
    (args.output_dir / "appendix_model_costs.json").write_text(json.dumps(document, indent=2) + "\n")
    fields = ["source", "rep", "model", "fit_minutes", "fit_n", "allocated_hours_lower_bound", "unit", "allocation", "search_records", "final_records", "grid_complete"]
    with (args.output_dir / "appendix_model_costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps({k: row.get(k) for k in fields}))


if __name__ == "__main__":
    main()
