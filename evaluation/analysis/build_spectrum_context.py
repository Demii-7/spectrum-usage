#!/usr/bin/env python3
"""Build one combined spectral-context results table from Ray/MinIO records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import boto3


MODELS = (
    "autoformer_csa",
    "lstmattn",
    "residualvanillalstm",
    "temporalconvnet",
    "vanillalstm",
    "linearar2d",
    "residuallinearar2d",
)
CONDITIONS = ("full_context", "region_only")
SEEDS = (40, 41, 42, 43, 44)
METRIC_FIELDS = (
    "objective",
    "objective_name",
    "test_mean_horizon_mae_db",
    "test_mean_horizon_rmse_db",
    "test_mae_db_t1",
    "test_rmse_db_t1",
    "test_mae_db_t15",
    "test_rmse_db_t15",
    "test_mae_db_t60",
    "test_rmse_db_t60",
    "test_n_values_t1",
    "test_n_values_t15",
    "test_n_values_t60",
)
FIELDS = (
    "model_name",
    "seed",
    "condition",
    "band_id",
    "start_mhz",
    "end_mhz",
    "run_status",
    "campaign_path",
    "trial_id",
    *METRIC_FIELDS,
)


def _records(client: Any, bucket: str, prefixes: list[str]) -> list[tuple[dict[str, Any], str]]:
    records: list[tuple[dict[str, Any], str]] = []
    for prefix in prefixes:
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith("/result.json"):
                    continue
                raw = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
                for line in reversed(raw.splitlines()):
                    try:
                        record = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
                if record.get("model_name") not in MODELS:
                    continue
                records.append((record, key))
    return records


def _campaign_path(key: str, bucket: str) -> str:
    path = key.removeprefix(bucket + "/")
    parts = path.split("/")
    return "/".join(parts[:4]) if len(parts) >= 4 else path


def _priority(record: dict[str, Any], key: str) -> tuple[int, int, float]:
    completed = bool(record.get("completed")) and "test_mean_horizon_mae_db" in record
    newer_campaign = "/production-20260730-spectral-context-v2/" in key
    return (int(completed), int(newer_campaign), float(record.get("timestamp", 0)))


def build(args: argparse.Namespace) -> None:
    credentials = {
        "aws_access_key_id": args.access_key,
        "aws_secret_access_key": args.secret_key,
    }
    client = boto3.client("s3", endpoint_url=args.endpoint, **credentials)
    prefixes = [
        f"{args.root_prefix.rstrip('/')}/production-20260730-spectral-context-v1/",
        f"{args.root_prefix.rstrip('/')}/production-20260730-spectral-context-v2/",
    ]
    records = _records(client, args.bucket, prefixes)

    by_cell: dict[tuple[str, int, str, str], tuple[dict[str, Any], str]] = {}
    for record, key in records:
        cell = (record["model_name"], int(record["seed"]), record["condition"], record["band_id"])
        if cell not in by_cell or _priority(record, key) > _priority(*by_cell[cell]):
            by_cell[cell] = (record, key)

    band_map: dict[str, tuple[Any, Any]] = {}
    for record, _ in records:
        band_map.setdefault(record["band_id"], (record.get("start_mhz", ""), record.get("end_mhz", "")))
    band_ids = tuple(sorted(band_map, key=lambda band: int(band.removeprefix("R"))))

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                for band_id in band_ids:
                    record_key = (model, seed, condition, band_id)
                    record_and_key = by_cell.get(record_key)
                    record, key = record_and_key if record_and_key else ({}, "")
                    completed = bool(record.get("completed")) and "test_mean_horizon_mae_db" in record
                    row = {
                        "model_name": model,
                        "seed": seed,
                        "condition": condition,
                        "band_id": band_id,
                        "start_mhz": record.get("start_mhz", band_map[band_id][0]),
                        "end_mhz": record.get("end_mhz", band_map[band_id][1]),
                        "run_status": "completed" if completed else "missing_metric",
                        "campaign_path": _campaign_path(key, args.bucket) if key else "",
                        "trial_id": record.get("trial_id", "") if completed else "",
                    }
                    for field in METRIC_FIELDS:
                        row[field] = record.get(field, "") if completed else ""
                    rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    completed_count = sum(row["run_status"] == "completed" for row in rows)
    print(f"wrote {len(rows)} rows ({completed_count} completed) to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--root-prefix", default="spectrum-usage/ray")
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
