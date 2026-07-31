#!/usr/bin/env python3
"""Combine spatial-ablation artifacts and calculate paired MAE deltas."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

KEY_FIELDS = ("model", "seed", "horizon", "receiver", "region_id")


def _cell(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def paired_deltas(
    rows: Iterable[dict[str, Any]],
    condition: str,
    reference_condition: str,
) -> list[dict[str, Any]]:
    rows = list(rows)
    references = {
        _cell(row): row for row in rows if row["condition"] == reference_condition
    }
    output = []
    for row in rows:
        if row["condition"] != condition:
            continue
        reference = references.get(_cell(row))
        if reference is None:
            continue
        output.append(
            {
                **{field: row[field] for field in KEY_FIELDS},
                "condition": condition,
                "reference_condition": reference_condition,
                "permutation_seed": row.get("permutation_seed", ""),
                "is_noise_floor": row["is_noise_floor"],
                "n_origins": row["n_origins"],
                "mae_db": row["mae_db"],
                "reference_mae_db": reference["mae_db"],
                "delta_mae_db": float(row["mae_db"]) - float(reference["mae_db"]),
            }
        )
    return output


def _artifact_rows(
    client: Any,
    bucket: str,
    prefix: str,
    artifact_name: str,
) -> list[dict[str, Any]]:
    rows = []
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(f"/checkpoint_000000/{artifact_name}"):
                continue
            text = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
            trial = key.split("/checkpoint_000000/", 1)[0].rsplit("/", 1)[-1]
            for row in csv.DictReader(StringIO(text)):
                row["artifact_key"] = key
                row["trial"] = trial
                rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
    )
    rows = _artifact_rows(
        client,
        args.bucket,
        args.prefix.rstrip("/") + "/",
        "region_metrics.csv",
    )
    map_rows = _artifact_rows(
        client,
        args.bucket,
        args.prefix.rstrip("/") + "/",
        "map_metrics.csv",
    )
    _write(args.output, rows)
    _write(args.map_output, map_rows)
    _write(
        args.geometry_deltas,
        paired_deltas(rows, "geometry_permuted", "geometry_correct"),
    )
    _write(
        args.new_receiver_deltas,
        paired_deltas(rows, "eight_to_new", "six_to_new"),
    )
    _write(
        args.guesthouse_deltas,
        paired_deltas(rows, "seven_to_guesthouse", "six_to_guesthouse"),
    )
    completed = {
        (row["model"], row["seed"], row["condition"], row.get("permutation_seed", ""))
        for row in rows
    }
    print(f"wrote {len(rows)} regional rows from {len(completed)} completed cells")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="spectrum-usage")
    parser.add_argument("--endpoint", default="http://minio:9000")
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--geometry-deltas", type=Path, required=True)
    parser.add_argument("--new-receiver-deltas", type=Path, required=True)
    parser.add_argument("--guesthouse-deltas", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
