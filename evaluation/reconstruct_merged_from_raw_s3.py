#!/usr/bin/env python3
"""Reconstruct merged per-minute 1 MHz power CSVs from raw S3 objects."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys

import numpy as np


MINUTE_RE = re.compile(r"/raw/minute_(\d{4})_[^/]+/")


def parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None or not path.exists():
        return values

    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise ValueError(f"invalid .env line {line_no}: missing '='")
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"invalid .env line {line_no}: empty key")
            values[key] = parse_env_value(value)
    return values


def build_s3_client(env_values: dict[str, str]):
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("Install dependency first: python3 -m pip install boto3 numpy") from exc

    endpoint_url = env_values.get("S3_ENDPOINT_URL") or None
    region_name = env_values.get("AWS_DEFAULT_REGION") or env_values.get("AWS_REGION") or "us-east-1"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=env_values.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env_values.get("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )


def parse_run_id(run_id: str) -> datetime:
    return datetime.strptime(run_id, "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)


def parse_bands(value: str) -> list[tuple[int, int]]:
    bands = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid band {item!r}; expected START:STOP")
        start_text, stop_text = item.split(":", 1)
        start_mhz = int(start_text)
        stop_mhz = int(stop_text)
        if stop_mhz <= start_mhz:
            raise ValueError(f"invalid band {item!r}; stop must exceed start")
        bands.append((start_mhz, stop_mhz))
    if not bands:
        raise ValueError("at least one band is required")
    return bands


def band_label(band: tuple[int, int]) -> str:
    return f"{band[0]}_{band[1]}"


def list_common_prefixes(s3, bucket: str, prefix: str, delimiter: str = "/") -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
        prefixes.extend(item["Prefix"] for item in page.get("CommonPrefixes", []))
    return prefixes


def list_objects(s3, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []) if not item["Key"].endswith("/"))
    return keys


def discover_nodes(s3, bucket: str, dataset: str, run_id: str, requested_nodes: list[str] | None) -> list[str]:
    if requested_nodes:
        return requested_nodes

    dataset_prefix = f"{dataset.strip('/')}/"
    nodes = []
    for prefix in list_common_prefixes(s3, bucket, dataset_prefix):
        node = prefix.rstrip("/").split("/")[-1]
        run_prefix = f"{dataset_prefix}{node}/{run_id}/"
        if list_common_prefixes(s3, bucket, run_prefix):
            nodes.append(node)
    return sorted(nodes)


def group_raw_keys_by_minute(keys: list[str]) -> dict[int, list[str]]:
    by_minute: dict[int, list[str]] = {}
    for key in keys:
        if not key.endswith(".npz"):
            continue
        match = MINUTE_RE.search("/" + key)
        if match is None:
            continue
        minute_idx = int(match.group(1))
        by_minute.setdefault(minute_idx, []).append(key)
    for minute_keys in by_minute.values():
        minute_keys.sort()
    return by_minute


def download_key(s3, bucket: str, key: str, cache_dir: Path) -> Path:
    local_path = cache_dir / key
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))
    return local_path


def aggregate_to_1mhz(sweeps: list[tuple[np.ndarray, np.ndarray, float]], start_mhz: int, stop_mhz: int) -> np.ndarray:
    bin_edges = np.arange(start_mhz, stop_mhz + 1)
    n_bins = stop_mhz - start_mhz

    bin_weighted_powers = []
    bin_weights = []
    for freqs_hz, psd_db, center_freq_hz in sweeps:
        if freqs_hz.size == 0 or psd_db.size == 0:
            continue

        max_offset = np.nanmax(np.abs(freqs_hz - center_freq_hz))
        weights = 1.0 - np.abs(freqs_hz - center_freq_hz) / max(max_offset, 1.0)
        weights = np.clip(weights, 0.05, 1.0)

        nan_mask = np.isnan(psd_db)
        weights = np.where(nan_mask, 0.0, weights)
        freqs_mhz = freqs_hz / 1e6
        bin_indices = np.digitize(freqs_mhz, bin_edges) - 1

        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        if not np.any(valid):
            continue

        binned_power = np.full(n_bins, np.nan)
        binned_weight = np.full(n_bins, np.nan)
        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx) & valid & (~nan_mask)
            if np.any(mask):
                linear_power = 10.0 ** (psd_db[mask] / 10.0)
                bin_weights_for_samples = weights[mask]
                binned_power[bin_idx] = np.average(linear_power, weights=bin_weights_for_samples)
                binned_weight[bin_idx] = np.mean(bin_weights_for_samples)

        bin_weighted_powers.append(binned_power)
        bin_weights.append(binned_weight)

    if not bin_weighted_powers:
        return np.full(n_bins, np.nan)

    bin_weighted_powers_arr = np.array(bin_weighted_powers)
    bin_weights_arr = np.array(bin_weights)
    total_weight = np.nansum(bin_weights_arr, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_linear = np.nansum(bin_weighted_powers_arr * bin_weights_arr, axis=0) / total_weight
    mean_linear = np.where(total_weight == 0, np.nan, mean_linear)
    return np.where(np.isnan(mean_linear), np.nan, 10.0 * np.log10(np.maximum(mean_linear, 1e-30)))


def reconstruct_band_minute(s3, bucket: str, keys: list[str], cache_dir: Path, band: tuple[int, int]) -> np.ndarray:
    sweeps = []
    for key in keys:
        local_path = download_key(s3, bucket, key, cache_dir)
        try:
            with np.load(local_path) as npz:
                freqs_hz = np.asarray(npz["freq_hz"], dtype=float)
                power_db = np.asarray(npz["power_db"], dtype=float)
                center_freq_hz = float(np.asarray(npz["center_freq_hz"]))
        except Exception as exc:
            print(f"warning: could not read {key}: {exc}", file=sys.stderr)
            continue
        sweeps.append((freqs_hz, power_db, center_freq_hz))
    return aggregate_to_1mhz(sweeps, band[0], band[1])


def format_value(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.6f}"


def write_node_csv(
    s3,
    bucket: str,
    dataset: str,
    node: str,
    run_id: str,
    bands: list[tuple[int, int]],
    cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    run_start = parse_run_id(run_id)
    run_prefix = f"{dataset.strip('/')}/{node}/{run_id}/"
    raw_by_band: dict[tuple[int, int], dict[int, list[str]]] = {}

    for band in bands:
        raw_prefix = f"{run_prefix}{band_label(band)}/raw/"
        keys = list_objects(s3, bucket, raw_prefix)
        raw_by_band[band] = group_raw_keys_by_minute(keys)
        print(f"{node} {band_label(band)}: {len(raw_by_band[band])} raw minute(s), {len(keys)} object(s)")

    minute_indices = sorted(set().union(*(set(v) for v in raw_by_band.values())))
    if not minute_indices:
        print(f"warning: no raw minutes found for {node}; skipping", file=sys.stderr)
        return None

    node_output_dir = output_dir / dataset / node / run_id
    node_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = node_output_dir / "power_merged_1mhz_avg_per_minute.csv"

    header = ["timestamp_utc"]
    for start_mhz, stop_mhz in bands:
        header.extend(f"{freq + 0.5:.1f}" for freq in range(start_mhz, stop_mhz))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_no, minute_idx in enumerate(minute_indices, start=1):
            timestamp = (run_start + timedelta(minutes=minute_idx)).strftime("%Y-%m-%dT%H:%M:%SZ")
            row = [timestamp]
            for band in bands:
                keys = raw_by_band[band].get(minute_idx)
                if keys:
                    values = reconstruct_band_minute(s3, bucket, keys, cache_dir, band)
                else:
                    values = np.full(band[1] - band[0], np.nan)
                row.extend(format_value(value) for value in values)
            writer.writerow(row)
            if row_no % 25 == 0 or row_no == len(minute_indices):
                print(f"{node}: wrote {row_no}/{len(minute_indices)} merged row(s)")

    return output_path


def upload_csv(s3, bucket: str, dataset: str, node: str, run_id: str, output_path: Path) -> str:
    key = f"{dataset.strip('/')}/{node}/{run_id}/{output_path.name}"
    s3.upload_file(str(output_path), bucket, key)
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("evaluation/.env"))
    parser.add_argument("--bucket", default="spectrum")
    parser.add_argument("--dataset", required=True, help="Dataset prefix, e.g. ara, powder, cosmos.")
    parser.add_argument("--run-id", required=True, help="Run ID in YYYYMMDDTHHMMZ format.")
    parser.add_argument(
        "--bands",
        required=True,
        help="Comma-separated bands as START:STOP, e.g. 600:800,2400:2600,3500:3700.",
    )
    parser.add_argument("--node", action="append", help="Process one node. Repeat to process multiple nodes. Default: all nodes with the run ID.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/spectrum-raw-s3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructed"))
    parser.add_argument("--upload", action="store_true", help="Upload reconstructed CSVs to S3.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        env_values = load_env_file(args.env_file)
        bands = parse_bands(args.bands)
        parse_run_id(args.run_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    s3 = build_s3_client(env_values)
    nodes = discover_nodes(s3, args.bucket, args.dataset, args.run_id, args.node)
    if not nodes:
        print(f"error: no nodes found for s3://{args.bucket}/{args.dataset}/{args.run_id}", file=sys.stderr)
        return 1

    print(f"processing {len(nodes)} node(s): {', '.join(nodes)}")
    for node in nodes:
        output_path = write_node_csv(
            s3=s3,
            bucket=args.bucket,
            dataset=args.dataset,
            node=node,
            run_id=args.run_id,
            bands=bands,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
        )
        if output_path is None:
            continue
        print(f"wrote {output_path}")
        if args.upload:
            key = upload_csv(s3, args.bucket, args.dataset, node, args.run_id, output_path)
            print(f"uploaded s3://{args.bucket}/{key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
