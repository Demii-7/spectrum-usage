#!/usr/bin/env python3
"""Download per-band power CSVs from S3-compatible object storage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


CSV_NAME = "power_1mhz_avg_per_minute.csv"
EVALUATION_ROOT = Path(__file__).resolve().parent


def parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0

    loaded = 0
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
            if key in os.environ:
                continue

            os.environ[key] = parse_env_value(value)
            loaded += 1
    return loaded


def build_fs_kwargs(endpoint_url: str) -> dict[str, object]:
    fs_kwargs: dict[str, object] = {
        "skip_instance_cache": True,
        "use_listings_cache": False,
    }
    if endpoint_url:
        fs_kwargs["client_kwargs"] = {"endpoint_url": endpoint_url}
        fs_kwargs["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return fs_kwargs


def children(fs, prefix: str) -> list[str]:
    prefix = prefix.rstrip("/")
    try:
        paths = fs.ls(prefix, detail=False, refresh=True)
    except FileNotFoundError:
        return []
    return sorted(path.rstrip("/") for path in paths if path.rstrip("/") != prefix)


def candidate_paths(root: str, values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [f"{root.rstrip('/')}/{value.strip('/')}" for value in values]


def discover_power_csvs(
    fs,
    bucket_root: str,
    datasets: list[str] | None,
    nodes: list[str] | None,
    run_ids: list[str] | None,
    bands: list[str] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    bucket_root = bucket_root.rstrip("/")

    dataset_paths = candidate_paths(bucket_root, datasets) or children(fs, bucket_root)
    for dataset_path in dataset_paths:
        dataset = dataset_path.rsplit("/", 1)[-1]
        node_paths = candidate_paths(dataset_path, nodes) or children(fs, dataset_path)
        for node_path in node_paths:
            node = node_path.rsplit("/", 1)[-1]
            run_paths = candidate_paths(node_path, run_ids) or children(fs, node_path)
            for run_path in run_paths:
                run_id = run_path.rsplit("/", 1)[-1]
                band_paths = candidate_paths(run_path, bands) or children(fs, run_path)
                for band_path in band_paths:
                    band = band_path.rsplit("/", 1)[-1]
                    csv_path = f"{band_path}/{CSV_NAME}"
                    if not fs.exists(csv_path):
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "node": node,
                            "run_id": run_id,
                            "band": band,
                            "remote_path": csv_path,
                        }
                    )
    return rows


def local_csv_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / row["dataset"] / row["node"] / row["run_id"] / row["band"] / CSV_NAME


def remote_size_matches(fs, remote_path: str, local_path: Path) -> bool:
    if not local_path.exists():
        return False
    try:
        info = fs.info(remote_path)
    except FileNotFoundError:
        return False
    return info.get("size") == local_path.stat().st_size


def download_csvs(fs, rows: list[dict[str, str]], output_dir: Path, force: bool, dry_run: bool) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    failed = 0
    for row in rows:
        remote_path = row["remote_path"]
        local_path = local_csv_path(output_dir, row)

        if not force and remote_size_matches(fs, remote_path, local_path):
            skipped += 1
            print(f"skip existing {local_path}")
            continue

        if dry_run:
            downloaded += 1
            print(f"would download s3://{remote_path} -> {local_path}")
            continue

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(remote_path, str(local_path))
            downloaded += 1
            print(f"downloaded s3://{remote_path} -> {local_path}")
        except Exception as exc:
            failed += 1
            print(f"ERROR: failed to download s3://{remote_path}: {exc}", file=sys.stderr)
    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=EVALUATION_ROOT / ".env", help="Load credentials from this file.")
    parser.add_argument("--endpoint-url", default="", help="S3-compatible endpoint URL. Defaults to S3_ENDPOINT_URL.")
    parser.add_argument("--bucket", default="spectrum", help="Object-store bucket name.")
    parser.add_argument("--prefix", default="", help="Optional root prefix inside the bucket.")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_ROOT, help="Local output root.")
    parser.add_argument("--dataset", action="append", help="Dataset/site prefix, e.g. ara, powder, cosmos. Repeatable.")
    parser.add_argument("--node", action="append", help="Node name. Repeatable.")
    parser.add_argument("--run-id", action="append", help="Run ID. Repeatable.")
    parser.add_argument("--band", action="append", help="Band label, e.g. 600_800. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Download even when the local file size matches the remote object.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned downloads without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        loaded = load_env_file(args.env_file)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        import fsspec
    except ImportError:
        print("ERROR: fsspec is not installed. Install with: pip install fsspec s3fs", file=sys.stderr)
        return 2

    endpoint_url = args.endpoint_url or os.environ.get("S3_ENDPOINT_URL", "")
    bucket_root = args.bucket.strip("/")
    if args.prefix:
        bucket_root = f"{bucket_root}/{args.prefix.strip('/')}"

    print(f"env_file: {args.env_file if args.env_file.exists() else '(not found)'}")
    print(f"env_values_loaded: {loaded}")
    print(f"endpoint_url: {endpoint_url if endpoint_url else '(default AWS endpoint)'}")
    print(f"bucket_root: s3://{bucket_root}")

    fs = fsspec.filesystem("s3", **build_fs_kwargs(endpoint_url))
    rows = discover_power_csvs(fs, bucket_root, args.dataset, args.node, args.run_id, args.band)
    print(f"csv_count: {len(rows)}")

    downloaded, skipped, failed = download_csvs(fs, rows, args.output_dir, args.force, args.dry_run)
    verb = "would_download" if args.dry_run else "downloaded"
    print(f"complete: {verb}={downloaded}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
