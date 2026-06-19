#!/usr/bin/env python3
"""Verify fsspec access to S3-compatible object storage."""

import argparse
import os
import sys
from pathlib import Path


def parse_env_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path):
    if path is None or not path.exists():
        return 0

    loaded = 0
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
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


def build_fs_kwargs(endpoint_url):
    fs_kwargs = {
        "skip_instance_cache": True,
        "use_listings_cache": False,
    }
    if endpoint_url:
        fs_kwargs["client_kwargs"] = {"endpoint_url": endpoint_url}
        fs_kwargs["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return fs_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Load credentials from this file before connecting (default: .env if present).",
    )
    parser.add_argument(
        "--endpoint-url",
        default="",
        help="S3-compatible endpoint URL. Defaults to S3_ENDPOINT_URL if set.",
    )
    parser.add_argument(
        "--bucket",
        default="",
        help="Bucket to list. If omitted, list available buckets.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix inside --bucket to list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_file = args.env_file.resolve() if args.env_file else None

    try:
        loaded = load_env_file(env_file)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    endpoint_url = args.endpoint_url or os.environ.get("S3_ENDPOINT_URL", "")
    print(f"env_file: {env_file if env_file and env_file.exists() else '(not found)'}")
    print(f"env_values_loaded: {loaded}")
    print(f"endpoint_url: {endpoint_url if endpoint_url else '(default AWS endpoint)'}")
    print(f"has_access_key: {'yes' if os.environ.get('AWS_ACCESS_KEY_ID') else 'no'}")
    print(f"has_secret_key: {'yes' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'no'}")

    try:
        import fsspec
    except ImportError:
        print("ERROR: fsspec is not installed. Install with: pip install fsspec s3fs", file=sys.stderr)
        return 2

    try:
        fs = fsspec.filesystem("s3", **build_fs_kwargs(endpoint_url))
        if args.bucket:
            path = args.bucket.strip("/")
            if args.prefix:
                path = f"{path}/{args.prefix.strip('/')}"
            print(f"listing: s3://{path}")
            entries = fs.ls(path, refresh=True)
        else:
            print("listing buckets")
            entries = fs.ls("", refresh=True)
    except Exception as exc:
        print(f"ERROR: object storage check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("OK")
    for entry in entries:
        print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
