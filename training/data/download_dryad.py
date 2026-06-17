#!/usr/bin/env python3
"""
Download the AERPAW sub-6 GHz spectrum monitoring dataset from Dryad.

This script handles Dryad's Anubis proof-of-work challenge automatically
by solving the SHA256 PoW using all available CPU cores.

Usage:
    python3 training/data/download_dryad.py [--dir DIR]

The three ZIP files (CC1 ~19 GB, CC2 ~21 GB, LW1 ~12 GB) are downloaded
into the specified directory (default: current working directory).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import requests

BASE = "https://datadryad.org"

FILES = [
    ("ResultsLW1Feb2022_SigMF.zip", "/downloads/file_stream/4677590"),
    ("ResultsCC1Feb2022_SigMF.zip", "/downloads/file_stream/4677592"),
    ("ResultsCC2Feb2022_SigMF.zip", "/downloads/file_stream/4677591"),
]


def _solve_worker(args):
    rd, diff, lo, hi = args
    for n in range(lo, hi):
        h = hashlib.sha256(f"{rd}{n}".encode()).hexdigest()
        if all(c == "0" for c in h[:diff]):
            return n, h
    return None, None


def solve(rd, diff):
    nw = max(1, cpu_count() - 1)
    cs = 50000
    with Pool(nw) as pool:
        while True:
            for n, h in pool.map(
                _solve_worker,
                [(rd, diff, i * cs, (i + 1) * cs) for i in range(nw)],
            ):
                if n is not None:
                    return n, h


def auth_and_download(url, filepath):
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    r = s.get(url, timeout=30)
    if r.status_code == 403:
        return False

    m = re.search(
        r'<script id="anubis_challenge" type="application/json">(.+?)</script>',
        r.text,
        re.DOTALL,
    )
    if not m:
        print("  No challenge, trying direct download...", flush=True)
        r2 = s.get(url, timeout=30, stream=True)
    else:
        c = json.loads(m.group(1))
        rd = c["challenge"]["randomData"]
        diff = c["challenge"]["difficulty"]
        cid = c["challenge"]["id"]
        print("  Solving Anubis PoW...", flush=True)
        start = time.time()
        nonce, h = solve(rd, diff)
        elapsed = time.time() - start
        print(f"  Solved in {elapsed:.0f}s", flush=True)
        params = {
            "id": cid,
            "response": h,
            "nonce": str(nonce),
            "redir": url,
            "elapsedTime": "1000",
        }
        s.get(
            f"{BASE}/.within.website/x/cmd/anubis/api/pass-challenge",
            params=params,
            timeout=30,
            allow_redirects=False,
        )
        r2 = s.get(url, timeout=30, stream=True)

    if r2.status_code != 200:
        print(f"  Download failed: HTTP {r2.status_code}", flush=True)
        return False

    total = int(r2.headers.get("Content-Length", 0))
    print(f"  Size: {total / 1e9:.2f} GB", flush=True)

    with open(filepath, "wb") as f:
        dl = 0
        start = time.time()
        last_log = time.time()
        for chunk in r2.iter_content(256 * 1024):
            if chunk:
                f.write(chunk)
                dl += len(chunk)
                now = time.time()
                if now - last_log >= 15:
                    elapsed = now - start
                    rate = dl / elapsed / 1024 / 1024
                    pct = dl / total * 100 if total else 0
                    print(
                        f"  {dl/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)"
                        f" @ {rate:.0f} MB/s",
                        flush=True,
                    )
                    last_log = now

    elapsed = time.time() - start
    rate = dl / elapsed / 1024 / 1024 if elapsed else 0
    print(
        f"  Done: {dl/1e9:.2f} GB in {elapsed:.0f}s ({rate:.0f} MB/s)",
        flush=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download AERPAW SigMF dataset from Dryad"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    out_dir = Path(args.dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in FILES:
        filepath = out_dir / name
        if filepath.exists() and filepath.stat().st_size > 1_000_000:
            print(f"Skip {name} (already exists, {filepath.stat().st_size/1e9:.1f} GB)", flush=True)
            continue

        url = f"{BASE}{path}"
        print(f"\n{name}:", flush=True)
        while True:
            if auth_and_download(url, str(filepath)):
                break
            print("  Rate limited, waiting 60s...", flush=True)
            time.sleep(60)

    print("\nAll downloads complete!", flush=True)


if __name__ == "__main__":
    main()
