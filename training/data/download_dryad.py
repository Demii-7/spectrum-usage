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

# Each tuple: (local_filename, Dryad API download path)
FILES = [
    ("ResultsLW1Feb2022_SigMF.zip", "/downloads/file_stream/4677590"),
    ("ResultsCC1Feb2022_SigMF.zip", "/downloads/file_stream/4677592"),
    ("ResultsCC2Feb2022_SigMF.zip", "/downloads/file_stream/4677591"),
]


def _solve_worker(args):
    """Worker process: search a range of nonces for a SHA256 matching the difficulty.

    Dryad uses Anubis PoW: find a nonce such that sha256(randomData || nonce)
    starts with `difficulty` zero hex digits. Each worker searches an exclusive
    chunk of the nonce space.

    Args:
        args: Tuple of (random_data_hex, difficulty, lo, hi) where lo and hi
              define the nonce range [lo, hi).

    Returns:
        Tuple of (nonce, hash_hex) on success, or (None, None) if not found
        in the assigned range.
    """
    rd, diff, lo, hi = args
    for n in range(lo, hi):
        h = hashlib.sha256(f"{rd}{n}".encode()).hexdigest()
        # Check if the first `diff` hex chars are all '0'
        if all(c == "0" for c in h[:diff]):
            return n, h
    return None, None


def solve(rd, diff):
    """Solve Anubis PoW challenge using parallel brute-force search across all CPUs.

    Spawns a worker pool that repeatedly searches 50000-nonce chunks until a
    valid nonce is found. Each worker gets a disjoint nonce range to avoid
    redundant work.

    Args:
        rd: Random data hex string from the challenge.
        diff: Required difficulty (number of leading zero hex digits).

    Returns:
        Tuple of (nonce: int, hash_hex: str) for the solved challenge.
    """
    nw = max(1, cpu_count() - 1)  # Leave one CPU free for system responsiveness
    cs = 50000                     # Chunk size per worker per iteration
    with Pool(nw) as pool:
        while True:
            # Map each worker to a disjoint 50000-nonce slice, stagger by worker index
            for n, h in pool.map(
                _solve_worker,
                [(rd, diff, i * cs, (i + 1) * cs) for i in range(nw)],
            ):
                if n is not None:
                    return n, h


def auth_and_download(url, filepath):
    """Handle Anubis PoW authentication and download a file with progress reporting.

    Anubis is Dryad's anti-scraping gateway. It presents a SHA256 proof-of-work
    challenge via an embedded JSON script tag. This function:
      1. Fetches the URL and checks for HTTP 403 (rate-limited).
      2. Looks for the Anubis challenge script tag.
      3. If found, solves the PoW and submits the solution.
      4. Downloads the file with streaming progress logging every 15 seconds.
      5. Retries on rate-limit (caller handles retry logic in main()).

    Args:
        url: Full download URL.
        filepath: Local filesystem path to save the downloaded file.

    Returns:
        True if download succeeded; False if rate-limited (HTTP 403).
    """
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
    # HTTP 403 means Anubis rejected us; caller will retry after a delay
    if r.status_code == 403:
        return False

    # Dryad embeds the Anubis challenge parameters in a JSON script tag
    m = re.search(
        r'<script id="anubis_challenge" type="application/json">(.+?)</script>',
        r.text,
        re.DOTALL,
    )
    if not m:
        # No challenge found — probably a direct download URL
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
        # Submit the solution via Anubis API to get a session cookie
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
            allow_redirects=False,  # Prevent redirect loop; cookie is set in response headers
        )
        # Now the session cookie should allow us to download directly
        r2 = s.get(url, timeout=30, stream=True)

    if r2.status_code != 200:
        print(f"  Download failed: HTTP {r2.status_code}", flush=True)
        return False

    total = int(r2.headers.get("Content-Length", 0))
    print(f"  Size: {total / 1e9:.2f} GB", flush=True)

    # Stream download with periodic progress reporting
    with open(filepath, "wb") as f:
        dl = 0
        start = time.time()
        last_log = time.time()
        for chunk in r2.iter_content(256 * 1024):  # 256 KB chunks
            if chunk:
                f.write(chunk)
                dl += len(chunk)
                now = time.time()
                # Log progress every 15 seconds to avoid spamming the terminal
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
    """Parse arguments and download each file with rate-limit retry."""
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
        # Skip if file exists and is > 1 MB (partial downloads smaller than this are retried)
        if filepath.exists() and filepath.stat().st_size > 1_000_000:
            print(f"Skip {name} (already exists, {filepath.stat().st_size/1e9:.1f} GB)", flush=True)
            continue

        url = f"{BASE}{path}"
        print(f"\n{name}:", flush=True)
        # Retry loop: if rate-limited, wait 60 seconds and try again
        while True:
            if auth_and_download(url, str(filepath)):
                break
            print("  Rate limited, waiting 60s...", flush=True)
            time.sleep(60)

    print("\nAll downloads complete!", flush=True)


if __name__ == "__main__":
    main()
