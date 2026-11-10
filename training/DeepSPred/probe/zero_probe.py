"""DeepSPred probe 4: which files produce exact-zero (floor) bins under a rectangular window, and why?

The full-range dB minimum of the cross-validation set is exactly -120.0 under the rectangular window (the 1e-12 floor of
10*log10(|S|^2 + 1e-12)), but not under Hann/Hamming/Tukey. For every file of both datasets this records, for the rectangular
hop-126 STFT: how many (frame, bin) entries sit at or below -100 dB, which frames and bins, and simple facts about the raw samples
(count of exactly-zero samples, longest run of identical consecutive samples, number of distinct sample values). The same
count is taken for Hann so the two windows can be compared on the very same files.

  python training/DeepSPred/probe/zero_probe.py --data-root data/deepspred
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rendering_probe as rp  # noqa: E402

THRESH_DB = -100.0


def longest_run(a: np.ndarray) -> int:
    """Longest run of identical consecutive complex samples."""
    if len(a) < 2:
        return len(a)
    same = a[1:] == a[:-1]
    best = cur = 0
    for v in same:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best + 1


def inspect_file(path_str: str) -> dict:
    x = rp.read_iq(Path(path_str))
    out = {"file": Path(path_str).name, "n": int(len(x))}
    for w in ("rect126", "hann126"):
        db = rp.spectrogram_db(x, w + "~12")  # keep the 1e-12 floor so exact zeros show as -120.0
        low = db <= THRESH_DB
        out[w + "_min"] = float(db.min())
        out[w + "_n_low"] = int(low.sum())
        if low.any():
            rows, cols = np.where(low)  # rows = STFT time index, cols = frequency bin
            out[w + "_frames"] = sorted(set(int(r) for r in rows))[:12]
            out[w + "_bins"] = sorted(set(int(c) for c in cols))[:12]
    if out["rect126_n_low"] > 0 or out["hann126_n_low"] > 0:
        out["zero_samples"] = int(np.sum(x == 0))
        out["longest_identical_run"] = int(longest_run(x))
        out["distinct_samples"] = int(len(np.unique(x)))
        out["max_abs"] = float(np.abs(x).max())
        out["mean_I"], out["mean_Q"] = float(x.real.mean()), float(x.imag.mean())
    return out


def run_dataset(root: Path, name: str, cpus: int) -> dict:
    t0 = time.time()
    seconds = rp.list_seconds(root / rp.DATASETS[name])
    paths = [str(p) for _, p in seconds]
    times = [t.isoformat() for t, _ in seconds]
    mod = importlib.import_module("zero_probe")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(cpus) as pool:
        res = pool.map(mod.inspect_file, paths, chunksize=8)
    hits = []
    for r, t in zip(res, times):
        if r["rect126_n_low"] > 0 or r["hann126_n_low"] > 0:
            r["time"] = t
            hits.append(r)
    summary = {"n_files": len(paths), "n_files_rect_low": sum(1 for r in res if r["rect126_n_low"] > 0),
               "n_files_hann_low": sum(1 for r in res if r["hann126_n_low"] > 0),
               "rect_min_overall": min(r["rect126_min"] for r in res), "hann_min_overall": min(r["hann126_min"] for r in res),
               "files": hits[:200]}
    print(f"[{name}] {len(paths)} files in {time.time() - t0:.0f}s: rect has entries <= {THRESH_DB} dB in "
          f"{summary['n_files_rect_low']} files, hann in {summary['n_files_hann_low']}", flush=True)
    return summary


def work(data_root: str, out_dir: str, cpus: int) -> dict:
    root = Path(data_root)
    result = {"created": datetime.utcnow().isoformat() + "Z", "host": os.uname().nodename, "threshold_db": THRESH_DB, "datasets": {}}
    for name in rp.DATASETS:
        if (root / rp.DATASETS[name]).exists():
            result["datasets"][name] = run_dataset(root, name, cpus)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"zero_probe_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path} on {result['host']}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/deepspred")
    ap.add_argument("--out-dir", default="data/deepspred/probe")
    ap.add_argument("--cpus", type=int, default=16)
    args = ap.parse_args()
    data_root = str(Path(args.data_root).resolve())
    out_dir = str(Path(args.out_dir).resolve())
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"] and n.get("Resources", {}).get("CPU", 0) >= args.cpus]
    probe_dir = str(HERE)

    def has_data(path):
        import os.path
        return os.path.isdir(os.path.join(path, "FM-Spectrum-Dataset"))

    def remote_entry(pdir, data_root, out_dir, cpus):
        import importlib
        import sys
        if pdir not in sys.path:
            sys.path.insert(0, pdir)
        return importlib.import_module("zero_probe").work(data_root, out_dir, cpus)

    probes = {n["NodeID"]: ray.remote(num_cpus=0.01, scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(has_data).remote(data_root)
              for n in nodes}
    ok = []
    for nid, ref in probes.items():
        try:
            if ray.get(ref, timeout=120):
                ok.append(nid)
        except Exception as exc:
            print(f"probe on {nid[:8]} failed: {type(exc).__name__}", flush=True)
    if not ok:
        sys.exit(f"no node has {data_root}/{rp.DATASETS['fm']}")
    task = ray.remote(num_cpus=args.cpus, num_gpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(ok[0], soft=False))(remote_entry)
    ray.get(task.remote(probe_dir, data_root, out_dir, args.cpus))


if __name__ == "__main__":
    main()
