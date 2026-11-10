"""DeepSPred probe 3: how wide is the dB range under each window, depending on which frames set it?

Normalisation is the paper's biggest silent choice: with min-max scaling, MSE goes roughly as 1/span^2
(span = hi - lo in dB), so the absolute MSE level is nearly a free parameter. The collaborator's runs
use a span estimated from every 50th second. This probe measures, for each window and dataset:

  full      min/max over ALL frames (any frame, including test)
  sampled   min/max over every 50th frame (what the reproduction used)
  train     min/max over the first 2/3 of the capture span only
  p0.1/99.9 the 0.1 / 99.9 percentile of pixel values over ALL frames (a robust range)
  frame     median over frames of the per-frame (max - min)  (what per-frame scaling would use)

The scipy-default reading (tukey224d) would need spans of about 187 dB (FM) and 177 dB (cross-validation)
to bring the best-linear-predictor MSE from ~725 / ~1060 down to the paper's ~450 / ~531.

Run (UC cluster, repo root):
  python training/DeepSPred/probe/range_probe.py --data-root data/deepspred
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rendering_probe as rp  # noqa: E402

WINDOWS_USED = ("hann126", "hamm128", "rect126", "rect126d", "tukey126", "tukey126d", "tukey224", "tukey224d")
SAMPLE_EVERY = 50
# percentile cut points kept per frame: the global percentile is approximated by the mean of per-frame percentiles
PCTS = (0.1, 99.9)


def file_ranges(path_str: str) -> dict:
    x = rp.read_iq(Path(path_str))
    out = {}
    for w in WINDOWS_USED:
        db = rp.spectrogram_db(x, w)
        lo_p, hi_p = np.percentile(db, PCTS)
        out[w] = (float(db.min()), float(db.max()), float(lo_p), float(hi_p))
    return out


def run_dataset(root: Path, name: str, cpus: int) -> dict:
    t0 = time.time()
    seconds = rp.list_seconds(root / rp.DATASETS[name])
    times = [t for t, _ in seconds]
    paths = [str(p) for _, p in seconds]
    mod = importlib.import_module("range_probe")  # the module the pool workers will see (not __main__)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(cpus) as pool:
        res = pool.map(mod.file_ranges, paths, chunksize=8)
    print(f"[{name}] {len(paths)} files in {time.time() - t0:.0f}s", flush=True)
    t0_, span = times[0], (times[-1] - times[0]).total_seconds()
    train_cut = t0_ + timedelta(seconds=span * 2.0 / 3.0)
    train = np.array([t < train_cut for t in times])
    samp = np.zeros(len(paths), dtype=bool)
    samp[::SAMPLE_EVERY] = True
    out = {"n_files": len(paths), "windows": {}}
    for w in WINDOWS_USED:
        a = np.array([r[w] for r in res])  # (N, 4): min, max, p0.1, p99.9
        row = {}
        for label, m in (("full", np.ones(len(paths), bool)), ("sampled", samp), ("train", train)):
            lo, hi = float(a[m, 0].min()), float(a[m, 1].max())
            row[label] = {"lo": round(lo, 2), "hi": round(hi, 2), "span": round(hi - lo, 2)}
        plo, phi = float(a[:, 2].mean()), float(a[:, 3].mean())
        row["pct_0.1_99.9_mean_of_frames"] = {"lo": round(plo, 2), "hi": round(phi, 2), "span": round(phi - plo, 2)}
        row["frame_span_median"] = round(float(np.median(a[:, 1] - a[:, 0])), 2)
        row["frame_max_p99"] = round(float(np.percentile(a[:, 1], 99)), 2)
        row["frame_min_p1"] = round(float(np.percentile(a[:, 0], 1)), 2)
        out["windows"][w] = row
        print(f"[{name}] {w:10s} full {row['full']['lo']:7.1f}..{row['full']['hi']:7.1f} span {row['full']['span']:6.1f} | "
              f"sampled span {row['sampled']['span']:6.1f} | train span {row['train']['span']:6.1f} | "
              f"p0.1-99.9 span {row['pct_0.1_99.9_mean_of_frames']['span']:6.1f} | per-frame median span {row['frame_span_median']:6.1f}", flush=True)
    return out


def work(data_root: str, out_dir: str, cpus: int) -> dict:
    root = Path(data_root)
    result = {"created": datetime.utcnow().isoformat() + "Z", "host": os.uname().nodename, "datasets": {}}
    for name in rp.DATASETS:
        if (root / rp.DATASETS[name]).exists():
            result["datasets"][name] = run_dataset(root, name, cpus)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"range_probe_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path} on {result['host']}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/deepspred")
    ap.add_argument("--out-dir", default="data/deepspred/probe")
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    data_root = str(Path(args.data_root).resolve())
    out_dir = str(Path(args.out_dir).resolve())
    if args.local:
        work(data_root, out_dir, args.cpus)
        return
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"] and n.get("Resources", {}).get("CPU", 0) >= args.cpus]
    probe_dir = str(HERE)

    def has_data(path):  # only builtins/stdlib: cloudpickle ships it by value
        import os.path
        return os.path.isdir(os.path.join(path, "FM-Spectrum-Dataset"))

    def remote_entry(pdir, data_root, out_dir, cpus):
        import importlib
        import sys
        if pdir not in sys.path:
            sys.path.insert(0, pdir)
        return importlib.import_module("range_probe").work(data_root, out_dir, cpus)

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
    print(f"data present on {len(ok)} node(s); using {ok[0]}", flush=True)
    task = ray.remote(num_cpus=args.cpus, num_gpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(ok[0], soft=False))(remote_entry)
    ray.get(task.remote(probe_dir, data_root, out_dir, args.cpus))


if __name__ == "__main__":
    main()
