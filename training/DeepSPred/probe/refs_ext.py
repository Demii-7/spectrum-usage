"""Sweep probe: trivial references (persistence, lookback-mean, static train-mean) for a
rendering spec via the cache machinery. Usage: refs_ext.py <spec>"""
import sys
import numpy as np
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[2])
sys.path.insert(0, REPO + '/training/DeepSPred/repro')
sys.path.insert(0, REPO)
from fm_data import load_frames_spec, chronological_split
from run_paper_faithful import windows

spec = sys.argv[1]
frames, meta = load_frames_spec(REPO + '/data/datasets/deepspred/FM-Spectrum-Dataset', spec, range_scope='full')
(tr, va, te), segs = chronological_split(frames, meta)
idx = windows(te, 20, 20, segs['test'])
K = 20
pers = np.zeros(K); lbm = np.zeros(K); n = 0
mean_img = tr.mean(axis=0).round().clip(0, 255).astype(np.uint8).astype(np.float64)
static = 0.0; sn = 0
for i in range(0, len(idx), 128):
    sub = idx[i:i+128]
    true = np.stack([te[b:c] for _, b, c in sub]).astype(np.float64)
    p = np.stack([np.repeat(te[b-1][None], K, axis=0) for _, b, _ in sub]).astype(np.float64)
    l = np.stack([np.repeat(te[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), K, axis=0)
                  for a, b, _ in sub]).astype(np.float64)
    pers += ((p-true)**2).sum(axis=(0, 2, 3, 4)); lbm += ((l-true)**2).sum(axis=(0, 2, 3, 4)); n += true.size // K
    static += ((mean_img[None, None] - true)**2).sum(); sn += true.size
print(f"SWEEP {spec}: pers@8={pers[7]/n:.0f} lbm@8={lbm[7]/n:.0f} static={static/sn:.0f} "
      f"(floor~{pers[7]/n/2:.0f}, model would land ~{1.05*lbm[7]/n:.0f}-{1.15*lbm[7]/n:.0f})", flush=True)
