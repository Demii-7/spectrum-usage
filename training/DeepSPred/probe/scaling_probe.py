"""Normalization-variant probe: render the tukey224d test split with a different scaling
(p1p99 / zf2 / zf3 / frame) and compute persistence + lookback-mean. Streaming, no cache files.
Usage: scaling_probe.py <spec> <scaling>"""
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np

from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'training' / 'DeepSPred' / 'probe'))
sys.path.insert(0, str(REPO / 'training' / 'DeepSPred' / 'repro'))
import rendering_probe as rp
from fm_data import list_seconds, read_iq, contiguous_segments

spec, scaling = sys.argv[1], sys.argv[2]
seconds = list_seconds(REPO / 'data' / 'datasets' / 'deepspred' / 'FM-Spectrum-Dataset')
times = [t for t, _ in seconds]; paths = [p for _, p in seconds]

# stats pass: min/max + 1/99th percentiles of dB over every 10th frame (full capture)
import multiprocessing as mp
def _db(a):
    return rp.spectrogram_db(read_iq(a)[0], spec)
with mp.get_context('fork').Pool(16) as pool:
    dbs = pool.map(_db, [str(p) for p in paths[::10]], chunksize=8)
allv = np.concatenate([d.ravel() for d in dbs])
stats = {'lo': float(allv.min()), 'hi': float(allv.max()),
         'p1': float(np.percentile(allv, 1)), 'p99': float(np.percentile(allv, 99))}

# render the test split (last 1/6 of the span, matching chronological_split's time boundary)
t0, span = times[0], (times[-1] - times[0]).total_seconds()
va_cut = t0 + timedelta(seconds=span * 5.0 / 6.0)
sel = [i for i, t in enumerate(times) if t >= va_cut]
def _frame(i):
    db = rp.spectrogram_db(read_iq(paths[i])[0], spec)
    return rp.render(db[None], scaling, 'raw', False, stats)[0]
with mp.get_context('fork').Pool(16) as pool:
    frames = np.stack(pool.map(_frame, sel, chunksize=8))

# contiguous segments local to the test block; windows wholly inside one segment
tt = [times[i] for i in sel]
idx = [(s, s + 20, s + 40) for s in range(0, len(frames) - 39)]
idx = [(a, b, c) for (a, b, c) in idx
       if all((tt[k+1] - tt[k]).total_seconds() == 1.0 for k in range(a, c - 1))]
K = 20
pers = np.zeros(K); lbm = np.zeros(K); n = 0
for i in range(0, len(idx), 128):
    sub = idx[i:i+128]
    true = np.stack([frames[b:c] for _, b, c in sub]).astype(np.float64)
    p = np.stack([np.repeat(frames[b-1][None], K, axis=0) for _, b, _ in sub]).astype(np.float64)
    l = np.stack([np.repeat(frames[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), K, axis=0)
                  for a, b, _ in sub]).astype(np.float64)
    pers += ((p-true)**2).sum(axis=(0, 2, 3, 4)); lbm += ((l-true)**2).sum(axis=(0, 2, 3, 4)); n += true.size // K
print(f"SCALING {spec} {scaling} [{stats if scaling=='p1p99' else ''}]: "
      f"pers@8={pers[7]/n:.0f} lbm@8={lbm[7]/n:.0f} windows={len(idx)} "
      f"(model would land ~{1.05*lbm[7]/n:.0f}-{1.15*lbm[7]/n:.0f})", flush=True)
