"""Fast trivial-reference probe: persistence + lookback-mean MSE per frame k (uint8_mean_per_channel).
MSE-only (no SSIM/PSNR/LPIPS) so it runs in ~1 min per spec. Chunked to keep RSS low."""
import sys
from pathlib import Path

from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'training' / 'DeepSPred' / 'repro'))
sys.path.insert(0, str(REPO))

import numpy as np
from fm_data import load_frames_spec, chronological_split
from run_paper_faithful import windows


def refs(test, segs, T=20, K=20, chunk=128):
    idx = windows(test, T, K, segs['test'])
    n = len(idx)
    pers_sq = np.zeros(K)
    lbm_sq = np.zeros(K)
    for i in range(0, n, chunk):
        sub = idx[i:i + chunk]
        true = np.stack([test[b:c] for _, b, c in sub]).astype(np.float32)
        pers = np.stack([np.repeat(test[b - 1][None], K, axis=0) for _, b, _ in sub]).astype(np.float32)
        lbm = np.stack([np.repeat(test[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), K, axis=0)
                        for a, b, _ in sub]).astype(np.float32)
        pers_sq += ((pers - true) ** 2).sum(axis=(0, 2, 3, 4))
        lbm_sq += ((lbm - true) ** 2).sum(axis=(0, 2, 3, 4))
    npx = n * 256 * 256 * 3
    return pers_sq / npx, lbm_sq / npx, n


for spec in sys.argv[1:]:
    frames, meta = load_frames_spec(REPO / 'data' / 'datasets' / 'deepspred' / 'FM-Spectrum-Dataset', spec, range_scope='full')
    (train, val, test), segs = chronological_split(frames, meta)
    p, l, n = refs(test, segs)
    print(f"SPEC {spec}: n_windows={n}", flush=True)
    print(f"  persistence @1,4,8,12,16,20 : {[round(x,1) for x in p[[0,3,7,11,15,19]]]}", flush=True)
    print(f"  lookback   @1,4,8,12,16,20 : {[round(x,1) for x in l[[0,3,7,11,15,19]]]}", flush=True)
    print(f"  floor~persistence/2 @8 = {p[7]/2:.0f} | anchor 448.76 is: "
          f"{'REACHABLE & NON-TRIVIAL' if p[7]/2 < 448.76 < p[7] else ('below floor (impossible)' if 448.76 < p[7]/2 else 'at/below persistence (trivial)')}",
          flush=True)
