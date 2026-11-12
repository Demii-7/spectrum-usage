"""Ensemble scoring: average the predictions of two seeds' checkpoints (same rendering),
score test MSE, compare with each seed alone and the paper. Usage:
ensemble_score.py <sha12_a> <sha12_b> <spec> <scope>"""
import sys, json
import numpy as np, torch

from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/training/DeepSPred/repro')
from run_paper_faithful import build, windows, PRIMARY_SPACE
from fm_data import load_frames_spec, chronological_split

sha_a, sha_b, spec, scope = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
models = []
for sha in (sha_a, sha_b):
    ck = torch.load(f'{REPO}/checkpoints/repro/deepspred/{sha}.pt', map_location='cuda')
    arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']
    m = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T)
    m.load_state_dict(ck['model_state_dict']); m.to('cuda').eval(); models.append(m)

frames, meta = load_frames_spec(REPO + '/data/datasets/deepspred/FM-Spectrum-Dataset', spec, range_scope=scope)
(tr, va, te), segs = chronological_split(frames, meta)
idx = windows(te, T, K, segs['test'])
sq = np.zeros(K); sq_a = np.zeros(K); n = 0
with torch.no_grad():
    for i in range(0, len(idx), 8):
        sub = idx[i:i+8]
        x = torch.from_numpy(np.stack([te[a:b] for a, b, _ in sub]).astype(np.float32)/255.).permute(0,1,4,2,3).cuda()
        pa = models[0](x); pb = models[1](x)
        pens = ((pa+pb)/2).cpu().numpy().transpose(0,1,3,4,2)*255
        pens = pens.round().clip(0,255).astype(np.float64)
        pa_u = (pa.cpu().numpy().transpose(0,1,3,4,2)*255).round().clip(0,255).astype(np.float64)
        t = np.stack([te[b:c] for _, b, c in sub]).astype(np.float64)
        sq += ((pens-t)**2).sum(axis=(0,2,3,4))
        sq_a += ((pa_u-t)**2).sum(axis=(0,2,3,4)); n += len(sub)
mse = sq/(n*256*256*3); mse_a = sq_a/(n*256*256*3)
line = (f"ENSEMBLE {spec} seeds {sha_a[:6]}+{sha_b[:6]}: MSE@8={mse[7]:.1f} @16={mse[15]:.1f} @18={mse[17]:.1f} "
        f"(seed-a alone @8={mse_a[7]:.1f}; paper 448.76/453.04/454.18)")
print(line, flush=True)
open('/home/cc/verification_log.txt', 'a').write(line + '\n')
