"""Blend scoring on tukey224d (the closest window): prediction = 0.5*model_s0 + 0.5*model_s1
+ 0.5*lookback-mean ensemble variants, scored against test. Complements the seed ensemble."""
import sys
import numpy as np, torch

from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/training/DeepSPred/repro')
from run_paper_faithful import build, windows, PRIMARY_SPACE
from fm_data import load_frames_spec, chronological_split

CK_A, CK_B = '3c96fc0dc0cd', 'c5a4e4c918ca'
_free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
DEV = f'cuda:{_free.index(max(_free))}'
models = []
for sha in (CK_A, CK_B):
    ck = torch.load(f'{REPO}/checkpoints/repro/deepspred/{sha}.pt', map_location=DEV)
    arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']
    m = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T)
    m.load_state_dict(ck['model_state_dict']); m.to(DEV).eval(); models.append(m)

frames, meta = load_frames_spec(REPO + '/data/datasets/deepspred/FM-Spectrum-Dataset', 'tukey224d', range_scope='full')
(tr, va, te), segs = chronological_split(frames, meta)
idx = windows(te, T, K, segs['test'])
sq = {'ens': np.zeros(K), 'ens_lbm': np.zeros(K), 's0_lbm': np.zeros(K)}
n = 0
for i in range(0, len(idx)):
    a, b, c = idx[i]
    x = torch.from_numpy(te[a:b][None].astype(np.float32)/255.).permute(0, 1, 4, 2, 3).to(DEV)
    with torch.no_grad():
        pa = models[0](x).cpu().numpy()[0]     # (K, 3, H, W) in [0,1]
        pb = models[1](x).cpu().numpy()[0]
    ens = (pa + pb) / 2
    lbm = np.repeat(te[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), K, axis=0)
    lbm01 = lbm.astype(np.float32).transpose(0, 3, 1, 2) / 255.
    t = te[b:c].astype(np.float64)
    for name, pred01 in (('ens', ens), ('ens_lbm', 0.5*ens + 0.5*lbm01), ('s0_lbm', 0.5*pa + 0.5*lbm01)):
        p = (np.ascontiguousarray(pred01.transpose(0, 2, 3, 1)) * 255.0).round().clip(0, 255)
        sq[name] += ((p - t) ** 2).sum(axis=(1, 2, 3))
    n += 1
mse = {k: v / (n * 256 * 256 * 3) for k, v in sq.items()}
line = (f"BLEND tukey224d: ensemble@8={mse['ens'][7]:.1f} | 0.5*ens+0.5*lbm@8={mse['ens_lbm'][7]:.1f} | "
        f"0.5*s0+0.5*lbm@8={mse['s0_lbm'][7]:.1f} | @16={mse['ens_lbm'][15]:.1f} @18={mse['ens_lbm'][17]:.1f} "
        f"(paper 448.76/453.04/454.18)")
print(line, flush=True)
open('/home/cc/verification_log.txt', 'a').write(line + '\n')
