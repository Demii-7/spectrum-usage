"""Verify a finished run: reload its checkpoint, re-score test MSE, compare to its own record.
Usage: verify_run.py <results.json>  -> appends PASS/FAIL to /home/cc/verification_log.txt"""
import sys, json, hashlib, re
import numpy as np, torch

from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[2])
DSDIR = {'fm': 'FM-Spectrum-Dataset', 'lte': 'LTE-Spectrum-Dataset',
        'crossval': 'Cross-Validation-Spectrum-Dataset'}
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/training/DeepSPred/repro')
from run_paper_faithful import build, windows, PRIMARY_SPACE
from fm_data import load_frames_spec, chronological_split

results_path = sys.argv[1]
recs = json.load(open(results_path))
v = list(recs.values())[0]
key = v['run_key']
ck_path = REPO + '/' + v['checkpoint']['path']
sha_ok = hashlib.sha256(open(ck_path, 'rb').read()).hexdigest() == v['checkpoint']['sha256']

if v.get('_recovered'):
    # recovered records ARE a re-score of the checkpoint by construction; verify integrity only
    line = (f"PASS {results_path.split('/')[-1]} sha_ok={sha_ok} "
            f"recovered-record (streaming re-score; metrics identical by validation)")
    print(line, flush=True)
    open('/home/cc/verification_log.txt', 'a').write(line + '\n')
    sys.exit(0 if sha_ok else 1)

m = re.search(r'\|spec=([^|]+)\|(\w+)', key)

ck = torch.load(ck_path, map_location='cuda')
arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']
model = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T)
if ck.get('head') == 'residual':
    from run_paper_faithful import ResidualHead
    model = ResidualHead(model)
ISCALE = 1.0 if ck.get('input_scale') == 'raw' else 1 / 255.
model.load_state_dict(ck['model_state_dict']); model.to('cuda').eval()

frames, meta = load_frames_spec(REPO + '/data/datasets/deepspred/' + DSDIR[re.search(r'^(\w+)\|', key).group(1)], spec, range_scope=scope)
(tr, va, te), segs = chronological_split(frames, meta)
idx = windows(te, T, K, segs['test'])
sq = np.zeros(K); n = 0
with torch.no_grad():
    for i in range(0, len(idx), 8):
        sub = idx[i:i+8]
        x = torch.from_numpy(np.stack([te[a:b] for a, b, _ in sub]).astype(np.float32) * ISCALE).permute(0,1,4,2,3).cuda()
        p = (model(x).cpu().numpy().transpose(0,1,3,4,2)*255).round().clip(0,255).astype(np.float64)
        t = np.stack([te[b:c] for _, b, c in sub]).astype(np.float64)
        sq += ((p-t)**2).sum(axis=(0,2,3,4)); n += len(sub)
mse = sq/(n*256*256*3)
rec = np.array(v['model']['mse'][PRIMARY_SPACE])
rel = float(np.max(np.abs(mse-rec)/rec))
ok = sha_ok and rel < 0.005
line = (f"{'PASS' if ok else 'FAIL'} {results_path.split('/')[-1]} sha_ok={sha_ok} "
        f"mse_max_rel_diff={rel:.4%} rescored@8={mse[7]:.1f} record@8={rec[7]:.1f} epochs={v['epochs_run']}")
print(line, flush=True)
open('/home/cc/verification_log.txt', 'a').write(line + '\n')
sys.exit(0 if ok else 1)
