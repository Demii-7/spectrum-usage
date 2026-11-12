"""Fast streaming record regeneration for runs whose runner process was killed after the
checkpoint was saved. Reproduces the run_paper_faithful record schema with EXACT numbers by
calling the original metrics functions on window-chunks and combining with weights (every
metric is a mean over the window axis). Peak RAM ~8 GB instead of ~75 GB.

Usage: regen_fast.py <sha12> <out.json>"""
import sys, json, hashlib, re
import numpy as np, torch

from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[3])
DSDIR = {'fm': 'FM-Spectrum-Dataset', 'lte': 'LTE-Spectrum-Dataset',
        'crossval': 'Cross-Validation-Spectrum-Dataset'}
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/training/DeepSPred/repro')
from run_paper_faithful import build, windows, PRIMARY_SPACE
from fm_data import load_frames_spec, chronological_split
from metrics import SPACES, mse_in_space, psnr_from_mse, ssim

sha12, out = sys.argv[1], sys.argv[2]
ck_path = f'{REPO}/checkpoints/repro/deepspred/{sha12}.pt'
# pick the freest GPU so re-scoring never collides with trainers on a busy device
_free = [torch.cuda.mem_get_info(i)[0] if torch.cuda.is_available() else 0
         for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)]
DEV = f'cuda:{_free.index(max(_free))}' if _free else 'cpu'
ck = torch.load(ck_path, map_location=DEV)
key = ck['run_key']
spec = re.search(r'\|spec=([^|]+)\|(\w+)', key).group(1)
scope = re.search(r'\|spec=([^|]+)\|(\w+)', key).group(2)
seed = int(re.search(r'\|s(\d+)\|', key).group(1))
arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']

model = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T)
if ck.get('head') == 'residual':
    from run_paper_faithful import ResidualHead
    model = ResidualHead(model)
ISCALE = 1.0 if ck.get('input_scale') == 'raw' else 1 / 255.
model.load_state_dict(ck['model_state_dict']); model.to(DEV).eval()

frames, meta = load_frames_spec(REPO + '/data/datasets/deepspred/' + DSDIR[re.search(r'^(\w+)\|', key).group(1)], spec, range_scope=scope)
(tr, va, te), segs = chronological_split(frames, meta)
idx = windows(te, T, K, segs['test'])

CH = 1  # batch-1 inference: bit-identical predictions to the runner (chunked flips round() boundaries)
acc = {name: {sp: np.zeros(K) for sp in SPACES} for name in ('model', 'persistence', 'lookback_mean')}
n = 0
for i in range(0, len(idx), CH):
    sub = idx[i:i+CH]
    x = torch.from_numpy(np.stack([te[a:b] for a, b, _ in sub]).astype(np.float32) * ISCALE).permute(0, 1, 4, 2, 3).to(DEV)
    with torch.no_grad():
        pred = (model(x).cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.0).round().clip(0, 255).astype(np.uint8)
    true = np.stack([te[b:c] for _, b, c in sub])
    pers = np.stack([np.repeat(te[b-1][None], K, axis=0) for _, b, _ in sub])
    lbm = np.stack([np.repeat(te[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), K, axis=0)
                    for a, b, _ in sub])
    for j in range(K):
        for name, arr in (('model', pred), ('persistence', pers), ('lookback_mean', lbm)):
            p, t = arr[:, j], true[:, j]
            for sp in SPACES:
                acc[name][sp][j] += mse_in_space(p, t, sp) * len(sub)
    n += len(sub)

def fm(name):
    mse = {sp: (acc[name][sp] / n).tolist() for sp in SPACES}
    return {'mse': mse,
            'psnr_max255': [psnr_from_mse(m, 255.0) for m in mse[PRIMARY_SPACE]],
            'ssim': None,
            'ssim_note': 'skipped per campaign decision 2026-11-11 (MSE-only records; SSIM '
                         'recomputable from the checkpoint with verify_checkpoints.py if needed)'}

record = {
    'run_key': key, 'dataset': 'fm', 'horizon_K': K, 'lookback_T': T, 'seed': seed,
    'metric_space_primary': PRIMARY_SPACE, 'max_i': 255.0, 'arch': arch, 'n_params': ck['n_params'],
    'epochs_run': ck['regime']['epochs_run'], 'best_val_loss': None, 'train_seconds': None,
    'device': f'{DEV} (re-score)', 'torch': torch.__version__, 'history': None, 'init_from': None,
    'dataset_meta': {'source': meta['source'], 'n_files': meta['n_files'], 'first': meta['first'],
                     'last': meta['last'], 'n_segments': len(meta['segments']),
                     'split_sizes': {'train': len(tr), 'val': len(va), 'test': len(te)},
                     'windows': {'train': len(windows(tr, T, K, segs['train'])),
                                 'val': len(windows(va, T, K, segs['val'])), 'test': len(idx)},
                     'db_range': meta['db_range'], 'switches': meta['switches']},
    'model': fm('model'),
    'trivial_references': {'persistence': fm('persistence'), 'lookback_mean': fm('lookback_mean'),
                           'n_windows': n,
                           'seasonal_naive': 'omitted deliberately — a 3 h capture at 1 s with a '
                                            f'{T}-frame lookback exposes no period (DEEPSPRED_REFERENCE section 7)'},
    'checkpoint': {'path': f'checkpoints/repro/deepspred/{sha12}.pt',
                   'sha256': hashlib.sha256(open(ck_path, 'rb').read()).hexdigest()},
    '_recovered': True,
    'recovery_note': ('runner process was terminated after its checkpoint was saved (OOM-avoidance '
                      'policy / kernel OOM during the record phase); record regenerated with the '
                      'streaming re-scorer (identical metric functions, chunk-weighted); per-epoch '
                      'history, best_val_loss and train_seconds were never persisted'),
}
json.dump({key: record}, open(out, 'w'), indent=2)
m = record['model']['mse'][PRIMARY_SPACE]
print(f"REGEN-FAST {sha12}: MSE@8={m[7]:.2f} @16={m[15]:.2f} @18={m[17]:.2f} windows={n}", flush=True)
