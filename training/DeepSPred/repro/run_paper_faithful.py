'''Reproduce Pan et al.'s FM-regime results for 3D-SwinSTB.

Regime constants below are properties of the paper, not knobs (REPRODUCTION_VS_BENCHMARK.md section 3).
What varies across the paper's OWN runs is exposed on the command line: dataset, horizon (K = T),
seed, and the four Table II / Table III ablation axes.

Conventions (section 5): resumable by run key; metric space recorded in every row; a checkpoint with
config + normalisation statistics + sha256 per run; trivial references carried in the same row;
per-epoch history kept in the record.

Run:
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --embed-dim 48      # Table II
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --depths 2 6 2      # Table II
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --lr 0.0001         # Table III
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --patch 4 4 4 --window 4 7 7
  python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --render-spec tukey224d --range-scope full   # alternative rendering (see probe/COLLABORATOR_RUN.md)

Rendering switch: --jpeg-quality Q round-trips every frame through JPEG at quality Q before use.
The paper never states how its "spectrograms" were stored; a JPEG pass is the only reading under
which the paper's MSE panel (~449) sits ABOVE the data's irreducible noise floor (measured
sigma^2 ~ 780 raw vs ~210 after JPEG), so it is carried as an explicit, recorded hypothesis and
kept out of the raw run keys. See DEEPSPRED_REFERENCE.md section 4.14.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from models.DeepSPred import SwinSTB3D
from fm_data import load_frames, load_frames_spec, chronological_split
from metrics import SPACES, mse_in_space, psnr_from_mse, ssim, lpips_available

H = W = 256
CHANNELS = 3
HORIZONS = (8, 16, 20)
EPOCHS = 20
BATCH_SIZE = 1
PATIENCE = 4
THETA_PER = 0.01
N_EP = 4
OPTIMIZER = 'adamw'
PRIMARY_SPACE = 'uint8_mean_per_channel'
WEIGHT_DECAY = 0.05
DROP_PATH = 0.0


def run_key(ds, horizon, seed, embed_dim, depths, lr, patch, window, space,
            jpeg_quality=None, init_from=None) -> str:
    tl = '|tl' if init_from else ''
    return f'''{ds}|K{horizon}|s{seed}|C{embed_dim}|d{'-'.join(map(str, depths))}|lr{lr}|p{'-'.join(map(str, patch))}|w{'-'.join(map(str, window))}|{'jpeg' + str(jpeg_quality) if jpeg_quality else 'raw'}{tl}|{space}'''


def build(embed_dim, depths, heads, patch, window, horizon, channels=3) -> SwinSTB3D:
    return SwinSTB3D({
        'model': {
            'embed_dim': embed_dim,
            'depths': list(depths),
            'num_heads': list(heads),
            'patch_size': list(patch),
            'window_size': list(window),
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': DROP_PATH,
            'in_chans': channels,
            'out_chans': channels,
            'head': 'paper'},
        'frames': {
            'minutes_per_frame': H,
            'w_pad': W,
            'w_orig': W},
        'windowing': {
            'input_frames': horizon,
            'output_frames': horizon}})


def windows(frames: np.ndarray, t: int, k: int, segs=None):
    '''(x_start, input_end, target_end) triples; stride 1.

    A window is kept only if it lies wholly inside one contiguous segment of the
    release (the FM capture is missing 7 seconds inside its span, the cross-val
    set 234), and by construction never straddles a split boundary.
    '''
    idx = [(s, s + t, s + t + k) for s in range(0, len(frames) - (t + k) + 1)]
    if segs:
        idx = [(a, b, c) for (a, b, c) in idx if any(a >= s and c <= e for s, e in segs)]
    return idx


def to_model(batch: np.ndarray) -> torch.Tensor:
    '''(n, T, H, W, 3) uint8 -> (n, T, 3, H, W) float in [0,1].'''
    return torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3).contiguous()


def frame_metrics(pred_u8: np.ndarray, true_u8: np.ndarray, k: int, want_lpips: bool) -> dict:
    '''Frame-wise metrics in every candidate space; the paper plots these against frame index.'''
    out = {'mse': {sp: [] for sp in SPACES}, 'psnr_max255': [], 'ssim': [], 'lpips': None}
    for j in range(k):
        p, t = pred_u8[:, j], true_u8[:, j]
        for sp in SPACES:
            out['mse'][sp].append(mse_in_space(p, t, sp))
        out['psnr_max255'].append(psnr_from_mse(out['mse'][PRIMARY_SPACE][-1], 255.0))
        out['ssim'].append(ssim(p, t, max_i=255.0))
    if want_lpips:
        ok, note = lpips_available()
        out['lpips_note'] = note
        if ok:
            import lpips as _l
            net = _l.LPIPS(net='alex')
            vals = []
            for j in range(k):
                a = torch.from_numpy(pred_u8[:, j].astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)
                b = torch.from_numpy(true_u8[:, j].astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)
                with torch.no_grad():
                    vals.append(float(net(a, b).mean()))
            out['lpips'] = vals
    return out


def trivial_references(test: np.ndarray, t: int, k: int, segs=None) -> dict:
    '''Persistence and lookback mean on the same windows. The paper reports neither.'''
    idx = windows(test, t, k, segs)
    true = np.stack([test[b:c] for _, b, c in idx])
    pers = np.stack([np.repeat(test[b - 1][None], k, axis=0) for _, b, _ in idx])
    lbm = np.stack([np.repeat(test[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), k, axis=0) for a, b, _ in idx])
    return {
        'persistence': frame_metrics(pers, true, k, want_lpips=False),
        'lookback_mean': frame_metrics(lbm, true, k, want_lpips=False),
        'n_windows': len(idx),
        'seasonal_naive': f'''omitted deliberately — a 3 h capture at 1 s with a {t}-frame lookback exposes no period (DEEPSPRED_REFERENCE section 7)'''}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--dataset', default='fm', choices=['fm', 'lte', 'crossval'])
    ap.add_argument('--horizon', type=int, default=20, choices=HORIZONS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--embed-dim', type=int, default=96)
    ap.add_argument('--depths', type=int, nargs=3, default=[2, 4, 2])
    ap.add_argument('--heads', type=int, nargs=3, default=[4, 8, 16])
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--patch', type=int, nargs=3, default=[2, 4, 4])
    ap.add_argument('--window', type=int, nargs=3, default=[2, 7, 7])
    ap.add_argument('--jpeg-quality', type=int, default=None, help='rendering hypothesis: round-trip frames through JPEG at this quality')
    ap.add_argument('--render-spec', default=None, help="alternative rendering from probe/rendering_probe.py, e.g. rect126, rect127c~12, tukey224@54-218, tukey224d (default: the original Hann hop-126 rendering)")
    ap.add_argument('--range-scope', default='full', choices=['full', 'train'], help="dB range for the min-max scaling with --render-spec: all frames, or the first 2/3 of the capture")
    ap.add_argument('--gray', action='store_true', help="Appendix-A grayscale ablation: 1-channel 'gray'-colormap frames, model in/out channels 1 (pass --embed-dim 32 per the paper's one-third-capacity note). LPIPS is skipped (the paper reports only MSE/PSNR for this ablation).")
    ap.add_argument('--init-from', type=Path, default=None, help='Transfer learning (paper Algorithm 4): fine-tune from this checkpoint instead of training from scratch. Recorded in the run record.')
    ap.add_argument('--lpips', action='store_true', help='compute LPIPS if the package is present')
    ap.add_argument('--results', type=Path, default=HERE / 'results.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    key = run_key(args.dataset, args.horizon, args.seed, args.embed_dim, args.depths, args.lr, args.patch, args.window, PRIMARY_SPACE, args.jpeg_quality, args.init_from)
    if args.render_spec:  # the rendering is part of the run identity (results key and checkpoint hash)
        key += f'|spec={args.render_spec}|{args.range_scope}'
    records = json.loads(args.results.read_text()) if args.results.exists() else {}
    if key in records:
        print(f'''already done, skipping: {key}''')
        return 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    channels = 1 if args.gray else 3
    if args.render_spec:
        frames, fmeta = load_frames_spec(args.data, args.render_spec, range_scope=args.range_scope)
    else:
        frames, fmeta = load_frames(args.data, expect_hw=(H, W), jpeg_quality=args.jpeg_quality, channels=channels)
    (train, val, test), segs = chronological_split(frames, fmeta)
    T = K = args.horizon
    dev = torch.device(args.device)
    model = build(args.embed_dim, args.depths, args.heads, args.patch, args.window, T, channels).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    init_note = None
    if args.init_from is not None:
        state = torch.load(args.init_from, map_location=dev)
        model.load_state_dict(state['model_state_dict'])
        init_note = {'checkpoint': str(args.init_from), 'run_key': state.get('run_key'), 'sha256_note': 'source run recorded in its own results file'}
        print(f'''  init from {args.init_from} (run_key {state.get('run_key')})''', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    lossf = nn.MSELoss()
    tr_idx, va_idx = windows(train, T, K, segs['train']), windows(val, T, K, segs['val'])
    print(f'''{key}\n  params {n_par:,} | train windows {len(tr_idx)} | val {len(va_idx)} | test {len(windows(test, T, K, segs['test']))}''', flush=True)
    history, best, best_state, stall = [], float('inf'), None, 0
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        order = np.random.permutation(len(tr_idx))
        for (a, b, cc) in (tr_idx[i] for i in order):
            x, y = to_model(train[a:b][None]).to(dev), to_model(train[b:cc][None]).to(dev)
            opt.zero_grad()
            l = lossf(model(x), y)
            l.backward()
            opt.step()
            tot += float(l.detach())
        tr_loss = tot / max(len(tr_idx), 1)
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for (a, b, cc) in va_idx:
                x, y = to_model(val[a:b][None]).to(dev), to_model(val[b:cc][None]).to(dev)
                vtot += float(lossf(model(x), y))
        va_loss = vtot / max(len(va_idx), 1)
        history.append({'epoch': ep, 'train_loss': tr_loss, 'val_loss': va_loss})
        print(f'''  epoch {ep:02d} train {tr_loss:.6f} val {va_loss:.6f}''', flush=True)
        rel = (best - va_loss) / max(abs(best), 1e-12) * 100.0 if best < float('inf') else 1e+09
        if va_loss < best:
            best, best_state = va_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stall = stall + 1 if rel < THETA_PER else 0
        if stall >= N_EP or stall >= PATIENCE:
            print(f'''  early stop at epoch {ep} (criterion: <{THETA_PER}% over {N_EP} epochs)''')
            break
    train_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    te_idx = windows(test, T, K, segs['test'])
    model.eval()
    preds = []
    with torch.no_grad():
        for (a, b, _) in te_idx:
            preds.append(model(to_model(test[a:b][None]).to(dev)).cpu().numpy())
    pred_u8 = (np.concatenate(preds).transpose(0, 1, 3, 4, 2) * 255.0).round().clip(0, 255).astype(np.uint8)
    true_u8 = np.stack([test[b:cc] for (_, b, cc) in te_idx])
    ck_dir = REPO / 'checkpoints' / 'repro' / 'deepspred'
    ck_dir.mkdir(exist_ok=True)
    ck_path = ck_dir / f'''{hashlib.sha1(key.encode()).hexdigest()[:12]}.pt'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'run_key': key,
        'n_params': n_par,
        'normalization': {'scheme': 'uint8/255 -> [0,1]; sigmoid head; x255 to score', 'max_i': 255.0},
        'regime': {'H': H, 'W': W, 'T': T, 'K': K, 'epochs_run': len(history), 'lr': args.lr, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE},
        'arch': {'embed_dim': args.embed_dim, 'depths': args.depths, 'heads': args.heads, 'patch': args.patch, 'window': args.window}}, ck_path)
    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    records[key] = {
        'run_key': key,
        'dataset': args.dataset,
        'horizon_K': K,
        'lookback_T': T,
        'seed': args.seed,
        'metric_space_primary': PRIMARY_SPACE,
        'max_i': 255.0,
        'arch': {'embed_dim': args.embed_dim, 'depths': args.depths, 'heads': args.heads, 'patch': args.patch, 'window': args.window},
        'n_params': n_par,
        'epochs_run': len(history),
        'best_val_loss': best,
        'train_seconds': round(train_s, 1),
        'device': str(dev),
        'torch': torch.__version__,
        'history': history,
        'init_from': init_note,
        'dataset_meta': {'source': fmeta['source'], 'n_files': fmeta['n_files'], 'first': fmeta['first'], 'last': fmeta['last'],
                         'n_segments': len(fmeta['segments']),
                         'split_sizes': {'train': len(train), 'val': len(val), 'test': len(test)},
                         'windows': {'train': len(tr_idx), 'val': len(va_idx), 'test': len(te_idx)},
                         'db_range': fmeta['db_range'], 'switches': fmeta['switches']},
        **{
            'model': frame_metrics(pred_u8, true_u8, K, want_lpips=args.lpips and not args.gray),
            'trivial_references': trivial_references(test, T, K, segs['test']),
            'checkpoint': {'path': str(ck_path.relative_to(REPO)), 'sha256': sha},
            'paper_text_audited': 'published IEEE TWC 24(1):509-525 (numbers identical to arXiv:2408.06870v3 in every quoted cell; the MSE/PSNR inconsistency and undefined lambda/w persist in print)'}}
    args.results.write_text(json.dumps(records, indent=2))
    m8 = records[key]['model']['mse'][PRIMARY_SPACE]
    print(f'''\n  MSE[{PRIMARY_SPACE}] frame 8 = {m8[7]:.4f}  (paper: 448.7628)''' if K >= 8 else '')
    print(f'''  wrote {args.results} and {ck_path}''')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
