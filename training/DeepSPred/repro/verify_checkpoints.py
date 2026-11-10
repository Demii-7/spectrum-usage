"""Verify every saved checkpoint against the results documented in REPRODUCTION.md.

Loads each checkpoints/<sha>.pt, rebuilds the model via models.DeepSPred.SwinSTB3D, runs
inference over the run's test split, computes the same frame-wise metrics as
run_paper_faithful.frame_metrics (batched where the original loops per image; the two are
verified numerically identical on a sample), and rebuilds repro/results.json in the
run_paper_faithful record schema so the runner resumes ("already done, skipping").

Fields the saved checkpoints never carried (per-epoch history, best_val_loss, train_seconds)
are recorded as null; every number quoted in RESULTS_FM_REGIME.md is recomputed and compared.

Run:  python verify_checkpoints.py [--only <sha12>] [--device cuda] [--no-lpips]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from models.DeepSPred import SwinSTB3D
from fm_data import load_frames, chronological_split
from metrics import SPACES, mse_in_space, psnr_from_mse, ssim as ssim_orig, lpips_available
from run_paper_faithful import (build, to_model, windows, H, W, PRIMARY_SPACE, run_key,
                                WEIGHT_DECAY, BATCH_SIZE)

DATA = REPO / 'data' / 'datasets' / 'deepspred'
CKPT_DIR = REPO / 'checkpoints' / 'repro' / 'deepspred'
RESULTS = HERE / 'results.json'

# Documented cells from RESULTS_FM_REGIME.md (audit 2026-11-03) keyed by checkpoint stem.
EXPECTED = {
    '954d763a44c4': dict(mse8=845.9, mse16=845.3, mse18=846.5, pers_mse8=1614.7,
                         lbm_mse8=857.6, ssim18=0.4713, lpips8=0.2522, gssim18=0.8039),
    'f6426d84dd5d': dict(mse8=266.6, mse16=266.4, mse18=267.8, pers_mse8=464.9,
                         lbm_mse8=252.0, ssim18=0.5827, lpips8=0.2853),
    'a9486d198417': dict(mse8=259.6),
    '172d4271c596': dict(mse8=259.4),
    'b7787a09b65b': dict(mse8=261.3),
    '1f25298f21b9': dict(mse8=142.2),
    '52749e3ffdd0': {},
    '46051ae4197d': {},
    'f79c9f671e00': {},
    '949a9e195592': dict(mse12=205.4, pers_mse12=375.3, lbm_mse12=203.0),
    'e51cac94d6d0': dict(mse12=201.4),
    '142297339c71': dict(avg_mse16=347.9),
    '6545adec4af7': dict(avg_mse16=347.3),
}


def ssim_batched(pred, true, max_i=255.0, sigma=1.5, k1=0.01, k2=0.03):
    """Identical to metrics.ssim (Wang et al., per channel then averaged), batched over
    the leading axis: gaussian_filter with sigma 0 along it filters each image
    independently, and mean-of-per-image-means == global mean for equal-size images."""
    c1, c2 = (k1 * max_i) ** 2, (k2 * max_i) ** 2
    sig = (0, sigma, sigma)
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    per_channel = []
    for ch in range(p.shape[-1]):
        x, y = p[..., ch], t[..., ch]
        mx, my = gaussian_filter(x, sig), gaussian_filter(y, sig)
        mxx = gaussian_filter(x * x, sig) - mx * mx
        myy = gaussian_filter(y * y, sig) - my * my
        mxy = gaussian_filter(x * y, sig) - mx * my
        s = (2 * mx * my + c1) * (2 * mxy + c2) / ((mx ** 2 + my ** 2 + c1) * (mxx + myy + c2))
        per_channel.append(float(s.mean()))
    return float(np.mean(per_channel))


def global_ssim(pred, true, max_i=255.0, k1=0.01, k2=0.03):
    """Eq. (32) LITERAL: one mean/variance/covariance per image (no sliding window),
    per channel then averaged over images and channels, mirroring metrics.ssim's structure."""
    c1, c2 = (k1 * max_i) ** 2, (k2 * max_i) ** 2
    out = []
    for p, t in zip(pred.astype(np.float64), true.astype(np.float64)):
        per_channel = []
        for ch in range(p.shape[-1]):
            x, y = p[..., ch], t[..., ch]
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cxy = ((x - mx) * (y - my)).mean()
            s = (2 * mx * my + c1) * (2 * cxy + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
            per_channel.append(float(s))
        out.append(sum(per_channel) / len(per_channel))
    return float(np.mean(out))


def frame_metrics_fast(pred_u8, true_u8, k, want_lpips, lpips_net=None, dev=None):
    """Same output as run_paper_faithful.frame_metrics, batched SSIM."""
    out = {'mse': {sp: [] for sp in SPACES}, 'psnr_max255': [], 'ssim': [], 'lpips': None}
    for j in range(k):
        p, t = pred_u8[:, j], true_u8[:, j]
        for sp in SPACES:
            out['mse'][sp].append(mse_in_space(p, t, sp))
        out['psnr_max255'].append(psnr_from_mse(out['mse'][PRIMARY_SPACE][-1], 255.0))
        out['ssim'].append(ssim_batched(p, t, max_i=255.0))
    if want_lpips:
        ok, note = lpips_available()
        out['lpips_note'] = note
        if ok and lpips_net is not None:
            vals = []
            for j in range(k):
                a = torch.from_numpy(pred_u8[:, j].astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).to(dev)
                b = torch.from_numpy(true_u8[:, j].astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).to(dev)
                # chunked to bound GPU memory; the mean over per-image distances is
                # identical to the original single net(a, b).mean() call
                d = []
                with torch.no_grad():
                    for lo in range(0, a.shape[0], 256):
                        d.append(lpips_net(a[lo:lo + 256], b[lo:lo + 256]).reshape(-1))
                vals.append(float(torch.cat(d).mean()))
            out['lpips'] = vals
    return out


def trivial_references_fast(test, t, k, segs, want_lpips=False, lpips_net=None, dev=None):
    """Same output as run_paper_faithful.trivial_references, batched SSIM."""
    idx = windows(test, t, k, segs)
    true = np.stack([test[b:c] for _, b, c in idx])
    pers = np.stack([np.repeat(test[b - 1][None], k, axis=0) for _, b, _ in idx])
    lbm = np.stack([np.repeat(test[a:b].mean(axis=0, keepdims=True).round().clip(0, 255).astype(np.uint8), k, axis=0) for a, b, _ in idx])
    return {
        'persistence': frame_metrics_fast(pers, true, k, want_lpips, lpips_net, dev),
        'lookback_mean': frame_metrics_fast(lbm, true, k, want_lpips, lpips_net, dev),
        'n_windows': len(idx),
        'seasonal_naive': f'''omitted deliberately — a 3 h capture at 1 s with a {t}-frame lookback exposes no period (DEEPSPRED_REFERENCE section 7)'''}


def parse_run_key(rk):
    """Handle every run-key generation the runner went through: the rendering marker
    ('raw'/'jpegNNN') was added after the first runs, so raw keys have one field fewer;
    transfer-learning keys carry an extra 'tl' field [RESULTS_FM_REGIME section 1]."""
    parts = rk.split('|')
    tl = 'tl' in parts
    parts = [p for p in parts if p != 'tl']
    ds, K, seed, C, d, lr, p, w = parts[:8]
    rest = parts[8:]
    space = rest[-1]
    rend = rest[0] if len(rest) > 1 else 'raw'
    return dict(dataset=ds, K=int(K[1:]), seed=int(seed[1:]), C=int(C[1:]),
                jpeg=int(rend[4:]) if rend.startswith('jpeg') else None,
                gray=False, tl=tl)


def evaluate(ck_path, dev, lpips_net, want_lpips=True):
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    rk, a, r = ck['run_key'], ck['arch'], ck['regime']
    info = parse_run_key(rk)
    T = K = r['K']
    channels = ck['model_state_dict']['patch_embed.proj.weight'].shape[1]
    info['gray'] = channels == 1
    print(f"\n{'='*100}\n{ck_path.name}  {rk}\n  arch={a} regime={r}", flush=True)
    t0 = time.perf_counter()

    ds_dir = {'fm': 'FM', 'crossval': 'Cross-Validation', 'lte': 'LTE'}[info['dataset']]
    frames, fmeta = load_frames(DATA / f"{ds_dir}-Spectrum-Dataset",
                                expect_hw=(H, W), jpeg_quality=info['jpeg'], channels=channels)
    (train, val, test), segs = chronological_split(frames, fmeta)
    test = np.asarray(test)
    te_idx = windows(test, T, K, segs['test'])
    tr_idx = windows(np.asarray(train), T, K, segs['train'])
    va_idx = windows(np.asarray(val), T, K, segs['val'])

    model = build(a['embed_dim'], a['depths'], a['heads'], a['patch'], a['window'], T, channels)
    model.load_state_dict(ck['model_state_dict'])
    model.eval().to(dev)
    preds = []
    with torch.no_grad():
        for (i, b, c) in te_idx:
            preds.append(model(to_model(test[i:b][None]).to(dev)).cpu().numpy())
    pred_u8 = (np.concatenate(preds).transpose(0, 1, 3, 4, 2) * 255.0).round().clip(0, 255).astype(np.uint8)
    true_u8 = np.stack([test[b:c] for (_, b, c) in te_idx])
    print(f"  {len(te_idx)} test windows, inference done in {time.perf_counter()-t0:.0f}s", flush=True)

    do_lpips = want_lpips and not info['gray'] and lpips_net is not None
    t1 = time.perf_counter()
    model_m = frame_metrics_fast(pred_u8, true_u8, K, do_lpips, lpips_net, dev)
    print(f"  model metrics in {time.perf_counter()-t1:.0f}s", flush=True)
    t2 = time.perf_counter()
    triv = trivial_references_fast(test, T, K, segs['test'])
    print(f"  trivial references in {time.perf_counter()-t2:.0f}s", flush=True)

    m = model_m['mse'][PRIMARY_SPACE]
    got = dict(mse8=m[7] if K >= 8 else None, mse12=m[11] if K >= 12 else None,
               mse16=m[15] if K >= 16 else None, mse18=m[17] if K >= 18 else None,
               avg_mse16=float(np.mean(m[:16])) if K >= 16 else None,
               pers_mse8=triv['persistence']['mse'][PRIMARY_SPACE][7] if K >= 8 else None,
               pers_mse12=triv['persistence']['mse'][PRIMARY_SPACE][11] if K >= 12 else None,
               lbm_mse8=triv['lookback_mean']['mse'][PRIMARY_SPACE][7] if K >= 8 else None,
               lbm_mse12=triv['lookback_mean']['mse'][PRIMARY_SPACE][11] if K >= 12 else None,
               ssim18=model_m['ssim'][17] if K >= 18 else None,
               lpips8=model_m['lpips'][7] if model_m.get('lpips') and K >= 8 else None)
    if K >= 18 and info['dataset'] == 'fm':
        got['gssim18'] = global_ssim(pred_u8[:, 17], true_u8[:, 17])
    exp = EXPECTED.get(ck_path.stem, {})
    verdicts = []
    for name, e in exp.items():
        g = got.get(name)
        if g is None:
            verdicts.append((name, e, None, 'MISSING'))
            continue
        rel = abs(g - e) / max(abs(e), 1e-9)
        verdicts.append((name, e, g, 'MATCH' if rel <= 0.01 else ('CLOSE' if rel <= 0.05 else 'DIFFER')))
    print("  documented-cell comparison:")
    for name, e, g, v in verdicts:
        gs = f"{g:.4f}" if isinstance(g, float) else str(g)
        print(f"    {name:>12}: documented {e:>10} | recomputed {gs:>10} | {v}")
    if not exp:
        print("    (no cell quoted in RESULTS_FM_REGIME.md; record rebuilt from checkpoint)")

    record = {
        'run_key': rk,
        'dataset': info['dataset'],
        'horizon_K': K,
        'lookback_T': T,
        'seed': info['seed'],
        'metric_space_primary': PRIMARY_SPACE,
        'max_i': 255.0,
        'arch': {'embed_dim': a['embed_dim'], 'depths': a['depths'], 'heads': a['heads'],
                 'patch': a['patch'], 'window': a['window']},
        'n_params': ck['n_params'],
        'epochs_run': r['epochs_run'],
        'best_val_loss': None,
        'train_seconds': None,
        'device': str(dev),
        'torch': torch.__version__,
        'history': [],
        'init_from': ({'checkpoint': None,
                       'run_key': None,
                       'sha256_note': 'transfer-learning source inferred: FM jpeg95 seed-0 checkpoint f6426d84dd5d (RESULTS_FM_REGIME section 3); the saved checkpoint does not record it'}
                      if info['tl'] else None),
        'dataset_meta': {'source': fmeta['source'], 'n_files': fmeta['n_files'],
                         'first': fmeta['first'], 'last': fmeta['last'],
                         'n_segments': len(fmeta['segments']),
                         'split_sizes': {'train': len(train), 'val': len(val), 'test': len(test)},
                         'windows': {'train': len(tr_idx), 'val': len(va_idx), 'test': len(te_idx)},
                         'db_range': fmeta['db_range'], 'switches': fmeta['switches']},
        **{
            'model': model_m,
            'trivial_references': triv,
            'checkpoint': {'path': str(ck_path.relative_to(REPO)),
                           'sha256': hashlib.sha256(ck_path.read_bytes()).hexdigest()},
            'paper_text_audited': 'published IEEE TWC 24(1):509-525 (numbers identical to arXiv:2408.06870v3 in every quoted cell; the MSE/PSNR inconsistency and undefined lambda/w persist in print)'}}
    record['_recovered'] = ('record rebuilt from its checkpoint by verify_checkpoints.py after '
                            'results.json was lost; history/best_val_loss/train_seconds were '
                            'never stored in checkpoints and are null')
    return record, got, verdicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='evaluate only this checkpoint stem (12 hex)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-lpips', action='store_true')
    ap.add_argument('--record-out', default=None, help='write each record to this JSON file instead of results.json (for parallel shards)')
    args = ap.parse_args()
    dev = torch.device(args.device)
    lpips_net = None
    if not args.no_lpips:
        ok, note = lpips_available()
        print(f"lpips: {'ok' if ok else note}")
        if ok:
            import lpips as _l
            lpips_net = _l.LPIPS(net='alex').to(dev).eval()

    # sanity: batched SSIM must equal the original per-image SSIM
    rng = np.random.RandomState(0)
    pe = rng.randint(0, 256, (6, 64, 64, 3)).astype(np.uint8)
    tr = np.clip(pe.astype(int) + rng.randint(-8, 9, pe.shape), 0, 255).astype(np.uint8)
    d = abs(ssim_batched(pe, tr, max_i=255.0) - ssim_orig(pe, tr, max_i=255.0))
    print(f"batched-vs-original SSIM check: |diff| = {d:.2e}")
    assert d < 1e-9, 'batched SSIM is not equivalent'

    records = {} if args.record_out else (json.loads(RESULTS.read_text()) if RESULTS.exists() else {})
    all_verdicts = {}
    for ck_path in sorted(CKPT_DIR.glob('*.pt')):
        if args.only and ck_path.stem != args.only:
            continue
        record, got, verdicts = evaluate(ck_path, dev, lpips_net, want_lpips=not args.no_lpips)
        if args.record_out:
            Path(args.record_out).write_text(json.dumps(record, indent=2))
            print(f"  record -> {args.record_out}")
        else:
            records[record['run_key']] = record
            RESULTS.write_text(json.dumps(records, indent=2))
        all_verdicts[ck_path.stem] = verdicts
    if not args.record_out:
        print(f"\nwrote {RESULTS} ({len([k for k in records if not k.startswith('_')])} run records)")
    n_bad = sum(1 for v in all_verdicts.values() for _, _, _, s in v if s == 'DIFFER')
    print(f"documented cells checked: {sum(len(v) for v in all_verdicts.values())}, DIFFER: {n_bad}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
