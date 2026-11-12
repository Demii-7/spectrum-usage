"""Score a DeepSPred reproduction checkpoint in dB instead of colormap pixels.

The trained model predicts jet (or gray) colour frames. Because we render the frames ourselves from the authors' raw I/Q, the true dB
spectrogram behind every target frame is known. This script:

1. rebuilds the test block exactly as the runner did (``load_frames_spec`` with the checkpoint's rendering, ``chronological_split``,
   stride-1 windows inside contiguous segments) and, from the same files, the float dB spectrograms (``rendering_probe.spectrogram_db``);
2. runs the model at batch 1 (bit-identical to the runner) and checks its uint8 MSE against the ``results.json`` record;
3. inverts the predicted float colours to dB: gray is linear (exact); jet takes the nearest entry of a 4096-entry jet table, then
   dB = lo + v (hi - lo) with the rendering's dataset-wide dB range;
4. scores model and baselines against the true dB, per horizon k = 1..K: MAE, MSE, RMSE (dB), per pixel (one STFT row x one 122 kHz bin)
   and per second (rows averaged in linear power, i.e. the 256-bin spectrum of that second);
5. baselines computed natively in dB on the same windows: persistence (last input frame), lookback mean (of dB; and of linear power),
   lookback median, and a residual linear AR (lookback mean + ridge-fitted linear correction on the T lags, one coefficient set per
   horizon shared across pixels, fitted on the validation block, as in probe/acf_probe.residual_linear_curve);
6. inversion control: the lookback mean computed on the uint8 colour frames (also off the colormap curve, like a model's prediction),
   inverted the same way, vs the lookback mean computed in dB. The gap bounds what the back-out itself adds to the model's dB error.

Usage: dbspace_score.py <sha12> --data-root data/datasets/deepspred --out repro/dbspace/<sha12>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'probe'))

from fm_data import chronological_split, list_seconds, load_frames_spec, read_iq  # noqa: E402
from run_paper_faithful import build, windows  # noqa: E402

DSDIR = {'fm': 'FM-Spectrum-Dataset', 'lte': 'LTE-Spectrum-Dataset', 'crossval': 'Cross-Validation-Spectrum-Dataset'}
LUT_SIZE = 4096


def split_indices(meta):
    """The file indices of chronological_split's val and test blocks (same rule: by time, at 2/3 and 5/6 of the span)."""
    times = [datetime.fromisoformat(t) for t in meta['times']]
    t0, span = times[0], (times[-1] - times[0]).total_seconds()
    tr_cut, va_cut = t0 + timedelta(seconds=span * 2 / 3), t0 + timedelta(seconds=span * 5 / 6)
    return ([i for i, t in enumerate(times) if tr_cut <= t < va_cut], [i for i, t in enumerate(times) if t >= va_cut])


def _db_worker(args):
    path, spec = args
    import rendering_probe as rp
    x, _ = read_iq(path)
    return rp.spectrogram_db(x, spec).astype(np.float32)


def db_block(paths, spec, workers):
    with mp.get_context('fork').Pool(workers) as pool:
        return np.stack(pool.map(_db_worker, [(str(p), spec) for p in paths], chunksize=8))


class Inverter:
    """Float colours (..., C) in [0, 1] -> scalar v in [0, 1]."""

    def __init__(self, gray: bool, device):
        import matplotlib
        self.gray = gray
        self.lut = torch.from_numpy(matplotlib.colormaps['jet'].resampled(LUT_SIZE)(np.linspace(0, 1, LUT_SIZE))[:, :3]
                                    .astype(np.float32)).to(device)

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.gray:
            return rgb[..., 0]
        flat = rgb.reshape(-1, 3)
        idx = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        for s in range(0, flat.shape[0], 65536):
            idx[s:s + 65536] = torch.cdist(flat[s:s + 65536], self.lut).argmin(dim=1)
        return (idx.float() / (LUT_SIZE - 1)).view(rgb.shape[:-1])

    def off_curve(self, rgb: torch.Tensor) -> float:
        if self.gray:
            return 0.0
        flat = rgb.reshape(-1, 3)[::97]
        return float(torch.cdist(flat, self.lut).min(dim=1).values.mean())


def fit_residual_ar(D: np.ndarray, segs, T: int, K: int, n_pix: int = 6000, lam: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Coefficients A (K, T): pred(t+k) = m + sum_j A[k, j] (x[t-j] - m), m = lookback mean, fitted on block D (dB) over a pixel subset."""
    rng = np.random.default_rng(seed)
    flat = D.reshape(len(D), -1)
    pix = rng.choice(flat.shape[1], size=min(n_pix, flat.shape[1]), replace=False)
    X = flat[:, pix].astype(np.float64)
    G, R = np.zeros((T, T)), np.zeros((K, T))
    for a, b, c in windows(X, T, K, segs):
        inp = X[a:b][::-1]
        m = inp.mean(axis=0)
        Z, Y = inp - m, X[b:c] - m
        G += Z @ Z.T
        R += Y @ Z.T
    return np.linalg.solve(G + lam * np.trace(G) / T * np.eye(T), R.T).T


def tune_lookback(D: np.ndarray, segs, T: int, K: int, dev, metric: str, step: int = 4) -> list:
    """Per horizon k, the lookback-mean window w in 1..T (most recent w frames) with the lowest validation error on block D
    (metric 'mae' or 'mse'; every ``step``-th pixel in each axis). Same selection rule as the tuned heuristics of the POWDER tables."""
    err = torch.zeros(T, K, dtype=torch.float64, device=dev)
    for a, b, c in windows(D, T, K, segs):
        x = torch.from_numpy(np.ascontiguousarray(D[a:b, ::step, ::step])).to(dev).float().flip(0)   # row 0 = newest frame
        means = x.cumsum(0) / torch.arange(1, T + 1, device=dev).view(-1, *([1] * (x.ndim - 1)))   # window 1..T (any trailing dims)
        y = torch.from_numpy(np.ascontiguousarray(D[b:c, ::step, ::step])).to(dev).float()
        e = means[:, None] - y[None]
        err += (e.abs() if metric == 'mae' else e ** 2).flatten(2).mean(2).double()
    return (err.argmin(dim=0) + 1).tolist()                                                           # window per horizon


class Acc:
    """Per-horizon sums of |e|, e^2 (per pixel) and the same per second (rows averaged in linear power)."""

    def __init__(self, K):
        self.ae, self.se, self.n = np.zeros(K), np.zeros(K), np.zeros(K)
        self.ae_s, self.se_s, self.n_s = np.zeros(K), np.zeros(K), np.zeros(K)

    def add(self, pred_db: torch.Tensor, true_db: torch.Tensor, true_sec: torch.Tensor):
        e = (pred_db - true_db).double()
        self.ae += e.abs().flatten(1).sum(1).cpu().numpy()
        self.se += (e ** 2).flatten(1).sum(1).cpu().numpy()
        self.n += e[0].numel()
        es = (per_second(pred_db) - true_sec).double()
        self.ae_s += es.abs().flatten(1).sum(1).cpu().numpy()
        self.se_s += (es ** 2).flatten(1).sum(1).cpu().numpy()
        self.n_s += es[0].numel()

    def out(self):
        mse, mse_s = self.se / self.n, self.se_s / self.n_s
        return {'mae_db': (self.ae / self.n).tolist(), 'mse_db': mse.tolist(), 'rmse_db': np.sqrt(mse).tolist(),
                'per_second': {'mae_db': (self.ae_s / self.n_s).tolist(), 'mse_db': mse_s.tolist(), 'rmse_db': np.sqrt(mse_s).tolist()}}


def per_second(db: torch.Tensor) -> torch.Tensor:
    """(K, H, W) dB -> (K, W): the STFT rows of each second averaged in linear power."""
    return 10 * torch.log10(torch.pow(10.0, db.double() / 10).mean(dim=1)).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sha12')
    ap.add_argument('--checkpoints', type=Path, default=REPO / 'checkpoints' / 'repro' / 'deepspred')
    ap.add_argument('--data-root', type=Path, default=REPO / 'data' / 'datasets' / 'deepspred')
    ap.add_argument('--results', type=Path, default=HERE / 'results.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--max-windows', type=int, default=0, help='score only the first N test windows (smoke test)')
    args = ap.parse_args()
    t_start = time.time()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck_path = args.checkpoints / f'{args.sha12}.pt'
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    key = ck['run_key']
    m = re.search(r'\|spec=([^|]+)\|(\w+)', key)
    if not m:
        raise SystemExit(f'{key}: not a spec-rendered run (JPEG/legacy renderings are out of scope)')
    spec, scope = m.group(1), m.group(2)
    gray = '|gray' in key
    dataset = key.split('|')[0]
    arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']
    channels = 1 if gray else 3
    model = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T, channels)
    if ck.get('head') == 'residual':
        from run_paper_faithful import ResidualHead
        model = ResidualHead(model)
    iscale = 1.0 if ck.get('input_scale') == 'raw' else 1 / 255.
    model.load_state_dict(ck['model_state_dict'])
    model.to(dev).eval()

    directory = args.data_root / DSDIR[dataset]
    frames, meta = load_frames_spec(directory, spec, range_scope=scope, gray=gray)
    (tr, va, te), segs = chronological_split(frames, meta)
    lo, hi = meta['db_range']
    paths = [p for _, p in list_seconds(directory)]
    i_va, i_te = split_indices(meta)
    assert len(i_te) == len(te) and len(i_va) == len(va), 'split mismatch'
    print(f'{args.sha12} {key}\n  dB range {lo:.2f}..{hi:.2f}; computing dB for {len(i_va)} val + {len(i_te)} test files', flush=True)
    db_va = db_block([paths[i] for i in i_va], spec, args.workers)
    db_te = db_block([paths[i] for i in i_te], spec, args.workers)
    idx = windows(te, T, K, segs['test'])
    if args.max_windows:
        idx = idx[:args.max_windows]

    A = fit_residual_ar(db_va, segs['val'], T, K)
    w_db = tune_lookback(db_va, segs['val'], T, K, dev, 'mae')
    w_px = tune_lookback(va, segs['val'], T, K, dev, 'mse')                                   # uint8 frames, all channels
    print(f'  tuned lookback windows: dB/MAE {w_db}; pixels/MSE {w_px}', flush=True)
    A_t = torch.from_numpy(A).float().to(dev)
    inv = Inverter(gray, dev)
    names = ('model', 'persistence', 'lookback_mean', 'lookback_mean_tuned', 'lookback_mean_linear', 'lookback_median', 'residual_linear_ar',
             'lookback_mean_rgb_inverted')
    acc = {n: Acc(K) for n in names}
    u8 = {n: np.zeros(K) for n in ('model', 'lookback_mean_rgb', 'lookback_mean_rgb_tuned')}
    off, n_w = [], 0
    with torch.no_grad():
        for a, b, c in idx:
            x = torch.from_numpy(te[a:b][None].astype(np.float32) * iscale).permute(0, 1, 4, 2, 3).to(dev)
            pred = model(x)[0].permute(0, 2, 3, 1)                        # (K, H, W, C) floats in [0,1]
            true_u8 = torch.from_numpy(np.ascontiguousarray(te[b:c])).to(dev).float()
            pred_u8 = (pred * 255.0).round().clamp(0, 255)
            u8['model'] += ((pred_u8 - true_u8) ** 2).flatten(1).mean(1).cpu().numpy()
            lbm_rgb = torch.from_numpy(np.ascontiguousarray(te[a:b])).to(dev).float().mean(0).round()   # as the runner's lookback mean
            u8['lookback_mean_rgb'] += ((lbm_rgb[None] - true_u8) ** 2).flatten(1).mean(1).cpu().numpy()
            inp_u8 = torch.from_numpy(np.ascontiguousarray(te[a:b])).to(dev).float().flip(0)          # row 0 = newest
            lbm_rgb_t = torch.stack([inp_u8[:w].mean(0).round() for w in w_px])                       # (K, H, W, C)
            u8['lookback_mean_rgb_tuned'] += ((lbm_rgb_t - true_u8) ** 2).flatten(1).mean(1).cpu().numpy()
            if n_w % 50 == 0:
                off.append(inv.off_curve(pred))
            true_db = torch.from_numpy(db_te[b:c]).to(dev)
            true_sec = per_second(true_db)
            inp = torch.from_numpy(db_te[a:b]).to(dev)                    # (T, H, W) dB
            mean = inp.mean(0)
            dev_lags = inp.flip(0) - mean                                  # row j = frame t-j
            preds = {
                'model': lo + inv(pred) * (hi - lo),
                'persistence': inp[-1].expand(K, -1, -1),
                'lookback_mean': mean.expand(K, -1, -1),
                'lookback_mean_tuned': torch.stack([inp.flip(0)[:w].mean(0) for w in w_db]),
                'lookback_mean_linear': (10 * torch.log10(torch.pow(10.0, inp.double() / 10).mean(0))).float().expand(K, -1, -1),
                'lookback_median': inp.median(0).values.expand(K, -1, -1),
                'residual_linear_ar': mean + torch.einsum('kj,jhw->khw', A_t, dev_lags),
                'lookback_mean_rgb_inverted': (lo + inv(lbm_rgb / 255.0) * (hi - lo)).expand(K, -1, -1),
            }
            for n in names:
                acc[n].add(preds[n], true_db, true_sec)
            n_w += 1
            if n_w % 200 == 0:
                print(f'  {n_w}/{len(idx)} windows ({time.time() - t_start:.0f}s)', flush=True)

    rec = json.loads(args.results.read_text()).get(key, {}) if args.results.exists() else {}
    rec_model = rec.get('model', {}).get('mse', {}).get('uint8_mean_per_channel')
    rec_lbm = rec.get('trivial_references', {}).get('lookback_mean', {}).get('mse', {}).get('uint8_mean_per_channel')
    u8_model, u8_lbm = (u8['model'] / n_w).tolist(), (u8['lookback_mean_rgb'] / n_w).tolist()
    out = {
        'sha12': args.sha12, 'run_key': key, 'dataset': dataset, 'render_spec': spec, 'range_scope': scope, 'gray': gray,
        'T': T, 'K': K, 'n_windows': n_w, 'db_range': [lo, hi],
        'checkpoint_sha256': hashlib.sha256(ck_path.read_bytes()).hexdigest(),
        'uint8_mse': {'model': u8_model, 'lookback_mean': u8_lbm, 'lookback_mean_tuned': (u8['lookback_mean_rgb_tuned'] / n_w).tolist()},
        'tuned_lookback_window': {'db_mae': w_db, 'uint8_mse': w_px,
                                  'rule': 'per horizon, window 1..T of the most recent frames with the lowest validation error'},
        'record_uint8_mse': {'model': rec_model, 'lookback_mean': rec_lbm},
        'record_match_max_rel': (max(abs(a - b) / b for a, b in zip(u8_model, rec_model)) if rec_model and not args.max_windows else None),
        'db': {n: acc[n].out() for n in names},
        'residual_ar_coefficients': A.tolist(),
        'jet_off_curve_mean_distance': float(np.mean(off)) if off else None,
        'inversion': 'gray: linear (exact)' if gray else f'jet: nearest of a {LUT_SIZE}-entry table to the float prediction',
        'seconds': round(time.time() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    d = out['db']
    print(f"DBSCORE {args.sha12} k=8: model MAE {d['model']['mae_db'][7]:.3f} dB | lookback mean {d['lookback_mean']['mae_db'][7]:.3f} | "
          f"res-AR {d['residual_linear_ar']['mae_db'][7]:.3f} | persistence {d['persistence']['mae_db'][7]:.3f} | "
          f"uint8 MSE {u8_model[7]:.1f} (record {rec_model[7] if rec_model else None}) | match {out['record_match_max_rel']}", flush=True)


if __name__ == '__main__':
    main()
