"""Does averaging errors before squaring explain DeepSPred's published PSNR?

The paper reports, on FM at frame 8, MSE 448.76 and PSNR 35.10 dB (MAX = 255). That PSNR implies MSE 20.08, 22.35x lower. In
Appendix A (frame 10) the RGB pair has the same gap (MSE 450.79, PSNR 35.0961, so 22.4x), while the grayscale pair is nearly
consistent (MSE 5185.25, PSNR 11.339, so implied 4777 and a ratio of 1.09). This script tests one candidate explanation: PSNR computed
from the squared *mean* signed error (prediction minus truth) over a group of images, instead of from the mean squared error.

For the trained model and for the lookback mean (the rounded mean of the 20 input frames, as the runner computes it), on the 8-bit test
images, it computes per predicted frame k:

* ``mse``: the ordinary per-pixel MSE (mean over windows, pixels and channels), and ``psnr`` = 10 log10(255^2 / mse);
* ``mean_psnr_per_image``: the per-image PSNR averaged over windows (another common convention; not the averaging under test);
* at frames 8, 10 and 18: ``groups[order][n]``, the MSE after averaging the signed error image over groups of n test windows before
  squaring (n = 1, 2, 4, 8, 16, 20, 32 and all windows), with the groups taken as consecutive windows or drawn at random
  (``RANDOM_DRAWS`` random partitions, mean reported; leftover windows that do not fill a group are dropped);
* ``within_window``: the MSE after averaging the signed error over the 20 predicted frames of each window before squaring (one value per
  run, not per k).

Every MSE gets its PSNR and ``ratio`` = ordinary MSE at that frame / the averaged MSE. The paper's jet ratio is 22.35 (frame 8) and
22.4 (frame 10); its grayscale ratio is 1.09 (frame 10).

The loading and prediction path is dbspace_score.py's: the same checkpoint, ``load_frames_spec`` rendering, ``chronological_split``,
stride-1 test windows inside contiguous segments, batch-1 inference, and prediction rounded to uint8. The ordinary MSE is checked
against the dbspace record. CPU only.

    python training/DeepSPred/repro/psnr_averaging.py 2fa9ff9a3d9b --out data/deepspred-repro/psnr/2fa9ff9a3d9b.json
    python -m training.DeepSPred.repro.psnr_averaging --ray 2fa9ff9a3d9b 3c96fc0dc0cd 6ec6d8e47cc1     # UC, one CPU task each
    python training/DeepSPred/repro/psnr_averaging.py --combine dir/with/the/three/jsons            # -> dbspace/psnr_averaging.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES = (8, 10, 18)                      # 1-based predicted frame index, as in the paper's figures
GROUP_SIZES = (1, 2, 4, 8, 16, 20, 32, 'all')
RANDOM_DRAWS = 5
PAPER = {'jet_ratio': {8: 448.7628 / (255 ** 2 / 10 ** (35.1039 / 10)), 10: 450.7880 / (255 ** 2 / 10 ** (35.0961 / 10))},
         'gray_ratio': {10: 5185.2540 / (255 ** 2 / 10 ** (11.3390 / 10))},
         'jet_psnr': {8: 35.1039, 10: 35.0961}, 'gray_psnr': {10: 11.3390}}


def psnr(mse: float) -> float:
    return 10 * math.log10(255 ** 2 / mse) if mse > 0 else float('inf')


def group_mse(err: np.ndarray, n, order: str, rng=None) -> float:
    """err: (N, P) signed errors (int16) of one predicted frame. Mean over groups of mean over pixels of (mean over the group's
    windows of err)^2."""
    N = len(err)
    size = N if n == 'all' else n
    idx = np.arange(N) if order == 'consecutive' else rng.permutation(N)
    g = N // size
    total = 0.0
    for j in range(g):
        m = err[idx[j * size:(j + 1) * size]].astype(np.float64).mean(axis=0)
        total += float((m * m).mean())
    return total / g


def score(sha12: str, checkpoints: Path, data_root: Path, threads: int, max_windows: int = 0) -> dict:
    import torch
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / 'probe'))
    from dbspace_score import DSDIR
    from fm_data import chronological_split, load_frames_spec
    from run_paper_faithful import build, windows
    torch.set_num_threads(threads)
    t0 = time.time()
    ck = torch.load(checkpoints / f'{sha12}.pt', map_location='cpu', weights_only=False)
    key = ck['run_key']
    m = re.search(r'\|spec=([^|]+)\|(\w+)', key)
    spec, scope, gray, dataset = m.group(1), m.group(2), '|gray' in key, key.split('|')[0]
    arch, T, K = ck['arch'], ck['regime']['T'], ck['regime']['K']
    model = build(arch['embed_dim'], arch['depths'], arch['heads'], arch['patch'], arch['window'], T, 1 if gray else 3)
    if ck.get('head') == 'residual':
        from run_paper_faithful import ResidualHead
        model = ResidualHead(model)
    iscale = 1.0 if ck.get('input_scale') == 'raw' else 1 / 255.
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    frames, meta = load_frames_spec(data_root / DSDIR[dataset], spec, range_scope=scope, gray=gray)
    (_, _, te), segs = chronological_split(frames, meta)
    idx = windows(te, T, K, segs['test'])
    if max_windows:
        idx = idx[:max_windows]
    N, P = len(idx), int(np.prod(te.shape[1:]))
    ks = [k - 1 for k in FRAMES]
    arms = ('model', 'lookback_mean')
    store = {a: np.zeros((len(ks), N, P), dtype=np.int16) for a in arms}        # signed uint8 errors at the reported frames
    se = {a: np.zeros(K) for a in arms}
    psnr_sum = {a: np.zeros(K) for a in arms}
    within = {a: 0.0 for a in arms}
    print(f'{sha12} {key}: {N} test windows, {P} values per frame, {threads} threads', flush=True)
    with torch.no_grad():
        for i, (a, b, c) in enumerate(idx):
            x = torch.from_numpy(te[a:b][None].astype(np.float32) * iscale).permute(0, 1, 4, 2, 3)
            pred = model(x)[0].permute(0, 2, 3, 1)                                  # (K, H, W, C) in [0, 1]
            true = torch.from_numpy(np.ascontiguousarray(te[b:c])).float()
            preds = {'model': (pred * 255.0).round().clamp(0, 255),
                     'lookback_mean': torch.from_numpy(np.ascontiguousarray(te[a:b])).float().mean(0).round()[None].expand(K, -1, -1, -1)}
            for arm in arms:
                e = (preds[arm] - true).reshape(K, -1)                               # integer-valued
                per = (e * e).mean(1).double().numpy()
                se[arm] += per
                psnr_sum[arm] += np.array([psnr(v) for v in per])
                within[arm] += float((e.mean(0).double() ** 2).mean())
                store[arm][:, i] = e[ks].to(torch.int16).numpy()
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{N} windows ({time.time() - t0:.0f}s)', flush=True)

    rng = np.random.default_rng(0)
    out = {'sha12': sha12, 'run_key': key, 'render_spec': spec, 'gray': gray, 'T': T, 'K': K, 'n_windows': N, 'values_per_frame': P,
           'frames_reported': list(FRAMES), 'group_sizes': [str(n) for n in GROUP_SIZES], 'random_draws': RANDOM_DRAWS, 'arms': {}}
    for arm in arms:
        mse = se[arm] / N
        r = {'mse': mse.tolist(), 'psnr': [psnr(v) for v in mse], 'mean_psnr_per_image': (psnr_sum[arm] / N).tolist(),
             'within_window': {'mse': within[arm] / N, 'psnr': psnr(within[arm] / N),
                               'ratio_vs_mean_mse_over_frames': float(mse.mean()) / (within[arm] / N)},
             'groups': {}}
        for j, k in enumerate(FRAMES):
            g = {}
            for order in ('consecutive', 'random'):
                g[order] = {}
                for n in GROUP_SIZES:
                    if order == 'random' and n not in (1, 'all'):
                        v = float(np.mean([group_mse(store[arm][j], n, 'random', rng) for _ in range(RANDOM_DRAWS)]))
                    else:
                        v = group_mse(store[arm][j], n, 'consecutive')
                    g[order][str(n)] = {'mse': v, 'psnr': psnr(v), 'ratio': mse[k - 1] / v}
            r['groups'][str(k)] = g
        out['arms'][arm] = r
    dbs = HERE / 'dbspace' / f'{sha12}.json'
    if dbs.exists() and not max_windows:
        ref = json.loads(dbs.read_text())['uint8_mse']['model']
        out['check_vs_dbspace_model_mse_max_rel'] = max(abs(x - y) / y for x, y in zip(out['arms']['model']['mse'], ref))
    out['seconds'] = round(time.time() - t0, 1)
    return out


def ray_main(shas, prefix, bucket, cpus):
    import ray
    ray.init(address='auto', ignore_reinit_error=True)

    def task(sha12):
        from training.DeepSPred.repro import ray_campaign as rc
        os.chdir(rc.REPO)
        sys.path.insert(0, rc.REPO)
        from training.ray.storage import spectrum_s3_client
        client = spectrum_s3_client()
        started = time.time()
        try:
            key, rec = rc.record_for(sha12)
            rc.stage_checkpoint(client, bucket, sha12, rec.get('checkpoint', {}).get('sha256'))
            import torch
            run_key = torch.load(Path(rc.REPO) / 'checkpoints' / 'repro' / 'deepspred' / f'{sha12}.pt', map_location='cpu',
                                 weights_only=False)['run_key']
            m = re.search(r'\|spec=([^|]+)\|(\w+)', run_key)
            ds_dir = rc.stage_dataset(client, bucket, run_key.split('|')[0])
            rc.build_cache(ds_dir, m.group(1), m.group(2), '|gray' in run_key)
            out = score(sha12, Path(rc.REPO) / 'checkpoints' / 'repro' / 'deepspred', ds_dir.parent, cpus)
            body = json.dumps(out, indent=1).encode()
            path = rc.OUT / 'psnr' / f'{sha12}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            client.put_object(Bucket=bucket, Key=f'{prefix}/psnr/{sha12}.json', Body=body)
            a = out['arms']['model']['groups']['8']
            return {'ok': True, 'sha12': sha12, 'seconds': round(time.time() - started),
                    'k8': {'mse': out['arms']['model']['mse'][7], 'psnr': out['arms']['model']['psnr'][7],
                           'consecutive_psnr': {n: round(v['psnr'], 2) for n, v in a['consecutive'].items()}}}
        except Exception as error:
            import traceback
            return {'ok': False, 'sha12': sha12, 'error': f'{type(error).__name__}: {error}'[:400], 'trace': traceback.format_exc()[-1500:]}

    remote = ray.remote(num_cpus=cpus, num_gpus=0, memory=24 * 1024 ** 3, max_retries=1)(task)
    refs = {remote.remote(s): s for s in shas}
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        try:
            r = ray.get(done[0])
        except Exception as error:
            r = {'ok': False, 'sha12': refs[done[0]], 'error': f'LOST {type(error).__name__}: {error}'[:300]}
        print(f"[psnr-averaging] {json.dumps({k: v for k, v in r.items() if k != 'trace'})} [{len(pending)} left]", flush=True)
        if not r.get('ok'):
            print(r.get('trace', ''), flush=True)


def combine(src: Path) -> None:
    runs = {}
    for p in sorted(src.glob('*.json')):
        d = json.loads(p.read_text())
        if 'arms' in d:
            runs[d['sha12']] = d
    out = {'description': (__doc__ or '').split('\n\n')[0], 'paper': {k: {str(f): v for f, v in d.items()} for k, d in PAPER.items()},
           'runs': runs}
    target = HERE / 'dbspace' / 'psnr_averaging.json'
    target.write_text(json.dumps(out, indent=1) + '\n')
    for sha, d in runs.items():
        print(f"\n{sha} {d['render_spec']}{' gray' if d['gray'] else ' jet'}  N={d['n_windows']}")
        for arm, r in d['arms'].items():
            ww = r['within_window']
            print(f"  {arm}: within-window (20 frames) MSE {ww['mse']:.2f} PSNR {ww['psnr']:.2f} ratio {ww['ratio_vs_mean_mse_over_frames']:.2f}")
            for k in map(str, d['frames_reported']):
                kk = int(k) - 1
                print(f"   k={k}: MSE {r['mse'][kk]:.1f} PSNR {r['psnr'][kk]:.2f} mean-per-image PSNR {r['mean_psnr_per_image'][kk]:.2f}")
                for order in ('consecutive', 'random'):
                    g = r['groups'][k][order]
                    print(f"     {order:11s} " + '  '.join(f"n={n}: {v['psnr']:.2f}dB x{v['ratio']:.1f}" for n, v in g.items()))
    print(f'\nwrote {target}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sha12', nargs='*')
    ap.add_argument('--ray', action='store_true', help='one CPU-only Ray task per checkpoint on the cluster')
    ap.add_argument('--combine', type=Path, help='directory of per-run JSONs -> dbspace/psnr_averaging.json')
    ap.add_argument('--cpus', type=int, default=16)
    ap.add_argument('--prefix', default='deepspred-repro/2026-11-12')
    ap.add_argument('--bucket', default=os.environ.get('SPECTRUM_S3_BUCKET', 'spectrum'))
    ap.add_argument('--checkpoints', type=Path, default=HERE.parents[2] / 'checkpoints' / 'repro' / 'deepspred')
    ap.add_argument('--data-root', type=Path, default=HERE.parents[2] / 'data' / 'datasets' / 'deepspred')
    ap.add_argument('--max-windows', type=int, default=0)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.combine:
        combine(args.combine)
    elif args.ray:
        ray_main(args.sha12, args.prefix, args.bucket, args.cpus)
    else:
        for s in args.sha12:
            out = score(s, args.checkpoints, args.data_root, args.cpus, args.max_windows)
            path = args.out or Path(f'{s}_psnr.json')
            path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
