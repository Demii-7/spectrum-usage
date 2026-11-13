"""Is a DeepSPred checkpoint's prediction a constant image?

For each checkpoint, on its test windows (the dbspace_score.py path: same rendering, split, stride-1 windows, batch-1 inference), this
reports per predicted frame k (1, 8 and K):

* ``across_windows_var``: the variance across test windows of each pixel/channel value, averaged over pixels and channels, for the
  model's float prediction x255, the true target frames and the lookback mean (all on the 0..255 scale). A constant-image model has 0.
* ``max_abs_dev_from_first``: the largest absolute difference (x255) between any window's prediction and the first window's.
* ``within_image_var``: the variance over pixels/channels within one frame, averaged over windows (0 for a single flat colour).
* ``mean_rgb``: the prediction's mean colour per channel (x255) and the number of distinct rounded colours in the first window's frame.

    python -m training.DeepSPred.repro.output_variance --ray 4dfd1d59c176 ef76c703ed96 dba70d28179a      # UC, CPU only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def measure(sha12: str, checkpoints: Path, data_root: Path, threads: int) -> dict:
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
    ks = sorted({0, 7, K - 1})
    arms = ('model', 'target', 'lookback_mean')
    s1 = {a: None for a in arms}
    s2 = {a: None for a in arms}
    within = {a: np.zeros(len(ks)) for a in arms}
    first, maxdev, n_colours = None, np.zeros(len(ks)), None
    with torch.no_grad():
        for i, (a, b, c) in enumerate(idx):
            x = torch.from_numpy(te[a:b][None].astype(np.float32) * iscale).permute(0, 1, 4, 2, 3)
            pred = model(x)[0].permute(0, 2, 3, 1)[ks].double().numpy() * 255.0          # (len(ks), H, W, C)
            vals = {'model': pred, 'target': te[b:c][ks].astype(np.float64),
                    'lookback_mean': np.repeat(te[a:b].astype(np.float64).mean(0).round()[None], len(ks), 0)}
            for arm, v in vals.items():
                s1[arm] = v if s1[arm] is None else s1[arm] + v
                s2[arm] = v * v if s2[arm] is None else s2[arm] + v * v
                within[arm] += v.reshape(len(ks), -1).var(axis=1)
            if first is None:
                first = pred
                n_colours = [int(len(np.unique(np.round(f).reshape(-1, f.shape[-1]), axis=0))) for f in pred]
            else:
                maxdev = np.maximum(maxdev, np.abs(pred - first).reshape(len(ks), -1).max(axis=1))
    n = len(idx)
    out = {'sha12': sha12, 'run_key': key, 'n_windows': n, 'frames': [k + 1 for k in ks], 'scale': '0..255', 'arms': {}}
    for arm in arms:
        mean = s1[arm] / n
        var = np.maximum(s2[arm] / n - mean ** 2, 0)
        out['arms'][arm] = {'across_windows_var': var.reshape(len(ks), -1).mean(axis=1).tolist(),
                            'within_image_var': (within[arm] / n).tolist(),
                            'mean_rgb': mean.reshape(len(ks), -1, mean.shape[-1]).mean(axis=1).tolist()}
    out['arms']['model'].update(max_abs_dev_from_first=maxdev.tolist(), distinct_colours_first_window=n_colours)
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
        try:
            key, rec = rc.record_for(sha12)
            rc.stage_checkpoint(client, bucket, sha12, rec.get('checkpoint', {}).get('sha256'))
            import torch
            run_key = torch.load(Path(rc.REPO) / 'checkpoints' / 'repro' / 'deepspred' / f'{sha12}.pt', map_location='cpu',
                                 weights_only=False)['run_key']
            m = re.search(r'\|spec=([^|]+)\|(\w+)', run_key)
            ds_dir = rc.stage_dataset(client, bucket, run_key.split('|')[0])
            rc.build_cache(ds_dir, m.group(1), m.group(2), '|gray' in run_key)
            out = measure(sha12, Path(rc.REPO) / 'checkpoints' / 'repro' / 'deepspred', ds_dir.parent, cpus)
            body = json.dumps(out, indent=1).encode()
            client.put_object(Bucket=bucket, Key=f'{prefix}/output_variance/{sha12}.json', Body=body)
            return {'ok': True, **out}
        except Exception as error:
            import traceback
            return {'ok': False, 'sha12': sha12, 'error': f'{type(error).__name__}: {error}'[:400], 'trace': traceback.format_exc()[-1500:]}

    remote = ray.remote(num_cpus=cpus, num_gpus=0, memory=16 * 1024 ** 3, max_retries=1)(task)
    refs = [remote.remote(s) for s in shas]
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        print(f"[output-variance] {json.dumps({k: v for k, v in r.items() if k != 'trace'})} [{len(pending)} left]", flush=True)
        if not r.get('ok'):
            print(r.get('trace', ''), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sha12', nargs='+')
    ap.add_argument('--ray', action='store_true')
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--prefix', default='deepspred-repro/2026-11-12')
    ap.add_argument('--bucket', default=os.environ.get('SPECTRUM_S3_BUCKET', 'spectrum'))
    args = ap.parse_args()
    if args.ray:
        ray_main(args.sha12, args.prefix, args.bucket, args.cpus)
    else:
        for s in args.sha12:
            print(json.dumps(measure(s, HERE.parents[2] / 'checkpoints' / 'repro' / 'deepspred',
                                     HERE.parents[2] / 'data' / 'datasets' / 'deepspred', args.cpus)))


if __name__ == '__main__':
    main()
