"""Colour statistics of DeepSPred frame renderings: is the Tukey-224 LTE rendering unusually close to the colormap's ends?

Motivation (2026-11-13): from-scratch 3D-SwinSTB on LTE with the Tukey hop-224 rendering (``tukey224d``) collapses in every seed to one
flat colour, (255, 255, 0), with a saturated sigmoid output; the same recipe on JPEG-q95 LTE frames trains. A rendering whose pixels sit
near 0 or 255 in some channel pushes the sigmoid's inputs toward large values, where it saturates easily.

For each (dataset, rendering) this reports, on the training split (the frames the model fits) unless noted:

* per channel (R, G, B, 0..255): mean, standard deviation, share of values <= 5 and >= 250, and the logit of the channel mean
  (the sigmoid input that outputs the mean colour; |logit| > 2.2 means the mean is within 10% of an end);
* the colormap scalar v in [0, 1] (jet inverted through a 4096-entry table, on every 4th pixel): mean, sd, 1/5/50/95/99th percentiles;
* on the validation split, the MSE in [0,1] units (the runner's loss) of: the flat image of the training mean colour, the per-pixel
  training mean image, the flat (255, 255, 0) image the collapsed models output, and the lookback mean of the 16 previous frames
  (stride 16 windows). The collapsed runs' validation loss was 0.019715.

    python -m training.DeepSPred.repro.colour_stats --ray            # UC, CPU only, one task per rendering
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RENDERINGS = [('lte', 'tukey224d'), ('lte', 'hann126'), ('lte', 'hann128s'), ('lte', 'hamm128'), ('lte', 'rect126'), ('lte', 'rect127c~12'),
              ('lte', 'jpeg95'), ('fm', 'tukey224d'), ('fm', 'rect127c~12')]
T = 16
LUT_SIZE = 4096


def stats(dataset: str, spec: str, data_root: Path) -> dict:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / 'probe'))
    from dbspace_score import DSDIR
    from fm_data import chronological_split, load_frames, load_frames_spec
    from run_paper_faithful import windows
    import matplotlib
    import torch
    t0 = time.time()
    directory = data_root / DSDIR[dataset]
    if spec == 'jpeg95':
        frames, meta = load_frames(directory, expect_hw=(256, 256), jpeg_quality=95)
    else:
        frames, meta = load_frames_spec(directory, spec, range_scope='full')
    (tr, va, _), segs = chronological_split(frames, meta)
    tr = np.asarray(tr)
    va = np.asarray(va)
    out = {'dataset': dataset, 'render_spec': spec, 'n_train': len(tr), 'n_val': len(va), 'db_range': meta.get('db_range'), 'channels': {}}
    s1 = np.zeros(3)
    s2 = np.zeros(3)
    lo = np.zeros(3)
    hi = np.zeros(3)
    for i in range(0, len(tr), 64):
        b = tr[i:i + 64].reshape(-1, 3).astype(np.float64)
        s1 += b.sum(0)
        s2 += (b * b).sum(0)
        lo += (b <= 5).sum(0)
        hi += (b >= 250).sum(0)
    n = tr.shape[0] * tr.shape[1] * tr.shape[2]
    mean = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    m01 = np.clip(mean / 255, 1e-6, 1 - 1e-6)
    for c, name in enumerate('RGB'):
        out['channels'][name] = {'mean': mean[c], 'sd': sd[c], 'share_le5': lo[c] / n, 'share_ge250': hi[c] / n,
                                 'logit_of_mean': float(np.log(m01[c] / (1 - m01[c])))}
    lut = torch.from_numpy(matplotlib.colormaps['jet'].resampled(LUT_SIZE)(np.linspace(0, 1, LUT_SIZE))[:, :3].astype(np.float32))
    vs = []
    for i in range(0, len(tr), 8):
        px = torch.from_numpy(tr[i, ::4, ::4].reshape(-1, 3).astype(np.float32) / 255)
        vs.append((torch.cdist(px, lut).argmin(1).float() / (LUT_SIZE - 1)).numpy())
    v = np.concatenate(vs)
    out['v'] = {'mean': float(v.mean()), 'sd': float(v.std()), 'percentiles': dict(zip(['p1', 'p5', 'p50', 'p95', 'p99'],
                                                                                          np.percentile(v, [1, 5, 50, 95, 99]).tolist()))}
    mean_img = tr.astype(np.float64).mean(0) / 255
    flat_mean = mean / 255
    yellow = np.array([1.0, 1.0, 0.0])
    acc = {'flat_train_mean_colour': 0.0, 'train_mean_image': 0.0, 'flat_255_255_0': 0.0}
    for f in va:
        x = f.astype(np.float64) / 255
        acc['flat_train_mean_colour'] += ((x - flat_mean) ** 2).mean()
        acc['train_mean_image'] += ((x - mean_img) ** 2).mean()
        acc['flat_255_255_0'] += ((x - yellow) ** 2).mean()
    val = {k: v_ / len(va) for k, v_ in acc.items()}
    idx = windows(va, T, T, segs['val'])[::16]
    lb = 0.0
    for a, b, c in idx:
        lbm = va[a:b].astype(np.float64).mean(0) / 255
        lb += ((va[b:c].astype(np.float64) / 255 - lbm[None]) ** 2).mean()
    val['lookback_mean_16'] = lb / max(len(idx), 1)
    val['n_lookback_windows'] = len(idx)
    out['val_mse_unit'] = val
    out['seconds'] = round(time.time() - t0, 1)
    return out


def ray_main(prefix, bucket, cpus):
    import ray
    ray.init(address='auto', ignore_reinit_error=True)

    def task(dataset, spec):
        from training.DeepSPred.repro import ray_campaign as rc
        os.chdir(rc.REPO)
        sys.path.insert(0, rc.REPO)
        from training.ray.storage import spectrum_s3_client
        client = spectrum_s3_client()
        try:
            ds_dir = rc.stage_dataset(client, bucket, dataset)
            if spec != 'jpeg95':
                rc.build_cache(ds_dir, spec, 'full', False)
            out = stats(dataset, spec, ds_dir.parent)
            client.put_object(Bucket=bucket, Key=f'{prefix}/colour_stats/{dataset}_{spec}.json', Body=json.dumps(out, indent=1).encode())
            return {'ok': True, **out}
        except Exception as error:
            import traceback
            return {'ok': False, 'dataset': dataset, 'spec': spec, 'error': f'{type(error).__name__}: {error}'[:400],
                    'trace': traceback.format_exc()[-1500:]}

    remote = ray.remote(num_cpus=cpus, num_gpus=0, memory=24 * 1024 ** 3, max_retries=1)(task)
    pending = [remote.remote(d, s) for d, s in RENDERINGS]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        print(f"[colour-stats] {json.dumps({k: v for k, v in r.items() if k != 'trace'})} [{len(pending)} left]", flush=True)
        if not r.get('ok'):
            print(r.get('trace', ''), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ray', action='store_true')
    ap.add_argument('--cpus', type=int, default=16)
    ap.add_argument('--prefix', default='deepspred-repro/2026-11-12')
    ap.add_argument('--bucket', default=os.environ.get('SPECTRUM_S3_BUCKET', 'spectrum'))
    args = ap.parse_args()
    if args.ray:
        ray_main(args.prefix, args.bucket, args.cpus)


if __name__ == '__main__':
    main()
