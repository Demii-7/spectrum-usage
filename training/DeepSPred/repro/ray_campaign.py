"""Ray driver for the DeepSPred reproduction follow-up (2026-11-12): dB-space scoring of existing checkpoints and new training runs.

Each task takes one whole GPU (no fractional shares) and enough host memory that two reproduction runs never share a node (the runner's
record phase peaks near 75 GB). Every task stages what it needs on its node from the object store (raw I/Q under
``datasets/deepspred/<dataset>/``, checkpoints under ``checkpoints/repro/deepspred/``), builds the frame cache under a node-local lock,
runs the script as a subprocess, and uploads its outputs to ``<prefix>/``:

    score  <sha12>                       -> <prefix>/dbspace/<sha12>.json            (dbspace_score.py)
    train  <name> <runner args...>       -> <prefix>/train/<name>/{record.json,log.txt}, checkpoint to checkpoints/repro/deepspred/,
                                            then scored in dB like ``score``

    python -m training.DeepSPred.repro.ray_campaign --score 3c96fc0dc0cd 2fa9ff9a3d9b --train gray32-tukey224d
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = '/workspace/spectrum-usage'
DSDIR = {'fm': 'FM-Spectrum-Dataset', 'lte': 'LTE-Spectrum-Dataset'}
# Every node-local output goes under the untracked data/ tree: tasks run as root, and root-owned directories inside tracked paths
# make sync-cluster-code.py fail on that host. The durable copies are on the object store (<prefix>/...).
OUT = Path(REPO) / 'data' / 'deepspred-repro'
EXPECTED_FILES = {'fm': 10800, 'lte': 2400}
TRAIN_RUNS = {                                                   # the Appendix-A grayscale comparison, without JPEG
    'gray32-tukey224d': ['--render-spec', 'tukey224d', '--gray', '--embed-dim', '32'],
    'gray96-tukey224d': ['--render-spec', 'tukey224d', '--gray', '--embed-dim', '96'],
    'gray32-rect127c': ['--render-spec', 'rect127c~12', '--gray', '--embed-dim', '32'],
    'gray96-rect127c': ['--render-spec', 'rect127c~12', '--gray', '--embed-dim', '96'],
}
# Transfer replicates on the authors' LTE data (paper Algorithm 4), scipy-spectrogram-default rendering, K = T = 16 as the existing pair
# (scratch 4dfd1d59c176, FM-initialized dba70d28179a, both seed 0): name -> (seed, FM checkpoint to start from or None)
LTE_RUNS = {
    'lte-scratch-s1': (1, None), 'lte-scratch-s2': (2, None),
    'lte-tl-s1': (1, 'c5a4e4c918ca'),                            # FM seed 1
    'lte-tl-s2': (2, '3c96fc0dc0cd'),                            # no FM seed 2 exists: FM seed 0, fine-tuned with seed 2
}


def _lock(name: str):
    path = Path(REPO) / 'data' / 'datasets' / 'deepspred' / f'.{name}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'w')
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def stage_dataset(client, bucket: str, dataset: str) -> Path:
    target = Path(REPO) / 'data' / 'datasets' / 'deepspred' / DSDIR[dataset]
    with _lock(f'stage-{dataset}'):
        if target.is_dir() and sum(1 for _ in target.rglob('*.xls')) >= EXPECTED_FILES[dataset]:
            return target
        prefix = f'datasets/deepspred/{DSDIR[dataset]}/'
        keys, token = [], None
        while True:
            kw = {'Bucket': bucket, 'Prefix': prefix, 'MaxKeys': 1000, **({'ContinuationToken': token} if token else {})}
            page = client.list_objects_v2(**kw)
            keys += [o['Key'] for o in page.get('Contents', [])]
            if not page.get('IsTruncated'):
                break
            token = page['NextContinuationToken']

        def get(key):
            out = target / key[len(prefix):]
            if out.exists():
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + '.part')
            client.download_file(bucket, key, str(part))
            os.replace(part, out)

        with ThreadPoolExecutor(32) as pool:
            list(pool.map(get, keys))
    return target


def stage_checkpoint(client, bucket: str, sha12: str, expect_sha256: str | None) -> Path:
    path = Path(REPO) / 'checkpoints' / 'repro' / 'deepspred' / f'{sha12}.pt'
    with _lock(f'ckpt-{sha12}'):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + '.part')
            client.download_file(bucket, f'checkpoints/repro/deepspred/{sha12}.pt', str(part))
            os.replace(part, path)
    if expect_sha256:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expect_sha256:
            raise RuntimeError(f'{sha12}: sha256 {got} != record {expect_sha256}')
    return path


def build_cache(dataset_dir: Path, spec: str, scope: str, gray: bool) -> None:
    sys.path.insert(0, f'{REPO}/training/DeepSPred/repro')
    from fm_data import load_frames_spec
    with _lock(f'cache-{dataset_dir.name}-{hashlib.sha1(f"{spec}|{scope}|{gray}".encode()).hexdigest()[:8]}'):
        load_frames_spec(dataset_dir, spec, range_scope=scope, gray=gray)


def record_for(sha12: str) -> tuple[str, dict]:
    records = json.loads((Path(REPO) / 'training' / 'DeepSPred' / 'repro' / 'results.json').read_text())
    for key, rec in records.items():
        if isinstance(rec, dict) and str(rec.get('checkpoint', {}).get('path', '')).endswith(f'{sha12}.pt'):
            return key, rec
    return '', {}


def _score(client, bucket: str, prefix: str, sha12: str, results: str | None = None) -> dict:
    import re
    key, rec = record_for(sha12)
    stage_checkpoint(client, bucket, sha12, rec.get('checkpoint', {}).get('sha256'))
    import torch
    ck = torch.load(Path(REPO) / 'checkpoints' / 'repro' / 'deepspred' / f'{sha12}.pt', map_location='cpu', weights_only=False)
    run_key = ck['run_key']
    m = re.search(r'\|spec=([^|]+)\|(\w+)', run_key)
    dataset = run_key.split('|')[0]
    ds_dir = stage_dataset(client, bucket, dataset)
    build_cache(ds_dir, m.group(1), m.group(2), '|gray' in run_key)
    out = OUT / 'dbspace' / f'{sha12}.json'
    cmd = [sys.executable, '-u', 'training/DeepSPred/repro/dbspace_score.py', sha12, '--out', str(out)]
    if results:
        cmd += ['--results', results]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    log = proc.stdout[-4000:] + proc.stderr[-4000:]
    client.put_object(Bucket=bucket, Key=f'{prefix}/dbspace/{sha12}.log', Body=log.encode())
    if proc.returncode != 0 or not out.exists():
        return {'ok': False, 'sha12': sha12, 'log': log[-1500:]}
    client.put_object(Bucket=bucket, Key=f'{prefix}/dbspace/{sha12}.json', Body=out.read_bytes())
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('DBSCORE')]
    return {'ok': True, 'sha12': sha12, 'summary': line[-1] if line else ''}


def task(spec: dict) -> dict:
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from training.ray.storage import spectrum_s3_client
    client, bucket, prefix = spectrum_s3_client(), spec['bucket'], spec['prefix']
    started = time.time()
    try:
        if spec['kind'] == 'score':
            result = _score(client, bucket, prefix, spec['sha12'])
        else:
            name = spec['name']
            if name in LTE_RUNS:
                seed, init = LTE_RUNS[name]
                ds_dir = stage_dataset(client, bucket, 'lte')
                extra = ['--render-spec', 'tukey224d', '--dataset', 'lte']
                if init:
                    init_rec = record_for(init)[1]
                    extra += ['--init-from', str(stage_checkpoint(client, bucket, init, init_rec.get('checkpoint', {}).get('sha256')))]
                horizon = '16'
            else:
                seed, extra, horizon = spec.get('seed', 0), TRAIN_RUNS[name], '20'
                ds_dir = stage_dataset(client, bucket, 'fm')
            build_cache(ds_dir, extra[1], 'full', '--gray' in extra)
            results = OUT / 'campaign' / f'{name}.json'
            results.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', 'training/DeepSPred/repro/run_paper_faithful.py', '--data', str(ds_dir), '--horizon', horizon,
                   '--seed', str(seed), '--range-scope', 'full', '--results', str(results), *extra]
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            log = proc.stdout[-20000:] + proc.stderr[-6000:]
            client.put_object(Bucket=bucket, Key=f'{prefix}/train/{name}/log.txt', Body=log.encode())
            if not results.exists():
                return {'ok': False, 'kind': 'train', 'name': name, 'log': log[-1500:], 'seconds': round(time.time() - started)}
            records = json.loads(results.read_text())
            client.put_object(Bucket=bucket, Key=f'{prefix}/train/{name}/record.json', Body=json.dumps(records, indent=1).encode())
            (key, rec), = records.items()
            ck = Path(REPO) / rec['checkpoint']['path']
            sha12 = ck.stem
            client.upload_file(str(ck), bucket, f'checkpoints/repro/deepspred/{sha12}.pt')
            scored = _score(client, bucket, prefix, sha12, results=str(results))
            result = {'ok': scored['ok'], 'kind': 'train', 'name': name, 'sha12': sha12, 'run_key': key,
                      'mse8': rec['model']['mse']['uint8_mean_per_channel'][7], 'score': scored}
    except Exception as error:
        import traceback
        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'[:400], 'trace': traceback.format_exc()[-1500:]}
    result.update(spec=spec, seconds=round(time.time() - started))
    try:
        tag = spec.get('sha12') or spec.get('name')
        client.put_object(Bucket=bucket, Key=f"{prefix}/status/{spec['kind']}-{tag}.json", Body=json.dumps(result, indent=1).encode())
    except Exception:
        pass
    return result


def main(argv=None) -> None:
    import ray
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--score', nargs='*', default=[], help='checkpoint sha12s to score in dB')
    p.add_argument('--train', nargs='*', default=[], choices=sorted(TRAIN_RUNS) + sorted(LTE_RUNS),
                   help='training runs (each scored in dB afterwards)')
    p.add_argument('--prefix', default='deepspred-repro/2026-11-12')
    p.add_argument('--bucket', default=os.environ.get('SPECTRUM_S3_BUCKET', 'spectrum'))
    args = p.parse_args(argv)
    ray.init(address='auto', ignore_reinit_error=True)
    specs = [{'kind': 'train', 'name': n} for n in args.train] + [{'kind': 'score', 'sha12': s} for s in args.score]
    gb = 1024 ** 3
    refs = {}
    for s in specs:
        s.update(bucket=args.bucket, prefix=args.prefix)
        # training: a whole GPU (1-channel runs: the record phase needs about a third of the RGB 75 GB of host memory);
        # scoring is inference only: a quarter GPU
        train = s['kind'] == 'train'
        opts = dict(num_cpus=16 if train else 8, num_gpus=1 if train else 0.25, memory=(40 if train else 16) * gb, max_retries=1)
        refs[ray.remote(**opts)(task).remote(s)] = s
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        try:
            r = ray.get(done[0])
        except Exception as error:
            r = {'ok': False, 'error': f'LOST {type(error).__name__}: {error}'[:300], 'spec': refs[done[0]]}
        print(f"[deepspred-repro] {json.dumps({k: v for k, v in r.items() if k not in ('trace', 'log')}, default=str)[:600]} "
              f"[{len(pending)} left]", flush=True)
        if not r.get('ok'):
            print((r.get('trace') or r.get('log') or '')[-1200:], flush=True)


if __name__ == '__main__':
    main()
