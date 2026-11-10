# Running the three leading renderings in the paper-faithful runner

For the collaborator. Goal: train the paper-faithful 3D-SwinSTB (same recipe as `repro/run_paper_faithful.py`) on three alternative renderings of the FM I/Q, and
report the frame-wise test MSE, so we can see which rendering the paper's curves are consistent with.
Everything else stays exactly as in the existing reproduction (C=96, depths 2-4-2, patch 2-4-4, window 2-7-7, lr 1e-3, batch 1, 20 epochs, early stopping, K=T=20, 4:1:1 split).

## 0. What is different from the current runs (three things)

1. **Range for the min-max scaling: all frames, not every 50th.** `load_frames` currently uses `sample_every=50`. That shortcut moves the level by up to 20-40% (the extreme pixels set the range). Use the whole dataset (`--range-scope full`).
2. **STFT window / hop / resize / crop** are chosen by a `--render-spec` string (below). These are not CLI flags today; `run_paper_faithful.py` always uses Hann, hop 126.
3. **Run key and frame cache must include the rendering.** Today neither does: a second rendering would be skipped as "already done" (same run key) and the cached frames would be overwritten. The patch below fixes both.

Everything else in the rendering is as before: complex I/Q (one `.xls` = one second = 32,508 samples), 256-sample window, two-sided fftshifted spectrum,
`10*log10(|S|^2 + 1e-12)`, min-max to [0,1], clip, matplotlib `jet`, uint8, no JPEG, 3 channels. Frame axes are (H = STFT time rows, W = frequency bins).

## 1. The renderings (`--render-spec`)

| # | Setting | Spec | What it does exactly |
|---|---|---|---|
| 1a | Rectangular, hop 126 | `rect126` | window = all ones, hop `(N-256)//255 = 126` so exactly 256 rows, no padding, no resize (the existing `window='rect'` path) |
| 1b | Rectangular, torch-style | `rect127c~12` | `torch.stft(x, n_fft=256, hop_length=127, return_complex=True)` defaults: window None (rectangular), `center=True` reflect padding of 128 each side, 256 rows exactly; dB floor 1e-12 |
| 2 | Tukey hop 224, band crop A | `tukey224@54-218` | Tukey(0.25), hop 224 (144 rows) resized to 256 rows (bilinear), no detrend; keep frequency bins 54..217 (164 bins = 90-110 MHz if the spectrum is not mirrored), resized back to 256 columns (bilinear) |
| 3 | scipy.signal.spectrogram defaults | `tukey224d` | `scipy.signal.spectrogram(x, nperseg=256)`: Tukey(0.25), hop 224, `detrend='constant'` (segment mean removed), 144 rows resized to 256 rows (bilinear), 256 columns |

The bilinear resize is `scipy.ndimage.zoom(db, (256/rows, 1), order=1)` (corner-aligned). cv2 or torch bilinear use half-pixel centres and differ slightly; the level changes by up to about 20%
between nearest, bilinear and cubic, so use the same function as in the probe to reproduce our numbers.

## 2. `training/DeepSPred/repro/fm_data.py`: `load_frames_spec` (ALREADY APPLIED on `integrate`; shown for reference)

Requires `training/DeepSPred/probe/rendering_probe.py` (the renderer that produced all the numbers in `ANALYSIS_NOTES.md`); it is added to `sys.path` inside the function. Pull the `integrate` branch and the commands in section 4 work as written.
Tested on synthetic data: the new `rect126` path reproduces `load_frames(window='rect', sample_every=1)` exactly (0 differing pixels), the cache reloads, all four specs run, and `chronological_split` accepts the metadata. It has not been run on the real FM data or through a training step.

```python
def _spec_minmax(args):
    path, spec = args
    import rendering_probe as rp
    x, _ = read_iq(path)
    db = rp.spectrogram_db(x, spec)
    return float(db.min()), float(db.max())


def _spec_render(args):
    path, spec, lo, hi = args
    import rendering_probe as rp
    x, _ = read_iq(path)
    return rp.render(rp.spectrogram_db(x, spec)[None], 'glob', 'raw', False, {'lo': lo, 'hi': hi})[0]


def load_frames_spec(directory, spec, *, range_scope='full', cache_dir=None, workers=None):
    """Frames rendered with the probe's renderer. range_scope: 'full' = min/max over all frames, 'train' = first 2/3 of the capture (by time)."""
    import hashlib, os, multiprocessing as mp
    here = Path(__file__).resolve().parent
    if str(here.parent / 'probe') not in sys.path:
        sys.path.insert(0, str(here.parent / 'probe'))
    directory = Path(directory)
    seconds = list_seconds(directory)
    times, paths = [t for t, _ in seconds], [p for _, p in seconds]
    tag = hashlib.sha1(f'{spec}|{range_scope}'.encode()).hexdigest()[:8]
    cache_dir = Path(cache_dir) if cache_dir else here.parents[1] / 'data' / 'datasets' / 'deepspred' / 'frames'
    stem = f'{directory.resolve().name}_frames_256x256_spec{tag}'
    npy, jsn = cache_dir / f'{stem}.npy', cache_dir / f'{stem}_meta.json'
    if npy.exists() and jsn.exists():
        return np.load(npy, mmap_mode='r'), json.loads(jsn.read_text())
    n_workers = workers or min(32, os.cpu_count() or 1)
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        mm = pool.map(_spec_minmax, [(str(p), spec) for p in paths], chunksize=16)
    cut = times[0] + (times[-1] - times[0]) * (2.0 / 3.0)
    sel = [True] * len(paths) if range_scope == 'full' else [t < cut for t in times]
    lo = min(a for (a, _), s in zip(mm, sel) if s)
    hi = max(b for (_, b), s in zip(mm, sel) if s)
    with ctx.Pool(n_workers) as pool:
        frames = np.stack(pool.map(_spec_render, [(str(p), spec, lo, hi) for p in paths], chunksize=16))
    segs = contiguous_segments(times)
    meta = {'source': str(directory.resolve()), 'n_files': len(paths), 'first': _iso(times[0]), 'last': _iso(times[-1]),
            'times': [_iso(t) for t in times], 'segments': [[int(a), int(b)] for a, b in segs], 'db_range': [lo, hi],
            'switches': {'render_spec': spec, 'range_scope': range_scope, 'size': 256, 'jpeg_quality': None, 'channels': 3},
            'axes': 'H = STFT time rows, W = frequency bins'}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, frames)
    jsn.write_text(json.dumps(meta))
    return frames, meta
```

(`json`, `np`, `Path`, `sys`, `list_seconds`, `read_iq`, `contiguous_segments`, `_iso` already exist in `fm_data.py`; add `import sys` if it is not imported.)

## 3. `training/DeepSPred/repro/run_paper_faithful.py`: `--render-spec` / `--range-scope` (ALREADY APPLIED on `integrate`; shown for reference)

```python
from fm_data import load_frames, load_frames_spec, chronological_split          # was: load_frames, chronological_split

ap.add_argument('--render-spec', default=None, help="rendering_probe spec, e.g. rect126, rect127c~12, tukey224@54-218, tukey224d (default: original Hann hop 126)")
ap.add_argument('--range-scope', default='full', choices=['full', 'train'], help="dB range for min-max: all frames or the first 2/3 (only with --render-spec)")

# right after the existing `key = run_key(...)` line, BEFORE the key is used for the checkpoint hash:
if args.render_spec:
    key += f'|spec={args.render_spec}|{args.range_scope}'

# where frames are loaded (replace the single load_frames call):
if args.render_spec:
    frames, fmeta = load_frames_spec(args.data, args.render_spec, range_scope=args.range_scope)
else:
    frames, fmeta = load_frames(args.data, expect_hw=(H, W), jpeg_quality=args.jpeg_quality, channels=channels)
```

## 4. Commands (FM, K=T=20, seed 0; add `--seed 1` for a second seed if time allows)

```bash
cd training/DeepSPred/repro
python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --render-spec rect126          --range-scope full   # 1a
python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --render-spec rect127c~12      --range-scope full   # 1b
python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --render-spec tukey224@54-218  --range-scope full   # 2
python run_paper_faithful.py --data data/fm --horizon 20 --seed 0 --render-spec tukey224d        --range-scope full   # 3
```

Optional (cross-validation, each with its own range): the same commands with `--dataset crossval --data data/cv`.
Cost (estimate, not measured on your GPU): about 0.25 s per step x 7,122 training windows, so roughly 30 minutes per epoch and 2.5-4.5 hours per run with early stopping (epochs 5-9). The four runs can go on separate GPUs.

## 5. Sanity check before training (a minute)

Look at the `trivial_references` that each run records. The lookback-mean MSE at frame 8 on the FM test block should come out near (within about 3%):

| spec | lookback mean @ k=8 |
|---|---|
| `rect126` | 340 |
| `rect127c~12` | 294 |
| `tukey224@54-218` | 464 |
| `tukey224d` | 481 |

If yours differ by more than about 5%, the rendering differs from ours (range scope, resize function, hop, or padding): fix that before training.
Our probes show that a trained model ends up within a few percent of the lookback mean (the existing raw-Hann run: 846 trained vs 858 lookback mean), so training mainly tests whether the architecture's
horizon curve (rise from k=1 to k=20, paper about +3.6%) is reproduced, not the absolute level.

## 6. What to send back

For each run: the record from `results.json` (frame-wise MSE for k=1..20 in `uint8_mean_per_channel`, persistence and lookback-mean rows, epochs run, `db_range`), plus the training time.
Paper anchors for comparison: FM 448.76 at k=8 (453.04 at k=16, 454.18 at k=18); rise about +3.6% from k=1 to k=20 (plot reading).
