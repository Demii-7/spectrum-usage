'''Load the DeepSPred authors' release: raw per-second I/Q -> 256x256 RGB spectrograms.

The release is NOT rendered spectrograms, despite the data README saying "these spectrograms
constitute the FM dataset". Each .xls is one second of tab-separated raw I/Q:

    row 0 : I <TAB> Q <TAB> hhmmss <TAB> longitude <TAB> latitude <TAB> altitude
    rows+ : I <TAB> Q

Sample count equals the paper's "STFT number": 32508 for FM, 16254 for LTE [P:L 1787-1795].
Row-0 coordinates match the paper's sensor to six decimals.

Fetch: the authors' released Google Drive folder. Upstream declares no licence.

VERIFIED (2026-11-01/03)
  * window 256 with the hop below gives exactly 256 time frames; a two-sided FFT gives 256 bins
  * FM: fs = 125 MHz / 4 = 31.25 MHz about 99 MHz -> 83.38..114.50 MHz, bracketing the paper's
    90-110 MHz band, with the strongest bins on real FM stations near 89 MHz
  * linear power spans ~122 dB, so dB scaling before 8-bit quantisation is unavoidable

NOT STATED BY THE PAPER - every one of these changes the pixel values and hence MSE/PSNR:
window type, one- vs two-sided FFT, dB vs linear, normalisation scope, colormap, training stride.
They are switches here, defaulted to the most defensible reading, and each is recorded per run.

THE RELEASE IS NOT GAP-FREE. Measured from the filenames: FM is missing 7 seconds inside its
span and the cross-validation set 234; only LTE is contiguous. The paper's chronological split
and sliding windows assume one sample per second, so windows that straddle a gap must be dropped
-- use `contiguous_segments()` and keep only windows wholly inside one segment.
'''
from __future__ import annotations
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy.signal import get_window

_TS = re.compile('-(\\d{4})-(\\d{4})-(\\d{6})\\.xls$')
FS_HZ = 3.125e+07
CENTRE_HZ = {'fm': 9.9e+07, 'crossval': 9.9e+07, 'lte': 7e+08}


def list_seconds(directory):
    '''Every .xls in `directory`, sorted chronologically by its filename timestamp.'''
    out = []
    for p in Path(directory).rglob('*.xls'):
        m = _TS.search(p.name)
        if m:
            out.append((datetime.strptime(f'''{m.group(1)}-{m.group(2)}-{m.group(3)}''', '%Y-%m%d-%H%M%S'), p))
    if not out:
        raise FileNotFoundError(f'''no timestamped .xls under {directory}''')
    out.sort(key=lambda t: t[0])
    return out


def contiguous_segments(times):
    '''Half-open [start, end) index runs whose timestamps step by exactly one second.'''
    segs, s = [], 0
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]).total_seconds() != 1.0:
            segs.append((s, i))
            s = i
    segs.append((s, len(times)))
    return segs


def read_iq(source):
    '''Return (complex64 samples, metadata). `source` is a path or the raw bytes of one .xls.'''
    raw = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    lines = raw.decode('utf-8', 'replace').splitlines()
    first = lines[0].split('\t')
    meta = {'hhmmss': first[2], 'lon': float(first[3]), 'lat': float(first[4]), 'alt': float(first[5]), 'n': len(lines)}
    a = np.empty((len(lines), 2), dtype=np.float32)
    a[0] = (float(first[0]), float(first[1]))
    for k in range(1, len(lines)):
        p = lines[k].split('\t')
        a[k, 0] = float(p[0])
        a[k, 1] = float(p[1])
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64), meta


def spectrogram(x, size=256, window='hann', two_sided=True):
    '''|STFT|^2 as (freq, time) = (size, size). Paper eq. (2).

    The hop is derived so the frame count is exactly `size`, which is what makes a 256x256
    image out of 32508 (FM) or 16254 (LTE) samples. The paper gives the window length (256)
    but never the hop; this is the only hop consistent with its stated output shape.
    '''
    nper = size
    hop = (len(x) - nper) // (size - 1)
    if hop < 1:
        raise ValueError(f'''{len(x)} samples cannot yield {size} frames of {nper}''')
    w = np.ones(nper, dtype=np.float32) if window == 'rect' else get_window(window, nper)
    seg = np.stack([x[i:i + nper] * w for i in np.arange(size) * hop])
    S = np.fft.fft(seg, axis=1)
    S = np.fft.fftshift(S, axes=1) if two_sided else S[:, :nper // 2 + 1]
    return (np.abs(S) ** 2).T.astype(np.float32)


def to_rgb(power, cmap='jet', scale='db', vmin=None, vmax=None):
    '''(F,T) power -> (F,T,3) uint8 via dB compression, min-max normalisation and a colormap.

    Pass dataset-wide `vmin`/`vmax` in dB so every frame shares one scale. Omitting them
    normalises per frame, which destroys absolute level information - a switch, because the
    paper specifies neither.
    '''
    import matplotlib
    v = 10.0 * np.log10(power + 1e-12) if scale == 'db' else power
    lo = v.min() if vmin is None else vmin
    hi = v.max() if vmax is None else vmax
    n = np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return (matplotlib.colormaps[cmap](n)[..., :3] * 255.0).round().astype(np.uint8)


def db_range(paths, sample_every=50, **kw):
    '''Dataset-wide dB min/max from every `sample_every`-th second, for a shared colour scale.'''
    lo, hi = np.inf, -np.inf
    for p in paths[::sample_every]:
        x, _ = read_iq(p)
        v = 10.0 * np.log10(spectrogram(x, **kw) + 1e-12)
        lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
    return lo, hi


def _file_to_frame(item):
    """Worker: one .xls -> (H, W, C) uint8, H = STFT time, W = frequency  [P:L 409-412].

    channels=3 renders the colormap RGB image; channels=1 renders the same normalised dB
    value through the 'gray' colormap and keeps a single channel (the paper's Appendix-A
    grayscale ablation).
    """
    (path, size, window, two_sided, cmap, vmin, vmax, jpeg_quality, channels) = item
    x, _ = read_iq(path)
    power = spectrogram(x, size=size, window=window, two_sided=two_sided)
    if channels == 1:
        cmap = 'gray'
    rgb = to_rgb(power, cmap=cmap, scale='db', vmin=vmin, vmax=vmax)
    frame = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    if channels == 1:
        frame = frame[..., :1]
    if jpeg_quality is not None:
        import io
        from PIL import Image
        im = Image.fromarray(frame[..., 0] if frame.shape[-1] == 1 else frame, mode='L' if frame.shape[-1] == 1 else 'RGB')
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=jpeg_quality)
        buf.seek(0)
        got = np.asarray(Image.open(buf))
        frame = got[..., None] if got.ndim == 2 else got
    return frame


def _iso(t):
    return t.isoformat()


def load_frames(directory, expect_hw=(256, 256), *, cache=True, cache_dir=None, workers=None,
                window='hann', two_sided=True, cmap='jet', sample_every=50, jpeg_quality=None,
                channels=3):
    '''Load every second under `directory` as one (H, W, 3) uint8 frame, in filename time order.

    Returns ``(frames, meta)`` where frames is (N, H, W, 3) uint8 with H = STFT time sub-steps
    and W = frequency bins  [P:L 409-412, L 1811], and meta records the preprocessing switches
    the paper leaves open (fm_data module docstring): window type, one- vs two-sided FFT, dB
    range (dataset-wide, sampled every `sample_every`-th second), colormap, JPEG round-trip
    quality (None = raw pixels), plus the per-frame timestamps and the contiguous segments.

    Caches the array next to the cache dir (default: data/datasets/deepspred/frames/, gitignored) keyed by the
    directory name and the switches; the cache is invalidated by any change in either.
    '''
    import os
    directory = Path(directory)
    cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent.parents[2] / 'data' / 'datasets' / 'deepspred' / 'frames'
    suffix = '' if jpeg_quality is None else f'''_jpeg{jpeg_quality}'''
    if channels == 1:
        suffix = '_gray' + suffix
    stem = f'''{directory.resolve().name}_frames_{expect_hw[0]}x{expect_hw[1]}{suffix}'''
    npy = cache_dir / f'''{stem}.npy'''
    jsn = cache_dir / f'''{stem}_meta.json'''
    if not directory.exists() and npy.exists() and jsn.exists():
        meta = json.loads(jsn.read_text())
        frames = np.load(npy, mmap_mode='r')
        if frames.shape[1:] == (expect_hw[0], expect_hw[1], channels):
            print(f'''cache-only mode: {npy} ({frames.shape[0]} frames; source dir absent)''', flush=True)
            return frames, meta
    seconds = list_seconds(directory)
    times = [t for t, _ in seconds]
    paths = [p for _, p in seconds]
    switches = {'window': window, 'two_sided': two_sided, 'cmap': cmap, 'size': int(expect_hw[0]),
                'sample_every': int(sample_every), 'jpeg_quality': jpeg_quality, 'channels': channels}
    if cache and npy.exists() and jsn.exists():
        meta = json.loads(jsn.read_text())
        if meta.get('switches') == switches and meta.get('n_files') == len(paths) and meta.get('source') == str(directory.resolve()):
            frames = np.load(npy, mmap_mode='r')
            if frames.shape[1:] == (expect_hw[0], expect_hw[1], channels):
                print(f'''loaded cache {npy} ({frames.shape[0]} frames)''', flush=True)
                return frames, meta
    lo, hi = db_range(paths, sample_every=sample_every, size=expect_hw[0], window=window, two_sided=two_sided)
    items = [(p, expect_hw[0], window, two_sided, cmap, lo, hi, jpeg_quality, channels) for p in paths]
    n_workers = workers or min(32, os.cpu_count() or 1)
    if n_workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(n_workers) as pool:
            frames = pool.map(_file_to_frame, items, chunksize=16)
    else:
        frames = [_file_to_frame(it) for it in items]
    frames = np.stack(frames)
    segs = contiguous_segments(times)
    meta = {'source': str(directory.resolve()), 'n_files': len(paths), 'first': _iso(times[0]),
            'last': _iso(times[-1]), 'times': [_iso(t) for t in times],
            'segments': [[int(a), int(b)] for a, b in segs], 'db_range': [float(lo), float(hi)],
            'switches': switches, 'axes': 'H = STFT time sub-steps, W = frequency bins [P:L 409-412]'}
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, frames)
        jsn.write_text(json.dumps(meta))
        print(f'''wrote cache {npy} ({frames.shape[0]} frames)''', flush=True)
    return frames, meta


def chronological_split(frames, meta=None):
    """The paper's 4:1:1 chronological split  [P:L 1788-1793].

    FM is named exactly: 7200 train (17:20-19:20) / 1800 val (19:20-19:50) /
    1800 test (19:50-20:20), i.e. the boundaries sit at 2/3 and 5/6 of the span.
    With per-frame timestamps in `meta` the boundaries are placed by TIME at those
    fractions of the observed span; without them, by COUNT (7200/1800/1800 when
    N = 10800). The release is missing seconds inside its span, so windows that
    straddle a gap or a split boundary must be dropped by the caller using the
    returned per-split segments.

    Returns ((train, val, test), segments) with segments a dict of half-open
    (start, end) index pairs local to each split array.
    """
    n = len(frames)
    if meta and meta.get('times'):
        from datetime import datetime
        times = [datetime.fromisoformat(t) for t in meta['times']]
        t0, span = times[0], (times[-1] - times[0]).total_seconds()
        tr_cut = t0 + timedelta(seconds=span * 2.0 / 3.0)
        va_cut = t0 + timedelta(seconds=span * 5.0 / 6.0)
        i_tr = [i for i, t in enumerate(times) if t < tr_cut]
        i_va = [i for i, t in enumerate(times) if tr_cut <= t < va_cut]
        i_te = [i for i, t in enumerate(times) if t >= va_cut]

        def _segs(idxs):
            out, s = [], 0
            for k in range(1, len(idxs)):
                if (times[idxs[k]] - times[idxs[k - 1]]).total_seconds() != 1.0:
                    out.append((s, k))
                    s = k
            out.append((s, len(idxs)))
            return out
    else:
        n_tr = int(round(n * 2.0 / 3.0))
        n_va = int(round(n / 6.0))
        i_tr = list(range(0, n_tr))
        i_va = list(range(n_tr, n_tr + n_va))
        i_te = list(range(n_tr + n_va, n))

        def _segs(idxs):
            return [(0, len(idxs))]

    splits = (frames[i_tr], frames[i_va], frames[i_te])
    segments = {'train': _segs(i_tr), 'val': _segs(i_va), 'test': _segs(i_te)}
    return splits, segments
