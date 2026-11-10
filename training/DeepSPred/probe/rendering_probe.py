"""DeepSPred rendering probe: which I/Q -> image rendering puts trivial predictors at the paper's levels?

No training. For each candidate rendering of the authors' raw I/Q release (one .xls = one second =
one 256x256 frame), score frame persistence and the 20-frame lookback mean on the test split of the
FM and cross-validation sets, in the paper's metric space (uint8 pixels, mean over pixels and
channels, MAX_I = 255), at horizons k = 1..20 of input-20-predict-20 windows.

The paper's published levels (Pan et al., TWC 2025, Figs. 7-8, Appendix A) to compare against:
  FM:        ConvLSTM ~418 at k=1 rising to ~460 at k=20; DCG/NN-ResNet/SAE-TSS flat ~460-485;
             3D-SwinSTB ~440 -> ~455 (448.76 at k=8); global SSIM ~0.76-0.78.
  Crossval:  baselines ~575-612, 3D-SwinSTB ~526-538 (531.43 at k=12).
  Grayscale: 3D-SwinSTB 5185 at k=10 (Appendix A).
If a rendering is right, trivial predictors should land near the *baseline* levels in BOTH FM and
crossval (the paper's ratio crossval/FM is ~1.2-1.3).

Candidate renderings are labelled <window>_<colour scaling>_<post>[_gray]:
  window   hann126  : Hann, nperseg 256, hop 126 (256 columns from 32508 samples; collaborator's choice)
           hamm128  : Hamming, hop 128 (MATLAB spectrogram defaults: 252 columns, resized to 256)
           rect126  : rectangular window, hop 126
  scaling  glob     : dB, min/max from every 50th second of the whole dataset (collaborator's choice)
           frame    : dB, min/max of each frame (MATLAB imagesc default colour limits)
           p1p99    : dB, global 1st-99th percentile clip
           lin      : linear power, global min/max
  post     raw / jpegQ (JPEG round trip at quality Q; MATLAB imwrite default is 75) / blur1 (Gaussian
           sigma 1 on the scaled field before the colormap)

Run (UC cluster, from the repo root):
  python training/DeepSPred/probe/rendering_probe.py --data-root data/deepspred
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SIZE = 256
T_IN = 20
K_OUT = 20
SAMPLE_EVERY = 50
DATASETS = {"fm": "FM-Spectrum-Dataset", "crossval": "Cross-Validation-Spectrum-Dataset"}
WINDOWS = ("hann126", "hamm128", "rect126", "tukey126", "tukey126d", "tukey224")
VARIANTS = [
    # (name, window, scaling, post, gray)
    ("hann126_glob_raw", "hann126", "glob", "raw", False),        # collaborator raw (sanity: persistence ~1559-1615)
    ("hann126_glob_jpeg95", "hann126", "glob", "jpeg95", False),  # collaborator JPEG (sanity: ~420-465)
    ("hann126_glob_jpeg75", "hann126", "glob", "jpeg75", False),
    ("hann126_glob_blur1", "hann126", "glob", "blur1", False),
    ("hann126_frame_raw", "hann126", "frame", "raw", False),
    ("hann126_frame_jpeg75", "hann126", "frame", "jpeg75", False),
    ("hann126_p1p99_raw", "hann126", "p1p99", "raw", False),
    ("hann126_lin_raw", "hann126", "lin", "raw", False),
    ("rect126_glob_raw", "rect126", "glob", "raw", False),
    ("hamm128_glob_raw", "hamm128", "glob", "raw", False),
    ("hamm128_glob_jpeg75", "hamm128", "glob", "jpeg75", False),
    ("hamm128_frame_raw", "hamm128", "frame", "raw", False),
    ("hamm128_frame_jpeg75", "hamm128", "frame", "jpeg75", False),  # MATLAB spectrogram + imagesc + imwrite defaults
    ("hann126_glob_raw_gray", "hann126", "glob", "raw", True),
    ("hann126_glob_jpeg75_gray", "hann126", "glob", "jpeg75", True),
]

_TS = re.compile(r"-(\d{4})-(\d{4})-(\d{6})\.xls$")


# ---------------------------------------------------------------- data -----

def list_seconds(directory: Path):
    out = []
    for p in directory.rglob("*.xls"):
        m = _TS.search(p.name)
        if m:
            out.append((datetime.strptime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m%d-%H%M%S"), p))
    if not out:
        raise FileNotFoundError(f"no timestamped .xls under {directory}")
    out.sort(key=lambda t: t[0])
    return out


def read_iq(path: Path) -> np.ndarray:
    a = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 2:
                a.append((float(parts[0]), float(parts[1])))
    a = np.asarray(a, dtype=np.float32)
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)


def _win(name: str, n: int) -> np.ndarray:
    from scipy.signal import get_window
    if name.startswith("rect"):
        return np.ones(n, dtype=np.float32)
    if name.startswith("tukey"):  # scipy.signal.spectrogram's default window, ('tukey', 0.25)
        return get_window(("tukey", 0.25), n).astype(np.float32)
    return get_window("hann" if name.startswith("hann") else "hamming", n).astype(np.float32)


def spectrogram_db(x: np.ndarray, window: str) -> np.ndarray:
    """10*log10 |STFT|^2 as (time, freq) = (256, 256); two-sided, fftshifted (complex I/Q).

    `window` = <shape><hop>[d]: shape in rect/hann/hamm/tukey, hop 126 = derived so that exactly 256
    columns fit in the file, any other number = that hop with the time axis resized to 256 columns;
    a trailing 'd' subtracts each segment's complex mean first (scipy detrend='constant').
    """
    nper = SIZE
    detrend = window.endswith("d")
    base = window[:-1] if detrend else window
    tail = int(base.lstrip("abcdefghijklmnopqrstuvwxyz"))
    if tail == 126:
        hop = (len(x) - nper) // (SIZE - 1)
        starts = np.arange(SIZE) * hop
    else:  # fixed hop, then resize the time axis to 256 (MATLAB default 128, scipy default 224)
        hop = tail
        starts = np.arange((len(x) - nper) // hop + 1) * hop
    w = _win(base, nper)
    if detrend:
        seg = np.stack([(x[s:s + nper] - x[s:s + nper].mean()) * w for s in starts])
    else:
        seg = np.stack([x[s:s + nper] * w for s in starts])
    S = np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)
    db = 10.0 * np.log10(np.abs(S) ** 2 + 1e-12).astype(np.float32)  # (n_time, 256)
    if db.shape[0] != SIZE:
        from scipy.ndimage import zoom
        db = zoom(db, (SIZE / db.shape[0], 1.0), order=1).astype(np.float32)
    return db


def _file_db(path_str: str):
    x = read_iq(Path(path_str))
    return {w: spectrogram_db(x, w) for w in WINDOWS}


def load_db(paths, workers: int):
    import importlib
    import multiprocessing as mp
    # Inside a Ray task this module is not importable as __main__, so hand the pool the function
    # from the file imported under its own name.
    probe_dir = str(Path(__file__).resolve().parent)
    if probe_dir not in sys.path:
        sys.path.insert(0, probe_dir)
    fn = importlib.import_module("rendering_probe")._file_db
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(fn, [str(p) for p in paths], chunksize=8)
    return {w: np.stack([r[w] for r in res]) for w in WINDOWS}


def test_split(times):
    """Paper 4:1:1 chronological split by time; returns test indices and their contiguous segments."""
    t0, span = times[0], (times[-1] - times[0]).total_seconds()
    va_cut = t0 + timedelta(seconds=span * 5.0 / 6.0)
    idx = [i for i, t in enumerate(times) if t >= va_cut]
    segs, s = [], 0
    for k in range(1, len(idx)):
        if (times[idx[k]] - times[idx[k - 1]]).total_seconds() != 1.0:
            segs.append((s, k))
            s = k
    segs.append((s, len(idx)))
    return idx, segs


# ----------------------------------------------------------- rendering -----

def render(db: np.ndarray, scaling: str, post: str, gray: bool, stats: dict) -> np.ndarray:
    """(N, 256, 256) dB -> (N, 256, 256, C) uint8."""
    import matplotlib
    if scaling == "glob":
        n = (db - stats["lo"]) / (stats["hi"] - stats["lo"])
    elif scaling == "p1p99":
        n = (db - stats["p1"]) / (stats["p99"] - stats["p1"])
    elif scaling == "lin":
        p = np.power(10.0, db / 10.0, dtype=np.float64)
        n = (p - stats["plo"]) / (stats["phi"] - stats["plo"])
    elif scaling == "frame":
        lo = db.min(axis=(1, 2), keepdims=True)
        hi = db.max(axis=(1, 2), keepdims=True)
        n = (db - lo) / np.maximum(hi - lo, 1e-12)
    else:
        raise ValueError(scaling)
    n = np.clip(n, 0.0, 1.0).astype(np.float32)
    if post.startswith("blur"):
        from scipy.ndimage import gaussian_filter
        sigma = float(post[4:])
        n = np.stack([gaussian_filter(f, sigma) for f in n])
    if gray:
        img = (n * 255.0).round().astype(np.uint8)[..., None]
    else:
        img = (matplotlib.colormaps["jet"](n)[..., :3] * 255.0).round().astype(np.uint8)
    if post.startswith("jpeg"):
        from PIL import Image
        q = int(post[4:])
        out = np.empty_like(img)
        for i, f in enumerate(img):
            buf = io.BytesIO()
            Image.fromarray(f[..., 0] if gray else f, mode="L" if gray else "RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            g = np.asarray(Image.open(buf))
            out[i] = g[..., None] if gray else g
        img = out
    return img


# ------------------------------------------------------------- metrics -----

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. (32) read literally: one mean/variance/covariance per image and channel, then averaged."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = []
    for ch in range(a.shape[-1]):
        x = a[..., ch].astype(np.float64).ravel()
        y = b[..., ch].astype(np.float64).ravel()
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cxy = ((x - mx) * (y - my)).mean()
        out.append((2 * mx * my + c1) * (2 * cxy + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return float(np.mean(out))


def score(frames: np.ndarray, segs) -> dict:
    """Persistence and lookback-mean MSE at k = 1..K over all stride-1 windows inside contiguous segments."""
    pers = np.zeros(K_OUT)
    lbm = np.zeros(K_OUT)
    ssim18 = []
    n_win = 0
    for a, b in segs:
        L = b - a
        if L < T_IN + K_OUT:
            continue
        seg = frames[a:b]
        run = seg[:T_IN].astype(np.float64).sum(axis=0)
        for s in range(L - T_IN - K_OUT + 1):
            if s > 0:
                run += seg[s + T_IN - 1].astype(np.float64) - seg[s - 1].astype(np.float64)
            last = seg[s + T_IN - 1]
            mean = run / T_IN
            for k in range(1, K_OUT + 1):
                tgt = seg[s + T_IN - 1 + k]
                pers[k - 1] += _mse(last, tgt)
                lbm[k - 1] += _mse(mean, tgt)
            if s % 5 == 0:
                ssim18.append(_global_ssim(last, seg[s + T_IN - 1 + 18]))
            n_win += 1
    pers /= max(n_win, 1)
    lbm /= max(n_win, 1)
    return {"n_windows": n_win, "persistence_mse": pers.round(2).tolist(), "lookback_mean_mse": lbm.round(2).tolist(),
            "persistence_global_ssim_k18": round(float(np.mean(ssim18)), 4) if ssim18 else None}


# --------------------------------------------------------------- driver -----

def run_dataset(root: Path, name: str, workers: int) -> dict:
    t0 = time.time()
    seconds = list_seconds(root / DATASETS[name])
    times = [t for t, _ in seconds]
    paths = [p for _, p in seconds]
    idx, segs = test_split(times)
    print(f"[{name}] {len(paths)} files; test {len(idx)} frames in {len(segs)} segments", flush=True)
    rng_db = load_db(paths[::SAMPLE_EVERY], workers)
    test_db = load_db([paths[i] for i in idx], workers)
    print(f"[{name}] loaded I/Q in {time.time() - t0:.0f}s", flush=True)
    stats = {}
    for w in WINDOWS:
        r = rng_db[w]
        p = np.power(10.0, r / 10.0, dtype=np.float64)
        stats[w] = {"lo": float(r.min()), "hi": float(r.max()), "p1": float(np.percentile(r, 1)),
                    "p99": float(np.percentile(r, 99)), "plo": float(p.min()), "phi": float(p.max())}
    out = {"n_files": len(paths), "n_test": len(idx), "stats": stats, "variants": {}}
    for vname, w, scaling, post, gray in VARIANTS:
        t1 = time.time()
        img = render(test_db[w], scaling, post, gray, stats[w])
        res = score(img, segs)
        out["variants"][vname] = res
        p, l = res["persistence_mse"], res["lookback_mean_mse"]
        print(f"[{name}] {vname:26s} pers k1/8/12/20 {p[0]:8.1f} {p[7]:8.1f} {p[11]:8.1f} {p[19]:8.1f} | "
              f"lbm k1/8/20 {l[0]:8.1f} {l[7]:8.1f} {l[19]:8.1f} | gSSIM18 {res['persistence_global_ssim_k18']} "
              f"({time.time() - t1:.0f}s)", flush=True)
    return out


def work(data_root: str, out_dir: str, workers: int) -> dict:
    root = Path(data_root)
    result = {"created": datetime.utcnow().isoformat() + "Z", "host": os.uname().nodename, "t_in": T_IN, "k_out": K_OUT,
              "metric": "uint8 MSE, mean over pixels and channels", "datasets": {}}
    for name in DATASETS:
        if (root / DATASETS[name]).exists():
            result["datasets"][name] = run_dataset(root, name, workers)
        else:
            print(f"[{name}] missing under {root}; skipped", flush=True)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"rendering_probe_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path} on {result['host']}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/deepspred")
    ap.add_argument("--out-dir", default="data/deepspred/probe")
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--local", action="store_true", help="run in this process instead of a pinned Ray task")
    args = ap.parse_args()
    data_root = str(Path(args.data_root).resolve())
    out_dir = str(Path(args.out_dir).resolve())
    if args.local:
        work(data_root, out_dir, args.cpus)
        return
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"] and n.get("Resources", {}).get("CPU", 0) >= args.cpus]

    def has_data(path):
        return Path(path, DATASETS["fm"]).exists()

    probes = {n["NodeID"]: ray.remote(num_cpus=0.01, scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(has_data).remote(data_root)
              for n in nodes}
    ok = []
    for nid, ref in probes.items():
        try:
            if ray.get(ref, timeout=120):
                ok.append(nid)
        except Exception as exc:  # node unschedulable or busy: skip it
            print(f"probe on {nid[:8]} failed: {type(exc).__name__}", flush=True)
    if not ok:
        sys.exit(f"no node has {data_root}/{DATASETS['fm']}")
    print(f"data present on {len(ok)} node(s); using {ok[0]}", flush=True)
    task = ray.remote(num_cpus=args.cpus, num_gpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(ok[0], soft=False))(work)
    ray.get(task.remote(data_root, out_dir, args.cpus))


if __name__ == "__main__":
    main()
