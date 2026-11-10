"""DeepSPred rendering probe 2: temporal structure and linear-predictability ceiling per rendering.

No persistence, no training. For each candidate rendering of the authors' raw I/Q (see
rendering_probe.py for the rendering definitions), on a fixed random subset of pixels:

  clim_mse   MSE of predicting every test frame with the fit-set mean image (zero-skill reference).
  rho[k]     lag-k autocorrelation of the deviation from that mean image, k = 1..20 (test region,
             contiguous segments only). rho[1] near 0 means consecutive frames share nothing the
             image-level noise has not destroyed; a slowly decaying rho is what produces
             "better at short horizons than long".
  ridge[k]   out-of-sample MSE at horizon k of the best linear predictor of the deviation at t+k from
             the deviations at t, t-1, ..., t-4 (one coefficient set per k, shared across pixels, fitted
             on the fit region by ridge-regularised autocovariance equations). This is the skill-vs-horizon
             curve a good model could reach in that rendering, with no reference to persistence.
  skill[k]   1 - ridge[k] / clim_mse.

Fit region = frames between 2/3 and 5/6 of the capture span (the paper's validation block); test
region = the last 1/6 (the paper's test block). Means and coefficients never see the test region.

Extra smoothing variants (avgM): linear-power box average over M adjacent STFT rows of a frame
(Welch-style averaging along the frame's time axis), applied before dB scaling.

Run (UC cluster, repo root):
  python training/DeepSPred/probe/acf_probe.py --data-root data/deepspred
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rendering_probe as rp  # noqa: E402

K_MAX = 20
L_IN = 5
N_PIX = 6000
RIDGE = 1e-3
SEED = 0

# name -> (window, row-average M or 0, scaling, post)
VARIANTS = {
    "hann126_glob_raw": ("hann126", 0, "glob", "raw"),
    "hann126_glob_jpeg95": ("hann126", 0, "glob", "jpeg95"),
    "hann126_glob_jpeg75": ("hann126", 0, "glob", "jpeg75"),
    "hann126_glob_jpeg50": ("hann126", 0, "glob", "jpeg50"),
    "hann126_glob_jpeg25": ("hann126", 0, "glob", "jpeg25"),
    "hann126_glob_blur1": ("hann126", 0, "glob", "blur1"),
    "hann126_glob_blur2": ("hann126", 0, "glob", "blur2"),
    "hann126_frame_raw": ("hann126", 0, "frame", "raw"),
    "hann126_frame_jpeg75": ("hann126", 0, "frame", "jpeg75"),
    "hann126_p1p99_raw": ("hann126", 0, "p1p99", "raw"),
    "hann126_lin_raw": ("hann126", 0, "lin", "raw"),
    "rect126_glob_raw": ("rect126", 0, "glob", "raw"),
    "hamm128_glob_raw": ("hamm128", 0, "glob", "raw"),
    "hamm128_frame_jpeg75": ("hamm128", 0, "frame", "jpeg75"),
    "hann126_avg4_glob_raw": ("hann126", 4, "glob", "raw"),
    "hann126_avg16_glob_raw": ("hann126", 16, "glob", "raw"),
    "hann126_avg64_glob_raw": ("hann126", 64, "glob", "raw"),
    "hann126_avg4_glob_jpeg75": ("hann126", 4, "glob", "jpeg75"),
}

# Window study (scipy.signal.spectrogram's default window is Tukey(0.25), default hop 224 = nperseg - nperseg//8;
# 'd' = scipy's default detrend='constant'), with the earlier front-runners as references.
TUKEY_VARIANTS = {
    "tukey126_glob_raw": ("tukey126", 0, "glob", "raw"),
    "tukey126d_glob_raw": ("tukey126d", 0, "glob", "raw"),
    "tukey224_glob_raw": ("tukey224", 0, "glob", "raw"),
    "tukey224d_glob_raw": ("tukey224d", 0, "glob", "raw"),  # scipy.signal.spectrogram defaults (tukey .25, hop 224, detrend)
    "tukey224d_frame_raw": ("tukey224d", 0, "frame", "raw"),
    "tukey224d_glob_jpeg95": ("tukey224d", 0, "glob", "jpeg95"),
    "tukey224d_glob_jpeg75": ("tukey224d", 0, "glob", "jpeg75"),
    "tukey126_glob_jpeg95": ("tukey126", 0, "glob", "jpeg95"),
    "tukey126_glob_jpeg75": ("tukey126", 0, "glob", "jpeg75"),
    "tukey126_frame_raw": ("tukey126", 0, "frame", "raw"),
    "tukey126_avg4_glob_raw": ("tukey126", 4, "glob", "raw"),
    "rect126_glob_raw": ("rect126", 0, "glob", "raw"),
    "hann126_glob_raw": ("hann126", 0, "glob", "raw"),
    "hamm128_glob_raw": ("hamm128", 0, "glob", "raw"),
}
# Rectangular window, 256 columns (hop 126), everything else scipy-style (detrend='constant'), with references.
RECT_VARIANTS = {
    "rect126d_glob_raw": ("rect126d", 0, "glob", "raw"),
    "rect126d_frame_raw": ("rect126d", 0, "frame", "raw"),
    "rect126d_glob_jpeg95": ("rect126d", 0, "glob", "jpeg95"),
    "rect126d_glob_jpeg75": ("rect126d", 0, "glob", "jpeg75"),
    "rect126_glob_raw": ("rect126", 0, "glob", "raw"),
    "tukey224d_glob_raw": ("tukey224d", 0, "glob", "raw"),
}
# Raw/global renderings re-normalised with the true dB range (all frames, or the first 2/3 of the capture only).
RANGE_VARIANTS = {
    "hann126_glob_raw": ("hann126", 0, "glob", "raw"),
    "hamm128_glob_raw": ("hamm128", 0, "glob", "raw"),
    "rect126_glob_raw": ("rect126", 0, "glob", "raw"),
    "tukey126_glob_raw": ("tukey126", 0, "glob", "raw"),
    "tukey224_glob_raw": ("tukey224", 0, "glob", "raw"),
    "tukey224d_glob_raw": ("tukey224d", 0, "glob", "raw"),
    "hann126_glob_jpeg95": ("hann126", 0, "glob", "jpeg95"),
    "tukey224d_glob_jpeg95": ("tukey224d", 0, "glob", "jpeg95"),
}
ALL_VARIANTS = {**VARIANTS, **TUKEY_VARIANTS, **RECT_VARIANTS, **RANGE_VARIANTS}

G: dict = {}  # filled in the parent before the pool forks (read-only in the workers)


def smooth_rows(db: np.ndarray, m: int) -> np.ndarray:
    """Box-average linear power over m adjacent rows of each frame (axis 1 = frame time axis), back to dB."""
    if m <= 1:
        return db
    from scipy.ndimage import uniform_filter1d
    out = np.empty_like(db)
    for a in range(0, len(db), 64):
        p = np.power(10.0, db[a:a + 64] / 10.0, dtype=np.float32)
        p = uniform_filter1d(p, size=m, axis=1, mode="nearest")
        out[a:a + 64] = 10.0 * np.log10(p + 1e-12)
    return out


def stats_of(db: np.ndarray) -> dict:
    p = np.power(10.0, db / 10.0, dtype=np.float64)
    return {"lo": float(db.min()), "hi": float(db.max()), "p1": float(np.percentile(db, 1)),
            "p99": float(np.percentile(db, 99)), "plo": float(p.min()), "phi": float(p.max())}


def region(times, lo_frac, hi_frac):
    t0, span = times[0], (times[-1] - times[0]).total_seconds()
    lo = t0 + timedelta(seconds=span * lo_frac)
    hi = t0 + timedelta(seconds=span * hi_frac + (1.0 if hi_frac >= 1.0 else 0.0))
    idx = [i for i, t in enumerate(times) if lo <= t < hi]
    segs, s = [], 0
    for k in range(1, len(idx)):
        if (times[idx[k]] - times[idx[k - 1]]).total_seconds() != 1.0:
            segs.append((s, k))
            s = k
    segs.append((s, len(idx)))
    return idx, segs


def autocov(D: np.ndarray, segs, max_lag: int) -> np.ndarray:
    """Pooled autocovariance c[0..max_lag] of deviations D (N, P), within contiguous segments."""
    num = np.zeros(max_lag + 1)
    cnt = np.zeros(max_lag + 1)
    for a, b in segs:
        X = D[a:b]
        for l in range(max_lag + 1):
            if len(X) > l:
                num[l] += float(np.einsum("ij,ij->", X[:len(X) - l], X[l:], dtype=np.float64))
                cnt[l] += (len(X) - l) * X.shape[1]
    return num / np.maximum(cnt, 1)


def ridge_curve(c_fit: np.ndarray, D_te: np.ndarray, segs_te) -> list:
    """MSE at k = 1..K_MAX of the shared-coefficient AR predictor built from the fit-region autocovariance."""
    R = np.array([[c_fit[abs(i - j)] for j in range(L_IN)] for i in range(L_IN)])
    R = R + RIDGE * c_fit[0] * np.eye(L_IN)
    out = []
    for k in range(1, K_MAX + 1):
        r = np.array([c_fit[k + j] for j in range(L_IN)])
        a = np.linalg.solve(R, r)
        se, n = 0.0, 0
        for s0, s1 in segs_te:
            X = D_te[s0:s1]
            T = len(X)
            if T < L_IN + k:
                continue
            t = np.arange(L_IN - 1, T - k)
            pred = sum(a[j] * X[t - j] for j in range(L_IN))
            err = X[t + k] - pred
            se += float(np.einsum("ij,ij->", err, err, dtype=np.float64))
            n += err.size
        out.append(se / max(n, 1))
    return out


def file_minmax(path_str: str) -> dict:
    """Per-window (min, max) dB of one file; the window list is set in G before the pool forks."""
    x = rp.read_iq(Path(path_str))
    out = {}
    for w in G["range_windows"]:
        db = rp.spectrogram_db(x, w)
        out[w] = (float(db.min()), float(db.max()))
    return out


def variant_worker(name: str) -> dict:
    window, avg_m, scaling, post = ALL_VARIANTS[name]
    g = G
    t0 = time.time()
    stats = g["stats"][(window, avg_m)]
    idx = g["pix"]
    fit_db = smooth_rows(g["db_fit"][window], avg_m)
    te_db = smooth_rows(g["db_te"][window], avg_m)
    fit_img = rp.render(fit_db, scaling, post, False, stats).reshape(len(fit_db), -1)[:, idx].astype(np.float32)
    te_img = rp.render(te_db, scaling, post, False, stats).reshape(len(te_db), -1)[:, idx].astype(np.float32)
    mu = fit_img.mean(axis=0, keepdims=True)
    Df, Dt = fit_img - mu, te_img - mu
    c_fit = autocov(Df, g["segs_fit"], K_MAX + L_IN)
    c_te = autocov(Dt, g["segs_te"], K_MAX)
    clim = float(np.mean(Dt.astype(np.float64) ** 2))
    ridge = ridge_curve(c_fit, Dt, g["segs_te"])
    rho = (c_te[1:] / c_te[0]).tolist()
    skill = [1.0 - r / clim for r in ridge]
    return {"name": name, "clim_mse": round(clim, 2), "ridge_mse": [round(x, 2) for x in ridge],
            "skill": [round(x, 4) for x in skill], "rho": [round(x, 4) for x in rho],
            "var_fit": round(float(c_fit[0]), 2), "seconds": round(time.time() - t0, 1)}


def run_dataset(root: Path, name: str, workers: int, pool_size: int, names, stats_override=None, label=None,
                range_modes=None):
    """Returns (result, stats). With `stats_override` (e.g. FM's dB range) the dataset is rendered with those
    normalisation statistics instead of its own."""
    label = label or name
    t0 = time.time()
    seconds = rp.list_seconds(root / rp.DATASETS[name])
    times = [t for t, _ in seconds]
    paths = [p for _, p in seconds]
    fit_idx, segs_fit = region(times, 2.0 / 3.0, 5.0 / 6.0)
    te_idx, segs_te = region(times, 5.0 / 6.0, 1.0)
    print(f"[{name}] {len(paths)} files; fit {len(fit_idx)} frames ({len(segs_fit)} segs), "
          f"test {len(te_idx)} frames ({len(segs_te)} segs)", flush=True)
    rng_db = rp.load_db(paths[::rp.SAMPLE_EVERY], workers)
    g = {"db_fit": rp.load_db([paths[i] for i in fit_idx], workers),
         "db_te": rp.load_db([paths[i] for i in te_idx], workers),
         "segs_fit": segs_fit, "segs_te": segs_te}
    g["pix"] = np.sort(np.random.default_rng(SEED).choice(rp.SIZE * rp.SIZE * 3, size=N_PIX, replace=False))
    g["stats"] = {}
    for (window, avg_m, _, _) in {ALL_VARIANTS[n] for n in names}:
        g["stats"][(window, avg_m)] = stats_of(smooth_rows(rng_db[window], avg_m))
    own_stats = dict(g["stats"])
    if stats_override is not None:
        g["stats"] = stats_override
    print(f"[{label}] loaded in {time.time() - t0:.0f}s", flush=True)
    mod = importlib.import_module("acf_probe")  # the module the pool workers will see (not __main__)
    mod.G.update(g)
    import multiprocessing as mp

    def run_pool(label_):
        out_ = {"n_files": len(paths), "n_fit": len(fit_idx), "n_test": len(te_idx), "variants": {}}
        with mp.get_context("fork").Pool(pool_size) as pool:
            for res in pool.imap_unordered(mod.variant_worker, list(names)):
                out_["variants"][res["name"]] = res
                s_, r_ = res["skill"], res["rho"]
                print(f"[{label_}] {res['name']:26s} clim {res['clim_mse']:8.1f} ridge k1/8/20 "
                      f"{res['ridge_mse'][0]:8.1f} {res['ridge_mse'][7]:8.1f} {res['ridge_mse'][19]:8.1f} | "
                      f"skill {s_[0]:.3f} {s_[7]:.3f} {s_[19]:.3f} | rho {r_[0]:.3f} {r_[7]:.3f} {r_[19]:.3f} ({res['seconds']:.0f}s)", flush=True)
        return out_

    if range_modes:
        wins = sorted({ALL_VARIANTS[n][0] for n in names if ALL_VARIANTS[n][1] == 0})
        mod.G["range_windows"] = wins
        with mp.get_context("fork").Pool(workers) as pool:
            mm = pool.map(mod.file_minmax, paths, chunksize=8)
        t0_, span_ = times[0], (times[-1] - times[0]).total_seconds()
        cut = t0_ + timedelta(seconds=span_ * 2.0 / 3.0)
        masks = {"full": [True] * len(paths), "train": [t < cut for t in times]}
        print(f"[{label}] per-file min/max done in {time.time() - t0:.0f}s", flush=True)
        results, mode_stats = {}, {}
        for mode in range_modes:
            sel = masks[mode]
            st = dict(own_stats)
            for w in wins:
                lo = min(r[w][0] for r, k in zip(mm, sel) if k)
                hi = max(r[w][1] for r, k in zip(mm, sel) if k)
                st[(w, 0)] = {**own_stats[(w, 0)], "lo": lo, "hi": hi}
            mode_stats[mode] = st
            mod.G["stats"] = st
            res = run_pool(f"{label}_{mode}")
            res["range"] = {w: [round(st[(w, 0)]["lo"], 2), round(st[(w, 0)]["hi"], 2)] for w in wins}
            results[mode] = res
        return results, mode_stats
    return run_pool(label), own_stats


def work(data_root: str, out_dir: str, workers: int, pool_size: int, variant_set: str = "all") -> dict:
    root = Path(data_root)
    names = {"all": VARIANTS, "tukey": TUKEY_VARIANTS, "rect": RECT_VARIANTS, "range": RANGE_VARIANTS}[variant_set]
    names = list(names)
    result = {"created": datetime.utcnow().isoformat() + "Z", "host": os.uname().nodename, "n_pix": N_PIX,
              "L_in": L_IN, "variant_set": variant_set, "datasets": {}}
    fm_stats = None
    if variant_set == "range":
        for name in rp.DATASETS:
            if (root / rp.DATASETS[name]).exists():
                res, mstats = run_dataset(root, name, workers, pool_size, names, range_modes=("full", "train"))
                for mode, out in res.items():
                    result["datasets"][f"{name}_{mode}"] = out
                if name == "fm":
                    fm_stats = mstats["full"]
        if fm_stats is not None and "crossval_full" in result["datasets"]:
            # cross-validation rendered with FM's full-dataset dB range
            result["datasets"]["crossval_fmfull"], _ = run_dataset(root, "crossval", workers, pool_size, names,
                                                                  stats_override=fm_stats, label="crossval_fmfull")
    else:
        for name in rp.DATASETS:
            if (root / rp.DATASETS[name]).exists():
                result["datasets"][name], stats = run_dataset(root, name, workers, pool_size, names)
                if name == "fm":
                    fm_stats = stats
        if variant_set != "all" and fm_stats is not None and "crossval" in result["datasets"]:
            # cross-validation rendered with FM's dB range instead of its own (does the paper's ratio then hold?)
            result["datasets"]["crossval_fmnorm"], _ = run_dataset(root, "crossval", workers, pool_size, names,
                                                                   stats_override=fm_stats, label="crossval_fmnorm")
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"acf_probe_{variant_set}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path} on {result['host']}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data/deepspred")
    ap.add_argument("--out-dir", default="data/deepspred/probe")
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--pool", type=int, default=6)
    ap.add_argument("--set", dest="variant_set", choices=["all", "tukey", "rect", "range"], default="all")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    data_root = str(Path(args.data_root).resolve())
    out_dir = str(Path(args.out_dir).resolve())
    if args.local:
        work(data_root, out_dir, args.cpus, args.pool, args.variant_set)
        return
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"] and n.get("Resources", {}).get("CPU", 0) >= args.cpus]
    probe_dir = str(HERE)

    def has_data(path):  # only builtins/stdlib: cloudpickle ships it by value
        import os.path
        return os.path.isdir(os.path.join(path, "FM-Spectrum-Dataset"))

    def remote_entry(pdir, data_root, out_dir, cpus, pool_size, variant_set):
        import importlib
        import sys
        if pdir not in sys.path:
            sys.path.insert(0, pdir)
        return importlib.import_module("acf_probe").work(data_root, out_dir, cpus, pool_size, variant_set)

    probes = {n["NodeID"]: ray.remote(num_cpus=0.01, scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))(has_data).remote(data_root)
              for n in nodes}
    ok = []
    for nid, ref in probes.items():
        try:
            if ray.get(ref, timeout=120):
                ok.append(nid)
        except Exception as exc:
            print(f"probe on {nid[:8]} failed: {type(exc).__name__}", flush=True)
    if not ok:
        sys.exit(f"no node has {data_root}/{rp.DATASETS['fm']}")
    print(f"data present on {len(ok)} node(s); using {ok[0]}", flush=True)
    task = ray.remote(num_cpus=args.cpus, num_gpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(ok[0], soft=False))(remote_entry)
    ray.get(task.remote(probe_dir, data_root, out_dir, args.cpus, args.pool, args.variant_set))


if __name__ == "__main__":
    main()
