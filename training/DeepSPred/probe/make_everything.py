"""Build results/EVERYTHING.csv and results/EVERYTHING.md: one row per (rendering, normalization convention) over ALL probe result files.

  python training/DeepSPred/probe/make_everything.py

Metrics (uint8 MSE, mean over pixels and channels), horizon k=8 unless stated:
  LBM    = lookback mean (mean of the last 20 input frames), pixel subset            lin5 = 5-lag linear model on deviations from the validation-block mean
  RES    = residual linear (lookback mean + linear correction over 20 lags)          rise = value at k=20 over value at k=1 (percent), for LBM (else lin5)
Datasets: FM; CV own = cross-validation normalised with its own range; CV FM-rng = cross-validation normalised with FM's range/statistics.
Conventions: sampled = range from every 50th frame (collaborator's shortcut); full = all frames; train = first 2/3; z2..z5 = dataset mean +- k std; the per-frame
standardization is part of the rendering name (zf3).  '-' = not run.  Image-domain rows (whole 256x256 images, no pixel subset) come from rendering_probe.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acf_probe as ap  # noqa: E402
import rendering_probe as rp  # noqa: E402

R = HERE / "results"
SHAPE = {"hann": "Hann", "hamm": "Hamming", "rect": "Rectangular", "tukey": "Tukey .25"}
SCALE_LABEL = {"glob": "", "frame": "per-frame min-max", "p1p99": "1-99% clip", "lin": "linear power", "zf3": "per-frame standardization +-3 std"}


def window_label(win: str, avg: int = 0) -> str:
    crop = eps = None
    resize = None
    w = win
    if "@" in w:
        w, crop = w.split("@")
    if "~" in w:
        w, e = w.split("~")
        eps = e
    if "^" in w:
        w, o = w.split("^")
        resize = {"0": "nearest", "1": "bilinear", "3": "cubic"}.get(o, o)
    detr = w.endswith("d")
    if detr:
        w = w[:-1]
    stft = w.endswith("s")
    if stft:
        w = w[:-1]
    cen = w.endswith("c")
    if cen:
        w = w[:-1]
    m = re.match(r"([a-z]+)(\d+)", w)
    shape, hop = SHAPE.get(m.group(1), m.group(1)), m.group(2)
    s = f"{shape}, hop {hop}"
    if stft:
        s += ", zero-padded edges (scipy.stft-style)"
    if cen:
        s += ", reflect-padded (torch-style)"
    if detr:
        s += ", detrend"
    if resize:
        s += f", {resize} resize"
    if crop:
        s += ", crop " + {"54-218": "A (90-110 MHz)", "38-202": "B (mirrored)", "46-210": "C (+-10 MHz of 99)"}.get(crop, crop)
    if eps:
        s += f", dB floor 1e-{eps}"
    if avg:
        s += f", row-average x{avg}"
    return s


def label_for(name: str) -> tuple[str, str]:
    """(group, readable label) for a variant name."""
    spec = ap.ALL_VARIANTS.get(name)
    gray = name.endswith("_gray")
    if spec is None and gray:
        spec = ap.ALL_VARIANTS.get(name[:-5]) or None
        if spec is None:
            m = re.match(r"([a-z0-9]+)_(glob|frame)_(raw|jpeg\d+)", name)
            spec = (m.group(1), 0, m.group(2), m.group(3)) if m else None
    if spec is None:
        return "Z. other", name
    win, avg, scale, post = spec
    lab = window_label(win, avg)
    sl = SCALE_LABEL.get(scale, scale)
    if sl:
        lab += f", {sl}"
    if post != "raw":
        lab += ", " + post.replace("jpeg", "JPEG q").replace("blur", "blur sigma ")
    if gray:
        lab += ", GRAY (1 channel)"
    if "^" in win and "hann1" in win and "nopad" in name:
        g = "G. hop / padding diagnostic (Hann)"
    elif "^" in win or "resize" in name:
        g = "F. resize interpolation"
    elif post.startswith(("jpeg", "blur")) or avg:
        g = "C. smoothing / codec / row-averaging"
    elif "@" in win:
        g = "D. band crop to 90-110 MHz"
    elif "c" == win.split("^")[0].split("@")[0].split("~")[0].rstrip("d")[-1:] or "s" == win.split("^")[0].split("@")[0].split("~")[0].rstrip("d")[-1:]:
        g = "E. torch.stft / scipy.signal.stft-style"
    elif scale != "glob":
        g = "B. normalization type"
    else:
        g = "A. window / hop / detrend (raw, min-max)"
    return g, lab


def classify(key: str):
    m = {"fm": ("fm", "sampled"), "crossval": ("cv", "sampled"), "crossval_fmnorm": ("cvfm", "sampled"),
         "fm_sampled": ("fm", "sampled"), "crossval_sampled": ("cv", "sampled"),
         "crossval_fmfull": ("cvfm", "full"), "crossval_fm_full": ("cvfm", "full")}
    if key in m:
        return m[key]
    g = re.match(r"(fm|crossval)_(full|train|z\d)$", key)
    if g:
        return ("fm" if g.group(1) == "fm" else "cv", g.group(2))
    g = re.match(r"crossval_fm_(z\d)$", key)
    if g:
        return ("cvfm", g.group(1))
    return None


def metrics(v: dict) -> dict:
    o = {}
    if "lbm_mse" in v:
        o["lbm"] = v["lbm_mse"]
    if "ridge_mse" in v:
        o["lin5"] = v["ridge_mse"]
    if "rlin_mse" in v:
        o["res"] = v["rlin_mse"]
    return o


def score(m: dict) -> int:
    return 4 * ("res" in m) + 2 * ("lbm" in m) + ("lin5" in m)


def main():
    rows: dict = {}
    sampled_span = {}
    rpj = json.loads((R / "range_probe_20261110T184251.json").read_text())["datasets"]
    for ds in ("fm", "crossval"):
        for w, r in rpj[ds]["windows"].items():
            sampled_span[(ds, w)] = r["sampled"]["span"]

    for f in sorted(glob.glob(str(R / "acf_probe_*.json"))):
        d = json.loads(Path(f).read_text())["datasets"]
        for dkey, dv in d.items():
            c = classify(dkey)
            if not c:
                continue
            role, conv = c
            for name, v in dv["variants"].items():
                spec = ap.ALL_VARIANTS.get(name)
                span = None
                if role == "fm" and spec:
                    rng = dv.get("range", {}).get(spec[0])
                    if rng:
                        span = rng[1] - rng[0]
                    elif conv == "sampled":
                        span = sampled_span.get(("fm", spec[0]))
                ent = rows.setdefault((name, conv), {"src": set()})
                m = metrics(v)
                if role not in ent or score(m) > score(ent[role]):
                    ent[role] = m
                if span is not None and "span" not in ent:
                    ent["span"] = span
                ent["src"].add(Path(f).name.replace("acf_probe_", "").replace(".json", ""))

    # image-domain rows (whole images, persistence + lookback mean) from the first probe
    img = {}
    rj = json.loads((R / "rendering_probe_20261110T172330.json").read_text())["datasets"]
    for ds, dv in rj.items():
        for name, v in dv["variants"].items():
            img.setdefault(name, {})[ds] = v

    out_rows = []
    for (name, conv), ent in rows.items():
        g, lab = label_for(name)
        fm, cv, cf = ent.get("fm", {}), ent.get("cv", {}), ent.get("cvfm", {})
        pick = lambda mm, k, i: (mm[k][i] if k in mm else None)
        base = fm.get("lbm") or fm.get("lin5")
        rise = (100 * (base[19] / base[0] - 1)) if base else None
        cbase = cv.get("lbm") or cv.get("lin5")
        crise = (100 * (cbase[19] / cbase[0] - 1)) if cbase else None
        out_rows.append({
            "group": g, "rendering": lab, "convention": conv, "variant": name,
            "fm_lbm_k8": pick(fm, "lbm", 7), "fm_res_k8": pick(fm, "res", 7), "fm_lin5_k8": pick(fm, "lin5", 7), "fm_rise_pct": rise,
            "cv_lbm_k8": pick(cv, "lbm", 7), "cv_res_k8": pick(cv, "res", 7), "cv_lin5_k8": pick(cv, "lin5", 7), "cv_rise_pct": crise,
            "cvfm_lbm_k8": pick(cf, "lbm", 7), "cvfm_lin5_k8": pick(cf, "lin5", 7),
            "fm_span_db": ent.get("span"), "sources": ";".join(sorted(ent["src"]))})
    for name, dv in img.items():
        g, lab = label_for(name)
        fm, cv = dv.get("fm"), dv.get("crossval")
        out_rows.append({
            "group": "H. image-domain (whole images: persistence and lookback mean)", "rendering": lab, "convention": "sampled", "variant": name,
            "fm_lbm_k8": fm["lookback_mean_mse"][7] if fm else None, "fm_res_k8": None, "fm_lin5_k8": None,
            "fm_rise_pct": 100 * (fm["lookback_mean_mse"][19] / fm["lookback_mean_mse"][0] - 1) if fm else None,
            "cv_lbm_k8": cv["lookback_mean_mse"][7] if cv else None, "cv_res_k8": None, "cv_lin5_k8": None,
            "cv_rise_pct": 100 * (cv["lookback_mean_mse"][19] / cv["lookback_mean_mse"][0] - 1) if cv else None,
            "cvfm_lbm_k8": None, "cvfm_lin5_k8": None, "fm_span_db": None,
            "fm_persistence_k8": fm["persistence_mse"][7] if fm else None, "cv_persistence_k8": cv["persistence_mse"][7] if cv else None,
            "sources": "rendering_probe_20261110T172330"})
    for r in out_rows:
        n = []
        lab = r["rendering"]
        if lab.startswith("Rectangular, hop 126, detrend"):
            n.append("ARTIFACT: exact-zero DC bin floors at -120 dB and stretches the range")
        if (lab.startswith("Rectangular") or "torch-style" in lab) and not lab.startswith("Rectangular, hop 126, detrend"):
            n.append("CV level set by ONE exact-zero pixel under full/train range" if r["convention"] in ("full", "train") else "")
        if r["convention"] == "sampled":
            n.append("range from every 50th frame (collaborator's shortcut)")
        if r["group"].startswith("H."):
            n.append("whole images, no pixel subset")
        r["note"] = "; ".join(x for x in n if x)
    conv_order = {"full": 0, "train": 1, "sampled": 2}
    out_rows.sort(key=lambda r: (r["group"], r["fm_lbm_k8"] if r["fm_lbm_k8"] is not None else (r["fm_lin5_k8"] if r["fm_lin5_k8"] is not None else 1e9), conv_order.get(r["convention"], 3)))

    cols = ["group", "rendering", "convention", "variant", "fm_lbm_k8", "fm_res_k8", "fm_lin5_k8", "fm_rise_pct", "fm_span_db", "cv_lbm_k8", "cv_res_k8", "cv_lin5_k8", "cv_rise_pct",
            "cvfm_lbm_k8", "cvfm_lin5_k8", "fm_persistence_k8", "cv_persistence_k8", "note", "sources"]
    with open(R / "EVERYTHING.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: ("" if r.get(c) is None else (round(r[c], 2) if isinstance(r[c], float) else r[c])) for c in cols})

    f0 = lambda x: "-" if x is None else f"{x:.0f}"
    fp = lambda x: "-" if x is None else f"{x:+.1f}%"
    md = ["# Everything we tested (generated by `make_everything.py`; sortable version in `EVERYTHING.csv`)\n",
          "MSE in uint8 pixel units at predicted frame k=8 unless stated. **LBM** = lookback mean (mean of the last 20 input frames); **RES** = residual linear (lookback mean + linear correction over 20 lags); "
          "**lin5** = 5-lag linear model on deviations from the validation-block mean (not a bound; stale under drift). **Rise** = k=20 over k=1 for LBM (lin5 where LBM was not run). "
          "**CV own** = cross-validation normalised with its own range; **CV FM-rng** = with FM's range/statistics. **Convention**: sampled = range from every 50th frame (the collaborator's shortcut; not a plausible reading), "
          "full = all frames, train = first 2/3, z2-z5 = dataset mean +- k std. **FM span** = dB range used (where recorded). "
          "Paper (trained 3D-SwinSTB): FM 448.76 at k=8 [exact], rise about +3.6% [plot]; cross-validation 531.43 at k=12 [exact], rise about +2.3% [plot]. "
          "Collaborator's trained Swin (raw Hann, sampled range): FM 845.9; (JPEG q95): 266.6. `-` = not run.\n",
          "| group | rendering | convention | FM LBM | FM RES | FM lin5 | FM rise | FM span dB | CV own LBM | CV own RES | CV own lin5 | CV rise | CV FM-rng LBM | CV FM-rng lin5 | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        md.append(f"| {r['group'].split('.')[0]} | {r['rendering']} | {r['convention']} | {f0(r['fm_lbm_k8'])} | {f0(r['fm_res_k8'])} | {f0(r['fm_lin5_k8'])} | {fp(r['fm_rise_pct'])} | "
                  f"{'-' if r['fm_span_db'] is None else format(r['fm_span_db'], '.0f')} | {f0(r['cv_lbm_k8'])} | {f0(r['cv_res_k8'])} | {f0(r['cv_lin5_k8'])} | {fp(r['cv_rise_pct'])} | "
                  f"{f0(r['cvfm_lbm_k8'])} | {f0(r['cvfm_lin5_k8'])} | {r['note']} |")
    md.append("\nGroups: A window/hop/detrend (min-max), B normalization type, C smoothing/codec/row-averaging, D band crop, E torch.stft/scipy.signal.stft-style, F resize interpolation, G Hann hop/padding diagnostic, "
              "H image-domain (whole images; persistence and lookback mean), Z other.\n")
    (R / "EVERYTHING.md").write_text("\n".join(md) + "\n")
    from collections import Counter
    print("rows:", len(out_rows), dict(Counter(r["group"].split(".")[0] for r in out_rows)))


if __name__ == "__main__":
    main()
