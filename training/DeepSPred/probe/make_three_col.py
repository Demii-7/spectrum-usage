"""Build results/THREE_COL.md: FM lookback mean and residual linear AR at k=8, one row per exact rendering + normalization.

  python training/DeepSPred/probe/make_three_col.py

Rows come from EVERYTHING.csv (only rows with both metrics). Per-frame standardization ignores the dataset range, so it is listed once per window;
the dB floor (1e-12 vs 1e-8) is identical on FM, so it is merged. The setting column spells out every varying field.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acf_probe as ap  # noqa: E402

R = HERE / "results"
N = 32508
SHAPE = {"hann": "Hann", "hamm": "Hamming", "rect": "Rectangular", "tukey": "Tukey(0.25)"}
CONV = {"full": "min-max, all frames", "train": "min-max, first 2/3 of capture", "sampled": "min-max, every 50th frame",
        "z2": "clip to dataset mean ± 2σ", "z3": "clip to dataset mean ± 3σ", "z4": "clip to dataset mean ± 4σ", "z5": "clip to dataset mean ± 5σ"}


def describe(win: str) -> tuple[str, int]:
    """(readable window/STFT description, number of STFT rows before resize)."""
    crop = None
    resize = "bilinear"
    eps = "1e-12"
    w = win
    if "@" in w:
        w, crop = w.split("@")
    if "~" in w:
        w, e = w.split("~")
        eps = f"1e-{e}"
    if "^" in w:
        w, o = w.split("^")
        resize = {"0": "nearest", "1": "bilinear", "3": "cubic"}[o]
    detr = w.endswith("d")
    w = w[:-1] if detr else w
    stft = w.endswith("s")
    w = w[:-1] if stft else w
    cen = w.endswith("c")
    w = w[:-1] if cen else w
    m = re.match(r"([a-z]+)(\d+)", w)
    shape, hop = SHAPE[m.group(1)], int(m.group(2))
    if stft:
        L = N + 256
        rem = (L - 256) % hop
        L += (hop - rem) if rem else 0
        rows, edges = (L - 256) // hop + 1, "zero-padded 128 each side + end (scipy.stft-style)"
    elif cen:
        rows, edges = 1 + N // hop, "reflect-padded 128 each side (torch-style)"
    elif hop == 126:
        rows, edges = 256, "no padding"
    else:
        rows, edges = (N - 256) // hop + 1, "no padding"
    parts = [f"{shape}", f"hop {hop}", edges, "detrend (segment mean removed)" if detr else "no detrend",
             "256 rows, no resize" if rows == 256 else f"{rows} rows → 256 ({resize} resize)"]
    if crop:
        a, b = crop.split("-")
        parts.append(f"keep freq bins {a}-{int(b) - 1} ({int(b) - int(a)}) → 256 (bilinear)")
    if eps != "1e-12":
        parts.append(f"dB floor {eps}")
    return " · ".join(parts), rows


def main():
    rows = [r for r in csv.DictReader(open(R / "EVERYTHING.csv")) if r["fm_lbm_k8"] and r["fm_res_k8"]]
    out, seen = [], set()
    for r in rows:
        spec = ap.ALL_VARIANTS[r["variant"]]
        win, avg, scale, post = spec
        desc, nrows = describe(win)
        extra = []
        if post != "raw":
            extra.append(post.replace("jpeg", "JPEG q").replace("blur", "Gaussian blur σ="))
        if avg:
            extra.append(f"row-average ×{avg}")
        if scale.startswith("zf"):
            norm = f"per-frame standardization, clip ±{scale[2:]}σ (ignores dataset range)"
            conv_key = "perframe"
        else:
            norm = CONV[r["convention"]]
            conv_key = r["convention"]
        # dedupe: per-frame rows repeat under every range label; the dB floor is irrelevant on FM unless it matters for the range (it does not here)
        key = (re.sub(r" · dB floor 1e-\d+", "", desc), tuple(extra), norm if conv_key != "perframe" else "perframe", scale)
        if key in seen:
            continue
        seen.add(key)
        if "rect127c" in win:
            desc = re.sub(r" · dB floor 1e-\d+", "", desc) + " · dB floor 1e-12 or 1e-8 (identical on FM)"
        setting = " · ".join([desc] + extra + [norm])
        out.append((r["group"][0], float(r["fm_lbm_k8"]), setting, float(r["fm_lbm_k8"]), float(r["fm_res_k8"])))
    names = {"A": "A. Window / hop / detrend / normalization", "B": "B. Per-frame standardization", "C": "C. Smoothing / codec", "D": "D. Band crop to 90-110 MHz",
             "E": "E. torch.stft / scipy.signal.stft style", "F": "F. Resize interpolation", "G": "G. Hann hop / padding diagnostic"}
    md = ["# FM lookback mean and residual linear AR at k=8 (paper: 448.8)\n",
          "Fixed in every row: complex I/Q (one second = 32,508 samples), nperseg 256, two-sided fftshifted spectrum, 10·log10(|S|²+1e-12) unless a dB floor is given, jet colormap → RGB uint8, "
          "no codec unless stated; MSE over pixels and the 3 channels on the FM test block (last 1/6), 6,000 random pixel positions, stride-1 windows, 20-frame lookback.\n"]
    cur = None
    for g, key, setting, l, rl in sorted(out, key=lambda t: (t[0], t[1])):
        if g != cur:
            md.append(f"\n**{names[g]}**\n\n| setting | lookback mean | residual linear AR |\n|---|---|---|")
            cur = g
        md.append(f"| {setting} | {l:.0f} | {rl:.0f} |")
    (R / "THREE_COL.md").write_text("\n".join(md) + "\n")
    print("rows:", len(out))


if __name__ == "__main__":
    main()
