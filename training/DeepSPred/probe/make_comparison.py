"""Build results/COMPARISON.md: our results vs the paper's, per horizon, for the different rendering options.

  python training/DeepSPred/probe/make_comparison.py

Ours:  best-linear-predictor ceiling (see ANALYSIS_NOTES.md 4.3) from the probe JSONs in results/, plus the collaborator's
       TRAINED Swin curves from commit 45dc184 (training/DeepSPred/repro/results.json, read through git).
Paper: exact values only where the text/tables print them (tag [T]); everything else is read off Fig. 7(a) / 8(a) by eye
       (tag [P], +-4 MSE units) and is for shape only.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
REPO = Path(__file__).resolve().parents[3]
KS = (1, 2, 4, 8, 12, 16, 20)

# ---- the paper -----------------------------------------------------------------------------------------------
# [T] exact, printed in the text / tables;  [P] read from the plot.
PAPER_FM = {
    "3D-SwinSTB (ours in paper)": {1: 439, 2: 441, 4: 444, 8: 448.76, 12: 452, 16: 453.04, 20: 455, "tags": "k=8,16 [T]; others [P]; k=18 = 454.18 [T]"},
    "ConvLSTM": {1: 418, 2: 427, 4: 443, 8: 455, 12: 461, 16: 465, 20: 467, "tags": "[P] (text: below 3D-SwinSTB for k < 4)"},
    "DCG": {1: 463, 2: 466, 4: 468, 8: 471, 12: 473, 16: 474, 20: 475, "tags": "k=18 = 475.64 [T]; others [P]"},
}
PAPER_CV = {
    "3D-SwinSTB (ours in paper)": {1: 526, 2: 526, 4: 527, 8: 530, 12: 531.43, 16: 533, 20: 538, "tags": "k=12 [T]; others [P]"},
    "DCG": {1: 586, 2: 588, 4: 589, 8: 590, 12: 593.41, 16: 595, 20: 596, "tags": "k=12 [T]; others [P]"},
}


def load(name: str) -> dict:
    return json.loads((R / name).read_text())["datasets"]


def collab_results() -> dict:
    out = subprocess.run(["git", "-C", str(REPO), "show", "45dc184:training/DeepSPred/repro/results.json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def curve(v: dict) -> dict:
    q = v["ridge_mse"]
    return {k: q[k - 1] for k in range(1, 21)}


def fmt(x) -> str:
    return "-" if x is None else (f"{x:.1f}" if isinstance(x, (int, float)) else str(x))


def row(label: str, c: dict, paper_k8: float, note: str = "") -> str:
    cells = [fmt(c[k]) for k in KS]
    rise = 100 * (c[20] / c[1] - 1)
    rise8 = 100 * (c[8] / c[1] - 1)
    return f"| {label} | " + " | ".join(cells) + f" | {rise:+.1f}% | {rise8:+.1f}% | {c[8] / paper_k8:.2f}x | {note} |"


def paper_rows(table: dict, paper_k8: float) -> list:
    out = []
    for name, d in table.items():
        c = {k: d.get(k) for k in KS}
        cells = [f"{c[k]:.1f}" if c[k] is not None else "-" for k in KS]
        rise = 100 * (d[20] / d[1] - 1)
        rise8 = 100 * (d[8] / d[1] - 1)
        out.append(f"| **PAPER: {name}** | " + " | ".join(cells) + f" | {rise:+.1f}% | {rise8:+.1f}% | {d[8] / paper_k8:.2f}x | {d['tags']} |")
    return out


def main():
    a = load("acf_probe_20261110T172942.json")        # 18 renderings, sampled range
    t = load("acf_probe_tukey_20261110T183333.json")  # tukey study, sampled range
    r = load("acf_probe_rect_20261110T183943.json")   # rect +- detrend, sampled range
    g = load("acf_probe_range_20261110T185554.json")  # full / train-only range
    c = load("acf_probe_crop_20261110T191214.json")   # band crops, full range
    tor = load("acf_probe_torch_20261110T193321.json")  # torch.stft-style rect, hop 127, reflect padding
    collab = collab_results()
    key_raw = "fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|uint8_mean_per_channel"
    key_jpg = "fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|jpeg95|uint8_mean_per_channel"
    key_cv = "crossval|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|jpeg95|uint8_mean_per_channel"

    def trained(key):
        m = collab[key]["model"]["mse"]["uint8_mean_per_channel"]
        return {k: m[k - 1] for k in range(1, 21)}

    # (label, FM-curve, CV-curve, note)
    def V(src, ds, name):
        return curve(src[ds]["variants"][name])

    options = [
        ("Hann hop126, sampled range, raw (collaborator's rendering)", V(a, "fm", "hann126_glob_raw"), V(a, "crossval", "hann126_glob_raw"), "ceiling"),
        ("Hann hop126, sampled range, JPEG q95 (collaborator's headline)", V(a, "fm", "hann126_glob_jpeg95"), V(a, "crossval", "hann126_glob_jpeg95"), "ceiling"),
        ("Hann hop126, FULL range, raw", V(g, "fm_full", "hann126_glob_raw"), V(g, "crossval_full", "hann126_glob_raw"), "ceiling"),
        ("Hamming hop128, sampled range, raw", V(a, "fm", "hamm128_glob_raw"), V(a, "crossval", "hamm128_glob_raw"), "ceiling"),
        ("Hamming hop128, FULL range, raw", V(g, "fm_full", "hamm128_glob_raw"), V(g, "crossval_full", "hamm128_glob_raw"), "ceiling"),
        ("Rectangular hop126, sampled range, raw", V(r, "fm", "rect126_glob_raw"), V(r, "crossval", "rect126_glob_raw"), "CV full range is a -120 dB artifact"),
        ("torch.stft-style (rect, hop127, reflect), sampled range", V(tor, "fm_sampled", "rect127c_eps12_glob_raw"), V(tor, "crossval_sampled", "rect127c_eps12_glob_raw"), "CV sampled misses the zero-bin frames"),
        ("torch.stft-style, FULL range, dB floor 1e-8 (-80 dB)", V(tor, "fm_full", "rect127c_eps8_glob_raw"), V(tor, "crossval_full", "rect127c_eps8_glob_raw"), "CV level set by the floor"),
        ("torch.stft-style, FULL range, dB floor 1e-6 (-60 dB)", V(tor, "fm_full", "rect127c_eps6_glob_raw"), V(tor, "crossval_full", "rect127c_eps6_glob_raw"), "CV level set by the floor"),
        ("Tukey.25 hop126, FULL range, raw", V(g, "fm_full", "tukey126_glob_raw"), V(g, "crossval_full", "tukey126_glob_raw"), "ceiling"),
        ("Tukey.25 hop224 (no detrend), sampled range, raw", V(t, "fm", "tukey224_glob_raw"), V(t, "crossval", "tukey224_glob_raw"), "ceiling"),
        ("Tukey.25 hop224 (no detrend), FULL range, raw", V(g, "fm_full", "tukey224_glob_raw"), V(g, "crossval_full", "tukey224_glob_raw"), "ceiling"),
        ("scipy defaults (Tukey.25 hop224 detrend), sampled range, raw", V(t, "fm", "tukey224d_glob_raw"), V(t, "crossval", "tukey224d_glob_raw"), "ceiling"),
        ("scipy defaults, FULL range, raw", V(g, "fm_full", "tukey224d_glob_raw"), V(g, "crossval_full", "tukey224d_glob_raw"), "ceiling"),
        ("scipy defaults, TRAIN-ONLY range, raw", V(g, "fm_train", "tukey224d_glob_raw"), V(g, "crossval_train", "tukey224d_glob_raw"), "ceiling"),
        ("Tukey.25 hop224, crop A (90-110 MHz), FULL range", V(c, "fm_full", "tukey224_cropA_glob_raw"), V(c, "crossval_full", "tukey224_cropA_glob_raw"), "ceiling"),
        ("Hamming hop128, crop A, FULL range", V(c, "fm_full", "hamm128_cropA_glob_raw"), V(c, "crossval_full", "hamm128_cropA_glob_raw"), "ceiling"),
        ("Tukey.25 hop224 detrend, crop A, FULL range", V(c, "fm_full", "tukey224d_cropA_glob_raw"), V(c, "crossval_full", "tukey224d_cropA_glob_raw"), "ceiling"),
        ("Tukey.25 hop224, crop C (+-10 MHz of 99), FULL range", V(c, "fm_full", "tukey224_cropC_glob_raw"), V(c, "crossval_full", "tukey224_cropC_glob_raw"), "ceiling"),
        ("Hamming hop128, crop C, FULL range", V(c, "fm_full", "hamm128_cropC_glob_raw"), V(c, "crossval_full", "hamm128_cropC_glob_raw"), "ceiling"),
    ]
    head = "| option | k=1 | k=2 | k=4 | k=8 | k=12 | k=16 | k=20 | rise k1->20 | rise k1->8 | k=8 vs paper | note |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    P = []
    P.append("# Our results vs the paper's, per horizon and rendering option\n")
    P.append("Generated by `make_comparison.py`. MSE in uint8 pixel units (mean over pixels and channels). **Ours are best-linear-predictor ceilings** "
             "(a bound a model could approach, not a trained model) except the rows marked *trained*, which are the collaborator's Swin runs. "
             "Paper values: **[T]** printed in text/tables (exact), **[P]** read by eye from Fig. 7(a)/8(a), +-4 units, for shape only. "
             "'k=8 vs paper' = our k=8 divided by the paper's 3D-SwinSTB k=8 (FM 448.76; cross-validation compared at k=8 against 530 [P], "
             "its exact value being 531.43 at k=12).\n")
    P.append("\n## FM: MSE by predicted frame k (input-20-predict-20)\n")
    P.append(head)
    P.append(sep)
    P += paper_rows(PAPER_FM, 448.76)
    P.append(row("*trained* Swin, collaborator, raw Hann (sampled range)", trained(key_raw), 448.76, "trained, flat in k"))
    P.append(row("*trained* Swin, collaborator, JPEG q95", trained(key_jpg), 448.76, "trained, not paper-faithful"))
    for label, fm, cv, note in options:
        P.append(row(label, fm, 448.76, note))
    P.append("\n## Cross-validation: MSE by predicted frame k\n")
    P.append(head)
    P.append(sep)
    P += paper_rows(PAPER_CV, 530.0)
    P.append(row("*trained* Swin, collaborator, JPEG q95, local", trained(key_cv), 530.0, "trained locally on cross-val data"))
    for label, fm, cv, note in options:
        P.append(row(label, cv, 530.0, note))

    P.append("\n## Cross-validation / FM ratio at k=8 and horizon rise side by side\n")
    P.append("The paper's ratio is 530 / 448.76 = 1.18 (k=8, cross-val read from the plot; 531.43 / 452 = 1.18 at k=12). Paper rises: FM +3.6% (439 -> 455) and +2.3% cross-val (526 -> 538) [P].\n")
    P.append("| option | FM k=8 | CV k=8 | CV/FM | FM rise k1->20 | CV rise k1->20 |")
    P.append("|---|---|---|---|---|---|")
    P.append("| **PAPER: 3D-SwinSTB** | 448.76 [T] | ~530 [P] | 1.18 | +3.6% [P] | +2.3% [P] |")
    for label, fm, cv, note in options:
        P.append(f"| {label} | {fm[8]:.1f} | {cv[8]:.1f} | {cv[8] / fm[8]:.2f} | {100 * (fm[20] / fm[1] - 1):+.1f}% | {100 * (cv[20] / cv[1] - 1):+.1f}% |")

    P.append("\n## Rise vs time, relative to k=1 (percent above the k=1 value)\n")
    P.append("This removes the level and shows only the shape of the horizon curve. FM only.\n")
    P.append("| option | k=2 | k=4 | k=8 | k=12 | k=16 | k=20 |")
    P.append("|---|---|---|---|---|---|---|")
    for name, d in PAPER_FM.items():
        P.append(f"| **PAPER: {name}** [P] | " + " | ".join(f"{100 * (d[k] / d[1] - 1):+.1f}%" for k in (2, 4, 8, 12, 16, 20)) + " |")
    P.append("| *trained* Swin, raw Hann | " + " | ".join(f"{100 * (trained(key_raw)[k] / trained(key_raw)[1] - 1):+.1f}%" for k in (2, 4, 8, 12, 16, 20)) + " |")
    for label, fm, cv, note in options:
        P.append(f"| {label} | " + " | ".join(f"{100 * (fm[k] / fm[1] - 1):+.1f}%" for k in (2, 4, 8, 12, 16, 20)) + " |")
    (R / "COMPARISON.md").write_text("\n".join(P) + "\n")
    print("wrote", R / "COMPARISON.md")


if __name__ == "__main__":
    main()
