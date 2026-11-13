"""DeepSPred appendix table: trained 3D-SwinSTB vs baselines under each rendering, in the paper's units and in dB.

Inputs: ``repro/results.json`` (model and lookback-mean pixel MSE per run) and ``repro/dbspace/<sha12>.json`` (dbspace_score.py; fetched
from ``s3://spectrum/deepspred-repro/2026-11-12/dbspace/``). Output: ``latex/deepspred-reproduction-table.tex`` and a markdown copy
``repro/dbspace/TABLE.md``. Frame k = 8 (the paper's FM anchor, 448.76); dB columns are per-pixel MAE at k = 8 (all horizons are within
a few percent; the JSONs carry every k and the per-second variant).

    python training/DeepSPred/repro/make_appendix_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
K8 = 7

# (label, sha12 of seed 0 [, seed 1]) in display order; FM, K = T = 20, full-range min-max, jet unless noted
ROWS = [
    ("Hann, hop 128 (\\texttt{scipy} stft)", ["1442b5d2022f"]),
    ("Hann, hop 126", ["0727eb8f06c1"]),
    ("Hamming, hop 128", ["513bd5b54a8f"]),
    ("Tukey, hop 224, 90--110\\,MHz", ["187560732831", "1b987cdc5bb2"]),
    ("Tukey, hop 224 (\\texttt{scipy} spectrogram)", ["3c96fc0dc0cd", "c5a4e4c918ca"]),
    ("Rect., hop 127 (\\texttt{torch.stft})", ["2fa9ff9a3d9b"]),
    ("Rect., hop 126", ["c58d32f53cad", "cc28f79072b2"]),
]
GRAY = "gray"          # rows for the grayscale runs are added from dbspace JSONs whose run key ends in |gray
# LTE transfer (Tukey hop 224, K = T = 16): seed 0 from the collaborator's campaign, seeds 1-2 from ray_campaign.LTE_RUNS (2026-11-12)
# runs whose model cells are printed as "--" in the LaTeX table (TABLE.md keeps the numbers): the LTE from-scratch run froze after epoch 1
# (train_o_lte_scratch_tukey224d_s0_rule-none.log), so its error is not a result; remove once the replicates are in
WITHHELD = {"4dfd1d59c176"}
LTE = [("LTE, from scratch", ["4dfd1d59c176", "ef76c703ed96"]), ("LTE, from FM model", ["dba70d28179a"])]
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def records():
    out = {}
    for key, rec in json.loads((HERE / "results.json").read_text()).items():
        if isinstance(rec, dict) and rec.get("checkpoint", {}).get("path"):
            out[Path(rec["checkpoint"]["path"]).stem] = (key, rec)
    return out


def scores():
    out = {}
    for p in sorted((HERE / "dbspace").glob("*.json")):
        d = json.loads(p.read_text())
        if "db" in d:                       # skip non-score JSONs (e.g. psnr_averaging.json, *_record.json)
            out[d["sha12"]] = d
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def row(shas, recs, dbs, k):
    got = [dbs[s] for s in shas if s in dbs]
    # pixel MSE: the collaborator's record where it exists, else the rescorer's (identical to 1e-7 where both exist)
    pix_m = mean([recs[s][1]["model"]["mse"]["uint8_mean_per_channel"][k] if s in recs else dbs[s]["uint8_mse"]["model"][k]
                  for s in shas if s in recs or s in dbs])
    pix_l = mean([recs[s][1]["trivial_references"]["lookback_mean"]["mse"]["uint8_mean_per_channel"][k] if s in recs
                  else dbs[s]["uint8_mse"]["lookback_mean"][k] for s in shas if s in recs or s in dbs])

    def db(name):
        return mean([g["db"][name]["mae_db"][k] for g in got]) if got else None

    pix_t = mean([g["uint8_mse"].get("lookback_mean_tuned", [None] * (k + 1))[k] for g in got]) if got else None
    return {"pix_model": pix_m, "pix_lbm": pix_l, "pix_lbm_t": pix_t, "db_model": db("model"), "db_lbm": db("lookback_mean"),
            "db_lbm_t": db("lookback_mean_tuned"), "db_med": db("lookback_median"),
            "db_res": db("residual_linear_ar"), "db_pers": db("persistence"), "n": len(shas), "n_db": len(got),
            "ctrl": mean([g["db"]["lookback_mean_rgb_inverted"]["mae_db"][k] - g["db"]["lookback_mean"]["mae_db"][k] for g in got])}


def fmt(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def main():
    recs, dbs = records(), scores()
    lines, md = [], ["| rendering | seeds | model MSE@8 (px) | lookback mean, full (px) | lookback mean, tuned (px) | model MAE@8 (dB) | lookback mean, full | lookback mean, tuned | residual AR | persistence | inversion control (dB) |",
                     "|---|---|---|---|---|---|---|---|---|---|---|"]
    body = [(label, shas, K8) for label, shas in ROWS]
    spec_name = {"rect127c~12": "rect., hop 127", "tukey224d": "Tukey, hop 224"}
    gray = [(f"Gray, C={d['run_key'].split('|')[3][1:]}, {spec_name.get(d['render_spec'], d['render_spec'])}", [s], K8)
            for s, d in sorted(dbs.items(), key=lambda kv: kv[1]['run_key']) if d.get("gray")]
    for label, shas, k in body + gray:
        r = row(shas, recs, dbs, k)
        if "C=96" not in label:  # LaTeX: paper-design grayscale (C=32) only; full-width grayscale stays in TABLE.md
            lines.append(f"{label.replace(', C=32', '')} & {fmt(r['pix_model'])} & {fmt(r['pix_lbm'])} & {fmt(r['db_model'], 2)} & {fmt(r['db_med'], 2)} \\\\")
        md.append(f"| {label} | {r['n']} | {fmt(r['pix_model'])} | {fmt(r['pix_lbm'])} | {fmt(r['pix_lbm_t'])} | {fmt(r['db_model'], 3)} | {fmt(r['db_lbm'], 3)} | {fmt(r['db_lbm_t'], 3)} | "
                  f"{fmt(r['db_res'], 3)} | {fmt(r['db_pers'], 3)} | {fmt(r['ctrl'], 3)} |")
    counts = [len(shas) for _, shas in ROWS] + [1] * len(gray) + [len(shas) for _, shas in LTE]
    lo_n, hi_n = min(counts), max(counts)
    runs = f"one run each" if hi_n == 1 else (f"{NUM[lo_n]} or {NUM[hi_n]} runs" if hi_n - lo_n == 1 else f"{NUM[lo_n]} to {NUM[hi_n]} runs")
    lte = []
    for label, shas in LTE:
        r = row(shas, recs, dbs, K8)
        hide = bool(WITHHELD & set(shas))
        lte.append(f"{label} & {'--' if hide else fmt(r['pix_model'])} & {fmt(r['pix_lbm'])} & {'--' if hide else fmt(r['db_model'], 2)} & {fmt(r['db_med'], 2)} \\\\")
        md.append(f"| {label} | {r['n']} | {fmt(r['pix_model'])} | {fmt(r['pix_lbm'])} | {fmt(r['pix_lbm_t'])} | {fmt(r['db_model'], 3)} | {fmt(r['db_lbm'], 3)} | {fmt(r['db_lbm_t'], 3)} | "
                  f"{fmt(r['db_res'], 3)} | {fmt(r['db_pers'], 3)} | {fmt(r['ctrl'], 3)} |")
    tex = [
        "% Generated by training/DeepSPred/repro/make_appendix_table.py from repro/results.json and repro/dbspace/*.json; do not edit.",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{DeepSPred reproduction: 3D-SwinSTB (model) against the per-pixel mean or median of its input frames (lookback mean, lookback median) "
        "at frame 8, under spectrogram renderings the paper leaves unspecified. MSE on 8-bit images is the paper's metric (paper: 448.76 on FM); MAE in dB "
        "is computed after converting predicted colors back to dB. Jet colormap unless marked gray. Mean over "
        f"{runs}.}}",
        "\\label{tab:deepspred-reproduction}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\noindent\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrr@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{MSE (pixels)} & \\multicolumn{2}{c}{MAE (dB)} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Rendering & Model & Lookback mean & Model & Lookback median \\\\",
        "\\midrule",
        *lines,
        "\\midrule",
        *lte,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}",
    ]
    (REPO / "latex" / "deepspred-reproduction-table.tex").write_text("\n".join(tex) + "\n")
    (HERE / "dbspace").mkdir(exist_ok=True)
    (HERE / "dbspace" / "TABLE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
