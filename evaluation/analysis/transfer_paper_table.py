"""Build the paper's transfer table (latex/transfer-table.tex, Table tab:transfer).

Inputs are the tables of the NCAR transfer batch (evaluation/analysis/ncar_transfer_tables.py):
  --tables   directory holding transfer_combined.csv, ncar_transfer_target_level.csv, ncar_transfer_coverage.csv
  --baselines evaluation/results/transfer_baselines.csv (reference-week baselines)

Every cell is test MAE in dB, mean over t+4, t+12, t+24 (1, 3, 6 h), seeds averaged, group = mean over receivers.
Shading: relative difference from the better of the two baselines in the same column (blue lower, red higher).
Cells whose model/group is incomplete in the coverage file are printed as "--".
"""
import argparse
from pathlib import Path

import pandas as pd

H = [4, 12, 24]
BANDS = ["600_800", "2400_2600"]
# (header, group, target band or None for same-band)
COLS = [
    ("ref.", "core4_oct", None),
    ("later", "core4_week2", None),
    ("new", "heldout_oct", None),
    ("same", "july_core4", None),
    ("other", "july_other", None),
    ("800--1000", "crossband_800_1000", "800_1000"),
    ("COSMOS", "cosmos", None),
    ("ARA", "ara", None),
    ("AERPAW", "aerpaw", None),
]
# (label, model, rep, condition) ; None = section rule
ROWS = [
    ("sec", "Level pathway: as trained"),
    ("RevIN-MLP", "revinmlp", "2d", "frozen"),
    ("PatchTST", "patchtst", "2d", "frozen"),
    ("OFA (GPT-2)", "ofa", "2d", "frozen"),
    ("Time-LLM (GPT-2)", "timellm", "2d", "frozen"),
    ("TimeRAN", "timeran_lp", "2d", "frozen"),
    ("Residual LinearAR 1D", "residuallinearar1d", "1d", "frozen"),
    ("Residual LinearAR 2D", "residuallinearar2d", "2d", "frozen"),
    ("sec", "No level pathway, or fixed scale: as trained, and refit"),
    ("TCN", "temporalconvnet", "2d", "frozen"),
    ("\\quad refit", "temporalconvnet", "2d", "refit"),
    ("LSTM 1D", "vanillalstm1d", "1d", "frozen"),
    ("\\quad refit", "vanillalstm1d", "1d", "refit"),
    ("3D-SwinSTB", "deepspred2d", "2d", "frozen"),
    ("\\quad refit", "deepspred2d", "2d", "refit"),
    ("LSTM 2D", "vanillalstm", "2d", "frozen"),
    ("\\quad refit", "vanillalstm", "2d", "refit"),
    ("LSTMAttn", "lstmattn", "2d", "frozen"),
    ("\\quad refit", "lstmattn", "2d", "refit"),
    ("Autoformer-CSA", "autoformer_csa", "2d", "frozen"),
    ("\\quad refit", "autoformer_csa", "2d", "refit"),
]
# Only what the paper's evaluation-set table lists: ARA is ames and horticulture, July run, 600--800 MHz.
ARA_KEEP = {"ara/ames-ue-000/20260702T1920Z/600_800", "ara/horticulture-ue-004/20260702T1920Z/600_800"}
BASE = [("Lookback median", "tuned_median"), ("Same time yesterday", "same_time_yesterday")]


def fmt(v):
    if pd.isna(v):
        return "--"
    return f"{v:.2f}" if v < 9.995 else f"{v:.1f}"


def shade(v, ref):
    if pd.isna(v) or pd.isna(ref):
        return fmt(v)
    rel = v / ref - 1.0
    k = min(52, int(round(130 * abs(rel))))
    if k == 0:
        return fmt(v)
    return f"\\cellcolor{{{'red' if rel > 0 else 'blue'}!{k}}}{fmt(v)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--baselines", default="evaluation/results/transfer_baselines.csv")
    ap.add_argument("--scope", default="evaluation/results/transfer_paper_scope.csv", help="AERPAW/ARA cells on the paper table's intervals (transfer_paper_scope.py)")
    ap.add_argument("--out", default="latex/transfer-table.tex")
    a = ap.parse_args()
    d = Path(a.tables)
    t = pd.read_csv(d / "ncar_transfer_target_level.csv")
    t = t[t.horizon.isin(H)]
    sc = pd.read_csv(a.scope)
    sc = sc[sc.horizon.isin(H)]
    # AERPAW and ARA come from the interval-restricted rescoring only (seed mean per target first)
    scm = sc[sc.rep != "baseline"].groupby(["model", "rep", "condition", "source_band", "target_band", "group", "target", "receiver", "horizon"], as_index=False).mae_db.mean()
    t = pd.concat([t[~t.group.isin(["aerpaw", "ara"])], scm], ignore_index=True)
    # group value = mean over receivers of (mean over that receiver's targets), then mean over horizons
    g = (t.groupby(["model", "rep", "condition", "source_band", "target_band", "group", "receiver", "horizon"]).mae_db.mean()
         .groupby(["model", "rep", "condition", "source_band", "target_band", "group", "horizon"]).mean()
         .groupby(["model", "rep", "condition", "source_band", "target_band", "group"]).mean())
    cov = pd.read_csv(d / "ncar_transfer_coverage.csv")
    bad = {(r.model, r.rep, r.group, r.source_band, r.target_band) for r in cov[~cov.complete].itertuples()}
    b = pd.read_csv(a.baselines)
    b = b[b.horizon.astype(str).isin([str(h) for h in H])]
    scb = sc[sc.rep == "baseline"].rename(columns={"model": "baseline", "target_band": "band"})[["group", "band", "baseline", "receiver", "horizon", "mae_db"]]
    scb = scb.groupby(["group", "band", "baseline", "receiver", "horizon"], as_index=False).mae_db.mean()
    scb = scb.groupby(["group", "band", "baseline", "horizon"], as_index=False).mae_db.mean()
    b = pd.concat([b[(b.level == "group") & ~b.group.isin(["aerpaw", "ara"])], scb])
    bg = b.groupby(["group", "band", "baseline"]).mae_db.mean()

    def base(name, group, band, tband):
        key = name
        if tband and name == "tuned_median":
            key = f"tuned_median_k{band}"  # window selected on the source band's validation data
        try:
            return bg[(group, tband or band, key)]
        except KeyError:
            return float("nan")

    def cell(model, rep, cond, group, band, tband):
        tb = tband or band
        if (model, rep, group, band, tb) in bad:
            return float("nan")
        try:
            return g[(model, rep, cond, band, tb, group)]
        except KeyError:
            return float("nan")

    n = len(COLS)
    out = []
    out.append("% Generated by evaluation/analysis/transfer_paper_table.py; do not edit by hand.")
    out.append("\\begin{table}[!tp]")
    out.append("\\centering")
    out.append("\\caption{\\textbf{Transfer of the 15-minute models.} MAE (dB), averaged over 1, 3, and 6 hours and seeds, of models trained on the four core POWDER receivers and applied without retraining to the evaluation sets of Table~\\ref{tab:transferability}: the test week (\\emph{ref.}), the following week (\\emph{later}), two receivers not used in training (\\emph{new}), the June--July collection at the same and at other receivers, a band not used in training (800--1000\\,MHz), and three other testbeds. \\emph{Refit}: the per-bin normalization statistics are re-estimated on the first 24 hours of the target, with the weights unchanged. The lookback median uses the window selected on POWDER validation data. Shading: MAE relative to the better of the two heuristics in the same column (blue: lower, red: higher).}")
    out.append("\\label{tab:transfer}")
    out.append("\\fontsize{6.8pt}{8pt}\\selectfont")
    out.append("\\setlength{\\tabcolsep}{2pt}")
    out.append("\\begin{tabular}{@{}l" + "r" * n + "@{}}")
    out.append("\\toprule")
    for bi, band in enumerate(BANDS):
        if bi:
            out.append("\\midrule")
        out.append(f"\\textbf{{({'ab'[bi]})}} Trained on {band.replace('_', '--')}\\,MHz & & & & \\multicolumn{{2}}{{c}}{{June--July}} & new band & \\multicolumn{{3}}{{c}}{{other testbed}} \\\\")
        out.append("\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(l){8-10}")
        out.append("Model & " + " & ".join(c[0] for c in COLS) + " \\\\")
        out.append("\\midrule")
        refs = {}
        for label, name in BASE:
            vals = []
            for _, group, tband in COLS:
                v = base(name, group, band, tband)
                if not pd.isna(v):
                    refs[group] = min(refs.get(group, float("inf")), v)
                vals.append(fmt(v))
            out.append(f"{label} & " + " & ".join(vals) + " \\\\")
        for row in ROWS:
            if row[0] == "sec":
                out.append("\\midrule")
                out.append(f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\textit{{{row[1]}}}}} \\\\")
                continue
            label, model, rep, cond = row
            vals = [shade(cell(model, rep, cond, group, band, tband), refs.get(group)) for _, group, tband in COLS]
            out.append(f"{label} & " + " & ".join(vals) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    Path(a.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
