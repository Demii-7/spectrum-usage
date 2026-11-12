"""Plot frame-wise test MSE curves (paper Fig. 7 style) from a results file.

Usage:
    python plot.py                          # repro/results.json -> FM campaign records
    python plot.py --all                    # include legacy (pre-campaign) and LTE records
    python plot.py --results myresults.json --out myplot

Solid line = trained model; dotted same-color = that record's lookback-mean reference;
black dashed = the paper's claimed 3D-SwinSTB (448.76 @8, 453.04 @16, 454.18 @18)."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / 'results.json'

PAPER_K = np.array([1, 8, 16, 18, 20])
PAPER_V = np.array([440.0, 448.7628, 453.0376, 454.1826, 456.0])

TAG_LABELS = [('|dp0.1', ' drop-path 0.1'), ('|rule=none', ' 20 ep, no early stop'),
              ('|head=residual', ' residual head'), ('|iscale=raw', ' raw input'), ('|tl', ' TL')]


def short_label(run_key):
    m = re.search(r'\|spec=([^|]+)\|(\w+)', run_key)
    if m:
        label = f"{m.group(1)}"
    else:
        rend = re.search(r'\|(jpeg\d+|raw)\|', run_key)
        label = f"legacy {rend.group(1)}" if rend else run_key[:24]
    seed = re.search(r'\|s(\d+)\|', run_key)
    if seed:
        label += f" s{seed.group(1)}"
    for tag, txt in TAG_LABELS:
        if tag in run_key:
            label += txt
    return label


def plot_frame_mse(results_path, out_base='frame_vs_mse', include_all=False, show_paper=True):
    records = json.load(open(results_path))
    fig, ax = plt.subplots(figsize=(9, 6))
    colors, plotted = plt.cm.tab10.colors, 0
    spec_colors, ci = {}, 0
    for key, v in records.items():
        if key.startswith('_') or 'model' not in v:
            continue
        is_campaign = '|spec=' in key
        is_fm = v.get('dataset', 'fm') == 'fm'
        if not (is_campaign and is_fm) and not include_all:
            continue
        try:
            m = np.array(v['model']['mse']['uint8_mean_per_channel'], dtype=float)
            l = np.array(v['trivial_references']['lookback_mean']['mse']['uint8_mean_per_channel'], dtype=float)
        except (KeyError, TypeError):
            continue
        spec = re.search(r'\|spec=([^|]+)\|', key)
        spec = spec.group(1) if spec else 'legacy'
        if spec not in spec_colors:
            spec_colors[spec] = colors[ci % len(colors)]
            ci += 1
        c = spec_colors[spec]
        k = np.arange(1, len(m) + 1)
        label = short_label(key) + (f" (@8={m[7]:.0f})" if len(m) > 7 else "")
        ax.plot(k, m, '-', color=c, lw=2, label=label)
        ax.plot(k, l, ':', color=c, lw=1.2, alpha=0.75)
        plotted += 1
    if show_paper and plotted:
        ax.plot(np.arange(1, 21), np.interp(np.arange(1, 21), PAPER_K, PAPER_V),
                'k--', lw=2, label="paper's claimed 3D-SwinSTB (448.76 @8)")
    ax.set_xlabel('predicted frame index k')
    ax.set_ylabel('MSE (uint8_mean_per_channel)')
    ax.set_title('FM, input-20-predict-20, test split' + ('' if not include_all else ' (all records)'))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(1, 21))
    fig.tight_layout()
    fig.savefig(out_base + '.png', dpi=150)
    fig.savefig(out_base + '.pdf')
    plt.close(fig)
    print(f"plot: {plotted} model curves -> {out_base}.png/.pdf")
    return plotted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(DEFAULT_RESULTS))
    ap.add_argument('--out', default=str(HERE.parents[3] / 'plots' / 'rendering_campaign_frame_vs_mse'))
    ap.add_argument('--all', action='help' if False else 'store_true',
                    help='include legacy (pre-campaign) and non-FM records')
    args = ap.parse_args()
    plot_frame_mse(args.results, args.out, include_all=args.all)
