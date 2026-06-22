#!/usr/bin/env python3
"""
Reconstruct the top-row (ground-truth) spectrograms of TSS-LCD Figure 9.
Uses the same sliding-window / 70-30 split as the paper's test.py.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = "/tmp/TSS-LCD/dataset/merged_power_data_sub6GHz_avg_per_minute.csv"
OUTPUT = "plots/figure9_ground_truth_reproduced.png"

# --- Load CSV ---
with open(CSV_PATH) as f:
    reader = csv.reader(f)
    data = np.array([[float(v) for v in row] for row in reader], dtype=np.float32)

T, D = data.shape  # (6839, 750)

# --- Sliding window (matching DataSetPrepare.py / test.py) ---
context_len = 50
future_len = 10
samples = np.array([data[i + context_len] for i in range(T - context_len - future_len + 1)])
split = int(samples.shape[0] * 0.7)
test_samples = samples[split:]  # (Ntest, 750)

# --- Split into nodes ---
nodes = ['CC1', 'CC2', 'LW1']
slices = [slice(0, 250), slice(250, 500), slice(500, 750)]

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
freqs = list(range(87, 337))

for col, (node, sl) in enumerate(zip(nodes, slices)):
    gt = test_samples[:, sl].T  # (250, Ntest)

    # Color limits: 1st/99th percentile (same as test.py)
    vmin, vmax = np.percentile(gt, [1, 99])

    im = axes[col].imshow(
        gt,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=vmin, vmax=vmax,
        extent=[0, gt.shape[1], freqs[0], freqs[-1]]
    )
    axes[col].set_title(f'{node}', fontsize=14)
    axes[col].set_xlabel('Test Sample Index', fontsize=11)
    if col == 0:
        axes[col].set_ylabel('Frequency (MHz)', fontsize=11)

fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label='Power (dBm)')
plt.suptitle('Ground Truth Spectrograms (87–336 MHz)', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
print(f"Saved: {OUTPUT}")
print(f"Test samples: {test_samples.shape[0]}, Freq bins: 250, Freq range: {freqs[0]}-{freqs[-1]} MHz")
