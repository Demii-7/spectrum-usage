"""Frozen-input shift tests for 3D-SwinSTB (``deepspred2d``, 15 minutes, 600-800 by default): a thin wrapper of ``training.ray.frequency_ablation_classes``.

3D-SwinSTB renders spectrogram windows into colour frames INSIDE the model (``models/DeepSPredForecaster.py``): the pipeline hands it per-frequency z-scored dB windows
(statistics of the checkpoint's training split, fixed) and the model maps z to [0, 1] with the checkpoint's own ``z_clip`` (a persistent buffer, restored by
``load_state_dict``; nothing is refitted on test or shifted data), renders the colormap and inverts the predicted colours back to z (then dB). The class-crossed driver
shifts the raw dB series of the disturbed bins in time BEFORE normalization (24 h = 96 steps, 36 h = 144 steps; the same wrap handling and window alignment as for every other
model), so the shift reaches the rendered image exactly as a shift of those columns of the z-scored window would. Conditions default to the four of the paper table:
all x shift96, all x shift144, P_only x shift144, N_only x shift144 (real is always kept); results under ``frequency-ablation-2026-11-classes-deepspred``.

    python -m training.ray.frequency_ablation_deepspred --seeds 40 41 42 43 44 --skip-done
"""
from __future__ import annotations

import sys

import training.ray.frequency_ablation as fa
import training.ray.frequency_ablation_classes as fc

MODEL = "deepspred2d"
for module in (fa, fc):
    if MODEL not in module.DIRECT:
        module.DIRECT = tuple(module.DIRECT) + (MODEL,)

DEFAULTS = ["--models", MODEL, "--bands", "600_800", "--prefix", "frequency-ablation-2026-11-classes-deepspred",
            "--only-conditions", "all_shift96", "all_shift144", "P_only_shift144", "N_only_shift144",
            "--gpu-fraction", "0.25", "--memory-gb", "12", "--skip-done"]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]          # explicit options override the defaults (last occurrence wins)
    fc.main()
