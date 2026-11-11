#!/usr/bin/env bash
# After the STS-PredNet variant queue (seed 40, then seeds 41-42) has ended, run STS-PredNet on the kriging map (S2 variant core4-krige).
cd /workspace/spectrum-usage || exit 1
C="python -u -m training.ray.spatial15 --tuned --skip-done --models stsprednet --experiments s2 --variants core4-krige --seeds 40 41 42 --max-parallel 6 --gpu-fraction 0.4 --memory-gb 24 --prefix spatial15-tuned-oct2026"
exec python -u training/ray/scripts/run_after.py --wait-ids claude-final-s15sts-seeds-1 \
  --then "$C --band 600_800" --then "$C --band 2400_2600"
