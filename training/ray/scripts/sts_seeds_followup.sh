#!/usr/bin/env bash
# After the seed-40 STS-PredNet variant runs and the Time-LLM 1-minute finals end, run STS-PredNet variants for seeds 41 and 42 (both bands).
cd /workspace/spectrum-usage || exit 1
C="python -u -m training.ray.spatial15 --tuned --skip-done --models stsprednet --experiments s2 s3 s5 --seeds 41 42 --max-parallel 6 --gpu-fraction 0.4 --memory-gb 24 --prefix spatial15-tuned-oct2026"
exec python -u training/ray/scripts/run_after.py --wait-ids claude-final-s15sts-full-600-2 claude-final-s15sts-full-2400-2 claude-final-timellm1min-finals-3 \
  --then "$C --band 600_800" --then "$C --band 2400_2600"
