import sys
from pathlib import Path

from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'training' / 'DeepSPred' / 'repro'))

from fm_data import load_frames_spec

spec = sys.argv[1]
data = REPO / 'data' / 'datasets' / 'deepspred' / 'FM-Spectrum-Dataset'
frames, meta = load_frames_spec(data, spec, range_scope='full', workers=20)
print(f"{spec} done: {frames.shape} {frames.dtype} db_range={meta['db_range']}", flush=True)
