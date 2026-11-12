import sys
from pathlib import Path
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'training' / 'DeepSPred' / 'repro'))
from fm_data import load_frames_spec
spec, ds = sys.argv[1], sys.argv[2]
frames, meta = load_frames_spec(REPO / 'data' / 'datasets' / 'deepspred' / ds, spec, range_scope='full', workers=20)
print(f"{ds}/{spec} done: {frames.shape} db_range={meta['db_range']}", flush=True)
