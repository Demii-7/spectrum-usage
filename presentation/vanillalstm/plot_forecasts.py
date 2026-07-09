import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.plot_forecasts import generate_all_plots

OUT = ROOT / "presentation" / "vanillalstm"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "VanillaLSTM"
MAX_STEPS = 500
BANDS = [
    {"id": "600_800", "label": "600-800 MHz", "path": "powder/600_800"},
    {"id": "2400_2600", "label": "2400-2600 MHz", "path": "powder/2400_2600"},
]

for band in BANDS:
    results_dir = ROOT / "training" / "results" / band["path"] / MODEL
    if not results_dir.exists():
        print(f"Skipping {results_dir} (not found)")
        continue
    print(f"Generating plots for {band['label']}...")
    generate_all_plots(
        results_dir=results_dir,
        model_name=MODEL,
        out_dir=OUT,
        bins=(30, 50),
        max_steps=MAX_STEPS,
    )
