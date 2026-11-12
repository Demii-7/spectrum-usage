import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("poller", Path(__file__).resolve().parents[1] / "training/ray/scripts/dswin_tuned_p4krige_poller.py")
poller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poller)


def test_decide():
    assert poller.decide(None)[0] == "wait"
    assert poller.decide({"verdict": "no completed trials"})[0] == "wait"
    assert poller.decide({"verdict": "improves", "improvement": 0.05, "winner": "w", "winner_index": 5})[0] == "run"
    assert poller.decide({"verdict": "does not improve", "improvement": 0.02, "winner": "w", "winner_index": 5})[0] == "skip"
    assert "NOT run" in poller.decide({"verdict": "does not improve", "improvement": 0.02, "winner": "w", "winner_index": 5})[1]
    assert poller.decide({"verdict": "improves", "improvement": 0.05, "winner": "w", "winner_index": 2})[0] == "skip"
    assert poller.decide({"verdict": "improves", "improvement": 0.03, "winner": "w", "winner_index": 0})[0] == "run"


def test_entrypoints_name_the_new_prefix_and_no_gpu_for_table():
    e = poller.entrypoint("600_800")
    assert "spatial15-tuned-dswin-oct2026" in e and "core4-p4 core4-krige" in e and "--fixed-config s3://spectrum/analysis-outputs/dswin_tuned_fixed_600_800.json" in e
    assert "gpu" not in poller.table_entrypoint().lower().replace("spectrum", "")
