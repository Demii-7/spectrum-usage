from training.ray.inference_benchmark import pick_checkpoints, summarize_calls


def test_pick_prefers_seed_42_else_lowest():
    entries = [{"prefix": "p", "rep": "1d", "model": "m", "seed": s} for s in (40, 41, 42)]
    entries += [{"prefix": "p", "rep": "2d", "model": "n", "seed": s} for s in (41, 44)]
    picked = {(e["rep"], e["model"]): e["seed"] for e in pick_checkpoints(entries)}
    assert picked == {("1d", "m"): 42, ("2d", "n"): 41}


def test_summarize_drops_warmup_and_partial_batch():
    seconds = [9.0, 9.0, 9.0, 0.10, 0.20, 0.30, 0.01]
    rows = [32, 32, 32, 32, 32, 32, 5]
    out = summarize_calls(seconds, rows)
    assert out["n_timed_calls"] == 3 and out["batch"] == 32
    assert abs(out["median_s_per_call"] - 0.20) < 1e-9 and abs(out["ms_per_window"] - 6.25) < 1e-9
    assert summarize_calls([1.0], [4])["n_timed_calls"] == 0
