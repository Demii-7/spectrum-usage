from training.ray.inference_benchmark import effective_stride, parse_direct_key, parse_meta_key, stride_key, pick_checkpoints, spectra_per_window, summarize_calls


def entry(source, rep, model, seed, band="600_800", horizon=None, prefix="p"):
    return {"source": source, "prefix": prefix, "band": band, "rep": rep, "model": model, "seed": seed, "horizon": horizon}


def test_pick_prefers_seed_42_else_lowest_and_keeps_horizons():
    entries = [entry("oct", "1d", "m", s) for s in (40, 41, 42)]
    entries += [entry("p15", "2d", "n", s, horizon=h) for s in (41, 44) for h in (4, 12)]
    picked = {(e["source"], e["model"], e["horizon"]): e["seed"] for e in pick_checkpoints(entries)}
    assert picked == {("oct", "m", None): 42, ("p15", "n", 4): 41, ("p15", "n", 12): 41}


def test_summarize_drops_warmup_and_partial_batch():
    out = summarize_calls([9.0, 9.0, 9.0, 0.10, 0.20, 0.30, 0.01], [32, 32, 32, 32, 32, 32, 5])
    assert out["n_timed_calls"] == 3 and out["batch"] == 32
    assert abs(out["median_s_per_call"] - 0.20) < 1e-9 and abs(out["ms_per_window"] - 6.25) < 1e-9
    assert summarize_calls([1.0], [4])["n_timed_calls"] == 0


def test_parse_meta_keys():
    rollout = parse_meta_key("powder15-results-w/weights/600_800/2d/vanillalstm/seed-42/powder15_600_800_vanillalstm.pt")
    assert rollout["band"] == "600_800" and rollout["rep"] == "2d" and rollout["model"] == "vanillalstm"
    assert rollout["horizon"] is None and rollout["seed"] == 42 and rollout["meta_key"].endswith(".pt.meta.json")
    direct = parse_meta_key("powder15-results-w/weights/2400_2600/1d/linearar1d/h12/seed-40/powder15_2400_2600_linearar1d.pt")
    assert direct["horizon"] == 12 and direct["prefix"] == "powder15-results-w"
    tuned = parse_meta_key("tuning/powder15/2400_2600/1d/linearar1d/weights/seed-40/powder15_2400_2600_linearar1d.pt")
    assert tuned["prefix"] == "tuning/powder15" and tuned["model"] == "linearar1d" and tuned["horizon"] is None
    assert parse_meta_key("powder15-results-w/weights/600_800/2d/x/site-humanities/seed-40/f.pt") is None


def test_stride_and_spectra_per_window():
    assert effective_stride("1d", 20) == 400 and effective_stride("2d", 20) == 20 and effective_stride("4d", 1) == 1
    assert spectra_per_window("1d") == 1 / 200 and spectra_per_window("2d") == 1.0 and spectra_per_window("4d") == 4.0


def test_direct_keys_and_stride_choice():
    e = parse_direct_key("direct-tuning/1min/2400_2600/timellm/weights/h60/seed-40/powder_2400_2600_spectrumllm.pt")
    assert (e["resolution"], e["band"], e["model"], e["pipeline"], e["horizon"], e["seed"]) == ("1min", "2400_2600", "timellm", "spectrumllm", 60, 40)
    assert e["meta_key"].endswith(".pt.meta.json") and e["rep"] == "2d"
    assert parse_direct_key("direct-tuning/15min/600_800/unknownmodel/weights/h4/seed-40/x.pt") is None
    assert parse_direct_key("direct-tuning/15min/600_800/ofa/trials/c00_h4.json") is None
    assert stride_key({"source": "direct", "resolution": "1min"}) == "oct" and stride_key({"source": "direct", "resolution": "15min"}) == "p15"
    assert stride_key({"source": "band"}) == "band"
