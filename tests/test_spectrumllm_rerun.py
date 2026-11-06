import numpy as np

from training.ray import spectrumllm_rerun as R


def test_val_baseline_matches_a_hand_computation_on_a_ramp():
    n = 1000
    raw = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    got = R.val_lookback_mean_mse_std(raw, horizon=10, seq=20)
    n_train = int(n * 0.7)
    sd = raw[:n_train].std(axis=0)
    num_test = int(n * 0.2)
    starts = range(n_train - 20, n_train + (n - n_train - num_test) - 20 - 10 + 1)
    want = np.mean([(((raw[s:s + 20].mean(0, keepdims=True) - raw[s + 20:s + 30]) / sd) ** 2).mean() for s in starts])
    assert abs(got - want) < 1e-9


def test_collapse_flag_needs_a_three_percent_margin_over_the_baseline():
    assert R.is_collapsed(0.51, 0.52)            # about 2% better: not a real improvement
    assert R.is_collapsed(0.52, 0.52)
    assert not R.is_collapsed(0.31, 0.52)


def test_result_key_layout():
    assert R.result_key("p/", "3e-4", 25, 2) == "p/lr3e-4/n25/seed-2.json"
