import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.common.forecasting import RolloutRemask, autoregressive_rollout, make_rollout_remask  # noqa: E402


class Echo(torch.nn.Module):
    """One-step model whose output is its last input timestep plus 1 (so a fed-back value is visible in the next output)."""

    def forward(self, window):
        return (window[:, -1:] + 1.0)


def reference_rollout(model, x, steps):
    predictions, window = [], x
    for _ in range(steps):
        nxt = model(window)[:, 0]
        predictions.append(nxt)
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(predictions, dim=1)


def make_window(batch=6, length=12, cols=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, cols, generator=g)
    x[:, :, 4:] = 3.0 + 0.5 * torch.randn(batch, length, 4, generator=g)       # columns 4..7: the masked (noise) columns, mean 3, std 0.5
    return x


def test_default_rollout_is_unchanged_when_the_flag_is_off():
    x = make_window()
    assert torch.equal(autoregressive_rollout(Echo(), x, 5), reference_rollout(Echo(), x, 5))
    assert torch.equal(autoregressive_rollout(Echo(), x, 5, remask=None), reference_rollout(Echo(), x, 5))


def test_remask_replaces_only_the_out_of_region_columns_of_the_fed_back_step_with_calibrated_noise_and_is_reproducible():
    x = make_window()
    remask = RolloutRemask(torch.arange(4, 8), seed=7)
    out = autoregressive_rollout(Echo(), x, 6, remask=remask)
    again = autoregressive_rollout(Echo(), x, 6, remask=remask)
    other = autoregressive_rollout(Echo(), x, 6, remask=RolloutRemask(torch.arange(4, 8), seed=8))
    assert torch.equal(out, again) and not torch.equal(out, other)
    # step 1 is a function of the real window only: unchanged; the returned forecast keeps the model's own output
    assert torch.equal(out[:, 0], reference_rollout(Echo(), x, 1)[:, 0])
    # in-region columns follow the plain recursion (+1 per step) because they are never replaced
    assert torch.allclose(out[:, :, :4], reference_rollout(Echo(), x, 6)[:, :, :4])
    # from step 2 the out-of-region outputs are the previous FRESH noise + 1: mean about 3 + 1, std about 0.5 (the window's calibration), not the plain recursion
    noise_out = out[:, 1:, 4:]
    assert not torch.allclose(noise_out, reference_rollout(Echo(), x, 6)[:, 1:, 4:])
    assert abs(float(noise_out.mean()) - 4.0) < 0.35 and 0.3 < float(noise_out.std()) < 0.8


def test_make_rollout_remask_needs_the_config_flag_and_picks_the_columns_outside_the_region():
    freqs = np.arange(600.5, 610.5, 1.0)
    cfg = {"data": {"mask": {"frequency_ranges": [[602.5, 604.5]], "seed": 3}}}
    assert make_rollout_remask(cfg, freqs) is None
    cfg["data"]["mask"]["rollout_remask"] = True
    r = make_rollout_remask(cfg, freqs)
    assert r.columns.tolist() == [0, 1, 5, 6, 7, 8, 9] and r.seed == 4
