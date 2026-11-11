import numpy as np
import pytest

from training.ray import moment_zeroshot as M


def test_masked_input_layout():
    context = np.arange(2 * 3 * 96, dtype=np.float32).reshape(2, 3, 96)
    x, input_mask, mask = M.build_masked_input(context, 48)
    assert x.shape == (2, 3, 512) and input_mask.shape == mask.shape == (2, 512)
    assert np.array_equal(x[:, :, :96], context) and not x[:, :, 96:].any()          # future and padding hold zeros
    assert input_mask[:, :144].all() and not input_mask[:, 144:].any()                # context and masked future attend; padding does not
    assert mask[:, :96].all() and not mask[:, 96:].any()                              # only the context is observed


def test_masked_steps_must_be_whole_patches_and_fit():
    context = np.zeros((1, 1, 96), dtype=np.float32)
    with pytest.raises(ValueError):
        M.build_masked_input(context, 12)
    with pytest.raises(ValueError):
        M.build_masked_input(context, 0)
    with pytest.raises(ValueError):
        M.build_masked_input(np.zeros((1, 1, 500), dtype=np.float32), 24)


def test_patches_and_read_out():
    assert [M.patches_for(h) for h in M.HORIZONS] == [1, 1, 2, 3, 6]
    recon = np.arange(512, dtype=np.float32)[None, None, :]
    got = M.read_out(recon, 96)
    assert [int(got[h][0, 0]) for h in M.HORIZONS] == [96, 99, 107, 119, 143]          # step context + h - 1
