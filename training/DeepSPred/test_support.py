import numpy as np
import pytest

from config_support import resolve_deepspred_config

try:
    from dataset import _pad_w
    from train_integrated import _frame_segments
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    _pad_w = None
    _frame_segments = None


MODEL = {"patch_size": [1, 2, 2], "embed_dim": 8, "depths": [1, 1, 1], "num_heads": [1, 1, 1], "window_size": [1, 1, 1]}


def test_normalized_and_legacy_config_are_callable():
    normalized = {"deepspred": {"model": MODEL, "train": {"batch_size": 3}, "frames": {"w_pad": 16}}}
    legacy = {"deepspred": {"model": MODEL, "batch_size": 3, "w_pad": 16}}
    assert resolve_deepspred_config(normalized, 10)["train"]["batch_size"] == 3
    assert resolve_deepspred_config(legacy, 10)["frames"]["w_pad"] == 16


def test_width_is_validated():
    with pytest.raises(ValueError, match="at least input width"):
        resolve_deepspred_config({"deepspred": {"model": MODEL, "w_pad": 8}}, 10)
    if _pad_w is not None:
        with pytest.raises(ValueError, match="at least frame width"):
            _pad_w(np.zeros((1, 2, 10, 3), dtype=np.float32), 8)


def test_frame_segments_keep_only_complete_frames():
    if _frame_segments is None:
        pytest.skip("PyTorch is not installed")
    class Segment:
        def __init__(self, start, end, label):
            self.start, self.end, self.label = start, end, label

    framed = _frame_segments((Segment(2, 18, "a"), Segment(20, 40, "b")), 0, 40, 10)
    assert [(s.start, s.end, s.label) for s in framed] == [(2, 4, "b")]
