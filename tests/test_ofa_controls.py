import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from models.OFA import OFAForecaster

BASE = {"input_size": 3, "input_sequence_length": 48, "prediction_horizon": 4, "patch_length": 8, "patch_stride": 4, "llm_layers": 2,
        "channel_independent": True}


def counts(**extra):
    model = OFAForecaster({"model": {**BASE, **extra}})
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    return model, trainable, sum(p.numel() for p in model.backbone.parameters())


def test_controls_have_the_same_shape_and_the_right_trainable_sets():
    pre, pre_trainable, total = counts()
    rand, rand_trainable, rand_total = counts(backbone_init="random")
    scratch, scratch_trainable, scratch_total = counts(backbone_init="random", backbone_trainable="all")
    assert total == rand_total == scratch_total                         # identical architecture
    assert pre_trainable == rand_trainable < total                      # OFA's rule: only ln and wpe train
    assert scratch_trainable == total                                   # every backbone weight trains
    x = torch.randn(2, 48, 3)
    assert pre(x).shape == rand(x).shape == scratch(x).shape == (2, 4, 3)


def test_random_init_differs_from_pretrained_weights():
    pre, _, _ = counts()
    rand, _, _ = counts(backbone_init="random")
    a = dict(pre.backbone.named_parameters())["h.0.attn.c_attn.weight"]
    b = dict(rand.backbone.named_parameters())["h.0.attn.c_attn.weight"]
    assert not torch.allclose(a, b)
