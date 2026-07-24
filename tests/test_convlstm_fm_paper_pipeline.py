import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).parents[1]
PIPELINE = ROOT / "training" / "ConvLSTM-FM"
sys.path.insert(0, str(PIPELINE))

from preprocessing import (  # noqa: E402
    SpectrogramConfig,
    iq_to_slices,
    recordings_to_sentences,
    sentences_to_tokens,
    tokens_to_sentences,
)
from segmentation import row_normalized_confusion_matrix, signal_noise_labels  # noqa: E402
from models.ConvLSTM_FM_Segmentation import ConvLSTMFMSegmentation  # noqa: E402


def test_iq_preprocessing_shapes_and_sentence_boundaries():
    config = SpectrogramConfig(
        sample_rate=8_000,
        n_fft=16,
        win_length=8,
        hop_length=8,
        sentence_slices=2,
        image_size=(32, 32),
        token_count=4,
    )
    first = np.ones(16 * 5, dtype=np.complex64)
    second = np.ones(16 * 3, dtype=np.complex64) * (2 + 1j)
    assert iq_to_slices(first, config).shape == (5, 16, 2)
    sentences, recording_ids = recordings_to_sentences([first, second], config)
    assert sentences.shape == (3, 1, 32, 32)
    assert recording_ids.tolist() == [0, 0, 1]


def test_token_shape_and_exact_roundtrip():
    sentences = torch.randn(2, 1, 256, 256)
    tokens = sentences_to_tokens(sentences)
    assert tokens.shape == (2, 16, 1, 256, 16)
    torch.testing.assert_close(tokens_to_sentences(tokens), sentences)


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(1, 64, 1)

    def forward(self, tokens):
        b, t, _, h, w = tokens.shape
        output = self.projection(tokens.reshape(b * t, 1, h, w)).reshape(b, t, 64, h, w)
        return [output], []


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()


def test_segmentation_concatenates_width_and_freezes_backbone():
    model = ConvLSTMFMSegmentation(_FakeBackbone())
    output = model(torch.randn(2, 4, 1, 8, 3))
    assert output.shape == (2, 3, 8, 12)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    model.train()
    assert not model.backbone.training


def test_segmentation_metrics_and_binary_conversion():
    labels = torch.tensor([[0, 1], [2, 2]])
    prediction = torch.tensor([[0, 2], [2, 1]])
    confusion = row_normalized_confusion_matrix(prediction, labels)
    torch.testing.assert_close(confusion.sum(1), torch.ones(3, dtype=torch.float64))
    assert signal_noise_labels(labels).tolist() == [[0, 1], [1, 1]]
