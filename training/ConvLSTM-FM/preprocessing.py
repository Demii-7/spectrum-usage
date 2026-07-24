"""Paper-native complex-IQ preprocessing for ConvLSTM-FM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: float
    slice_seconds: float = 0.002
    n_fft: int = 1024
    win_length: int = 512
    hop_length: int = 512
    value: str = "power"
    log_scale: bool = True
    log_floor: float = 1e-12
    fftshift: bool = True
    sentence_slices: int | None = None
    min_sentence_slices: int = 5
    max_sentence_slices: int = 10
    sentence_seed: int = 42
    image_size: tuple[int, int] = (256, 256)
    token_count: int = 16

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def iq_to_slices(iq: np.ndarray | torch.Tensor, config: SpectrogramConfig) -> torch.Tensor:
    """Convert one complex recording to non-overlapping 2 ms spectrogram slices."""
    samples = torch.as_tensor(iq)
    if not torch.is_complex(samples) or samples.ndim != 1:
        raise ValueError("IQ input must be a one-dimensional complex array")
    slice_samples = int(round(config.sample_rate * config.slice_seconds))
    if slice_samples < config.win_length:
        raise ValueError("A slice must contain at least win_length samples")
    count = samples.numel() // slice_samples
    if count == 0:
        return torch.empty((0, config.n_fft, 0), dtype=torch.float32)
    samples = samples[: count * slice_samples].reshape(count, slice_samples)
    if config.n_fft < config.win_length:
        raise ValueError("n_fft must be at least win_length")
    window = torch.hann_window(
        config.win_length, device=samples.device, dtype=samples.real.dtype
    )
    # Frame by win_length, not n_fft. torch.stft(center=False) frames by n_fft,
    # which would incorrectly discard valid paper windows when n_fft > 512.
    framed = samples.unfold(1, config.win_length, config.hop_length) * window
    spectra = torch.fft.fft(framed, n=config.n_fft, dim=-1).transpose(1, 2)
    if config.value == "power":
        spectra = spectra.abs().square()
    elif config.value == "magnitude":
        spectra = spectra.abs()
    else:
        raise ValueError("value must be 'power' or 'magnitude'")
    if config.log_scale:
        spectra = torch.log10(spectra.clamp_min(config.log_floor))
    if config.fftshift:
        spectra = torch.fft.fftshift(spectra, dim=1)
    return spectra.float()


def slices_to_sentences(slices: torch.Tensor, config: SpectrogramConfig) -> torch.Tensor:
    """Build successive 10-20 ms sentences and resize to the paper's 256 square."""
    if slices.ndim != 3:
        raise ValueError("slices must have shape (N, frequency, time)")
    if config.sentence_slices is not None:
        if config.sentence_slices <= 0:
            raise ValueError("sentence_slices must be positive")
        lengths = [config.sentence_slices] * (len(slices) // config.sentence_slices)
    else:
        if not 1 <= config.min_sentence_slices <= config.max_sentence_slices:
            raise ValueError("sentence slice bounds must satisfy 1 <= min <= max")
        generator = torch.Generator().manual_seed(config.sentence_seed)
        lengths = []
        remaining = len(slices)
        while remaining >= config.min_sentence_slices:
            upper = min(config.max_sentence_slices, remaining)
            length = int(torch.randint(
                config.min_sentence_slices,
                upper + 1,
                (1,),
                generator=generator,
            ).item())
            lengths.append(length)
            remaining -= length
    if not lengths:
        return slices.new_empty((0, 1, *config.image_size))
    images = []
    offset = 0
    for length in lengths:
        sentence = slices[offset:offset + length].permute(1, 0, 2).reshape(
            1, 1, slices.shape[1], -1
        )
        images.append(F.interpolate(
            sentence, size=config.image_size, mode="bilinear", align_corners=False
        ))
        offset += length
    return torch.cat(images)


def recordings_to_sentences(
    recordings: Iterable[np.ndarray | torch.Tensor], config: SpectrogramConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sentences independently per recording and return their recording IDs."""
    batches, recording_ids = [], []
    for recording_id, recording in enumerate(recordings):
        sentences = slices_to_sentences(iq_to_slices(recording, config), config)
        if len(sentences):
            batches.append(sentences)
            recording_ids.append(torch.full((len(sentences),), recording_id, dtype=torch.long))
    if not batches:
        return torch.empty((0, 1, *config.image_size)), torch.empty(0, dtype=torch.long)
    return torch.cat(batches), torch.cat(recording_ids)


def sentences_to_tokens(sentences: torch.Tensor, token_count: int = 16) -> torch.Tensor:
    """Split sentence width into ConvLSTM tokens: (B, T, 1, H, Wtoken)."""
    if sentences.ndim == 3:
        sentences = sentences.unsqueeze(1)
    if sentences.ndim != 4 or sentences.shape[1] != 1:
        raise ValueError("sentences must have shape (B, 1, H, W)")
    if sentences.shape[-1] % token_count:
        raise ValueError("sentence width must be divisible by token_count")
    return sentences.reshape(
        sentences.shape[0], 1, sentences.shape[2], token_count, sentences.shape[3] // token_count
    ).permute(0, 3, 1, 2, 4).contiguous()


def tokens_to_sentences(tokens: torch.Tensor) -> torch.Tensor:
    """Invert :func:`sentences_to_tokens` without interpolation."""
    if tokens.ndim != 5 or tokens.shape[2] != 1:
        raise ValueError("tokens must have shape (B, T, 1, H, Wtoken)")
    return tokens.permute(0, 2, 3, 1, 4).reshape(
        tokens.shape[0], 1, tokens.shape[3], tokens.shape[1] * tokens.shape[4]
    )
