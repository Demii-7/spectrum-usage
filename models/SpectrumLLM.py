"""SpectrumLLM forecaster after Liu et al., IEEE Wireless Communications 2026.

Pipeline, in the order the paper gives it (subsections "RSS Series Tokenization &
Embedding", "RSS Prompt Design & Embedding", "LLM Finetuning & Output"):

    RSS series -> RevIN -> sliding-window patch tokenization -> learnable patch
    embedding -> prototype cross-attention alignment -> prompt-as-prefix ->
    (frozen) pretrained LLM -> linear projection head -> RevIN inverse

RESOLUTION RULE (one implementation, no variant arms). The magazine article is a
framework description and leaves most numbers unassigned; it cites Time-LLM as its
instantiation. Where the article states something, the article wins. Where the
article is silent, the value is taken from Time-LLM's released code
(github.com/KimMeen/Time-LLM @ b13e881f86cd0475ce1b72c17110430663334955, Apache-2.0)
and tagged `[T:file:line]` below. The full item-by-item ledger is in
training/SpectrumLLM/repro/TIMELLM_OFA_REFERENCE.md.

From the ARTICLE:
  - one patch token spans every channel, x_i in R^{N x P} where P = patch_length = 16 (its eq. in sec. IV-B)
    (the article writes L; it is the patch, not the 192-step lookback)
  - f_embed is "a simple MLP"
  - backbone is a lightweight pretrained LLM (BERT / GPT-2), fully frozen
  - the head is a linear layer with output dimension K

From TIME-LLM (article silent):
  - patch_len 16, stride 8                               [T:models/TimeLLM.py:32]
  - 1000 prototypes, linearly probed from the vocabulary  [T:models/TimeLLM.py:178]
  - top-5 lags computed by FFT autocorrelation        [T:models/TimeLLM.py:257-264]
  - RevIN with affine=False                              [T:models/TimeLLM.py:192]
  - ReplicationPad1d((0, stride)) before unfolding, so the patch count is
    (L - P)/S + 2                                   [T:layers/Embed.py:166,180]
  - the LLM hidden states are truncated to their first d_ff dims before the head
                                                        [T:models/TimeLLM.py:244]
  - backbone truncated to llm_layers = 6                  [T:models/TimeLLM.py:46]
  - attention heads 8, dropout 0.1, d_ff 32, MSE loss              [T:run_main.py]

input:  (batch, input_sequence_length, input_size)
output: (batch, prediction_horizon, input_size)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022), as the paper requires.

    Statistics are taken per instance and per channel over the time axis, kept,
    and reapplied by denormalize() so the head predicts on the original scale.
    affine defaults to False, which is Time-LLM's setting [T:models/TimeLLM.py:192];
    the article does not say.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        out = (x - self._mean) / self._stdev
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("RevIN.denormalize called before normalize")
        out = x
        if self.affine:
            out = (out - self.bias) / (self.weight + self.eps * self.eps)
        return out * self._stdev + self._mean


class PatchTokenizer(nn.Module):
    """Sliding-window tokenization: one token per window of patch_length steps.

    One token spans every channel, which is the article's equation x_i in R^{N x L}.
    (Time-LLM instead patches each channel independently; the article's literal
    reading wins here, and no per-channel variant is kept.)

    The series is replication-padded by `stride` on the right before unfolding,
    which is Time-LLM's PatchEmbedding [T:layers/Embed.py:166,180]; the patch count
    is therefore (L - P)/S + 2 rather than (L - P)/S + 1.
    """

    def __init__(self, patch_length: int, stride: int) -> None:
        super().__init__()
        if patch_length < 1:
            raise ValueError("patch_length must be at least 1")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.patch_length = patch_length
        self.stride = stride

    def num_patches(self, sequence_length: int) -> int:
        if sequence_length < self.patch_length:
            raise ValueError(
                f"input_sequence_length {sequence_length} is shorter than "
                f"patch_length {self.patch_length}"
            )
        return (sequence_length + self.stride - self.patch_length) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, time, channels) -> (batch, patches, patch_length * channels)."""
        batch, _, channels = x.shape
        padded = F.pad(x.transpose(1, 2), (0, self.stride), mode="replicate")
        windows = padded.unfold(dimension=-1, size=self.patch_length, step=self.stride)
        # (batch, channels, patches, patch_length) -> (batch, patches, channels * patch_length)
        return windows.permute(0, 2, 1, 3).reshape(batch, windows.shape[2], -1)


class PrototypeReprogramming(nn.Module):
    """Align patch embeddings to a small prototype set drawn from the LLM vocabulary.

    The prototypes are a linear probe of the pretrained word-embedding matrix
    (paper eqs. (1)-(2), after Time-LLM): num_prototypes prototypes rather than the
    full vocabulary, then cross-attention with patch embeddings as queries.
    """

    def __init__(self, d_model: int, d_llm: int, vocab_size: int, num_prototypes: int,
                 num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_llm % num_heads != 0:
            raise ValueError(f"d_llm {d_llm} must be divisible by num_heads {num_heads}")
        self.mapping_layer = nn.Linear(vocab_size, num_prototypes)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_llm, num_heads=num_heads, dropout=dropout,
            kdim=d_llm, vdim=d_llm, batch_first=True,
        )
        self.query_projection = nn.Linear(d_model, d_llm)

    def forward(self, patch_embeddings: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        prototypes = self.mapping_layer(word_embeddings.transpose(0, 1)).transpose(0, 1)
        prototypes = prototypes.unsqueeze(0).expand(patch_embeddings.shape[0], -1, -1)
        queries = self.query_projection(patch_embeddings)
        aligned, _ = self.cross_attention(queries, prototypes, prototypes, need_weights=False)
        return aligned


def autocorrelation_lags(x: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k autocorrelation lags, computed by FFT as Time-LLM does.

    [T:models/TimeLLM.py:257-264]. x is (batch, time, channels); the autocorrelation
    is taken along time and averaged over channels, then the k strongest lags are
    returned as indices. The article asks the prompt to carry "lags" and does not
    say how they are obtained.
    """
    series = x.transpose(1, 2)                       # (batch, channels, time)
    spectrum = torch.fft.rfft(series, dim=-1)
    corr = torch.fft.irfft(spectrum * torch.conj(spectrum), n=series.shape[-1], dim=-1)
    mean_corr = corr.mean(dim=1)                     # (batch, time)
    k = min(top_k, mean_corr.shape[-1])
    return torch.topk(mean_corr, k, dim=-1).indices


def series_statistics(x: torch.Tensor, num_lags: int) -> list[dict[str, Any]]:
    """Per-instance descriptive statistics the paper's prompt template carries.

    min / max / median / trend / top-k lags, matching Fig. 3 of the article. The
    trend is the sign of the summed first difference [T:models/TimeLLM.py:211,226];
    the lags are FFT autocorrelation lags, not the largest samples.
    """
    detached = x.detach()
    if detached.dim() == 2:
        detached = detached.unsqueeze(-1)
    lags = autocorrelation_lags(detached, num_lags)
    flat = detached.reshape(detached.shape[0], -1)
    stats: list[dict[str, Any]] = []
    for index in range(flat.shape[0]):
        series = flat[index]
        trend = "upward" if float(detached[index].diff(dim=0).sum()) > 0 else "downward"
        stats.append({
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "trend": trend,
            "lags": [int(v) for v in lags[index].tolist()],
        })
    return stats


class SpectrumLLMForecaster(nn.Module):
    """Direct multi-step RSS forecaster built on a pretrained language model."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.input_size = int(model_config["input_size"])
        self.input_sequence_length = int(model_config["input_sequence_length"])
        self.prediction_horizon = int(model_config.get("prediction_horizon", 1))
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")

        self.patch_length = int(model_config.get("patch_length", 16))
        self.patch_stride = int(model_config.get("patch_stride", 8))
        self.d_model = int(model_config.get("d_model", 32))
        self.num_prototypes = int(model_config.get("num_prototypes", 1000))
        self.num_heads = int(model_config.get("attention_heads", 8))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.num_lags = int(model_config.get("prompt_top_lags", 5))
        self.prompt_mode = str(model_config.get("prompt_mode", "static")).lower()
        # Which series the prompt's "Input statistics" describe. Time-LLM builds the
        # prompt from the series as fed to the patcher, i.e. AFTER RevIN, so min/max/median
        # are z-scores and the article's "input statistics" lose their physical meaning.
        # "raw" (default) describes the un-normalised input instead, which is what the
        # article's template implies; "normalized" reproduces the earlier runs.
        self.prompt_stats = str(model_config.get("prompt_stats", "raw")).lower()
        if self.prompt_stats not in {"raw", "normalized"}:
            raise ValueError("prompt_stats must be 'raw' or 'normalized'")
        if self.prompt_mode not in {"static", "none"}:
            raise ValueError("prompt_mode must be 'static' or 'none'")
        self.backbone_name = str(model_config.get("backbone", "openai-community/gpt2"))
        self.llm_layers = int(model_config.get("llm_layers", 6))
        self.domain_hint = str(
            model_config.get("domain_hint", "radio spectrum power spectral density")
        )

        # Channel handling. The article's eq. (1) makes a token the WHOLE multivariate patch
        # x_i in R^{N x L}; Time-LLM, whose released code produced the article's tables, is
        # channel-independent -- it folds channels into the batch and embeds L -> d_model per
        # series, and the article's own sentence "a linear layer with an output dimension of K"
        # (not K x N) matches that. The all-channel token projects N*patch_length -> d_model
        # with one shared linear map: 64 -> 32 at N=4, 400 -> 32 at N=25, 2416 -> 32 at N=151.
        # Do NOT read that as a fixed information loss. The article states no d_model, no patch
        # length and no stride; d_model=32 is ours (Time-LLM's own default is 16) and stride=8
        # is Time-LLM's. The AGGREGATE representation is num_patches*d_model, which at stride 1
        # would be 178*32 = 5696 against 192*25 = 4800 input values -- an expansion, not a
        # bottleneck. Only the per-token width is pinched, and overlapping windows recover much
        # of it. Our measured all-channel penalty (1.37x at N=25, 1.08x at N=4) was obtained
        # with d_model held at 32 for every N and stride fixed at 8, so it CONFOUNDS channel
        # fusion with an under-provisioned embedding; a width-scaled control has not been run.
        # The default (False) follows the article's own equation; set True to follow the
        # Time-LLM code that produced its tables. Neither reading reproduces both scenarios:
        # on 4 channels all-channel is closer, on 25 channel-independent is.
        self.channel_independent = bool(model_config.get("channel_independent", False))
        self._series_width = 1 if self.channel_independent else self.input_size

        self.revin = RevIN(self.input_size, affine=bool(model_config.get("revin_affine", False)))
        self.tokenizer_module = PatchTokenizer(self.patch_length, self.patch_stride)
        self._patches = self.tokenizer_module.num_patches(self.input_sequence_length)

        self.patch_embedding = nn.Sequential(
            nn.Linear(self.patch_length * self._series_width, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.backbone, self.text_tokenizer, self.d_llm, vocab_size = self._load_backbone(
            model_config
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Width of the LLM hidden state actually fed to the head. Time-LLM keeps the
        # first d_ff dimensions [T:models/TimeLLM.py:244]; the article is silent.
        self.d_ff = int(model_config.get("d_ff", 32))
        if not 1 <= self.d_ff <= self.d_llm:
            raise ValueError(f"d_ff {self.d_ff} must be in [1, d_llm={self.d_llm}]")

        self.reprogramming = PrototypeReprogramming(
            d_model=self.d_model, d_llm=self.d_llm, vocab_size=vocab_size,
            num_prototypes=self.num_prototypes, num_heads=self.num_heads,
            dropout=self.dropout,
        )

        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(self.dropout),
            nn.Linear(self._patches * self.d_ff, self.prediction_horizon * self._series_width),
        )

    def _load_backbone(self, model_config: dict) -> tuple[nn.Module, Any, int, int]:
        """Return (backbone, text tokenizer, hidden width, vocabulary size).

        backbone: random builds an untrained transformer of the same shape.
        That is a structural control for the paper's transfer claim, NOT a
        reproduction of it; it never silently replaces the pretrained path.
        """
        if self.backbone_name == "random":
            d_llm = int(model_config.get("random_d_llm", 256))
            vocab_size = int(model_config.get("random_vocab_size", 2048))
            layer = nn.TransformerEncoderLayer(
                d_model=d_llm, nhead=self.num_heads,
                dim_feedforward=int(model_config.get("random_ffn_dim", 4 * d_llm)),
                dropout=self.dropout, batch_first=True,
            )
            backbone = _RandomBackbone(
                nn.TransformerEncoder(layer, num_layers=self.llm_layers),
                nn.Embedding(vocab_size, d_llm),
            )
            return backbone, None, d_llm, vocab_size

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "SpectrumLLM with a pretrained backbone requires `transformers`. "
                "Install it, or set model.backbone: random for the untrained control."
            ) from exc

        local_only = bool(model_config.get("local_files_only", False))
        backbone_config = AutoConfig.from_pretrained(
            self.backbone_name, local_files_only=local_only
        )
        if hasattr(backbone_config, "n_layer"):
            backbone_config.n_layer = self.llm_layers
        elif hasattr(backbone_config, "num_hidden_layers"):
            backbone_config.num_hidden_layers = self.llm_layers
        backbone_config.output_hidden_states = True
        backbone = AutoModel.from_pretrained(
            self.backbone_name, config=backbone_config, local_files_only=local_only
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            self.backbone_name, local_files_only=local_only
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token or text_tokenizer.sep_token
        d_llm = int(
            getattr(backbone_config, "n_embd", getattr(backbone_config, "hidden_size", 0))
        )
        if d_llm <= 0:
            raise ValueError(
                f"Could not determine hidden width for backbone {self.backbone_name!r}"
            )
        return backbone, text_tokenizer, d_llm, int(backbone_config.vocab_size)

    def _word_embeddings(self) -> torch.Tensor:
        return self.backbone.get_input_embeddings().weight

    def _prompt_prefix(self, normalized: torch.Tensor) -> torch.Tensor | None:
        """Prompt-as-Prefix: render the static template, embed it, return the prefix."""
        if self.prompt_mode == "none" or self.text_tokenizer is None:
            return None
        prompts = [
            (
                f"<|start_prompt|>Dataset description: {self.domain_hint} measurements. "
                f"Task description: forecast the next {self.prediction_horizon} steps "
                f"given the previous {self.input_sequence_length} steps. "
                "Input statistics: "
                f"min value {entry['min']:.3f}, max value {entry['max']:.3f}, "
                f"median value {entry['median']:.3f}, the trend of input is {entry['trend']}, "
                f"top {len(entry['lags'])} lags are {entry['lags']}<|end_prompt|>"
            )
            for entry in series_statistics(normalized, self.num_lags)
        ]
        encoded = self.text_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids.to(normalized.device)
        return self.backbone.get_input_embeddings()(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"SpectrumLLM expects (batch, time, features), got {tuple(x.shape)}"
            )
        if x.shape[1:] != (self.input_sequence_length, self.input_size):
            raise ValueError(
                "SpectrumLLM input dimensions do not match configuration: "
                f"got {tuple(x.shape[1:])}, expected "
                f"({self.input_sequence_length}, {self.input_size})"
            )

        batch_size = x.shape[0]
        normalized = self.revin.normalize(x)
        # The prompt is rendered from this tensor; see prompt_stats. It is folded exactly like
        # `normalized` so prompt row i always describes the series that produced token row i.
        prompt_source = x if self.prompt_stats == "raw" else normalized

        if self.channel_independent:
            # (B, T, N) -> (B*N, T, 1): every channel becomes its own univariate series,
            # exactly as Time-LLM does before patching. RevIN has already run per (sample,
            # channel), so each folded series carries its own statistics.
            fold = lambda a: (a.permute(0, 2, 1)
                              .reshape(batch_size * self.input_size, self.input_sequence_length, 1))
            normalized = fold(normalized)
            prompt_source = fold(prompt_source)

        patches = self.tokenizer_module(normalized)
        patch_embeddings = self.patch_embedding(patches)
        aligned = self.reprogramming(patch_embeddings, self._word_embeddings())

        prefix = self._prompt_prefix(prompt_source)
        if prefix is not None:
            sequence = torch.cat([prefix.to(aligned.dtype), aligned], dim=1)
            data_start = prefix.shape[1]
        else:
            sequence = aligned
            data_start = 0

        hidden = self.backbone(inputs_embeds=sequence).last_hidden_state
        hidden = hidden[:, data_start:, : self.d_ff]

        prediction = self.output_projection(hidden)
        if self.channel_independent:
            prediction = (prediction.reshape(batch_size, self.input_size, self.prediction_horizon)
                          .permute(0, 2, 1))
        else:
            prediction = prediction.reshape(
                batch_size, self.prediction_horizon, self.input_size
            )

        return self.revin.denormalize(prediction)


class _RandomBackbone(nn.Module):
    """Untrained stand-in exposing the two calls the forecaster needs."""

    def __init__(self, encoder: nn.Module, embeddings: nn.Embedding) -> None:
        super().__init__()
        self.encoder = encoder
        self.embeddings = embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, inputs_embeds: torch.Tensor):
        class _Output:
            def __init__(self, last_hidden_state: torch.Tensor) -> None:
                self.last_hidden_state = last_hidden_state

        return _Output(self.encoder(inputs_embeds))
