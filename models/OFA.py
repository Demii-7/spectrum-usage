"""OFA / GPT4TS arm (Zhou et al., NeurIPS 2023), as benchmarked in the SpectrumLLM article.

OFA is the article's third LLM row and its best average performer in both published tables.
It differs from Time-LLM in three ways, all of which matter:

  * no prototype reprogramming -- patch embeddings go straight into the backbone
  * no prompt prefix -- there is no text, so no tokenizer and no padding
  * PARTIAL fine-tuning -- the backbone is not fully frozen. Only parameters whose names
    contain "ln" (layer norms) or "wpe" (positional embeddings) are trainable, which is the
    rule in the released code [DAMO-DI-ML/NeurIPS2023-One-Fits-All @ 9f3875db].

Two controls isolate what the pretrained language model contributes (``config["model"]`` keys; the defaults reproduce OFA exactly):

  * ``backbone_init``: "pretrained" (default) or "random" -- the same GPT-2 architecture (same width, layers, heads, positional table)
    initialized from scratch instead of from the language-model weights.
  * ``backbone_trainable``: "ofa" (default; only layer norms and positional embeddings train) or "all" -- every backbone weight trains.

Everything else is shared with our SpectrumLLM port so the comparison is like for like:
the same RevIN, the same patch tokenizer, the same channel handling, the same raw-PSD loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # package import, as training/common/model_factory.py does it
    from models.SpectrumLLM import PatchTokenizer, RevIN
except ImportError:  # models/ placed directly on sys.path (repro scripts)
    from SpectrumLLM import PatchTokenizer, RevIN

TRAINABLE_SUBSTRINGS = ("ln", "wpe")


class OFAForecaster(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mc = config["model"]
        self.input_size = int(mc["input_size"])
        self.input_sequence_length = int(mc["input_sequence_length"])
        self.prediction_horizon = int(mc["prediction_horizon"])
        self.patch_length = int(mc.get("patch_length", 16))
        self.patch_stride = int(mc.get("patch_stride", 8))
        self.dropout = float(mc.get("dropout", 0.1))
        self.llm_layers = int(mc.get("llm_layers", 6))
        self.backbone_name = str(mc.get("backbone", "openai-community/gpt2"))
        self.channel_independent = bool(mc.get("channel_independent", False))
        self._series_width = 1 if self.channel_independent else self.input_size

        self.revin = RevIN(self.input_size, affine=bool(mc.get("revin_affine", False)))
        self.tokenizer_module = PatchTokenizer(self.patch_length, self.patch_stride)
        self._patches = self.tokenizer_module.num_patches(self.input_sequence_length)

        from transformers import AutoConfig, AutoModel

        bc = AutoConfig.from_pretrained(self.backbone_name)
        if hasattr(bc, "n_layer"):
            bc.n_layer = self.llm_layers
        elif hasattr(bc, "num_hidden_layers"):
            bc.num_hidden_layers = self.llm_layers
        bc.output_hidden_states = True
        init = str(mc.get("backbone_init", "pretrained")).lower()
        trainable = str(mc.get("backbone_trainable", "ofa")).lower()
        if init not in ("pretrained", "random") or trainable not in ("ofa", "all"):
            raise ValueError(f"backbone_init must be pretrained|random and backbone_trainable ofa|all, got {init!r}, {trainable!r}")
        if init == "random":
            self.backbone = AutoModel.from_config(bc)                     # same architecture, no pretrained weights
        else:
            self.backbone = AutoModel.from_pretrained(self.backbone_name, config=bc)
        self.d_llm = int(getattr(bc, "n_embd", getattr(bc, "hidden_size", 0)))

        # Partial fine-tuning (the OFA rule): ln and wpe trainable, everything else frozen; "all" trains every backbone weight.
        for name, p in self.backbone.named_parameters():
            p.requires_grad = True if trainable == "all" else any(s in name for s in TRAINABLE_SUBSTRINGS)

        # Patch embedding straight to the backbone width -- no reprogramming stage.
        self.patch_embedding = nn.Linear(self.patch_length * self._series_width, self.d_llm)
        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(self.dropout),
            nn.Linear(self._patches * self.d_llm, self.prediction_horizon * self._series_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != (self.input_sequence_length, self.input_size):
            raise ValueError(
                f"OFA input dimensions do not match configuration: got {tuple(x.shape[1:])}, "
                f"expected ({self.input_sequence_length}, {self.input_size})"
            )
        batch = x.shape[0]
        normalized = self.revin.normalize(x)
        if self.channel_independent:
            normalized = (normalized.permute(0, 2, 1)
                          .reshape(batch * self.input_size, self.input_sequence_length, 1))
        patches = self.tokenizer_module(normalized)
        hidden = self.backbone(inputs_embeds=self.patch_embedding(patches)).last_hidden_state
        prediction = self.output_projection(hidden)
        if self.channel_independent:
            prediction = (prediction.reshape(batch, self.input_size, self.prediction_horizon)
                          .permute(0, 2, 1))
        else:
            prediction = prediction.reshape(batch, self.prediction_horizon, self.input_size)
        return self.revin.denormalize(prediction)
