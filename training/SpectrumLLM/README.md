# SpectrumLLM

Reimplementation of the SpectrumLLM architecture for radio-spectrum-state (RSS) prediction.

**Source paper.** C. Liu, Y. Wang, S. Mao, D. Niyato, X. Wang, G. Gui, "SpectrumLLM: Large Language
Models for Next-Generation Spectrum Prediction", *IEEE Wireless Communications*, April 2026,
pp. 193-199.

## Quick Start

```bash
# tuning / training through the shared pipeline
python3 -m training.ray.run --config training/configs/ray_tuning_spectrumllm_2d.yaml \
    --models spectrumllm --candidates 12

# single train + evaluate
python3 training/common/train_integrated.py      --config training/configs/ray_tuning_spectrumllm_2d.yaml --name spectrumllm
python3 training/common/evaluation_integrated.py --config training/configs/ray_tuning_spectrumllm_2d.yaml --name spectrumllm

# tests (the repo has no conftest.py, so the root must be on PYTHONPATH)
PYTHONPATH=. pytest tests/test_spectrum_llm.py -q
```

The first run downloads the GPT-2 checkpoint from Hugging Face. Set
`model.local_files_only: true` to require a cached copy instead.

## Scripts Reference

This model has **no per-model `train_integrated.py` / `evaluate_integrated.py`**. It is registered in
`training/ray/registry.py` and constructed by `training/common/model_factory.py:build_model`, so the
shared runner trains and scores it directly. That is deliberate: the standalone trainers in the older
per-model folders duplicate the shared pipeline and drift from it (see `REPO_GUIDE.md` section 6.1).
Add wrappers only if this model needs a training contract the shared runner cannot express.

## File Structure

```
models/SpectrumLLM.py                            architecture (RevIN, PatchTokenizer,
                                                 PrototypeReprogramming, SpectrumLLMForecaster)
training/SpectrumLLM/README.md                   this file
training/SpectrumLLM/requirements.txt            torch, transformers
training/configs/ray_tuning_spectrumllm_2d.yaml  POWDER 600-800 MHz, 2D, horizons 1/15/60
tests/test_spectrum_llm.py                       10 tests (untrained-control backbone)
```

## Configuration Reference

| Key | Default | Meaning |
|---|---|---|
| `input_sequence_length` | required | lookback in minutes |
| `prediction_horizon` | 1 | direct multi-step output; 1 lets the shared rollout handle horizons |
| `patch_length` / `patch_stride` | 16 / 8 | sliding-window tokenization |
| `channel_independent` | false | false = one token spans all channels (the paper's equation); true = Time-LLM's per-channel patching |
| `d_model` | 32 | width of the learnable patch embedding |
| `num_prototypes` | 1000 | size of the prototype set probed from the LLM vocabulary |
| `attention_heads` | 8 | heads in the prototype cross-attention; must divide the backbone width |
| `prompt_mode` | static | `static` = Prompt-as-Prefix; `none` = no prompt tokens |
| `backbone` | openai-community/gpt2 | any `AutoModel` id, or `random` for the untrained control |
| `llm_layers` | 6 | backbone layers retained |
| `revin_affine` | true | learnable affine terms in RevIN |
| `local_files_only` | false | forbid network access when loading the backbone |

## Purpose

Tests whether a pretrained language model, fed tokenized spectrum patches aligned to its own word
embeddings and prefixed with a descriptive prompt, forecasts RSS better than the architectures
already in the benchmark.

## Input Format

`(batch, input_sequence_length, input_size)` on the 2D representation — time x frequency for one
receiver, in dBm. `build_model` fills `input_size` from `train_data.shape[-1]`.

## Architecture

1. **RevIN** - per-instance, per-channel normalization, inverted at the output.
2. **Patch tokenization** - sliding window of `patch_length`, stride `patch_stride`; one token per window.
3. **Patch embedding** - 2-layer MLP with GELU to `d_model`.
4. **Prototype reprogramming** - `num_prototypes` prototypes obtained by a linear probe of the
   backbone's word-embedding matrix, then multi-head cross-attention with the patch embeddings
   as queries.
5. **Prompt-as-Prefix** - a static template carrying the domain, the task, and per-instance
   min / max / median / trend / top-lags, embedded with the backbone's own embedding table and
   concatenated in front of the data tokens.
6. **Backbone** - frozen pretrained LLM (`requires_grad = False` on every parameter).
7. **Projection head** - flatten the data-token hidden states, dropout, one linear layer to
   `prediction_horizon x input_size`.

Only steps 1-5 and 7 train. With GPT-2 at `llm_layers: 6` the trainable fraction is a few percent
of the total; the smoke run in `smoke_spectrumllm_result.json` records the exact split.

## Output Format

`(batch, prediction_horizon, input_size)`, denormalized to dBm by the RevIN inverse.

## Training Pipeline

Shared runner, MSE loss on denormalized dBm, chronological splits, normalization statistics fitted
on the training split only, five-seed replication per the benchmark protocol.

## Assumptions

- The backbone is frozen, so the head is the model's entire learned capacity.
- `attention_heads` must divide the backbone hidden width (768 for GPT-2).
- The prompt is rebuilt per batch, which costs a tokenizer call per forward pass.

## Deviations from the Original Paper

1. **The paper publishes no network.** It is an architecture survey: it names the stages
   (preprocessing, tokenization, embedding, prototype alignment, prompt design, fine-tuning,
   projection) and cites Time-LLM, OFA and TEST as instantiations. Every layer count, width and
   hyperparameter here is ours.
2. **Backbone.** The paper's case study evaluates Time-LLM and OFA with BERT and GPT-2. We default
   to GPT-2 (repo id `openai-community/gpt2`; transformers v5 no longer resolves the bare `gpt2`
   alias for weights) and truncate it to `llm_layers`; the truncation is ours and is not in the paper.
3. **Patch semantics are ambiguous in the source.** The paper writes a patch as `x_i` in
   `R^{N x L}` - one token spanning all N channels - while Time-LLM, which it cites as its
   instantiation, patches each channel independently. We default to the paper's literal reading and
   expose `channel_independent` for the other.
4. **Prototype count** is not stated; default 1000, with 100/500/1000 across the registry anchors.
5. **Embedding network.** The paper says `f_embed` "can be instantiated as a simple MLP"; the depth,
   width and activation are ours.
6. **Prompt wording.** The paper shows an example prompt as a figure and lists the statistics it
   carries, but the exact template text is not recoverable from the figure. Ours is modelled on the
   fields the paper enumerates.
7. **Degree of fine-tuning.** The paper discusses full fine-tuning, partial freezing, and complete
   freezing. We freeze completely, which is Time-LLM's choice; OFA's "positional embeddings and
   layer-norm layers only" variant is not implemented.
8. **Input regime.** The paper uses a 192-step lookback at 15-minute sampling (48 h) with horizons
   of 12-192 steps. Our benchmark uses a 60-minute lookback at 1-minute sampling with horizons of
   1/15/60 minutes, so the long-horizon regime where the paper reports its gains is not tested here.
9. **Dataset.** The paper uses one ElectroSense sensor near Madrid, 500-800 MHz, one month from
   2021-05-28, 15-minute sampling, 7:3 split. Ours is POWDER, 600-800 MHz, 1-minute sampling, with
   the benchmark's chronological split.
10. **Metrics.** MSE loss matches the paper; reported errors are dB MAE/RMSE per the benchmark.
11. **`backbone: random` is an addition**, not from the paper: an untrained transformer of identical
    shape, as a control for the paper's transfer claim. A result from it is not a reproduction of
    SpectrumLLM.

## References

- Liu et al., "SpectrumLLM: Large Language Models for Next-Generation Spectrum Prediction",
  IEEE Wireless Communications, April 2026.
- Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", ICLR 2024
  (prototype reprogramming and Prompt-as-Prefix).
- Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", NeurIPS 2023 (OFA).
- Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against
  Distribution Shift", ICLR 2022 (RevIN).
