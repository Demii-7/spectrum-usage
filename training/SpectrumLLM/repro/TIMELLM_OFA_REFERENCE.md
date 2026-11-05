# Reference implementations behind the SpectrumLLM case study

The SpectrumLLM article publishes no network; its Tables I-II are **Time-LLM** and **OFA** run as
released by their own authors. Both repos are live, so the hyperparameters the article omits are
largely recoverable from them. Pinned 2026-10-30:

| Model | Repo | Commit | Pushed | License |
|---|---|---|---|---|
| Time-LLM (ICLR 2024) | `github.com/KimMeen/Time-LLM` | `b13e881f86cd0475ce1b72c17110430663334955` | 2025-10-15 | Apache-2.0 |
| OFA / "One Fits All" (NeurIPS 2023) | `github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All` | `9f3875dbba9f52ee534ac0902572502746f3b29d` | 2024-01-08 | none declared |
| Baseline library | `github.com/thuml/Time-Series-Library` | (not pinned by the authors) | 2026-04-18 | MIT |

Source tags below: `[T:file:line]` = Time-LLM's code, `[F:file:line]` = OFA's code, `[O:]` = ours.

## 1. Time-LLM's released defaults vs our port

| Item | Time-LLM released | Ours | Verdict |
|---|---|---|---|
| *(rows below re-verified 2026-11-05; the port has since been aligned on RevIN, patch count, lags and head geometry — original verdicts in parentheses where they changed)* | | | |
| Patch length / stride | 16 / 8 `[T:models/TimeLLM.py:32]` | 16 / 8 | **match** |
| Prototype count | **1000, hardcoded** `[T:models/TimeLLM.py:178]` | 1000 default | **match** |
| Top-k lags in prompt | 5 `[T:models/TimeLLM.py:38]` | 5 | match in count |
| **How "lags" are computed** | **FFT autocorrelation**, `topk` over the mean autocorrelation `[T:models/TimeLLM.py:257-264]` | FFT autocorrelation, `autocorrelation_lags` | **match** (was: ours wrong; aligned 2026-11) |
| Trend string | sign of `x.diff(dim=1).sum(dim=1)` `[T:...:211,226]` | sign of `series[-1] - series[0]` | equivalent (a telescoping sum), cosmetic |
| Channel handling | **strictly channel-independent**: `reshape(B*N, T, 1)` `[T:...:205]` | all-channel token by default; `channel_independent: true` available | ours defaults to the *article's* equation, not the instantiation's |
| RevIN | `Normalize(enc_in, affine=False)` `[T:...:192]` | `revin_affine: false` | **match** (was: differs; aligned 2026-11) |
| Patch count | `(L - P)/S + 2` - `ReplicationPad1d((0, stride))` pads before unfolding `[T:layers/Embed.py:166,180]` | replicate pad, `(L - P)/S + 2` | **match** (was: differs; aligned 2026-11) |
| Patch embedding | `TokenEmbedding` = a **Conv1d(patch_len -> d_model, k=3, circular padding)** `[T:layers/Embed.py:169]` | 2-layer MLP with GELU | **differs**; the article only says "a simple MLP", so ours follows the article and theirs does not |
| Post-LLM head | truncate hidden states to the **first `d_ff` dims** `[T:...:244]`, then `FlattenHead` over `d_ff x patch_nums` per variable | d_ff truncation, flatten, one linear to `K x N` | **match in geometry** (was: differs; aligned 2026-11) |
| Reprogramming layer | explicit Q/K/V projections with `d_keys = d_ff`, out-projection to `d_llm` `[T:...:267-276]` | `nn.MultiheadAttention(embed_dim=d_llm)` + query projection `d_model -> d_llm` | **differs** in width and parameter count |
| Backbone | LLaMA-7B default; GPT-2 and BERT also wired | GPT-2 | the article uses BERT and GPT-2 |
| **Backbone truncation** | **`num_hidden_layers = llm_layers`, default 6** `[T:...:46]`, `run_main.py` | truncated to 6 | **match - and this resolves a flagged uncertainty: truncating the backbone is Time-LLM's own practice, not an invention of ours** |
| Backbone frozen | `requires_grad = False` on every parameter `[T:...:163-164]` | identical | **match** |
| `d_model` / `d_ff` | 16 / 32 (argparse), 32 / 128 (ETTh1 script) | `d_model` 32, no `d_ff` concept | partially comparable |
| Attention heads | 8 | 8 | **match** |
| Dropout | 0.1 | 0.1 | **match** |
| Learning rate | 1e-4 (argparse); **0.01-0.02 in the ETTh1 script** | 1e-3 | differs; theirs is schedule-dependent (`lradj type1`) |
| Epochs / patience | 10 / 10 (argparse); 100 in the script | 30 / 5 | differs |
| Batch size | 32 (argparse), 24 (script, x8 processes) | 16 | differs |
| Seed | 2021 | 0/1/2 | ours replicates, theirs does not |
| Loss | MSE | MSE | **match** |
| Precision | bf16 via `accelerate`, 8 GPUs | fp32, 1 GPU | differs |
| Prompt template | `<|start_prompt|>Dataset description: ... Task description: forecast the next {K} steps given the previous {L} steps information; Input statistics: min value ..., max value ..., median value ..., the trend of input is {upward/downward}, top 5 lags are : [...]<|<end_prompt>|>` `[T:...:219-228]` | same fields, our own wording | **our wording differs; theirs is the one the SpectrumLLM article's Fig. 3 is drawn from** |
| Dataset description text | per-dataset file in `dataset/prompt_bank/` | `domain_hint` config string | ours is the same mechanism |

## 2. OFA's partial fine-tuning, exactly

The arm we have **not** implemented, and the best average performer in both of the article's tables:

- GPT-2 truncated by slicing the block list: `self.gpt2.h = self.gpt2.h[:configs.gpt_layers]` `[F:Long-term_Forecasting/models/GPT4TS.py:31]`.
- Trainable set: `if 'ln' in name or 'wpe' in name: param.requires_grad = True`, everything else frozen
  `[F:...:38-42]` - i.e. **layer norms and positional embeddings only**, which is what the SpectrumLLM
  article describes as "fine-tuned only the positional embeddings and layer normalization layers".
- No prototypes, no prompt: `in_layer = nn.Linear(patch_size, d_model)` straight into the backbone `[F:...:34]`.
- `patch_num = (seq_len - patch_size) // stride + 1` `[F:...:20]`.

## 3. What this changes for the model card

- `code_available` for **SpectrumLLM** stays `none` - there is still no SpectrumLLM network. But the two
  instantiations whose numbers the article reports are fully released, so the article's missing
  hyperparameters are **partially recoverable by proxy**, and §6's "In their code" column is no longer
  uniformly empty.
- Uncertainty #2 (patch length, prototype count) is **largely resolved**: 16/8 and 1000, both of which our
  port already matched.
- Uncertainty #8 (backbone depth) is **resolved in our favour**: 6 layers is Time-LLM's own default.
- Uncertainty #3 (training schedule) is **not** resolved - Time-LLM's own argparse and its shipped script
  disagree by 100x on learning rate and 10x on epochs, so "Time-LLM's defaults" is not a single number.
- A new, separate finding: **our port is not a faithful Time-LLM**. The head geometry (full `d_llm` flatten
  instead of a `d_ff` truncation), the reprogramming widths, `affine=True`, the patch count and the lag
  statistic all differ. The §13(b) numbers therefore measure *our* build, and cannot be read as "Time-LLM
  on ElectroSense".
