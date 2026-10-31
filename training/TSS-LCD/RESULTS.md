# TSS-LCD reproduction results

Generated from `training/TSS-LCD/repro/results/*/metrics.json`. All values dBm, t+1 only.
Narrative and chronology: `EXPERIMENTS.md`. Paper audit: `MODEL_CARD.md`.

Authors' published Table III: MSE 0.1541, RMSE 0.3926, MAE 0.2486, MAPE 0.2075%.

| arm | impl | split | smoothing | MSE | persistence | lookback mean | AE ceiling | model/persistence |
|---|---|---|---|---|---|---|---|---|
| `abl_temfe` | theirs | random | on (centred) | **0.2146** | 0.00677 | 0.5984 | - | 31.7x |
| `pf_chrono` | ours | chrono | - | **38.9486** | 4.14122 | 2.8603 | 1.4400 | 9.4x |
| `pf_chrono_s1` | ours | chrono | off | **7.3379** | 4.14122 | 2.8603 | 1.4187 | 1.8x |
| `pf_chrono_s2` | ours | chrono | off | **17.7880** | 4.14122 | 2.8603 | 1.6936 | 4.3x |
| `pf_chrono_smooth` | ours | chrono | on | **5.4458** | 0.00619 | 0.4683 | 0.7607 | 880.2x |
| `pf_random` | ours | random | - | **18.4655** | 4.24153 | 3.0750 | 1.6642 | 4.4x |
| `pf_random_s1` | ours | random | off | **10.2182** | 4.08177 | 2.9647 | 1.7025 | 2.5x |
| `pf_random_s2` | ours | random | off | **12.9849** | 4.28563 | 3.0682 | 1.6409 | 3.0x |
| `pf_random_smooth` | ours | random | on | **24.3888** | 0.00676 | 0.5997 | 0.3754 | 3609.6x |
| `theirs_random` | theirs | random | on (centred) | **0.1548** | 0.00677 | 0.5984 | - | 22.9x |

## Table III reproduction (`theirs_random`)

| metric | paper | ours | difference |
|---|---|---|---|
| mse | 0.1541 | 0.15477 | +0.43% |
| rmse | 0.3926 | 0.39340 | +0.20% |
| mae | 0.2486 | 0.24536 | -1.30% |
| mape_pct | 0.2075 | 0.20770 | +0.10% |

## Table IV ablations (their code, leaky random-split configuration)

| ablation | paper MSE | ours | ratio |
|---|---|---|---|
| w/o TemFE | 0.3027 | 0.2146 | 0.709 |
| w/o SpeFE | 0.2910 | not run | - |
| w/o SpaFE | 0.2861 | not run | - |
| w/o FFM | 0.3139 | not run | - |

Note: the paper text calls w/o TemFE the most severe ablation, but its own Table IV
ranks w/o FFM worse (0.3139 > 0.3027). The prose contradicts the table.

## Seed spread (our paper-faithful model, unsmoothed)

- chronological: [38.95, 7.34, 17.79] - spread 5.31x
- random pooled: [18.47, 10.22, 12.98] - spread 1.81x

The seed spread exceeds the chronological-vs-random difference (Welch p = 0.51),
so single-run comparisons in this project are not reliable.

## Caveat on the Table IV ablations — READ BEFORE QUOTING THESE

The ablation arms (`abl_*`) are **inconclusive as evidence about the paper**, because our
intervention is not matched to theirs. Recorded here so the numbers are not misread.

What each switch actually does in `repro/upstream/Context2CondNew.py`:

| switch | what we disabled | what still runs |
|---|---|---|
| `temfe` | `time_encoder` only | positional encoding, and the result still enters fusion |
| `spefe` | `freq_local_encoder` only | `freq_pool`, `freq_global_encoder`, `freq_token_proj`, fusion |
| `spafe` | `freq_global_encoder` only | local frequency attention, pooling, projection, fusion |
| `ffm` | cross-attention, `fusion = time_repr` | **drops the frequency branch entirely** |

Two problems:

1. **Three switches are mild, one is drastic.** `temfe`/`spefe`/`spafe` bypass one encoder
   inside a branch that otherwise keeps working; `ffm` removes a whole branch. They are not
   comparable to each other, let alone to the paper.
2. **Every one of our ablations is *less* damaging than the paper's** (w/o TemFE 0.2146 vs
   0.3027; w/o SpeFE 0.1631 vs 0.2910). That one-sided direction is what a systematically
   gentler intervention produces. It is most likely our implementation, not their result.

A matched experiment would remove each branch from the fusion input entirely, and implement
`w/o FFM` as concatenation/addition of both branches instead of cross-attention. Roughly
10 GPU-hours; not run before the host was released.

Also unresolved: each ablation arm is n=1. The full model reproduced to 0.43% because leakage
pins the achievable error, but an ablated model is less constrained by the leak, so run-to-run
variance may return — and this project has already measured a 5.31x seed spread on weakly
constrained models.

**Separately, and independent of our runs:** the paper's §V text says "discarding the temporal
modeling (w/o TemFE) results in the most severe performance drop", but its own Table IV ranks
w/o FFM worse (0.3139 > 0.3027). The prose contradicts the table. That is a finding about the
paper and does not depend on anything we ran.

## Trained checkpoints — where they are and how to get them back

Uploaded to the project's Chameleon object store, container `spectrum`:

    checkpoints/repro/tsslcd/<arm>/{ae_ctx.pth, ae_fut.pth, diffusion.pth, metrics.json}

793 MiB over three arms (`theirs_random`, `abl_temfe`, `abl_spefe`).

**Why `repro/` and not `2d/tsslcd/`.** The `checkpoints/{1d,2d,4d}/<model>/seed-<N>/`
tree and its `MANIFEST.tsv` describe benchmark runs: POWDER 600-800, seeds 40-44,
one `.pt` per seed produced by `train_integrated` through Ray, scored in
`evaluation/results/tables/seeds_*.csv`. These checkpoints match none of that --
they are trained on the *paper authors'* AERPAW 750-column data with the authors'
own scripts, carry three `.pth` per run, and are indexed by experimental arm rather
than by seed. Filing them under `2d/` would imply a comparability that does not
exist, and any `MANIFEST.tsv` row would have to cite a seeds table they do not
belong to. They are therefore deliberately absent from `MANIFEST.tsv`.

Nothing else reads them: `transfer_1d_eval` filters on `checkpoints/1d/`, the
spatial-ablation test names `checkpoints/4d/convlstm/...` directly, and no code in
the repo parses `MANIFEST.tsv`. The one visible effect is size -- `repro/` is 793 MiB
against 165 MiB for all of `1d`+`2d`+`4d`, so `--checkpoints` now pulls ~958 MiB.
Almost all of that is the three 264 MiB `ae_ctx.pth` context autoencoders.

**Retrieval.** `data/sync_object_storage.py --checkpoints` will NOT work with a
Chameleon application credential -- it speaks S3 via fsspec/s3fs and the S3 gateway
rejects that credential type with 403 InvalidAccessKeyId. Use rclone over Swift
(see `data/.env.example` for the full variable list):

    export RCLONE_CONFIG_CHI_TYPE=swift
    export RCLONE_CONFIG_CHI_AUTH_VERSION=3
    export RCLONE_CONFIG_CHI_AUTH=https://chi.tacc.chameleoncloud.org:5000/v3
    export RCLONE_CONFIG_CHI_APPLICATION_CREDENTIAL_ID=<id>
    export RCLONE_CONFIG_CHI_APPLICATION_CREDENTIAL_SECRET=<secret>
    rclone copy chi:spectrum/checkpoints/repro/tsslcd \
                ./checkpoints/repro/tsslcd --ignore-checksum

`--ignore-checksum` is required; Swift's multipart ETag is not the whole-object MD5
and rclone reads the mismatch as corruption.

A `README.md` restating the above is stored alongside the data at
`checkpoints/repro/README.md`, so the recipe travels with the checkpoints.
