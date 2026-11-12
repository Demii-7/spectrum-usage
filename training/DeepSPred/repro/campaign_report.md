# DeepSPred rendering campaign — final report

All runs: FM, K=T=20, C96/d2-4-2/p2-4-4/w2-7-7, lr 1e-3, batch 1, paper early-stopping (except tagged diagnostics), chronological 4:1:1, metric uint8_mean_per_channel.

| run | ep | MSE@8 | @16 | @18 | ratio@8 | ratio@16 | ratio@18 |
|---|---|---|---|---|---|---|---|
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=hamm128|full` *(recovered record)* | 9 | 590.2 | 589.9 | 590.1 | 1.315 | 1.302 | 1.299 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=hann126|full` | 8 | 829.9 | 829.4 | 829.8 | 1.849 | 1.831 | 1.827 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=hann128s|full` | 8 | 579.1 | 578.5 | 579.3 | 1.290 | 1.277 | 1.275 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=rect126|full` | 11 | 348.9 | 348.8 | 348.9 | 0.777 | 0.770 | 0.768 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=rect127c~12|full` | 11 | 302.1 | 302.0 | 302.0 | 0.673 | 0.667 | 0.665 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224@54-218|full` *(recovered record)* | 7 | 494.7 | 494.7 | 495.3 | 1.102 | 1.092 | 1.091 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d@54-218|full` *(recovered record)* | 6 | 596.1 | 595.9 | 596.7 | 1.328 | 1.315 | 1.314 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full` *(recovered record)* | 8 | 498.6 | 498.4 | 496.7 | 1.111 | 1.100 | 1.094 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full|dp0.1` *(recovered record)* | 8 | 494.6 | 494.3 | 494.0 | 1.102 | 1.091 | 1.088 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full|head=residual` *(recovered record)* | 5 | 481.7 | 485.6 | 487.3 | 1.073 | 1.072 | 1.073 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full|rule=none` *(recovered record)* | 20 | 498.6 | 498.4 | 496.7 | 1.111 | 1.100 | 1.094 |
| `fm|K20|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224|full` *(recovered record)* | 8 | 590.4 | 590.3 | 589.8 | 1.316 | 1.303 | 1.299 |
| `fm|K20|s1|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=rect126|full` | 8 | 342.6 | 342.5 | 342.6 | 0.764 | 0.756 | 0.754 |
| `fm|K20|s1|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224@54-218|full` *(recovered record)* | 6 | 491.2 | 490.9 | 491.7 | 1.094 | 1.084 | 1.083 |
| `fm|K20|s1|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full` *(recovered record)* | 7 | 496.8 | 496.6 | 496.8 | 1.107 | 1.096 | 1.094 |
| `lte|K16|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|tl|uint8_mean_per_channel|spec=tukey224d|full` | 12 | n/a | | | | | |
| `lte|K16|s0|C96|d2-4-2|lr0.001|p2-4-4|w2-7-7|raw|uint8_mean_per_channel|spec=tukey224d|full` *(recovered record)* | 5 | n/a | | | | | |

Paper anchors: @8=448.7628 @16=453.0376 @18=454.1826.

## Verification log
```
PASS results_rect126.json sha_ok=True mse_max_rel_diff=0.0000% rescored@8=348.9 record@8=348.9 epochs=11
PASS results_spec-hann126_s0.json sha_ok=True mse_max_rel_diff=0.0000% rescored@8=829.9 record@8=829.9 epochs=8
PASS results_spec-tukey224_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_tukey224_54-218.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_tukey224d.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-tukey224d_54-218_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-tukey224_54-218_s1.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-rect126_s1.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-tukey224d_s0_dp01.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
ENSEMBLE tukey224d seeds 3c96fc+c5a4e4: MSE@8=493.7 @16=493.5 @18=492.8 (seed-a alone @8=498.6; paper 448.76/453.04/454.18)
PASS results_spec-tukey224d_s1.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-lte_scratch_tukey224d_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-rect127c_12_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-lte_tl_tukey224d_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
BLEND tukey224d: ensemble@8=9862.9 | 0.5*ens+0.5*lbm@8=9398.7 | 0.5*s0+0.5*lbm@8=9424.3 | @16=9398.7 @18=9398.7 (paper 448.76/453.04/454.18)
PASS results_spec-tukey224d_s0_rule-none.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-hamm128_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
BLEND tukey224d: ensemble@8=493.7 | 0.5*ens+0.5*lbm@8=470.1 | 0.5*s0+0.5*lbm@8=471.4 | @16=471.8 @18=472.4 (paper 448.76/453.04/454.18)
PASS results_spec-hann128s_s0.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
PASS results_spec-tukey224d_s0_head-res.json sha_ok=True recovered-record (streaming re-score; metrics identical by validation)
```

## Checkpoint sha256 manifest
```
84a599ab75cdf8a0f08af9834aa07aa9fd725617a80e9c674b0671aeb99bb26e  /home/cc/spectrum-usage/checkpoints/repro/deepspred/c58d32f53cad.pt
aac9797fb265944c3a70437d0ce3cd53e27eee0db11b804b4d2136f480d726e9  /home/cc/spectrum-usage/checkpoints/repro/deepspred/513bd5b54a8f.pt
9a7c93599b21d9b24eaf670699fb98f3504266c6a49bf4bceda48f611de39701  /home/cc/spectrum-usage/checkpoints/repro/deepspred/0727eb8f06c1.pt
3975767a2a9dc90b4219c8724b0cd767d1f14626a5f23dc82f279a7dd271c6e3  /home/cc/spectrum-usage/checkpoints/repro/deepspred/1442b5d2022f.pt
efbb0497ceb513337bd6cdd4dcb85283e505553fcbe535eb9adb1e8eeb6a3054  /home/cc/spectrum-usage/checkpoints/repro/deepspred/4dfd1d59c176.pt
5aeecbfd60afd29b9d47b0a55172b3a55246502dcc5fb8a3340068118c5bff1e  /home/cc/spectrum-usage/checkpoints/repro/deepspred/dba70d28179a.pt
c4233dc4071b80be4ab798d52b94bcfba71ef69491e7324231edbe9b27f8a8e1  /home/cc/spectrum-usage/checkpoints/repro/deepspred/cc28f79072b2.pt
847b72e52c00e0e537aab4ac75a3cb403df5e94200bed461e69d3a017a235460  /home/cc/spectrum-usage/checkpoints/repro/deepspred/2fa9ff9a3d9b.pt
39bd069c2c71e719599c9a75494051738eb993a38387bad62222c8c2ff8d482e  /home/cc/spectrum-usage/checkpoints/repro/deepspred/1b987cdc5bb2.pt
6375a02b24050011b5d807378acdb03b9ec35c83fd7db6a49910ee7a6e604c73  /home/cc/spectrum-usage/checkpoints/repro/deepspred/d0419b0f1311.pt
fccaac8d95b39ad8676b35dea08f399ab578a5258dc9ab6ba63f38390c4bcc4a  /home/cc/spectrum-usage/checkpoints/repro/deepspred/b60bc8677c44.pt
0de2bf53cfd45ce8b20835c570e75f9db5ace9787ace1ec64846f45ec855609e  /home/cc/spectrum-usage/checkpoints/repro/deepspred/64f6b2d131a5.pt
758855e8fbacf30f748d1a9c5d67b551222d5fd79c41c1d14528d7b6067f5b94  /home/cc/spectrum-usage/checkpoints/repro/deepspred/371f09a3d475.pt
4cf3aeeca70467ef82df55637b7c71077c53d972710d7cac510e850e70d27b67  /home/cc/spectrum-usage/checkpoints/repro/deepspred/e7520c4cc040.pt
760d649480e4cddfc22134143022154f7ad090660489d454548774d853e1b67a  /home/cc/spectrum-usage/checkpoints/repro/deepspred/c5a4e4c918ca.pt
dd3663c70d1fdaf56f26e76d69e37aedf74d1cf68620602c5c72f985010b06f8  /home/cc/spectrum-usage/checkpoints/repro/deepspred/187560732831.pt
fc2e8b15f77df0aaf0ff59457eda7b5ea407cbd5a6f9a4fea6bef78a59d9e6a0  /home/cc/spectrum-usage/checkpoints/repro/deepspred/3c96fc0dc0cd.pt
```
Snapshot at server decommission: the LTE no-early-stop scratch (frozen at persistence, 9+ epochs) and the raw-input rect126 run (slow descent, epoch 4) were truncated without checkpoints; their trajectories are in repro/train_o_*.log.
