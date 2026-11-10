# FM lookback mean and residual linear AR at k=8 (paper: 448.8)

Fixed in every row: complex I/Q (one second = 32,508 samples), nperseg 256, two-sided fftshifted spectrum, 10·log10(|S|²+1e-12) unless a dB floor is given, jet colormap → RGB uint8, no codec unless stated; MSE over pixels and the 3 channels on the FM test block (last 1/6), 6,000 random pixel positions, stride-1 windows, 20-frame lookback.


**A. Window / hop / detrend / normalization**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Rectangular · hop 126 · no padding · no detrend · 256 rows, no resize · min-max, all frames | 340 | 340 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · min-max, all frames | 481 | 481 |
| Tukey(0.25) · hop 126 · no padding · no detrend · 256 rows, no resize · min-max, all frames | 548 | 548 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · min-max, all frames | 552 | 552 |
| Tukey(0.25) · hop 224 · no padding · no detrend · 144 rows → 256 (bilinear resize) · min-max, all frames | 571 | 571 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · clip to dataset mean ± 5σ | 575 | 575 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · min-max, all frames | 584 | 584 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · clip to dataset mean ± 5σ | 775 | 775 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · clip to dataset mean ± 5σ | 791 | 791 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · min-max, all frames | 831 | 831 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · clip to dataset mean ± 4σ | 865 | 864 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · clip to dataset mean ± 5σ | 957 | 957 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · clip to dataset mean ± 4σ | 1152 | 1152 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · clip to dataset mean ± 4σ | 1172 | 1172 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · clip to dataset mean ± 4σ | 1408 | 1408 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · clip to dataset mean ± 3σ | 1419 | 1419 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · clip to dataset mean ± 3σ | 1848 | 1848 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · clip to dataset mean ± 3σ | 1851 | 1851 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · clip to dataset mean ± 3σ | 2207 | 2207 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · clip to dataset mean ± 2σ | 2596 | 2596 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · clip to dataset mean ± 2σ | 3151 | 3150 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · clip to dataset mean ± 2σ | 3253 | 3252 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · clip to dataset mean ± 2σ | 3773 | 3773 |

**B. Per-frame standardization**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · per-frame standardization, clip ±3σ (ignores dataset range) | 1440 | 1440 |
| Tukey(0.25) · hop 126 · no padding · detrend (segment mean removed) · 256 rows, no resize · per-frame standardization, clip ±3σ (ignores dataset range) | 1877 | 1877 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · per-frame standardization, clip ±3σ (ignores dataset range) | 1890 | 1890 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · per-frame standardization, clip ±3σ (ignores dataset range) | 2262 | 2262 |

**C. Smoothing / codec**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · Gaussian blur σ=2 · min-max, all frames | 83 | 82 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · JPEG q75 · min-max, all frames | 187 | 187 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · JPEG q75 · min-max, all frames | 187 | 187 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · JPEG q25 · min-max, all frames | 190 | 189 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · JPEG q95 · min-max, all frames | 201 | 200 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · JPEG q95 · min-max, all frames | 201 | 201 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · JPEG q50 · min-max, all frames | 207 | 207 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · Gaussian blur σ=1 · min-max, all frames | 214 | 214 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · JPEG q75 · min-max, all frames | 223 | 223 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · JPEG q95 · min-max, all frames | 244 | 244 |

**D. Band crop to 90-110 MHz**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · keep freq bins 46-209 (164) → 256 (bilinear) · min-max, all frames | 334 | 333 |
| Tukey(0.25) · hop 224 · no padding · no detrend · 144 rows → 256 (bilinear resize) · keep freq bins 54-217 (164) → 256 (bilinear) · min-max, all frames | 464 | 464 |
| Tukey(0.25) · hop 224 · no padding · no detrend · 144 rows → 256 (bilinear resize) · keep freq bins 46-209 (164) → 256 (bilinear) · min-max, all frames | 468 | 467 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · keep freq bins 46-209 (164) → 256 (bilinear) · min-max, all frames | 515 | 515 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (bilinear resize) · keep freq bins 54-217 (164) → 256 (bilinear) · min-max, all frames | 519 | 518 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (bilinear resize) · keep freq bins 54-217 (164) → 256 (bilinear) · min-max, all frames | 575 | 574 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · keep freq bins 46-209 (164) → 256 (bilinear) · min-max, all frames | 617 | 616 |
| Hann · hop 126 · no padding · no detrend · 256 rows, no resize · keep freq bins 54-217 (164) → 256 (bilinear) · min-max, all frames | 785 | 785 |

**E. torch.stft / scipy.signal.stft style**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · min-max, all frames | 294 | 294 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · min-max, all frames | 584 | 584 |
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · clip to dataset mean ± 5σ | 664 | 664 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · clip to dataset mean ± 5σ | 733 | 733 |
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · clip to dataset mean ± 4σ | 968 | 968 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · clip to dataset mean ± 4σ | 1095 | 1095 |
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · clip to dataset mean ± 3σ | 1522 | 1522 |
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · per-frame standardization, clip ±3σ (ignores dataset range) | 1535 | 1535 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · clip to dataset mean ± 3σ | 1760 | 1760 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · per-frame standardization, clip ±3σ (ignores dataset range) | 1802 | 1802 |
| Rectangular · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · dB floor 1e-12 or 1e-8 (identical on FM) · clip to dataset mean ± 2σ | 2555 | 2555 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (bilinear resize) · clip to dataset mean ± 2σ | 3112 | 3111 |

**F. Resize interpolation**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (cubic resize) · min-max, all frames | 534 | 534 |
| Tukey(0.25) · hop 224 · no padding · detrend (segment mean removed) · 144 rows → 256 (nearest resize) · min-max, all frames | 579 | 579 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (cubic resize) · min-max, all frames | 586 | 586 |
| Hann · hop 128 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 255 rows → 256 (nearest resize) · min-max, all frames | 602 | 602 |
| Hann · hop 128 · reflect-padded 128 each side (torch-style) · no detrend · 254 rows → 256 (nearest resize) · min-max, all frames | 602 | 602 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (cubic resize) · min-max, all frames | 604 | 604 |
| Hamming · hop 128 · no padding · no detrend · 252 rows → 256 (nearest resize) · min-max, all frames | 615 | 615 |
| Hann · hop 127 · reflect-padded 128 each side (torch-style) · no detrend · 256 rows, no resize · min-max, all frames | 687 | 687 |
| Hann · hop 126 · zero-padded 128 each side + end (scipy.stft-style) · no detrend · 259 rows → 256 (nearest resize) · min-max, all frames | 759 | 759 |

**G. Hann hop / padding diagnostic**

| setting | lookback mean | residual linear AR |
|---|---|---|
| Hann · hop 128 · no padding · no detrend · 252 rows → 256 (nearest resize) · min-max, all frames | 604 | 604 |
| Hann · hop 120 · no padding · no detrend · 269 rows → 256 (nearest resize) · min-max, all frames | 651 | 651 |
| Hann · hop 136 · no padding · no detrend · 238 rows → 256 (nearest resize) · min-max, all frames | 652 | 652 |
| Hann · hop 129 · no padding · no detrend · 251 rows → 256 (nearest resize) · min-max, all frames | 655 | 655 |
| Hann · hop 130 · no padding · no detrend · 249 rows → 256 (nearest resize) · min-max, all frames | 664 | 664 |
| Hann · hop 125 · no padding · no detrend · 259 rows → 256 (nearest resize) · min-max, all frames | 703 | 703 |
| Hann · hop 127 · no padding · no detrend · 254 rows → 256 (nearest resize) · min-max, all frames | 725 | 725 |
| Hann · hop 132 · no padding · no detrend · 245 rows → 256 (nearest resize) · min-max, all frames | 741 | 741 |
| Hann · hop 124 · no padding · no detrend · 261 rows → 256 (nearest resize) · min-max, all frames | 748 | 748 |
