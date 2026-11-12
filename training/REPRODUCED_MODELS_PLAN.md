# Plan for the reproduced models

How we intend to use the models reproduced from other papers in the POWDER experiments.
These are plans, not results.

## 1. SpectrumLLM: both 1D and 2D

Evaluate it in both representations. In the original paper it did worse in the 2D format, so
running both lets us report whether that holds on our data and avoids penalizing it with the
weaker format.

## 2. TSS-LCD: 2D, all sites concatenated as columns

Use it as a 2D model, with all sites concatenated as columns as in the TSS-LCD paper, rather than
one model per site.

## 3. DeepSPred: to be decided

No decision yet on representation or setup (see `training/DeepSPred/DESIGN.md`).
