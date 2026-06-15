## Training data

This directory includes code necessary to retrieve the AERPAW dataset [February 2022: CC1, CC2, LW1 Spectrum Measurements](https://aerpaw.org/dataset/february-2022-cc1-cc2-lw1-spectrum-measurements/) and format it so that it may be used to train a machine learning model on the spectrum usage prediction task.

`build_training_csv.py` can stream SigMF zip archives directly and write a per-minute CSV into `training/data/` without fully extracting the archive contents to disk.
