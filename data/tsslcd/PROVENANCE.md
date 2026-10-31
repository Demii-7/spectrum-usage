# TSS-LCD authors' data

Recovered from github.com/Xlab2024/TSS-LCD commit `d7f1e51263` (2025-12-02,
"split dataset and divide the training part and the test part"). Both files were
DELETED from HEAD (`b156d0a007`); only the test split survives there.

| File here | Upstream name | Shape | Notes |
|---|---|---|---|
| `tsslcd_authors_train.csv` | `dataset/trainDataset.csv` | 3761 x 750 | = source rows 0-3760, byte-exact |
| `tsslcd_authors_test.csv`  | `dataset/testDataset.csv`  | 3079 x 750 | rows 1.. = source rows 3761-6838 |

750 columns = 3 AERPAW nodes (CC1, CC2, LW1) x 250 frequency bins, 85-335 MHz, dBm, 1-minute.

Facts established by inspection, not by the paper:
- the split is chronological 55:45, NOT the 7:3 the paper states;
- the source series is 6839 rows, NOT the 10,080 the paper states;
- row 0 of BOTH files holds a malformed cell at column 672 (`-134.20303.1`). Their
  loader consumes row 0 as a CSV header, so it never reaches the model. Effective
  data is train rows 1-3760 and test rows 1-3078.

NOT interchangeable with `training/data/merged_power_data_sub6GHz_avg_per_minute.csv`:
same 6839 x 750 shape and filename lineage, but values differ by mean -2.16 dB
(sd 3.33, range -17.05..+19.14). Different derivation of the same AERPAW campaign.
