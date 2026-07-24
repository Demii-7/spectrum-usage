# Evaluation Plan

## Trace inventory

Object paths and timestamps come from the remote
`power_1mhz_avg_per_minute.csv` files in the `spectrum` bucket at CHI@TACC,
inspected on 2026-07-24. Start UTC is the earliest first `timestamp_utc` value and
end UTC is the latest final `timestamp_utc` value among the listed bands at a
collection point. An end marks the start of the final one-minute CSV row, rather
than an exclusive interval boundary. Individual bands can end a few minutes or
hours earlier; the evaluation manifest must retain per-band bounds. All CSV bounds
below come from the power CSV itself, not from raw-minute object names.

Missingness counts empty or NaN power cells between each CSV's first and last row,
divided by all power cells in that interval and aggregated across the listed bands.
Contiguity treats each consecutive missing sequence in one frequency bin as a run;
`runs/median/max` reports the run count and median and maximum lengths in minutes.

Annotation status refers to frequency-region boundary files under
`data/annotations/`; it does not imply time-event or occupancy labels.

| Trace | Testbed | Collection point | Object-store run | Start UTC | Last observed minute UTC | Bands (MHz) | Internal missingness; runs/median/max | Annotations | Role and caveats |
|---|---|---|---|---|---|---|---|---|---|
| T1 | AERPAW | CC1 | `aerpaw/CC1/20220208T1750Z` | 2022-02-08 17:50:00 | 2022-02-25 00:59:00 | 87--6020 | 0%; no runs | No | Centennial Campus; spatial transfer from CC2. |
| T1 | AERPAW | CC2 | `aerpaw/CC2/20220208T1748Z` | 2022-02-08 17:48:00 | 2022-02-15 17:47:00 | 87--6020 | 0%; no runs | No | Centennial Campus reference receiver; 10,080 CSV rows. |
| T1 | AERPAW | LW1 | `aerpaw/LW1/20220208T1757Z` | 2022-02-08 17:57:00 | 2022-02-22 15:42:00 | 87--6020 | 0%; no runs | No | Rural Lake Wheeler receiver; spatial transfer from CC2. |
| T2 | COSMOS | sdr2-md1 | `cosmos/sdr2-md1/20260619T1816Z` | 2026-06-19 18:16:00 | 2026-06-24 11:47:00 | 600--800; 2400--2600; 3500--3700 | 0.0019%; 76/1/1 min | Yes: all bands | One of the rooftop/street-level pair. Confirm mounting identity from COSMOS metadata before labeling height. |
| T2 | COSMOS | sdr2-s1-lg1 | `cosmos/sdr2-s1-lg1/20260619T1815Z` | 2026-06-19 18:15:00 | 2026-06-24 11:46:00 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Yes: all bands | One of the rooftop/street-level pair. Confirm mounting identity from COSMOS metadata before labeling height. |
| T3 | POWDER | guesthouse-nuc1 | `powder/guesthouse-nuc1/20260618T0036Z` | 2026-06-18 00:36:00 | 2026-06-24 15:35:00 | 600--800; 2400--2600; 3500--3700 | 0.0105%; 400/1.5/2 min | Yes: all bands | Long simultaneous campus run. |
| T3 | POWDER | humanities-nuc1 | `powder/humanities-nuc1/20260618T0036Z` | 2026-06-18 00:36:00 | 2026-06-24 15:43:00 | 600--800; 2400--2600; 3500--3700 | 1.3791%; 10,948/2/31 min | Yes: all bands | Long simultaneous campus run. |
| T3 | POWDER | law73-nuc1 | `powder/law73-nuc1/20260618T0036Z` | 2026-06-18 00:36:00 | 2026-06-24 15:54:00 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Long simultaneous campus run. |
| T4 | POWDER | cpg-nuc1 | `powder/cpg-nuc1/20260628T0437Z` | 2026-06-28 04:37:19 | 2026-07-03 02:57:19 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | ebc-nuc1 | `powder/ebc-nuc1/20260628T0436Z` | 2026-06-28 04:37:01 | 2026-07-03 03:01:01 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | guesthouse-nuc1 | `powder/guesthouse-nuc1/20260628T0436Z` | 2026-06-28 04:37:12 | 2026-07-03 03:01:12 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Yes: all bands | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | humanities-nuc1 | `powder/humanities-nuc1/20260628T0436Z` | 2026-06-28 04:37:13 | 2026-07-03 03:00:13 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Yes: all bands | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | madsen-nuc1 | `powder/madsen-nuc1/20260628T0437Z` | 2026-06-28 04:37:15 | 2026-07-03 02:54:15 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | moran-nuc1 | `powder/moran-nuc1/20260628T0437Z` | 2026-06-28 04:37:16 | 2026-07-03 02:59:16 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Seven-receiver spatial run; overlaps T5. |
| T4 | POWDER | sagepoint-nuc1 | `powder/sagepoint-nuc1/20260628T0437Z` | 2026-06-28 04:37:14 | 2026-07-03 02:55:14 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Seven-receiver spatial run; overlaps T5. |
| T5 | POWDER | law73-nuc1 | `powder/law73-nuc1/20260630T1949Z` | 2026-06-30 19:49:25 | 2026-07-03 18:37:25 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Adds two receivers during T4; usable nine-receiver overlap ends near 2026-07-03 03:00 UTC. |
| T5 | POWDER | web-nuc1 | `powder/web-nuc1/20260630T1949Z` | 2026-06-30 19:49:25 | 2026-07-03 18:37:25 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | Adds two receivers during T4; usable nine-receiver overlap ends near 2026-07-03 03:00 UTC. |
| T6 | POWDER | cpg-nuc1 | `powder/cpg-nuc1/20260703T1839Z` | 2026-07-03 18:39:38 | 2026-07-06 09:14:38 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0%; no runs | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | ebc-nuc1 | `powder/ebc-nuc1/20260703T1839Z` | 2026-07-03 18:39:29 | 2026-07-06 10:00:29 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0.0263%; 1,000/1/1 min | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | guesthouse-nuc1 | `powder/guesthouse-nuc1/20260703T1839Z` | 2026-07-03 18:39:15 | 2026-07-06 10:00:15 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 30.2049%; 4,000/274.5/597 min | Partial: 600--800 | Overheating caused long outages; exclude the backup-before-merge object prefix. |
| T6 | POWDER | humanities-nuc1 | `powder/humanities-nuc1/20260703T1839Z` | 2026-07-03 18:39:19 | 2026-07-06 09:49:19 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0%; no runs | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | madsen-nuc1 | `powder/madsen-nuc1/20260703T1839Z` | 2026-07-03 18:39:28 | 2026-07-06 09:58:28 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0.0527%; 2,000/1/1 min | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | moran-nuc1 | `powder/moran-nuc1/20260703T1839Z` | 2026-07-03 18:39:33 | 2026-07-06 10:00:33 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0.0264%; 1,000/1/1 min | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | sagepoint-nuc1 | `powder/sagepoint-nuc1/20260703T1839Z` | 2026-07-03 18:39:32 | 2026-07-06 09:58:32 | 600--800; 800--1000; 2400--2600; 3500--3700; 5725--5925 | 0%; no runs | Partial: 600--800 | Nine-receiver expanded-band run. |
| T6 | POWDER | law73-nuc1 | `powder/law73-nuc1/20260630T1949Z` | 2026-06-30 19:49:25 | 2026-07-03 18:37:25 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | The bucket has no five-band July 3 run for this point; treat it as temporal overlap only, not a full T6 member. |
| T6 | POWDER | web-nuc1 | `powder/web-nuc1/20260630T1949Z` | 2026-06-30 19:49:25 | 2026-07-03 18:37:25 | 600--800; 2400--2600; 3500--3700 | 0%; no runs | Partial: 600--800 | The bucket has no five-band July 3 run for this point; treat it as temporal overlap only, not a full T6 member. |
| T7 | ARA | horticulture-ue-004 | `ara/horticulture-ue-004/20260618T1255Z` | 2026-06-18 12:55:00 | 2026-06-25 11:34:00 | 600--800; 2400--2600; 3500--3700 | 14.3503%; 851,400/1/3 min | No | Long rural trace; missing values are frequent but short. |
| T8 | ARA | ames-ue-000 | `ara/ames-ue-000/20260702T1920Z` | 2026-07-02 19:20:04 | 2026-07-06 14:01:04 | 600--800; 2400--2600 | 0.5147%; 11,200/1/1 min | No | Three-receiver run; this point ends on July 6. |
| T8 | ARA | feedmill-ue-001 | `ara/feedmill-ue-001/20260702T1920Z` | 2026-07-02 19:20:05 | 2026-07-08 14:20:05 | 600--800; 2400--2600 | 42.1682%; 1,132,586/1/4 min | No | Three-receiver run; extended collection. |
| T8 | ARA | horticulture-ue-004 | `ara/horticulture-ue-004/20260702T1920Z` | 2026-07-02 19:20:04 | 2026-07-08 14:24:04 | 600--800; 2400--2600 | 1.8940%; 63,200/1/1 min | No | Three-receiver run; extended collection. |

The remote inventory exposes two discrepancies that must be resolved before freezing
the benchmark manifest. First, the paper assigns nine points and five bands to T6,
but only seven points have a July 3 five-band run. Law73 and WEB end before that run
and contain three bands. Second, the bucket contains short ARA runs from June 25
through July 17 beyond T7 and T8. These belong to T9--TX or later collections and
need explicit trace IDs before use.

### Additional ARA runs

The rows below cover every ARA power-CSV run in the bucket that is not already
listed as T7 or T8. `T9--TX?` marks the runs that still need stable benchmark IDs.
Some nominally short runs continued for multiple days; the CSV bounds expose those
extensions. None of these additional ARA runs has a matching annotation file.

| Trace | Collection point | Object-store run | Start UTC | Last observed minute UTC | Bands (MHz) | Internal missingness; runs/median/max |
|---|---|---|---|---|---|---|
| T9--TX? | agronomyfarm-ue-000 | `ara/agronomyfarm-ue-000/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:53:00 | 600--800; 2400--2600; 3500--3700 | 19.2308%; 27,600/1/4 min |
| T9--TX? | agronomyfarm-ue-000 | `ara/agronomyfarm-ue-000/20260630T1802Z` | 2026-06-30 18:02:09 | 2026-06-30 22:28:09 | 600--800; 2400--2600 | 3.7807%; 4,000/1/1 min |
| T9--TX? | agronomyfarm-ue-001 | `ara/agronomyfarm-ue-001/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:55:00 | 600--800; 2400--2600; 3500--3700 | 1.4981%; 800/2.5/6 min |
| T9--TX? | agronomyfarm-ue-001 | `ara/agronomyfarm-ue-001/20260626T1716Z` | 2026-06-26 17:16:53 | 2026-06-26 21:52:53 | 600--800; 2400--2600; 3500--3700 | 0%; no runs |
| T9--TX? | agronomyfarm-ue-001 | `ara/agronomyfarm-ue-001/20260630T1802Z` | 2026-06-30 18:06:44 | 2026-06-30 22:18:44 | 600--800; 2400--2600 | 0%; no runs |
| T9--TX? | agronomyfarm-ue-001 | `ara/agronomyfarm-ue-001/20260702T0115Z` | 2026-07-02 01:15:03 | 2026-07-04 06:44:03 | 600--800; 2400--2600 | 0%; no runs |
| T9--TX? | agronomyfarm-ue-010 | `ara/agronomyfarm-ue-010/20260626T1716Z` | 2026-06-26 17:17:06 | 2026-06-26 21:58:06 | 600--800; 2400--2600; 3500--3700 | 100%; 600/282/282 min |
| T9--TX? | agronomyfarm-ue-010 | `ara/agronomyfarm-ue-010/20260715T2200Z` | 2026-07-15 22:00:22 | 2026-07-16 00:35:22 | 600--800; 2400--2600 | 100%; 400/156/156 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:53:00 | 600--800; 2400--2600; 3500--3700 | 6.0575%; 9,200/1/2 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260626T1716Z` | 2026-06-26 17:16:54 | 2026-06-26 21:56:54 | 600--800; 2400--2600; 3500--3700 | 16.1252%; 26,800/1/1 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260630T1802Z` | 2026-06-30 18:02:11 | 2026-06-30 22:25:11 | 600--800; 2400--2600 | 0.3861%; 400/1/1 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260702T0115Z` | 2026-07-02 01:15:03 | 2026-07-02 06:03:03 | 600--800; 2400--2600 | 0.3521%; 400/1/1 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260702T1410Z` | 2026-07-02 14:10:04 | 2026-07-02 18:55:04 | 600--800; 2400--2600 | 0.3527%; 400/1/1 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260708T1700Z` | 2026-07-08 17:00:13 | 2026-07-08 21:40:13 | 600--800; 2400--2600 | 0.7168%; 800/1/1 min |
| T9--TX? | ames-ue-000 | `ara/ames-ue-000/20260715T2200Z` | 2026-07-15 22:00:31 | 2026-07-17 18:48:31 | 600--800; 2400--2600 | 0.5582%; 6,000/1/1 min |
| T9--TX? | curtissfarm-ue-000 | `ara/curtissfarm-ue-000/20260630T1802Z` | 2026-06-30 18:02:15 | 2026-06-30 22:20:15 | 600--800; 2400--2600 | 32.9457%; 29,600/1/2 min |
| T9--TX? | curtissfarm-ue-000 | `ara/curtissfarm-ue-000/20260702T1410Z` | 2026-07-02 14:10:04 | 2026-07-02 19:03:04 | 600--800; 2400--2600 | 25.5613%; 27,600/1/2 min |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:55:00 | 600--800; 2400--2600; 3500--3700 | 1.6565%; 1,404/1/5 min |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260626T1716Z` | 2026-06-26 17:16:51 | 2026-06-26 21:02:51 | 600--800; 2400--2600; 3500--3700 | 0%; no runs |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260630T1802Z` | 2026-06-30 18:02:15 | 2026-06-30 22:25:15 | 600--800; 2400--2600 | 10.0580%; 10,000/1/2 min |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260702T0115Z` | 2026-07-02 01:15:04 | 2026-07-02 06:04:04 | 600--800; 2400--2600 | 9.8246%; 10,400/1/2 min |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260702T1410Z` | 2026-07-02 14:10:04 | 2026-07-09 12:49:04 | 600--800; 2400--2600 | 10.4904%; 411,990/1/3 min |
| T9--TX? | curtissfarm-ue-001 | `ara/curtissfarm-ue-001/20260715T2200Z` | 2026-07-15 22:00:38 | 2026-07-17 18:47:38 | 600--800; 2400--2600 | 11.3903%; 120,800/1/2 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:55:00 | 600--800; 2400--2600; 3500--3700 | 59.8065%; 60,190/1/5 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260626T1716Z` | 2026-06-26 17:16:52 | 2026-06-26 21:54:52 | 600--800; 2400--2600; 3500--3700 | 60.7914%; 60,600/2/4 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260630T1802Z` | 2026-06-30 18:02:18 | 2026-06-30 22:20:18 | 600--800; 2400--2600 | 38.0583%; 31,400/1/3 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260702T0115Z` | 2026-07-02 01:15:05 | 2026-07-02 05:55:05 | 600--800; 2400--2600 | 38.1038%; 35,400/1/2 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260702T1410Z` | 2026-07-02 14:10:04 | 2026-07-02 18:56:04 | 600--800; 2400--2600 | 40.3509%; 40,000/1/3 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260708T1700Z` | 2026-07-08 17:00:05 | 2026-07-08 21:37:05 | 600--800; 2400--2600 | 44.5848%; 39,800/1/2 min |
| T9--TX? | feedmill-ue-001 | `ara/feedmill-ue-001/20260715T2200Z` | 2026-07-15 22:00:05 | 2026-07-17 18:55:05 | 600--800; 2400--2600 | 42.2351%; 371,200/1/3 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260625T1723Z` | 2026-06-25 17:23:00 | 2026-06-25 21:54:00 | 600--800; 2400--2600; 3500--3700 | 15.8950%; 24,200/1/4 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260626T1716Z` | 2026-06-26 17:16:50 | 2026-06-26 21:46:50 | 600--800; 2400--2600; 3500--3700 | 14.4654%; 23,000/1/1 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260630T1802Z` | 2026-06-30 18:02:18 | 2026-06-30 22:26:18 | 600--800; 2400--2600 | 1.1561%; 1,200/1/1 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260702T0115Z` | 2026-07-02 01:15:04 | 2026-07-02 05:53:04 | 600--800; 2400--2600 | 1.8083%; 2,000/1/1 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260708T1700Z` | 2026-07-08 17:00:12 | 2026-07-08 21:34:12 | 600--800; 2400--2600 | 1.1009%; 1,200/1/1 min |
| T9--TX? | horticulture-ue-004 | `ara/horticulture-ue-004/20260715T2200Z` | 2026-07-15 22:05:26 | 2026-07-17 18:50:26 | 600--800; 2400--2600 | 1.7145%; 18,400/1/1 min |


## Basic dataset

The evaluation uses POWDER T4 for model fitting and validation, then evaluates the
selected checkpoint on a later complete interval from T6. Source timestamps are
floored to UTC minutes. All boundaries are inclusive.

| Split | Trace | Start UTC | End UTC | Minutes | Selection rule |
|---|---|---|---|---:|---|
| Training | T4 | 2026-06-28 04:37 | 2026-07-02 02:23 | 5,627 | Common seven-site interval before the final T4 day. |
| Validation | T4 | 2026-07-02 02:24 | 2026-07-03 02:23 | 1,440 | Final 24 hours of the common T4 interval. |
| Test | T6 | 2026-07-03 18:39 | 2026-07-04 14:48 | 1,210 | Longest interval complete at all seven sites and all selected frequency bins. |

Every split uses `cpg`, `ebc`, `guesthouse`, `humanities`, `madsen`, `moran`, and
`sagepoint`, and the same 200 bins from 600.5 through 799.5 MHz. T6 has no complete
24-hour interval for this site-frequency set. The selected test interval is 20
hours and 10 minutes. Build every test window wholly inside this interval, and do
not extend it with imputed T6 rows. Use the maximum required history and rollout,
60 minutes each, to define a common set of eligible window starts for all models
and horizons.

Representation definitions:

- 1D: Treat each site-frequency pair as an independent scalar time series. Train/evaluate over all 7 × 200 = 1,400 series, with macro-averaged results so every site and frequency contributes equally.
- 2D: Treat each site as an independent time-frequency matrix with shape T × 200. Use all seven site matrices as separate segments of one dataset; windows must never cross site boundaries.
- 4D: Use all seven sites jointly to construct each frequency × height × width spatial map. The temporal input has shape T × 200 × H × W. Use one fixed grid and interpolation configuration fitted without test targets.

Models included in the common evaluation:

- 1D: Lookback Mean 1D, Linear AR 1D, Residual Linear AR 1D, Vanilla LSTM, Residual Vanilla LSTM, TimeRAN, TCN, LSTM-Attention, and ARIMA.
- 2D: Lookback Mean 2D, Linear AR 2D, Residual Linear AR 2D, Autoformer-CSA, TSS-LCD, and DeepSPred.
- 4D: Lookback Mean 4D, Linear AR 4D, Residual Linear AR 4D, ConvLSTM, ConvLSTM-FM, Residual ConvLSTM, DSwinLSTM-I, STS-PredNet in IDW-map mode, and Autoformer-CSA in spatial-map mode.

The 4D category requires an IDW-interpolated geographic map. RGB channels and a
site-by-frequency matrix do not qualify as spatial map dimensions.

Common comparison rules:
- Use identical target timestamps, sites, frequency bins, lookback, and horizons.
- Fit normalization on the training interval only.
- Allow validation and test targets to use earlier input rows from the same split; do not draw test inputs from outside the selected T6 interval.
- Use validation for early stopping and model selection, then evaluate the selected checkpoint once on T6.


## Common data preparation

1. Build and commit a small manifest generated from object paths, metadata, CSV
   dimensions, and CSV timestamps. Record one row per testbed, point, run, and
   band, including first and last minute, expected minute count, observed row
   count, duplicate timestamps, missing fraction, and longest gap.
2. Align simultaneous receivers on exact UTC minute bins. Preserve missing values
   and an observation mask. Interpolate only within training inputs when a model
   requires dense arrays; never interpolate labels across a train/test boundary.
3. Use chronological splits. Fit normalization, occupancy thresholds, frequency
   region definitions, and model-selection criteria on training data only. Retain
   raw dB values for reported MAE and RMSE.
4. Evaluate 600--800, 2400--2600, and 3500--3700 MHz as separate 200 MHz tasks.
   T6 can add 800--1000 and 5725--5925 MHz after resolving its membership.
5. Report persistence, lookback mean, and a seasonal baseline alongside every
   learned model. Use horizons of 1, 5, 15, and 60 minutes and a 60-minute lookback
   for the first controlled comparison.
6. Report per-frequency and per-region MAE/RMSE, then macro-average regions so
   quiet bins cannot dominate a full-band average. Publish sample count and missing
   fraction beside each metric.

## Evaluations enabled by the traces


### Temporal forecasting

Use T1, T2, T3, T4, and T7 for multi-day forecasting. Train on an initial
contiguous interval and reserve at least the final 48 hours for testing. Compare
short horizons with 24-hour and day-of-week seasonal baselines, and stratify errors
by hour of day, weekday/weekend status, activity level, and manually defined
frequency region. T6 and T8 support shorter-horizon tests but do not contain enough
complete days for the same weekly analysis.

### Spectral context

For each 200 MHz band, compare independent per-bin predictors against models that
consume the full frequency vector. Evaluate the target bins in fixed regions and
ablate neighboring-frequency context. T6 supports cross-band tests on five spans at
the same receivers if the benchmark restricts the comparison to the seven points
present in the bucket.

### Spatial prediction and receiver holdout

Use the common interval of T4 and T5, approximately 2026-06-30 19:50 through
2026-07-03 02:53 UTC, for the densest POWDER evaluation. Hold out one receiver at a
time, interpolate maps from the remaining receivers, and score predictions at the
physical held-out receiver. Repeat with distance-based holdouts and with fewer
input receivers. Use T3 for a three-point replication and T8 for a rural
three-point replication. Do not score interpolated grid points as if they were
independent measurements.

### Cross-location and cross-testbed transfer

Train on AERPAW CC2 and test zero-shot on CC1 and LW1 to separate nearby-campus
from campus-to-rural transfer. For the common PAWR bands, train on one testbed and
test on another after fitting normalization on the source only. Report zero-shot
results first, then fine-tuning curves using fixed target budgets such as 60, 360,
and 1,440 minutes. Keep a final target interval untouched by adaptation.

### Cross-band transfer

Train the same architecture on one 200 MHz span and test or fine-tune on another
span with the same number of frequency bins. Compare absolute-frequency inputs
with bin-relative inputs. T2, T3, T4, and T7 provide the common three-band design;
T6 can test transfer to 800--1000 and 5725--5925 MHz.

### Missing-data robustness

Measure natural gap distributions first. Evaluate models at those observed gaps
and under synthetic masks matched by gap length and receiver correlation. Report
performance against missing fraction and longest input gap. T6 guesthouse and the
ARA traces provide natural failure cases, but comparisons must use a shared set of
observable target minutes so a model cannot benefit from easier retained samples.

### Predictability characterization

Compute difference entropy, autocorrelation at 1, 60, and 1,440 minutes, spectral
occupancy rate, and temporal variance from training intervals. Relate each measure
to held-out forecast error by frequency region and trace. Use grouped confidence
intervals or a mixed-effects analysis with trace and receiver as groups; individual
minute-frequency cells are not independent samples.

### Model reporting

Publish aggregate, per-trace, per-receiver, per-band, and per-horizon tables. Include
the exact UTC split boundaries, usable target count, missing-data policy,
normalization scope, model seed, parameter count, training time, and inference
time. Save predictions with trace, receiver, timestamp, frequency, horizon, target,
prediction, and observation-mask fields so every reported metric can be rebuilt.
