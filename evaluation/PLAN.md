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


## Basic test

The evaluation uses POWDER T4 for model fitting and validation, then evaluates the
selected checkpoint on the full common interval of six reliable receivers from T6.
Source timestamps are
floored to UTC minutes. All boundaries are inclusive.

| Split | Trace | Start UTC | End UTC | Minutes | Selection rule |
|---|---|---|---|---:|---|
| Training | T4 | 2026-06-28 04:37 | 2026-07-02 02:23 | 5,627 | Common six-site interval before the final T4 day. |
| Validation | T4 | 2026-07-02 02:24 | 2026-07-03 02:23 | 1,440 | Final 24 hours of the common T4 interval. |
| Test | T6 | 2026-07-03 18:39 | 2026-07-06 09:14 | 3,756 | Full common span of the six selected sites; isolated short gaps (less than 10 minutes) may be interpolated. |

Every split uses `cpg`, `ebc`, `humanities`, `madsen`, `moran`, and `sagepoint`,
and the same 200 bins from 600.5 through 799.5 MHz. Exclude `guesthouse` because
its T6 receiver suffered long daytime outages. This permits use of the full common
T6 span, 62 hours and 36 minutes, rather than selecting test times according to
Guesthouse availability. T6 contains only isolated one-minute gaps for the six
selected sites. Fill those gaps by temporal linear interpolation within each
site-frequency series; do not interpolate longer gaps or extrapolate beyond a
site's observed bounds. Fit no interpolation parameters from validation or test
data. Build every test window wholly inside this interval. Use the maximum required
history and rollout, 60 minutes each, to define a common set of eligible window
starts for all models and horizons.

Representation definitions:

- 1D: Treat each site-frequency pair as an independent scalar time series. Train/evaluate over all 6 × 200 = 1,200 series, with macro-averaged results so every site and frequency contributes equally.
- 2D: Treat each site as an independent time-frequency matrix with shape T × 200. Use all six site matrices as separate segments of one dataset; windows must never cross site boundaries.
- 4D: Use all six sites jointly to construct each frequency × height × width spatial map on a fixed 10 × 10 geographic grid. The temporal input has shape T × 200 × 10 × 10. Keep the same bounding box, cell centers, receiver coordinates, IDW exponent, and distance handling across T4 and T6; fit the interpolation configuration without test targets.

Models included in the common evaluation:

- 1D: Lookback Mean 1D, Linear AR 1D, Residual Linear AR 1D, Vanilla LSTM, Residual Vanilla LSTM, TimeRAN, TCN 1D, LSTM-Attention 1D, and ARIMA.
- 2D: Lookback Mean 2D, Linear AR 2D, Residual Linear AR 2D, Vanilla LSTM 2D, Residual Vanilla LSTM 2D, Autoformer-CSA, LSTM-Attention 2D, TCN 2D, TSS-LCD, and DeepSPred.
- 4D: Lookback Mean 4D, Linear AR 4D, Residual Linear AR 4D, ConvLSTM, Residual ConvLSTM, ConvLSTM-FM, DSwinLSTM-I, STS-PredNet, and Autoformer-CSA in spatial-map mode.

(The 4D category requires an IDW-interpolated geographic map. RGB channels and a
site-by-frequency matrix do not qualify as spatial map dimensions.)

Common comparison rules:
- Use identical target timestamps, sites, frequency bins, lookback, and horizons.
- Fit normalization on the training interval only.
- Allow validation and test targets to use earlier input rows from the same split; do not draw test inputs from outside the selected T6 interval.
- Use validation for early stopping and model selection, then evaluate the selected checkpoint once on T6.
- do each for 3 different seeds

## Test vs input sequence/prediction horizon

Forecast difficulty and the value of additional input history may depend on the
temporal correlation structure of each frequency region. Evaluate the following
selected combinations, where both input and horizon are measured in minutes:

| Input history | 1-minute horizon | 15-minute horizon | 60-minute horizon | 120-minute horizon | 480-minute horizon |
|---:|:---:|:---:|:---:|:---:|:---:|
| 10 minutes | Yes | Yes | | | |
| 60 minutes | Yes | Yes | Yes | Yes | Yes |
| 120 minutes | Yes | Yes | | Yes | Yes |
| 480 minutes | Yes | Yes | | | Yes |

This design tests the effect of input length at fixed one- and 15-minute horizons,
the effect of forecast horizon with a fixed 60-minute input, matched input and
forecast lengths, and whether a longer history improves a 480-minute forecast.
Retrain each learned model for every applicable cell.

Use the same forecast origins for every cell. An eligible origin must have the full
480 minutes of preceding history and 480 minutes of future ground truth within its
split and segment. The 3,756-minute T6 test interval therefore provides about 2,797
minute-aligned origins before accounting for inclusive-boundary details. These
origins overlap and are not independent samples.

Report error at the exact horizon `t+h` for every cell. If a model produces the
full intermediate trajectory, cumulative error over `t+1` through `t+h` may be
reported as a secondary rollout metric, but do not use it for the
horizon-versus-autocorrelation analysis. Include Lookback Mean and persistence
baselines in every cell. Report results by frequency bin and annotated frequency
region, and compare model performance with the temporal dependence at the
corresponding scale.

Estimate the autocorrelation function for every site-frequency series using only
the training split. Evaluate lags of 1, 5, 10, 15, 30, 60, 120, 240, 480, 720, and
1,440 minutes. For each annotated frequency region, report the median and
interquartile range across sites and frequency bins. Also report signal variance
and the first lag at which the autocorrelation falls below 0.5. Interpret the
720- and 1,440-minute estimates with caution because T4 contains only a few daily
cycles.

Estimate uncertainty with blocked resampling rather than treating overlapping
windows as independent. Use blocks of at least 480 minutes. For stochastic models,
combine this sampling uncertainty with variation across training seeds.


## Test relevance of spectral structure

Test whether simultaneous measurements from other frequencies improve prediction
at a target frequency. Apply these tests to the 2D and 4D models; a 1D model has no
cross-frequency input to ablate. Mask historical inputs only and retain the original
targets. Replace a masked input with that site-frequency bin's training-set mean,
which is zero after training-only normalization. Retrain each learned model under
each mask rather than applying a new mask only at test time. Keep splits, forecast
origins, horizons, seeds, architecture, and optimization settings fixed between a
masked run and its full-band control.

The 600--800 MHz behavior annotations define the following non-noise target
regions. Evaluate every 1 MHz bin centered at 0.5 MHz within each inclusive range:

| Region | Range (MHz) | Bins | Behavior |
|---:|---:|---:|---|
| 1 | 600.5--607.5 | 8 | Mixed activity |
| 3 | 622.5--641.5 | 20 | Bursty, short timescale |
| 4 | 642.5--646.5 | 5 | Mixed activity |
| 5 | 647.5--655.5 | 9 | Diurnal pattern |
| 6 | 656.5--675.5 | 20 | Mixed activity |
| 8 | 691.5--728.5 | 38 | Mixed activity |
| 9 | 729.5--734.5 | 6 | Mixed activity |
| 10 | 735.5--740.5 | 6 | Intermittent occupancy |
| 11 | 741.5--745.5 | 5 | Intermittent occupancy |
| 12 | 746.5--755.5 | 10 | Bursty, short timescale |
| 13 | 756.5--768.5 | 13 | Diurnal pattern |
| 14 | 769.5--776.5 | 8 | Bursty, short timescale |
| 16 | 789.5--794.5 | 6 | Mixed activity |

Do not use the annotated noise-floor regions as prediction targets for these
ablations: 608.5--621.5 MHz (14 bins), 676.5--690.5 MHz (15 bins),
777.5--788.5 MHz (12 bins), and 795.5--799.5 MHz (5 bins). The target set therefore
contains 154 bins.

### Other-bin context

For each of the 154 target bins, compare:

- **Full band:** provide all 200 historical frequency bins.
- **Target bin only:** provide the historical values of the target bin and mask the
  other 199 bins with noise floor-like pattern.

Score only the target bin. Report the paired change in MAE or RMSE from target-bin
only to full-band input for every target bin, then summarize it within each of the
13 regions and across sites. This isolates the value of simultaneous observations
from other bins, including bins inside and outside the target's annotated region.

### Other-region context

For each of the 13 non-noise target regions, compare:

- **Full band:** provide all 200 historical frequency bins.
- **Target region only:** provide every historical bin in the target region and
  mask all bins outside it with noise-floor like pattern.

Score only bins in the retained target region. Report macro-averaged error across
its bins and sites, and the paired change from target-region-only to full-band
input. This tests whether other annotated regions provide information beyond the
joint history of bins that belong to the target transmission region.


## Test relevance of spatial structure

Spatial tests must distinguish three questions: whether other receivers help,
whether their geographic arrangement helps, and whether forecasting an interpolated
map directly helps more than forecasting receiver measurements and interpolating
afterward. Score every comparison at physical receiver locations. Do not treat the
100 IDW grid cells as independent ground truth.

### Spatial-correlation strata

Estimate spatial correlation for every frequency bin using only the training
portion of the relevant experiment. For each bin, calculate Spearman correlation
over time for every receiver pair, then average the pairwise correlations. Aggregate
bins within the fixed behavior annotations using the median. Define low correlation
as a region median below 0.25, moderate correlation as 0.25 through 0.50, and high
correlation as greater than 0.50. Freeze these assignments before validation and
test evaluation.

The existing nine-receiver T4+T5 analysis gives the following preliminary strata.
Recompute the values on the final training interval rather than copying these
numbers into the results:

| Stratum | Annotated non-noise regions and preliminary median pairwise temporal Spearman correlation |
|---|---|
| Low | R1 600.5--607.5 MHz (0.04) |
| Moderate | R3 622.5--641.5 (0.38); R4 642.5--646.5 (0.32); R6 656.5--675.5 (0.46); R8 691.5--728.5 (0.41); R9 729.5--734.5 (0.44); R10 735.5--740.5 (0.48); R11 741.5--745.5 (0.37); R12 746.5--755.5 (0.48); R14 769.5--776.5 (0.36); R16 789.5--794.5 MHz (0.45) |
| High | R5 647.5--655.5 MHz (0.65); R13 756.5--768.5 MHz (0.52) |

Exclude the four noise-floor regions from claims about useful spatial structure.
Report every spatial ablation separately for low-, moderate-, and high-correlation
regions. Also report each region separately because receiver hardware, antenna
response, local interference, and propagation conditions can weaken or reverse a
simple relationship between geographic distance and correlation. As a descriptive
check, report the association between receiver-pair distance and temporal
correlation within each region; do not use distance alone to assign the strata.

### Direct map forecasting versus receiver forecasting

For architectures that support both receiver-vector and map inputs, compare two
pipelines on the six-site Basic split:

- **Receiver forecast then IDW:** train on the six `T × 200` receiver matrices,
  forecast receiver power, and apply the fixed 10 × 10 IDW operator to each
  forecast time and frequency.
- **Direct map forecast:** apply the same IDW operator before training, train on
  `T × 200 × 10 × 10` inputs, and forecast the 10 × 10 maps directly.

Run this comparison for Lookback Mean, Linear AR, Residual Linear AR, ConvLSTM,
Residual ConvLSTM, STS-PredNet, and Autoformer-CSA where both modes are supported.
The first three provide architecture-neutral controls. ConvLSTM, STS-PredNet, and
Autoformer-CSA are map-based in the cited spectrum-prediction setups but can accept
receiver or frequency-vector adaptations; this comparison tests whether the map
construction itself adds predictive value. DSwinLSTM-I and ConvLSTM-FM remain in
the direct-map comparison only unless a receiver-vector implementation is added.

Use identical receiver targets, forecast origins, input lengths, horizons, seeds,
and IDW settings. Score both pipelines at the six physical receiver coordinates,
before and after IDW when applicable. Report training time, peak accelerator memory,
parameter count, and inference time alongside error because direct 4D forecasting
has a larger computational cost.

### Value of additional receivers

For each target receiver `X`, train and evaluate the following conditions:

- **Target only:** use only receiver `X` as input and predict `X`.
- **All receivers:** use all available receivers, including `X`, and predict `X`.
- **Other receivers only:** use every available receiver except `X` and predict
  `X`. This is receiver holdout and tests spatial transfer rather than additional
  context for an already observed receiver.

For 4D models, construct each input map only from the receivers allowed by that
condition. Keep the 10 × 10 grid bounds and cell centers fixed. Never include the
held-out receiver in the IDW input or target-map construction. Score only at the
physical coordinate of `X`. Compare the all-receiver and target-only conditions to
measure the value of simultaneous multi-receiver context. Report the other-receiver
condition separately because it is a harder interpolation-and-forecasting task.

### Geometry permutation test

Test whether a 4D model uses the geographic arrangement rather than only the
collection of receiver streams. Train the model with the correct map geometry.
At test time, randomly permute the assignment of receiver streams to coordinates
before IDW while leaving target values, target coordinates, grid bounds, and model
weights unchanged. Use at least 100 permutations shared across models and seeds.
For each region, report

`delta_error = error_permuted - error_original`.

A positive paired delta for distance-structured regions indicates that the learned
forecast depends on the correct geometry. Report the full permutation distribution
and a blocked confidence interval over forecast origins. Run the same receiver
permutations through 1D and 2D models as negative controls; their predictions must
not change because those models receive no coordinates. A permutation effect in a
low-correlation region indicates sensitivity to map construction, not useful
geographic structure, so interpret the result together with the region's measured
pairwise and distance-dependent correlation.

### Dense T4+T5 spatial test

Use T5 to add `law73` and `web` to the seven T4 receivers during their common
600--800 MHz interval, approximately 2026-06-30 19:49 through 2026-07-03 02:54 UTC.
Guesthouse is complete in T4 and should also be used in this separate nine-receiver spatial
experiment; its T6 outage is the reason it is excluded from the Basic T4-to-T6
test, not a defect in this overlap.

Determine the exact common endpoint from the minute-aligned files before generating windows.

Split the common interval chronologically into an initial training interval,
followed by 12 hours of validation and 12 hours of test data. Restrict this compact
experiment to a 60-minute input and horizons of 1, 15, and 60 minutes. Repeat the
target-only, all-receiver, other-receiver, and geometry-permutation comparisons for
each of the nine target receivers. Use one fixed 10 × 10 bounding box covering all
nine sites.

The T4+T5 experiment is the primary receiver-holdout and geometry test because it
has the densest simultaneous POWDER layout. The six-site T4-to-T6 experiment remains
the primary temporal generalization test because it has a later test period. Do not
pool their metrics: report the nine-site within-period spatial results separately
from the six-site later-period results.
