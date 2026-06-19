# Evaluation data

This directory contains spectrum data collected from POWDER, ARA, COSMOS, and
AERPAW testbed endpoints, plus the materials used to acquire
it.

## Acquisition script

`collect_spectrum.py` sweeps one or more 200 MHz bands with a USRP software radio,
computes PSD via Welch's method, and saves:

- **Per-minute aggregated data**: `power_1mhz_avg_per_minute.csv` — one row
  per minute, exactly 200 columns (one per 1 MHz bin).
- **Optional raw per-tune data**: `.npz` files with frequency and power vectors
  for every tuned center frequency. This is disabled by default and enabled
  with `--save-raw`.
- **Metadata**: `metadata.json` alongside every band directory.

with additional configuration via command-line arguments.


### All arguments

| Argument | Default | Description |
|---|---|---|
| `--site` | *required* | Site name (e.g. POWDER, ARA, AERPAW, COSMOS) |
| `--device-args` | `""` | UHD device arguments |
| `--rx-channel` | `0` | RX channel index |
| `--antenna` | `RX2` | RX antenna port |
| `--gain` | `35` | RX gain in dB |
| `--sample-rate` | `30.72e6` | Sample rate in Hz |
| `--bandwidth` | = sample-rate | RX bandwidth in Hz |
| `--fft-size` | `4096` | FFT size for Welch PSD |
| `--cutoff` | `0.836` | Fraction of FFT bins retained per tune |
| `--sample-seconds` | `0.2` | Seconds of IQ data per tune |
| `--bands` | *required* | Comma-separated 200 MHz bands, e.g. `3400:3600,3600:3800` |
| `--duration-minutes` | *required* | Total collection duration in minutes |
| `--output-dir` | *required* | Output directory root |
| `--save-raw` | disabled | Save optional per-tune raw PSD `.npz` files |
| `--overwrite` | disabled | Allow writing into an output directory that already has CSV, metadata, or raw files |
| `--sdr` | auto-detected | SDR model string for metadata |


### Output structure

Each collection process writes data on its remote node under the directory
passed to `--output-dir`, e.g.

```bash
./cosmos/$RUN_ID/$NODE_ID/
```

For example, a COSMOS run on `sdr1-md1` might save to
`./cosmos/20260615T1800Z/sdr1-md1/` on its experiment server.

```
<output-dir>/
  metadata.json                          # site-level
  3400_3600/
    metadata.json                        # per-band metadata
    power_1mhz_avg_per_minute.csv        # T rows × 200 columns
    raw/                                 # only present if --save-raw is used
      minute_0000_20260616T183000Z/
        tune_00_fc_3413MHz.npz          # per-tune PSD data
        tune_01_fc_3439MHz.npz
        ...
```


For each band, `power_1mhz_avg_per_minute.csv` has:

- A header row with bin-center labels (e.g. `3400.5,3401.5,...,3599.5`).
- One data row per minute of collection.
- Exactly 200 numeric columns (power in dB, uncalibrated).

Example:

```
3400.5,3401.5,3402.5,...,3599.5
-112.3,-111.8,-112.1,...,-114.0
-111.9,-112.2,-111.5,...,-113.7
...
```

### Plotting a band CSV

Use `plot_band_ground_truth.py` to render heatmaps from a collection tarball,
an extracted run directory, a band output directory, or a CSV file:

```bash
python3 plot_band_ground_truth.py \
  20260619T1509Z.tgz \
  --output plots/20260619T1509Z
```

For one band, point it at the band directory or CSV and choose an output file:

```bash
python3 plot_band_ground_truth.py \
  data/ara/ara-node/3400_3600/power_1mhz_avg_per_minute.csv \
  --output evaluation/ara_ground_truth.png
```

Flags:

- `--max-time N` plots only the first `N` rows. Use `--from-end` for the last `N` rows.
- `--no-smooth` disables the same light temporal smoothing used by the other plotting scripts.


## Data collection on PAWR testbeds

- [ARA portal](https://portal.arawireless.org/)
- [COSMOS portal](https://www.cosmos-lab.org/portal/)
- [POWDER portal](https://powderwireless.net/)
- [AERPAW portal](https://user-web-portal.aerpaw.ncsu.edu/)


### POWDER

POWDER fixed-endpoint `nuc1` resources provide USRP B210 access for passive
spectrum sensing, with a wideband antenna designed for operation at 698-6000MHz. 
These are mounted on the sides of buildings throughout the University of Utah campus 
(Central Parking Garage, Moran, EBC, Guest
House, Sage Point, Madsen, Law 73, Bookstore, Humanities, WEB)
at roughly human height (5-6ft). Available fixed-endpoint sites: .


These resources require advance reservation. Use the [POWDER portal](https://powderwireless.net/)
to make reservations for one or more fixed-endpoint `nuc1` resources - we used `law73`, `humanities`, 
and `guesthouse` locations. Since these measurements only receive
spectrum, no frequency reservation is needed.

At the beginning of the reservation time, bring up resources using the GNURadio profile: 

[https://www.powderwireless.net/show-profile.php?uuid=de1dc9d5-f79d-11ee-9f39-e4434b2381fc](https://www.powderwireless.net/show-profile.php?uuid=de1dc9d5-f79d-11ee-9f39-e4434b2381fc)

In the "Parameterize" tab, expand the "Fixed endpoint NUC+B210/COTSUE radios to allocate" section. 
Use the "+" button to add each of your reserved fixed endpoint nodes, one at a time. Then, launch the experiment.

Wait for all of the resources to be ready, then open an SSH session to each. You'll repeat the remainder of this section on every node.


On each node, install prerequisites:

```bash
sudo apt update
sudo apt -y  install python3-numpy python3-scipy uhd-host python3-uhd libuhd-dev screen
sudo python3 /lib/uhd/utils/uhd_images_downloader.py --types b2xx
```

and verify that the USRP is visible:

```bash
uhd_usrp_probe
```

Optionally, you can explore the spectrum to visually identify interesting bands. Use

```bash
/usr/local/lib/uhd/examples/rx_ascii_art_dft \
  --ref-lvl -60  --dyn-rng 40 \
  --step 10000000 --freq 600e6 \
  --num-bins 1024 --rate 31.25e6 \
  --frame-rate 2 --ant RX2
```

Use the right and left arrow keys to browse, and Ctrl+C to stop.

Then, clone this repository:

```bash
git clone https://github.com/Demii-7/spectrum-usage
cd spectrum-usage/evaluation
```

now you can start the spectrum monitoring job:


```bash
export RUN_ID=$(date -u +%Y%m%dT%H%MZ)
export NODE=$(hostname -s)

screen -dmS collect-spectrum \
  python3 collect_spectrum.py   --site POWDER \
   --device-args ""   --rx-channel 0   --antenna RX2   \
   --gain 25   --sample-rate 30.72e6   --bandwidth 30.72e6   \
   --bands "600:800,2400:2600,3500:3700"   \
   --duration-minutes 10000   --sample-seconds 0.2   \
   --tune-step-mhz 10   --center-notch-mhz 0.5   --dc-offset-mhz 0   \
   --output-dir powder/$RUN_ID   --save-raw   --overwrite
```

**Optional: Sync to Chameleon Object Storage**. To avoid filling the disk on the POWDER resource, we are periodically syncing the data to the Chameleon object storage service. 
A `spectrum` bucket should have been created at CHI@TACC, and S3 credentials should have been generated. Then,

```bash
cp .env.example .env
nano .env
```

fill in the credentials, and save. 

Also run 

```bash
python3 -m pip install fsspec s3fs
```

Now you can 

```bash
screen -dmS spectrum-sync \
  python3 sync_object_storage.py \
    powder/$RUN_ID \
    s3://spectrum/powder/$NODE/$RUN_ID \
    --env-file .env \
    --tool fsspec \
    --interval-seconds 300 \
    --prune-age-seconds 3600
```

to start syncing data to Chameleon (and pruning it locally).


### ARA

AraRAN UE resources provide USRP B210 access for passive
spectrum sensing. These radios have an antenna designed to operate from 698-4200 MHz,
and a front-end booster that operates on the frequency band 3400-3600.
They are deployed throughout the Ames, Iowa campus.

These resources require advance reservation. To make a reservation, 
log on to the [ARA portal](https://portal.arawireless.org/).
Then, create a lease: Go to "Reservations > Leases" and click "Create lease".

   - Set a start date and end date in UTC format,
     e.g. `2026-06-12T18:51:27Z`. The start date must be later than the
     current time.
   - Choose "AraRAN" as the resource type and "User equipment" as the
     device type.
   - Select a site and enter the DEVICE-ID. For example, site "Horticulture"
     with device ID 004.
   - Leave the "Wireless" tab blank.
   - Click "Create lease".

> Note: We arranged a week-long reservation via ARA support. Usually, reservations are limited to 5 hours.

Once the start date passes, the lease state changes from `PENDING` to `ACTIVE`.
At that point, you can create a container in the ARA Portal:

   - Go to "Container > Containers" and click "Create container".
   - Use the device name for Name (e.g. "horticulture-ue-004") and use the image `arawirelesshub/uhd:4.7.0.0`.
   - On the "Spec" page, use the device name again for "Hostname". Set the number of CPUs to 4 and the memory request to 8192. Select the lease name you just created.
   - On the "Networks" tab, select "ARA_Shared_Net".
   - Click "Create".

After the container reaches "Running" state, you can configure it. Click on the "Console" tab to open a shell. Then, set up SSH:

```bash
apt update
apt -y install openssh-server
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
passwd root # enter password twice
service ssh start
```

From the container Overview page, check the "Spec" section for the floating IP address.
Then, you can SSH to the instance and run the remaining steps in the SSH session. Use e.g.

```
ssh -J USERNAME@jbox.arawireless.org  root@10.189.X.Y
```

substituting your own jumpbox `USERNAME` and `X` and `Y`. 
You can find your jumpbox username by clicking on your username in the top right of the Horizon GUI, then on "Upload Public Key". 


Inside the SSH session, install prerequisites:

```bash
apt update
apt -y  install python3-numpy python3-scipy uhd-host python3-uhd libuhd-dev screen rsync
python3 /lib/uhd/utils/uhd_images_downloader.py --types b2xx
```

and verify that the USRP is visible:

```bash
uhd_usrp_probe
```

Optionally, you can explore the spectrum to visually identify interesting bands. Use

```bash
/usr/local/lib/uhd/examples/rx_ascii_art_dft \
  --ref-lvl -60  --dyn-rng 40 \
  --step 10000000 --freq 600e6 \
  --num-bins 1024 --rate 31.25e6 \
  --frame-rate 2 --ant RX2
```

Use the right and left arrow keys to browse, and Ctrl+C to stop.

Then, get the code:

```bash
git clone https://github.com/Demii-7/spectrum-usage
cd spectrum-usage/evaluation
```

and now you can start the spectrum monitoring job:


```bash
export RUN_ID=$(date -u +%Y%m%dT%H%MZ)
export NODE=$(hostname -s)

screen -dmS collect-spectrum \
  python3 collect_spectrum.py   --site ARA \
   --device-args ""   --rx-channel 0   --antenna RX2   \
   --gain 25   --sample-rate 30.72e6   --bandwidth 30.72e6   \
   --bands "600:800,2400:2600,3500:3700"   \
   --duration-minutes 10000   --sample-seconds 0.2   \
   --tune-step-mhz 10   --center-notch-mhz 0.5   --dc-offset-mhz 0   \
   --output-dir ara/$RUN_ID   --save-raw   --overwrite
```

**Note**: Currently, the network restrictions on ARA prevent you from sending data
to the Chameleon object store directly from ARA. To save data, you will have to retrieve the 
data from the ARA resource to another device, and then sync to the object store from there.

### COSMOS

COSMOS provides access to an outdoor deployment at Columbia University's West Harlem campus, 
with USRP N310 software radios that we will use for spectrum sensing. 
The large nodes types are on the Mudd building rooftop. The medium nodes 
are installed at the Mudd building near street level. 
These are available as part of the main `bed.cosmos-lab.org` deployment,
which can be requested for up to 5 hours at a time.

First, reserve the main `bed` domain via the [COSMOS portal](https://www.cosmos-lab.org/portal/). 

At the beginning of your reserved time, SSH to `bed.cosmos-lab.org` using your ORBIT username.

Then, find out what nodes are available:

```bash
omf stat -t all
```

and load a disk image onto the large and medium nodes that are available. The UHD version on the disk image
should match the firmware on the device reasonably well, or it will not work; we used `ubuntu2404-uhd4.9-gr3.10.ndz`.
(On other COSMOS sites, e.g. `sb1`, use `ubuntu2204-uhd4.4-gr3.10.ndz` instead.)

```bash
omf load -i ubuntu2404-uhd4.9-gr3.10.ndz -t sdr2-md1.bed.cosmos-lab.org,sdr2-s1-lg1.bed.cosmos-lab.org
omf tell -a on -t sdr2-md1.bed.cosmos-lab.org,sdr2-s1-lg1.bed.cosmos-lab.org
```

Then, you can SSH to the node as `root` user, from the `bed.cosmos-lab.org` console. e.g.

```
ssh -J <cosmos-user>@bed.cosmos-lab.org root@sdr2-s1-lg1.bed.cosmos-lab.org
```


Inside the SSH session, install prerequisites:

```bash
apt update
apt -y  install python3-pip python3-numpy python3-scipy python3-uhd libuhd-dev uhd-host
```

```bash
sudo sysctl -w net.core.rmem_max=25000000
sudo sysctl -w net.core.wmem_max=25000000
```

On each node, verify that the USRP is visible, and look for the serial number of a N3XX node specifically
(there are other USRP types attached to these nodes, but we will use the N3XX type).


```bash
uhd_find_devices --args "type=n3xx"
```

Optionally, you can explore the spectrum to visually identify interesting bands. Use

```bash
/usr/local/lib/uhd/examples/rx_ascii_art_dft \
  --gain 30 --ref-lvl -70  --dyn-rng 40 \
  --step 5000000 --freq 600e6 \
  --num-bins 1024 --rate 15.625e6 \
  --frame-rate 2 --ant RX2 --args "type=n3xx"
```


Use the right and left arrow keys to browse, and Ctrl+C to stop.

Then, clone this repository:

```bash
git clone https://github.com/Demii-7/spectrum-usage
cd spectrum-usage/evaluation
```

now you can start the spectrum monitoring job.

> **Note**: For simultaneous collection from multiple radios on the same testbed, specify serial number
> of each device in `--device-args` to prevent multiple nodes from trying to claim the same radio.


```bash
export RUN_ID=$(date -u +%Y%m%dT%H%MZ)
export NODE=$(hostname -s)

screen -dmS collect-spectrum \
  python3 collect_spectrum.py   --site COSMOS \
   --device-args  "type=n3xx" --rx-channel 0   --antenna RX2   \
   --gain 25   --sample-rate 15.625e6   --bandwidth 20e6   \
   --bands "600:800,2400:2600,3500:3700"   \
   --duration-minutes 10000   --sample-seconds 0.05  \
   --tune-step-mhz 5   --center-notch-mhz 0.5   --dc-offset-mhz 0   \
   --output-dir cosmos/$RUN_ID   --save-raw   --overwrite
```


**Optional: Sync to Chameleon Object Storage**. To avoid filling the disk on the COSMOS resource, we are periodically syncing the data to the Chameleon object storage service. 
A `spectrum` bucket should have been created at CHI@TACC, and S3 credentials should have been generated. Then,

```bash
cp .env.example .env
nano .env
```

fill in the credentials, and save. 

Also run 

```bash
python3 -m pip install fsspec s3fs
```

Now you can 

```bash
screen -dmS spectrum-sync \
  python3 sync_object_storage.py \
    cosmos/$RUN_ID \
    s3://spectrum/powder/$NODE/$RUN_ID \
    --env-file .env \
    --tool fsspec \
    --interval-seconds 300 \
    --prune-age-seconds 3600
```

to start syncing data to Chameleon (and pruning it locally).
