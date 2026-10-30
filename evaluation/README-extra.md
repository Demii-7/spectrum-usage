

### AERPAW

AERPAW fixed and portable nodes have USRP B205mini and USRP B210 radios
available for experimentation by default. The AERPAW manual also says USRP N310
and USRP X310 radios are deployed in fixed nodes, but they are not accessible
to experimenters by default; contact the AERPAW team if those are required.

The radios themselves support broad tuning ranges, but the standard fixed-node
RF front end constrains the usable receive path. The documented fixed-node RX
chain includes a Mini-Circuits `VBFZ-3590-S+` band-pass filter with an
approximately `3–4.2 GHz` passband. Use `3400–3600 MHz` for the standard fixed
node path.

Do not use low bands such as `85:285` on the standard fixed-node RF path. Those
frequencies are outside the fixed-node receive filter and would require a
different RF path, bypass, or explicit platform support from AERPAW staff.

For the Lake Wheeler site, use the `fixed-measure-lw` setup with fixed `afrn`
nodes and no vehicle. The portal lists these target resources with UHD `4.3`:

- `LW1`: Near Animal Health Building
- `LW2`: Westernmost edge of flying area
- `LW3`: Near Agroecology Education Farm
- `LW4`: Northern tip of non-wooded sub-area
- `LW5`: Far north, close to Tryon Road

Setup steps:

1. **Log in** to the [AERPAW portal](https://user-web-portal.aerpaw.ncsu.edu/).
   Create or select the `fixed-measure-lw` experiment at the Lake Wheeler site.

2. **Select target resources** from the Lake Wheeler fixed nodes: `LW1`, `LW2`,
   `LW3`, `LW4`, and/or `LW5`. The node type should be `afrn`, node UHD should
   be `4.3`, and node vehicle should be `vehicle_none`.

3. **SSH into the E-VM** using the credentials provided by the portal.

4. **Install the AERPAW launcher as `Radio/startRadio.sh`.** Clone this
   repository on the E-VM:

   ```bash
   cd /root
   git clone https://github.com/Demii-7/spectrum-usage
   ```

   Copy the launcher into the active radio profile script:

   ```bash
   cd /root/Profiles/ProfileScripts/Radio
   cp /root/spectrum-usage/evaluation/startSpectrumUsageCollection.sh startRadio.sh
   chmod +x startRadio.sh
   ```

   The launcher handles the collection settings. It uses the standard
   `3400:3600` band and writes results under `/root/Results`.

   The same launcher runs in both AERPAW Testbed mode and Digital Twin mode.
   In Digital Twin mode, UHD should discover the V-USRP from the normal AERPAW
   environment. If you need explicit UHD device arguments for a test, set only
   `SPECTRUM_USAGE_DEVICE_ARGS`.

5. **Make sure `/root/startexperiment.sh` runs the radio profile.** The usual
   AERPAW entry point is:

   ```text
   /root/startexperiment.sh
     -> /root/Profiles/ProfileScripts/startAll.sh
     -> Radio/startRadio.sh
   ```

   If your local `/root/startexperiment.sh` has the radio line commented out,
   uncomment `./Radio/startRadio.sh` or the equivalent radio start command.

6. **Start the experiment** from the E-VM using the normal AERPAW entry point:

    ```bash
    /root/startexperiment.sh
    ```

7. **Stop the experiment** with the normal AERPAW stop script:

    ```bash
    /root/stopexperiment.sh
    ```

8. **Retrieve the data** from the timestamped `spectrum_usage_*` directory
   under `$RESULTS_DIR` after the run completes.

### COSMOS

COSMOS provides USRP N310 access in the main `bed.cosmos-lab.org` deployment
in West Harlem. The N310 supports 1 MHz – 6 GHz with up to 100 MHz bandwidth,
and it can collect multiple 200 MHz bands in sequence.

For the Columbia University deployment, use the large and medium fixed nodes.
The large node `lg1` is on the Mudd building rooftop. The medium nodes `md1`
and `md2` are installed at the Mudd building near street level. Each large-node
sector and each medium node includes a USRP N310 and a USRP 2974; use the
`sdr1-*` resources for N310 access.

Available N310 resources:

- `sdr1-s1-lg1.bed.cosmos-lab.org`
- `sdr1-s2-lg1.bed.cosmos-lab.org`
- `sdr1-s3-lg1.bed.cosmos-lab.org`
- `sdr1-md1.bed.cosmos-lab.org`
- `sdr1-md2.bed.cosmos-lab.org`

Example command:

```bash
python3 collect_spectrum.py \
  --site COSMOS \
  --device-args "mgmt_addr=10.37.2.1,addr=10.39.2.1" \
  --antenna RX2 \
  --gain 30 \
  --bands 3400:3600,3600:3800,3800:4000 \
  --duration-minutes 60 \
  --output-dir ./data/$RUN_ID/cosmos-<sdr>
```

Setup steps:

1. **Reserve time** on the main `bed` domain via the
   [COSMOS portal](https://www.cosmos-lab.org/portal/). Select one N310 SDR
   resource from the large or medium Columbia sites, plus an experiment server
   with network access to that SDR.

ssh gw.cosmos-lab.org

ssh bed.cosmos-lab.org
ssh outdoor.orbit-lab.org


2. **Load the UHD disk image** onto the experiment server from the COSMOS
   console. 

#### BED


*NEEDS* newer UHD

   ```bash
   omf load -i ubuntu2404-uhd4.9-gr3.10.ndz \
      -t sdr2-md1.bed.cosmos-lab.org,sdr2-s1-lg1.bed.cosmos-lab.org
   omf tell -a on \
      -t sdr2-md1.bed.cosmos-lab.org,sdr2-s1-lg1.bed.cosmos-lab.org
   ```


uhd_find_devices --args "type=n3xx"

uhd_usrp_probe --args "type=n3xx,serial=3176DF9"


wget https://raw.githubusercontent.com/Demii-7/spectrum-usage/refs/heads/main/evaluation/collect_spectrum.py

python3 collect_spectrum.py \
  --site ORBIT \
  --sdr N310 \
  --device-args "type=n3xx,serial=3176DF9,mgmt_addr=10.110.10.5,addr=10.112.10.5" \
  --rx-channel 0 \
  --antenna RX2 \
  --gain 35 \
  --sample-rate 15.625e6 \
  --bandwidth 20e6 \
  --bands "3400:3600" \
  --duration-minutes 20 \
  --sample-seconds 0.016 \
  --output-dir ./n310_1gbe_15p625 \
  --overwrite


python3 collect_spectrum.py \
  --site ORBIT \
  --sdr N310 \
  --device-args "type=n3xx,serial=3176DFE,mgmt_addr=10.110.11.1,addr=10.112.11.1" \
  --rx-channel 0 \
  --antenna RX2 \
  --gain 35 \
  --sample-rate 15.625e6 \
  --bandwidth 20e6 \
  --bands "3400:3600" \
  --duration-minutes 20 \
  --sample-seconds 0.016 \
  --output-dir ./n310_3176DFE_1gbe_15p625 \
  --overwrite


---

#### OUTDOOR

omf load -i ubuntu2404-uhd4.9-gr3.10.ndz -t node1-1,node1-3,node1-4,node1-9 # this is too new for the USRPs here

omf load -i baseline20.04-sdr.ndz -t node1-1,node1-3,node1-4,node1-9

omf tell -a on -t node1-1,node1-3,node1-4,node1-9



Try: omf load -i ubuntu2204-uhd4.4-gr3.10.ndz -t node1-1

THIS IS THE ONE!!!!!

sudo sysctl -w net.core.wmem_max=2500000

apt update
apt install python3-numpy python3-scipy python3-uhd libuhd-dev uhd-host


node1-3 and 1-4 and 1-9 have Exxx devices

wget https://raw.githubusercontent.com/Demii-7/spectrum-usage/refs/heads/main/evaluation/collect_spectrum.py

To avoid center artifacts:

python3 collect_spectrum.py   --site ORBIT   --sdr E310   --gain 35   --rx-channel 0   --antenna RX2   --sample-rate 2e6   --bandwidth 2e6   --bands "3400:3600"   --duration-minutes 1   --sample-seconds 0.016   --tune-step-mhz 1   --center-notch-mhz 0.1   --output-dir ./e310_2MHz_overlap_notch_0p1   --overwrite

python3 collect_spectrum.py \
  --site ORBIT \
  --sdr E310 \
  --gain 35 \
  --rx-channel 0 \
  --antenna RX2 \
  --sample-rate 2e6 \
  --bandwidth 2e6 \
  --bands "3400:3600" \
  --duration-minutes 1 \
  --sample-seconds 0.016 \
  --tune-step-mhz 1 \
  --output-dir ./e310_2MHz_overlap \
  --overwrite


```
  _____________________________________________________
 /
|       Device: E300-Series Device
|     _____________________________________________________
|    /
|   |       Mboard: ni-e31x-316E0E0
|   |   dboard_0_pid: 272
|   |   dboard_0_serial: 316D307
|   |   fs_version: 20230131233832
|   |   mender_artifact: v4.4.0.0_e310_orbit.mender
|   |   mpm_sw_version: 4.4.0.0-g5fac246b
|   |   pid: 30675
|   |   product: e310_sg3
|   |   rev: 7
|   |   rpc_connection: remote
|   |   serial: 316E0E0
|   |   type: e3xx
|   |   MPM Version: 4.4
|   |   FPGA Version: 6.0
|   |   FPGA git hash: 92c09f7.clean
|   |   RFNoC capable: Yes
|   |   
|   |   Time sources:  internal, external, gpsdo
|   |   Clock sources: internal
|   |   Sensors: ref_locked, gps_locked, temp_fpga, temp_mb, gps_gpgga, gps_sky, gps_time, gps_tpv
```

wget https://raw.githubusercontent.com/Demii-7/spectrum-usage/refs/heads/main/evaluation/collect_spectrum.py

on node1-4

python3 collect_spectrum.py \
  --site test \
  --device-args "type=e3xx,addr=10.40.250.4" \
  --sdr "E310 SG3" \
  --rx-channel 0 \
  --antenna RX2 \
  --gain 35 \
  --sample-rate 30.72e6 \
  --bands "3400:3600" \
  --duration-minutes 1 \
  --output-dir ./e310_spectrum_test

3. **SSH into the experiment server** using the credentials provided:

   ```bash
   ssh root@<server>.bed.cosmos-lab.org
   ```




4. **Install missing Python dependencies** if needed. The UHD image includes
   Python 3 and `numpy`; install `scipy` if it is missing:

   ```bash
   sudo apt update
   sudo apt install python3-scipy
   python3 -c "import uhd; print('UHD Python OK')"
   ```

5. **Clone the repository** onto the node:

   ```bash
   git clone git@github.com:Demii-7/spectrum-usage.git
   cd spectrum-usage/evaluation
   ```

6. **Configure the N310 data interface** on the experiment server. The N310 has
   a 1Gb control interface and a 10Gb data interface. COSMOS docs use `DATA1`
   as the data-interface placeholder; replace it with the actual interface name
   on the reserved server if needed:

   ```bash
   sudo ip addr add 10.39.1.1/16 dev DATA1
   sudo ip link set DATA1 mtu 8000
   sudo sysctl -w net.core.wmem_max=62500000
   sudo sysctl -w net.core.rmem_max=62500000
   ```

7. **Discover and verify the N310**. First list reachable N3xx devices:

   ```bash
   uhd_find_devices --args="type=n3xx"
   ```

   Then probe the selected N310 using the discovered `mgmt_addr` and `addr`:

   ```bash
   uhd_usrp_probe --args="mgmt_addr=10.37.2.1,addr=10.39.2.1"
   ```

   Confirm the probe shows `N300-Series Device` or `N310`, with
   `Antennas: TX/RX, RX2` and `Freq range: 1.000 to 6000.000 MHz`.

   If the selected N310 uses different IPs, update the `--device-args`
   accordingly.

8. **Run a short test** to confirm data collection works:

   ```bash
   python3 collect_spectrum.py \
     --site COSMOS \
     --device-args "mgmt_addr=10.37.2.1,addr=10.39.2.1" \
     --antenna RX2 \
     --gain 30 \
     --bands 3400:3600 \
     --duration-minutes 1 \
     --output-dir ./data/$RUN_ID/test-cosmos-<sdr>
   ```

9. **Launch the full collection** using `tmux` or `screen`:

   ```bash
   screen -dmS spectrum \
     python3 collect_spectrum.py \
       --site COSMOS \
       --device-args "mgmt_addr=10.37.2.1,addr=10.39.2.1" \
       --antenna RX2 \
       --gain 30 \
       --bands 3400:3600,3600:3800,3800:4000 \
       --duration-minutes 60 \
       --output-dir ./data/$RUN_ID/cosmos-<sdr>
   ```

   The N310 can support multiple 200 MHz bands in sequence each minute.

10. **Retrieve the data** from `./data/$RUN_ID/cosmos-<sdr>/` after the run
   completes.

