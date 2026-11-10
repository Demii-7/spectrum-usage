# Current Chameleon cluster

As of November 1, 2026, stack `spectrum` uses these hosts:

| Role | SSH address | Private address | Hostname |
|---|---|---|---|
| Manager, head, GPU worker, MinIO | `cc@129.114.27.86` | `192.168.1.43` | `ray-worker-1` |
| GPU worker | `cc@129.114.26.81` | `192.168.1.49` | `ray-worker-2` |
| GPU worker | `cc@129.114.25.125` | `192.168.1.26` | `ray-worker-3` |
| GPU worker | `cc@129.114.25.135` | `192.168.1.186` | `ray-worker-4` |

Each host contributes one H100 and 48 CPUs, for 4 GPUs and 192 CPUs total.
Ray reports five logical nodes: the CPU-free head container and four GPU
worker containers. We copy the
`training/` and `models/` source snapshot to `/home/cc/spectrum-usage` on all four
hosts. Containers mount that directory at `/workspace/spectrum-usage`.

The image `spectrum-ray:2.54.0-chameleon` contains Ray 2.54.0 and PyTorch
2.5.1 with CUDA 12.4. We built it once on the manager and copied the image to
the worker with `docker image save/load`.

## Access and jobs

Open an SSH tunnel from your workstation:

```bash
ssh -N -L 8265:localhost:8265 -L 9001:localhost:9001 cc@129.114.27.86
```

Visit `http://localhost:8265` for the Ray dashboard and Jobs API. MinIO's
console uses `http://localhost:9001`. For CLI access on the head:

```bash
container=$(docker ps -q --filter label=com.docker.swarm.service.name=spectrum_ray-head)
docker exec "$container" ray status
docker exec "$container" ray job submit --address http://127.0.0.1:8265 -- \
  python -m training.ray.cluster_probe --fractional-gpu-check 0.25 \
  --check-model-imports --check-minio-cross-worker --check-spectrum-read
```

We passed that probe through the Jobs API as `chameleon-acceptance-1`.
It checked GPU UUIDs, CUDA, model imports, eight fractional GPU tasks, a
MinIO write/read/delete across workers, and a spectrum object read on each
worker. Both workers read 4096 bytes with the same SHA-256 checksum.

After adding `ray-worker-3`, we passed `chameleon-acceptance-3-workers` with
all three expected GPU UUIDs, twelve fractional GPU tasks, model imports,
and matching spectrum reads on all three workers. The head stores the log at
`/home/cc/chameleon-acceptance-3-workers.log`.

On November 1, we replaced the failed worker 3 at `129.114.25.96` with
`129.114.25.125` and added worker 4 at `129.114.25.135`. We removed the old
Swarm node `ziy7mp488k4ihq0t0i4klp5kc` after confirming its Down state.
We installed Ubuntu host prerequisites and copied the existing Ray image to
the new hosts. Their source snapshots use `integrate` commit `fd60eae` and
include `data/locations/`, including `powder.json`.

We kept the new nodes outside the `ray.gpu=true` worker placement constraint
while preparing them. A temporary Swarm service mounted the existing storage
secrets for data downloads and checks. Each new host received 208 processed
CSVs (3,856,317,115 bytes) and 251 checkpoint objects (24,609,999,755 bytes).
Full-file SHA-256 checksums matched between the new hosts for both selections.
Both hosts passed CUDA computation, model imports, spectrum reads, and MinIO
write/read/delete checks before we enabled Ray scheduling. Both POWDER runs
`20261015T1434Z` and `20261025T1355Z` contain 14 processed CSVs on each new host.

We removed the temporary service and applied `ray.gpu=true` after preparing
the files. We kept the head and worker 2 containers running throughout the
replacement. After the user requested a refresh on all four hosts, we synced
`integrate` commit `241528c`, including `data/locations/`, and verified 531
files per host. The current four-host inventory and this runbook overlay the
committed deployment metadata. No services required a restart.

The subsequent all-host asset sync verified matching SHA-256 checksums for
208 processed CSVs (3,856,421,295 bytes per host) and 255 checkpoint objects
(24,887,148,638 bytes per host). Ray reported four H100 workers and 192 CPUs.
Node-affinity tasks on all four workers verified the location JSON, both
October POWDER runs, source-bucket reads, and a MinIO object written on worker
3 and read on all four workers.

Hosts use copied source snapshots from the local `integrate` checkout.
They do not track the branch with Git. Pushes to `integrate` require an explicit
source sync to update running hosts. The add-worker script copies the manager's
current snapshot to each new host.

## Data and outputs

We copied the object-store credentials from local `evaluation/.env` into
Swarm secrets `ray_spectrum_access_key` and `ray_spectrum_secret_key`.
The head and GPU workers mount those secrets. Use this client inside tasks:

```python
import os
from training.ray.storage import spectrum_s3_client

client = spectrum_s3_client()
bucket = os.environ["SPECTRUM_S3_BUCKET"]
objects = client.list_objects_v2(Bucket=bucket, MaxKeys=10)
```

The source data lives in bucket `spectrum` at
`https://chi.tacc.chameleoncloud.org:7480`. Containers use the
`SPECTRUM_S3_*` and `SPECTRUM_AWS_*` variables for this store.

Ray outputs use MinIO at `http://minio:9000`, bucket `spectrum-usage`.
The standard `AWS_*` and `S3_BUCKET` variables refer to MinIO. Swarm stores
its generated credentials in `ray_minio_root_user` and
`ray_minio_root_password`. MinIO keeps data in the manager's Docker volume
`spectrum_minio-data`.

Quay returned HTTP 401 for both requested images, including
`quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z` and
`quay.io/minio/mc:RELEASE.2025-06-13T11-33-47Z-cpuv1`. We used accessible
Bitnami archive images for the April 2025 server and client versions. The
stack invokes the server binary at `/opt/bitnami/minio/bin/minio`; change
that entrypoint if you switch to the upstream image.

To redeploy from the manager:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE=spectrum-ray:2.54.0-chameleon
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml spectrum
```

## Add workers

From your workstation, with your Chameleon key loaded in `ssh-agent`:

```bash
bash training/ray/scripts/add-chameleon-worker.sh NEW_PUBLIC_IP NEW_PRIVATE_IP
```

The script installs Docker and the NVIDIA toolkit, copies the manager's source
and image through the private network, joins the Swarm, and labels the new
node. The global service starts a worker with both storage configurations.
Use idle Ubuntu 24.04 hosts with one H100 each and private network access to
the manager. Permit TCP 2377, TCP/UDP 7946, UDP 4789, and IP protocol 50
(ESP, for the encrypted overlay) between cluster hosts.

Add each new private address and hostname to `chameleon-nodes.sh` for the
inventory-based inspection scripts. Run the acceptance command above after
joining nodes and confirm that `gpu_worker_count` matches the host count.
The manager and its MinIO volume stay in place.

For an expansion during active campaigns, prepare code, locations, processed
data, and checkpoints before applying `ray.gpu=true`. The basic add-worker
script applies that label as soon as it joins the node. Use a temporary
secret-bearing preparation service for the staged workflow described above,
then remove it before enabling the Ray worker. A preparation service can use
the same image, repository mount, network, environment, and secrets as the
worker service, with its entrypoint replaced by `bash -c 'sleep infinity'`.

## Sync processed data

Run the cluster wrapper from your workstation:

```bash
bash training/ray/scripts/sync-cluster-data.sh --dry-run
bash training/ray/scripts/sync-cluster-data.sh
# Select a subset instead of the default, which covers all sites:
bash training/ray/scripts/sync-cluster-data.sh --site powder
```

The wrapper uses `RAY_NODE_SSH_HOSTS` in `chameleon-nodes.sh` to reach each
host. It copies and runs the existing `data/sync_object_storage.py` inside each
GPU worker container. That script reads the source-store secrets with
`--spectrum-storage`, leaving the worker's default MinIO environment intact.

The first host lists processed `power_1mhz_avg_per_minute.csv` objects and
skips `raw/` subtrees. The wrapper shares that inventory with the other hosts,
downloads the selected files on each host, and compares full-file SHA-256
checksums across all hosts. Each download uses a temporary `.part` file before
replacing its destination. Repeat runs skip files with matching size and
last-modified timestamp. Ray containers see the data through their existing
repository bind mount; a data sync does not require a Ray restart.

On October 30, 2026, we synced 206 files (3,772,142,226 bytes per host):
3 AERPAW, 93 ARA, 6 COSMOS, and 104 POWDER files. All three hosts returned
matching checksums. Files reside at `/home/cc/spectrum-usage/data/<object-key>`
on the host and `/workspace/spectrum-usage/data/<object-key>` in containers.
Each host keeps the inventory and checksums under `data/.spectrum-sync/`.
The source bucket can receive new data between runs; rerun the wrapper to
refresh the inventory and download updates.

## Sync checkpoints

Use the same script with checkpoint mode:

```bash
bash training/ray/scripts/sync-cluster-data.sh --checkpoints --dry-run
bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

The wrapper downloads `s3://spectrum/checkpoints/` to the repository root on
each host. It preserves paths such as
`checkpoints/1d/linearar1d/seed-40/powder_600_800_linearar1d.pt` and includes
`checkpoints/MANIFEST.tsv`. Containers read them under
`/workspace/spectrum-usage/checkpoints/`.

On October 30, 2026, we synced 67 model checkpoint files and the manifest,
173,142,276 bytes per host. All three hosts returned matching SHA-256
checksums. Checkpoint mode keeps its inventory and checksums separate from
the processed-data records, in `data/.spectrum-sync/checkpoint-inventory.json`
and `data/.spectrum-sync/checkpoint-checksums.json`. Run both sync modes when
preparing a new worker for training or evaluation.
