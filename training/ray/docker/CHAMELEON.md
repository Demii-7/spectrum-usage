# Current Chameleon cluster

On October 30, 2026, we deployed stack `spectrum` on these hosts:

| Role | SSH address | Private address | Hostname |
|---|---|---|---|
| Manager, head, GPU worker, MinIO | `cc@129.114.27.86` | `192.168.1.43` | `ray-worker-1` |
| GPU worker | `cc@129.114.26.81` | `192.168.1.49` | `ray-worker-2` |
| GPU worker | `cc@129.114.25.96` | `192.168.1.136` | `ray-worker-3` |

Each host contributes one H100 and 48 CPUs, for 3 GPUs and 144 CPUs total.
Ray reports four logical nodes: the CPU-free head container and three GPU
worker containers. We copied the local
`training/` and `models/` source snapshot to `/home/cc/spectrum-usage` on all three
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
