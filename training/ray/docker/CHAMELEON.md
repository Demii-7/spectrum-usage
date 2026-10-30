# Current Chameleon cluster

On October 30, 2026, we deployed stack `spectrum` on these hosts:

| Role | SSH address | Private address | Hostname |
|---|---|---|---|
| Manager, head, GPU worker, MinIO | `cc@129.114.27.86` | `192.168.1.43` | `ray-worker-1` |
| GPU worker | `cc@129.114.26.81` | `192.168.1.49` | `ray-worker-2` |

Each host contributes one H100 and 48 CPUs. Ray reports three logical nodes:
the CPU-free head container and two GPU worker containers. We copied the local
`training/` and `models/` source snapshot to `/home/cc/spectrum-usage` on both
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
