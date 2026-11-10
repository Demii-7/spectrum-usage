# Ray cluster with mixed GPU hosts

**Historical deployment:** the four-GPU head instance expired on November 7.
For the replacement head on `ray-worker-2` and six one-GPU hosts, use
[H100.md](H100.md) and `h100-ssh.conf`. The addresses and deploy commands below
describe the previous topology.

The independent cluster uses stack `standalone`, with its manager and Ray head
at `cc@129.114.25.96`. The head advertises zero CPUs and zero GPUs. The
three GPU worker containers advertise six H100 GPUs and 288 CPUs in total.
Ray reports four logical nodes, including the head.

| Hostname | SSH address | Private address | GPUs | CPUs | Worker service |
|---|---|---|---:|---:|---|
| `ray-standalone` | `cc@129.114.25.96` | `192.168.1.175` | 4 | 192 | `standalone_ray-worker` |
| `ray-worker-1` | `cc@129.114.26.81` | `192.168.1.39` | 1 | 48 | `standalone_ray-single-gpu-worker` |
| `ray-worker-2` | `cc@129.114.27.86` | `192.168.1.185` | 1 | 48 | `standalone_ray-single-gpu-worker` |

## Services and shared memory

Deploy `stack.yaml` with the `standalone.yaml` overlay. The overlay sets the
worker's GPU count to four, provides a 256 GiB `/dev/shm` tmpfs, and sets Ray's
object-store limit to 64 GiB. PyTorch DataLoader processes can use the
remaining shared-memory capacity. The tmpfs size is a limit; the kernel
allocates RAM as processes write to it. The head has a separate 2 GiB
`/dev/shm`, a 1 GiB object-store limit, and a 1 GiB logical Ray memory resource
to avoid advertising the host's RAM twice.

The one-GPU service uses a 64 GiB `/dev/shm` tmpfs and a 16 GiB Ray object
store on each new host. Those hosts carry `ray.single_gpu=true` and do not
carry `ray.gpu=true`, so the original service cannot launch a four-GPU worker
on a one-GPU machine. The service definition in `standalone-workers.yaml`
contains only the new worker service and references the existing overlay
network and Swarm secrets.

The deployed image is `spectrum-ray:2.54.0-standalone-921290800551`, image ID
`sha256:921290800551c46441627be9ed385baa77d96a7afd24210f4efce0e3944f72ad`.
It contains Ray 2.54.0 and PyTorch 2.5.1 with CUDA 12.4. We built it from the
existing pinned cluster image with the memory-aware entrypoints. Rebuild with
a new tag if the requirements, Dockerfile, or entrypoints change.

The Ray containers mount `/home/cc/spectrum-usage` at
`/workspace/spectrum-usage`. The head and four-GPU worker share one host copy
of `data/` and `checkpoints/`; the new machines keep their own synced copies.
Give jobs distinct output directories.

To inspect the running worker on the standalone host:

```bash
worker=$(docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-worker)
docker exec "$worker" df -h /dev/shm
docker exec "$worker" nvidia-smi -L
docker stack services standalone
```

## Storage and source

The code snapshot uses `integrate` commit `0b18e33`, plus the standalone
deployment files, memory-aware entrypoints, and acceptance probe. The host
records its source commit in `/home/cc/spectrum-source-revision`. The snapshot
includes `data/locations/powder.json`.

This Swarm has its own `ray_spectrum_access_key` and
`ray_spectrum_secret_key` secrets, populated from local `evaluation/.env`.
Use `training.ray.storage.spectrum_s3_client()` for the source bucket
`spectrum` at `https://chi.tacc.chameleoncloud.org:7480`. The containers'
`SPECTRUM_*` variables select those credentials and that endpoint.

The default `AWS_*` variables select this cluster's MinIO at
`http://minio:9000`, bucket `spectrum-usage`. Its generated credentials live
in this Swarm's `ray_minio_root_user` and `ray_minio_root_password` secrets.
MinIO stores outputs in Docker volume `standalone_minio-data` on this host.

On November 2, 2026, the sync downloaded and verified sizes and SHA-256
checksums for 208 processed CSVs (3,890,686,143 bytes) and 281 checkpoint
objects (26,482,797,254 bytes). Both Ray containers see those files through
the shared repository mount.

To refresh assets from your workstation, select the standalone inventory:

```bash
RAY_INVENTORY=training/ray/docker/standalone-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
RAY_INVENTORY=training/ray/docker/standalone-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

`standalone-nodes.sh` selects all three hosts and maps each address to its
worker service through `RAY_NODE_SERVICES`. Add `--dry-run` to preview
downloads. A data sync does not require restarting Ray. For rebuilt instances
at reused IPs, the wrapper accepts `RAY_SSH_KNOWN_HOSTS` for a scoped SSH
host-key record. The inventory uses the ignored
`secrets/standalone_known_hosts` file when it exists, preserving old entries
in the workstation's normal SSH known-hosts file.

On November 3, 2026, we prepared and verified the new hosts before enabling
their Ray workers. All three hosts then returned matching SHA-256 checksums
for 208 processed CSVs (3,937,499,834 bytes per host) and 297 checkpoint
objects (27,591,393,491 bytes per host). The processed data includes 35 POWDER
`2400_2600` CSVs on each host, with seven receiver files for each of
`20261015T1434Z` and `20261025T1355Z`. The path inside a container is:

```text
/workspace/spectrum-usage/data/powder/<receiver>/<run>/2400_2600/power_1mhz_avg_per_minute.csv
```

Node-affinity tasks verified the band files, source-bucket reads, the location
JSON, and a MinIO object written on one new host and read on all three.
The expansion check records its result in
`/home/cc/spectrum-usage/runs/infra/six-gpu-expansion-acceptance.json`.

## Access and jobs

From your workstation, use distinct local ports if you also have a tunnel to
the multi-host cluster:

```bash
ssh -N -L 8266:localhost:8265 -L 9002:localhost:9001 cc@129.114.25.96
```

Open `http://localhost:8266` for the standalone Ray dashboard and Jobs API.
Open `http://localhost:9002` for its MinIO console.

On the standalone host, submit jobs through its head:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-head)
docker exec "$head" ray status
docker exec "$head" ray job submit --address http://127.0.0.1:8265 -- \
  python -m training.ray.cluster_probe --check-model-imports \
  --check-minio-cross-worker --check-spectrum-read
```

Before expansion, `standalone_probe` passed as job `standalone-acceptance-v1`.
That single-host probe expects just one GPU worker. It checked four concurrent
one-GPU actors against the host GPU UUIDs, a single task using all four GPUs,
eight half-GPU actors, and sixteen quarter-GPU actors. It also checked CUDA
computation, PyTorch shared-memory allocation, model imports, the 256 GiB
mount, shared files written from the head and worker, source-bucket reads,
and MinIO write/read/delete across the GPU actors. The host retains the full
log at `/home/cc/standalone-acceptance.log` and the result at
`/home/cc/standalone-acceptance.json`.

## Redeploy

On `129.114.25.96`:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE=spectrum-ray:2.54.0-standalone-921290800551
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/standalone.yaml standalone
```

Changing the image, entrypoint, or shared-memory settings replaces the affected
containers. A source-only sync can serve future job submissions without
restarting services.

To deploy or update only the one-GPU service on the manager:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE=spectrum-ray:2.54.0-standalone-921290800551
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/standalone-workers.yaml standalone
```

Keep `--prune` off both deployment commands: each manifest omits services
that the other command manages. During the November 3 expansion, we used a
temporary secret-bearing preparation service for downloads and GPU checks,
then removed it and applied `ray.single_gpu=true` to the new nodes. The
original head, four-GPU worker, and MinIO container IDs stayed unchanged.
