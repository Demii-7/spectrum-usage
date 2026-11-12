# Three-GPU H100 allocation on h4-lm

Stack `h100three` is an independent Swarm on `cc@129.114.109.32`, private
`10.52.2.210`, hostname `h4-lm`. The host runs Ubuntu 22.04, Docker CE 29.9.3,
and NVIDIA Container Toolkit. It has four H100 GPUs reporting 95,830 MiB each,
192 logical CPUs (96 cores with two SMT threads), and about 1 TiB memory.

## Reserved resources

**Physical GPU 0 belongs to the collaborator. Use only physical GPUs 1–3.**
The worker's `NVIDIA_VISIBLE_DEVICES` contains only the three allowed UUIDs:

| Physical GPU | UUID | Allocation |
|---|---|---|
| 0 | `GPU-209249b1-08e8-9f45-3957-1b919bcded1a` | Collaborator; excluded |
| 1 | `GPU-4568a833-3e66-8937-71fb-a8ded13589fd` | Ray |
| 2 | `GPU-77efff73-64f0-41fb-0c7f-12818d7fcc5c` | Ray |
| 3 | `GPU-6f62c84a-184f-e2fe-8140-539d2e6164b8` | Ray |

CUDA numbers the allowed GPUs 0–2 inside the worker; compare UUIDs to identify
the physical devices. We verified that `/dev/nvidia0` is absent from the
worker and from the one-off preparation container.

The allocation uses **144 of 192 logical CPUs**, on CPU IDs
**`0-71,96-167`**. IDs `72-95,168-191` remain outside the allocation. This
reserves 24 complete cores, including both SMT threads, rather than sharing
physical cores with the reserved quarter.

The hard service CPU quotas are:

| Service | CPU quota | Ray-advertised CPUs |
|---|---:|---:|
| `h100three_ray-worker` | 140 | 140 |
| `h100three_ray-head` | 2 | 0 |
| `h100three_minio` | 2 | n/a |

The steady-state total is 144. The 2-CPU MinIO initialization task completes
before we enable the GPU worker.

Swarm service specs support CPU quotas but no cpuset field. The enabled
`h100-three-cpuset.service` runs the stack-scoped guard
`training/ray/scripts/h100-three-cpuset.py`, which applies the allowed cpuset
to containers labeled `com.docker.stack.namespace=h100three`, including
replacement tasks. It does not modify other users' containers or the Docker
daemon's default cgroup parent. The Ray entrypoint waits for the exact affinity
before starting Ray. The kernel rejects attempts by a Ray task to select a
CPU outside its cgroup cpuset.

Keep the guard enabled and include `h100-three.yaml` on deployment. Deploying
`stack.yaml` alone would lose the GPU/CPU restrictions.

## Access

From the repository root, with the key in your SSH agent:

```bash
ssh -F training/ray/docker/h100-three-ssh.conf ray-h100-three-1
```

The config uses ignored `docker/secrets/h100_three_known_hosts`. This public
IP previously belonged to the TACC SSH gateway; its old inventory is now
historical and excluded from default source syncs.

Dashboard `http://localhost:8271`, MinIO console `http://localhost:9007`:

```bash
ssh -F training/ray/docker/h100-three-ssh.conf -N \
  -L 8271:localhost:8265 -L 9007:localhost:9001 ray-h100-three-1
```

The enabled `h100-three-ports.service` restricts this stack's Ray/MinIO ports
(8265, 10001, 9000, 9001) to SSH tunnels/private sources. It leaves other
published ports under their existing policy for the collaborator.

## Runtime and deployment

The immutable image is `spectrum-ray:2.54.0-h100-three-64bd487e533c`, ID
`sha256:64bd487e533c42ad11d89d959fcf5c5bcffffc8c8373211944559b17c2105ee5`.
We copied the tested x86 image from the final cluster without changing its
services. It contains Python 3.11, PyTorch 2.5.1/CUDA 12.4, Ray 2.54.0, and
NumPy 2.1.3. The host already had Docker CE and containerd.io; we retained those
packages and configured the installed NVIDIA runtime as Docker's default.

Source lives at `/home/cc/spectrum-usage`, mounted at
`/workspace/spectrum-usage`. The worker also mounts its own host directories
`/home/cc/h100-three-runtime/{pylibs,hf-cache,pretrained}` at
`/workspace/{pylibs,hf-cache,pretrained}`. It has 64 GiB `/dev/shm` and a
16 GiB Ray object store; the separate head has 2 GiB `/dev/shm` and a 1 GiB
object store.

The Swarm advertises `10.52.2.210`, uses overlay pool `10.245.0.0/16`, and
holds its own source and MinIO secrets. Source reads use `spectrum_s3_client()`
and `SPECTRUM_*`; default `AWS_*` selects this cluster's MinIO bucket
`spectrum-usage`. The `h100three_minio-data` volume is independent of the
other clusters. The user has not supplied a lease end.

For a deliberate redeploy:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE="$(</home/cc/h100-three-image-tag)"
sudo systemctl is-active h100-three-cpuset.service
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/h100-three.yaml h100three
```

The custom Ray entrypoint explicitly starts the worker with three GPUs and
140 CPUs. The head advertises zero CPUs/GPUs. Treat GPU UUIDs and the CPU
allocation as host-specific: inspect and adapt them before using this overlay
on different hardware.

## Source, data, and preparation

```bash
python3 training/ray/scripts/sync-cluster-code.py --cluster h100-three --revision HEAD
RAY_INVENTORY=training/ray/docker/h100-three-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
```

The host keeps code and datasets, excluding the bulk checkpoint archive.
`ncar_prep.setup_on_node` prepared native Transformers 4.44.2, momentfm 0.1.4,
GPT-2/BERT/MOMENT caches, and the selected TimeRAN pretrained weights in the
host-mounted runtime directories. We constrained NumPy to 2.1.3 and fsspec to
2025.7.0. Its fresh offline process loaded all requested backbones. The
TimeRAN setup loaded 78 tensors with no unexpected tensors.

Preparation ran before worker admission in a one-off container with only the
three allowed GPU UUIDs, `--cpuset-cpus 0-71,96-167`, and `--cpus 140`. CUDA
and bf16 matrix multiplication passed on all three GPUs, with GPU 0 absent.
We synced and SHA-256 checked 211 measurement CSVs (4,246,783,056 bytes),
prepared ElectroSense caches and 81 transfer/POWDER conversion files, and
verified source S3 access. Source credentials traveled through stdin rather
than files, logs, or command arguments.

## Acceptance, November 12, 2026

Jobs API submission **`h100-three-acceptance-20261112` succeeded**. It checked:

- Exactly three GPUs and 140 CPUs in Ray, with the allowed GPU UUIDs and no
  exposure of physical GPU 0.
- The 144-CPU affinity and a rejected attempt to select reserved CPU 72.
- Twelve quarter-GPU tasks distributed across all three allowed GPUs.
- Offline model loads, source storage access, and a 1 MiB MinIO roundtrip
  between the logical head and worker.

We inspected the running Docker containers after the cpuset guard updated
them: CPU quotas remained 140 + 2 + 2, totaling 144. The manager keeps
`runs/infra/h100-three-preparation.json` and
`runs/infra/h100-three-acceptance.json` under the source directory, plus the
log `/home/cc/h100-three-acceptance.log`.
