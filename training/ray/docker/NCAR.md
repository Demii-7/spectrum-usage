# NCAR mixed-architecture cluster

Stack `ncar` has its own Swarm and MinIO store. The first host runs a separate
Ray head and one GPU worker; the second runs another GPU worker. Two x86 hosts
run CPU-only workers. Ray advertises **2 GH200 GPUs and 192 CPUs** across five
logical nodes on four hosts (144 ARM CPUs plus 48 x86 CPUs).

| Host | Public IP | Private IP | Hardware |
|---|---|---|---|
| `ray-ncar-1` | `128.117.250.76` | `172.20.0.252` | 72 ARM CPUs, one GH200 GPU |
| `ray-ncar-2` | none (through head) | `172.20.0.132` | 72 ARM CPUs, one GH200 GPU |
| `ray-ncar-cpu-3` | none (through head) | `172.20.0.136` | 16 online x86 CPUs, about 60 GiB RAM |
| `ray-ncar-cpu-4` | none (through head) | `172.20.0.196` | 32 online x86 CPUs, about 60 GiB RAM |

The host runs Ubuntu 24.04 on aarch64. Linux reports about 565 GiB host memory;
`nvidia-smi` reports 97,871 MiB GPU memory. The GPU model string is
`NVIDIA GH200 480GB`. We use the root filesystem for the deployment and leave
the separate unmounted disk untouched. The user has not supplied a lease end.

## Access

From the repository root, with your Chameleon key in the SSH agent:

```bash
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-1
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-2
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-cpu-3
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-cpu-4
```

The config enables agent forwarding and uses the ignored public host-key
record `training/ray/docker/secrets/ncar_known_hosts`.

Open Ray at `http://localhost:8270` and the MinIO console at
`http://localhost:9006` through this tunnel:

```bash
ssh -F training/ray/docker/ncar-ssh.conf -N \
  -L 8270:localhost:8265 -L 9006:localhost:9001 ray-ncar-1
```

## Native ARM runtime

Build `Dockerfile.ncar` on the ARM host. The x86 clusters' images and isolated
Python libraries contain native x86 binaries and cannot serve this worker.

The NCAR Dockerfile uses a native ARM/SBSA CUDA 12.6.3 Ubuntu 24.04 base,
Python 3.12, official aarch64 PyTorch 2.6.0/CUDA 12.6 and torchvision 0.21.0
wheels, and the shared core requirements, including Ray 2.54.0 and NumPy 2.1.3.
The other x86 GPU clusters retain PyTorch 2.5.1/CUDA 12.4. The NCAR head and
GPU workers use the same ARM image and Python version. NCAR's x86 CPU workers
use the separate CPU image described below.

```bash
docker build --progress=plain -t spectrum-ray:ncar-build \
  -f training/ray/docker/Dockerfile.ncar .
```

The manager records the deployed immutable image tag in
`/home/cc/ncar-image-tag`. Docker uses the NVIDIA runtime by default. Native
ARM variants exist for the pinned MinIO and MinIO-client images in `stack.yaml`.

The deployed tag is `spectrum-ray:2.54.0-ncar-arm64-e01f1841be61`, image ID
`sha256:e01f1841be6122897586d97e8bd6e935b854a4e80f882089bc5ba0340cd398b1`.

The deployment combines `stack.yaml`, `ncar.yaml`, `ncar-extra.yaml`, and
`ncar-cpu.yaml`.
The head advertises
zero CPUs/GPUs, with 1 GiB logical RAM, a 1 GiB object store, and 2 GiB
`/dev/shm`. Each GPU worker exposes one GPU and 72 CPUs, with a 16 GiB object
store and 64 GiB `/dev/shm`. CPU workers use a smaller 8 GiB object store,
16 GiB `/dev/shm`, and 40 GiB logical memory. The worker mounts are:

| Host path | Container path |
|---|---|
| `/home/cc/spectrum-usage` | `/workspace/spectrum-usage` |
| `/home/cc/ncar-runtime/pylibs` | `/workspace/pylibs` |
| `/home/cc/ncar-runtime/hf-cache` | `/workspace/hf-cache` |
| `/home/cc/ncar-runtime/pretrained` (node 2 and CPU workers) | `/workspace/pretrained` |

Each host owns its runtime directories. `ncar_ray-worker` runs on node 1
(`ray.gpu=true`). `ncar_ray-worker-extra` runs on node 2, labeled
`ray.ncar-extra=true` with `ray.gpu` unset. Keep these labels separate to avoid
starting two workers on node 2. The extra service preserves the TimeRAN
checkpoint across container recreation without changing the busy original
worker's mounts.

Install isolated Transformers 4.44.2 libraries with each node's matching image Python
into the host-mounted `pylibs` directory. Pin NumPy 2.1.3 and fsspec 2025.7.0
during that install to retain the core versions; remove the target directory's
duplicate NumPy package so jobs use the core installation. Cache GPT-2 and BERT
under `hf-cache`. Prepare these paths and validate with a one-off container
before labeling the host for a worker.

For a deliberate redeploy on the manager:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE="$(</home/cc/ncar-image-tag)"
export NCAR_CPU_IMAGE="$(</home/cc/ncar-cpu-image-tag)"
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/ncar.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/ncar-extra.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/ncar-cpu.yaml ncar
```

## Code, data, and storage

The host keeps a source snapshot at `/home/cc/spectrum-usage`, not a Git clone.
Read `/home/cc/spectrum-source-revision` for its commit. From the workstation:

```bash
python3 training/ray/scripts/sync-cluster-code.py --cluster ncar --revision HEAD
RAY_INVENTORY=training/ray/docker/ncar-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
```

NCAR keeps **code and datasets only**, excluding the checkpoint archive.
Its inventory selects processed data from `aerpaw`, `ara`, `cosmos`, and
`powder`, and excludes all NCAR hosts from checkpoint syncs. Prepare ElectroSense
and 15-minute POWDER caches before starting concurrent jobs.

The Swarm holds its own source-storage and MinIO secrets. Use
`spectrum_s3_client()` and `SPECTRUM_*` for durable source bucket `spectrum`.
Default `AWS_*` variables select the NCAR MinIO store at `http://minio:9000`,
bucket `spectrum-usage`, with volume `ncar_minio-data`. The completed
`ncar_minio-init` service creates the bucket and exits; `0/1` is normal.

The manager advertises `172.20.0.252`, with overlay pool `10.244.0.0/16`.
We trusted its exact private address in firewalld's runtime and permanent
configuration. The node has `ray.head=true` and `ray.gpu=true`; we added these
labels after the native one-off preparation passed.

## Jobs

On the manager:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=ncar_ray-head)
docker exec "$head" ray status
docker exec "$head" ray job list
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id ncar-probe-$(date +%Y%m%d-%H%M%S) -- \
  python -m training.ray.cluster_probe --fractional-gpu-check 0.25 \
  --check-model-imports --check-spectrum-read
```

The generic probe's `--check-minio-cross-worker` option now works across the
two GPU workers. Use unique job IDs and output prefixes. Check resource
availability before submitting GPU probes while training jobs are active.

## Verified setup, November 12, 2026

We seeded the code from committed revision `b5d516a`, plus pending NCAR
deployment files. The one-off container ran on aarch64 with Python 3.12.3,
PyTorch 2.6.0+cu126, torchvision 0.21.0, Ray 2.54.0, and NumPy 2.1.3. It passed
CUDA and native bf16 matrix multiplication, shared-memory allocation, and a
DLinear training/backward/optimizer step with an in-memory state checkpoint
roundtrip. We installed the isolated Transformers 4.44.2 packages on ARM and
loaded both GPT-2 and BERT.

The same container synced and hashed **211 measurement CSVs totaling
4,246,783,056 bytes**, prepared all three ElectroSense caches, and generated
12 POWDER 15-minute CSVs for each of `600_800` and `2400_2600`. Source S3
access passed. We supplied credentials through stdin for this pre-Swarm test,
keeping values out of files, command arguments, and logs. No checkpoint
archive exists on this host.

Jobs API submission **`ncar-acceptance-20261112` succeeded** with two logical
Ray nodes, one GPU worker, 72 CPUs, and four quarter-GPU tasks. The probe
checked model imports, the expected GH200 UUID, and source storage access.
The head wrote a 1 MiB MinIO object; both logical Ray nodes read matching
contents, and the worker deleted it. Native worker/version checks passed.

The manager keeps the one-off result at
`/home/cc/spectrum-usage/runs/infra/ncar-oneoff-validation.json`, the Ray result
at `/home/cc/spectrum-usage/runs/infra/ncar-acceptance-20261112.json`, and the
job log at `/home/cc/ncar-acceptance.log`.

## Node 2 preparation and admission, November 12, 2026

We prepared `ray-ncar-2` while it remained outside Swarm, with no worker label.
It has the same image ID as node 1 and its own initially empty `pylibs`,
`hf-cache`, and `pretrained` directories. We installed ARM dependencies on
node 2; we did not reuse another node's Python installation.

In a one-off container on node 2, we synced and hashed all 211 processed CSVs
(4,246,783,056 bytes), prepared ElectroSense caches, and called the conversion
functions in `training.ray.ncar_prep`. All **81 converted transfer/POWDER files**
passed and matched the head's SHA-256 fingerprints.

We then called **`ncar_prep.setup_on_node` on node 2 itself**, rather than
dispatching an unpinned Ray `setup` task that could run on the head. It installed
Transformers 4.44.2 and momentfm 0.1.4 in node 2's `pylibs` and cached GPT-2,
BERT, and MOMENT-1-small in its `hf-cache`. We constrained the install to
NumPy 2.1.3 and fsspec 2025.7.0, and verified that NumPy loads from the core
image rather than a duplicate in `pylibs`.

The setup's fresh offline subprocess loaded both GPT-2 names, both BERT names
and tokenizers, and `AutonLab/MOMENT-1-small`. The TimeRAN setup loaded 78
tensors with no unexpected tensors, and built MOMENT and random controls.
Its requested 151,643,630-byte TimeRAN checkpoint lives at
`/home/cc/ncar-runtime/pretrained/timeran/TimeRAN_small.pth`, mounted as
`/workspace/pretrained/timeran/TimeRAN_small.pth`. This is a selected pretrained
backbone, not the bulk checkpoint archive.

Before admission we checked CUDA/bf16, S3 access, all prepared file hashes,
and the source snapshot against the head (691 source/config/location files).
We refreshed a concurrent source change before proceeding. A second fresh
container with networking disabled then loaded every cached backbone and the
host-persisted TimeRAN checkpoint. Only after these checks passed did we join
Swarm, add `ray.ncar-extra=true`, and create the extra worker service.

Jobs API submission **`ncar-node2-acceptance-20261112` succeeded**. It verified
the two GPU UUIDs, 144 CPUs, source access on both workers, and a 1 MiB MinIO
write/read/delete across the workers. Offline model loading also passed from
the new Ray worker. We compared container IDs/creation times and confirmed
that the original head, worker, and MinIO never restarted.

Both hosts keep `runs/infra/ncar-node2-preparation.json`. The manager also has
`runs/infra/ncar-node2-acceptance-20261112.json` under its source directory and
the log `/home/cc/ncar-node2-acceptance.log`.

## x86 CPU workers 3 and 4, November 12, 2026

We prepared `ray-ncar-cpu-3` (`172.20.0.136`) and `ray-ncar-cpu-4`
(`172.20.0.196`), then admitted them to the existing Swarm. **Node 5 at
`172.20.0.195` is not ready and remains outside the inventory and cluster.**
We made no SSH connection to that address.

Both hosts have EPYC 4545P processors and about 60 GiB usable RAM. Node 3
currently exposes 16 logical CPUs, one thread per core; node 4 exposes 32.
We retained those settings and advertise the actual CPU counts rather than
assuming the catalog's 32 threads on both hosts.

`Dockerfile.ncar-cpu` builds a native x86_64 CPU image from Ubuntu 24.04.
It pins **Python 3.12.3, Ray 2.54.0, cloudpickle 3.1.2, NumPy 2.1.3,
PyTorch 2.6.0+cpu, and torchvision 0.21.0+cpu**. Python, Ray, and cloudpickle
match the ARM nodes; Torch has the same base version with a CPU-only build.
No CUDA runtime or NVIDIA toolkit is installed on the CPU workers.
Do not use `Dockerfile.core` for these workers: its Python 3.11 runtime does
not match NCAR's Python 3.12.3 head.

The deployed CPU tag is `spectrum-ray:2.54.0-ncar-cpu-9ad6644552b2`, image ID
`sha256:9ad6644552b25ec5dce145b9ac44b594f481068341358c85c1b4269297d8a186`.
Both CPU hosts have that exact image. The head records its tag in
`/home/cc/ncar-cpu-image-tag`, separate from the ARM image tag.

The global service **`ncar_ray-worker-cpu`** targets only
**`ray.ncar-cpu=true`**. Both hosts have that label, with neither `ray.gpu`
nor `ray.ncar-extra`. The service sets `RAY_NUM_GPUS=0` and
`NVIDIA_VISIBLE_DEVICES=void`; its Ray resources have no GPU allocation.
Their node-local `pylibs`, `hf-cache`, and `pretrained` mounts use the same
container paths as node 2, with native x86 libraries installed independently.

Before either host joined Swarm, we synced and hashed 211 measurement CSVs
(4,246,783,056 bytes), prepared all three ElectroSense caches and 81 transfer
conversion files, and compared all converted hashes with the head. We also
verified 736 source/config/location files against the head's live snapshot
before admission. CPU DLinear training/backpropagation and an in-memory state
roundtrip passed. Node-local `ncar_prep.setup_on_node` installed Transformers
4.44.2 and momentfm 0.1.4 and cached GPT-2/BERT/MOMENT plus the selected
TimeRAN weights. A fresh network-disabled container on each host loaded all
backbones and the persisted TimeRAN checkpoint. No bulk checkpoint archive
was copied.

Jobs API submission **`ncar-cpu-acceptance-20261112` succeeded**. It verified
192 CPUs and two GPUs, CPU-only resources on nodes 3 and 4, and native
Python/Torch versions. The project baseline calculations matched on ARM and
x86. NumPy arrays, CPU Torch tensors, and Arrow IPC payloads crossed ARM→x86,
x86→ARM, and x86→x86 with matching contents. A CPU worker wrote a 1 MiB MinIO
object; all four workers read matching contents. Source storage and offline
model checks also passed. We preserved the original GPU services' container
IDs and creation times.

The manager keeps `runs/infra/ncar-cpu-preparation-20261112.json` and
`runs/infra/ncar-cpu-acceptance-20261112.json`, plus
`/home/cc/ncar-cpu-acceptance.log`. Each CPU host also keeps its own
`runs/infra/preparation-ray-ncar-cpu-N.json` report under the source directory.
