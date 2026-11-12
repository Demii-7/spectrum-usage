# UC cluster: eight RTX hosts and one four-GPU V100 host

Stack `uc` runs in its own Swarm, independent of the H100 and TACC clusters.
The head is `ray-uc-1`, reached at `cc@192.5.87.177`. The original eight hosts
each run one GPU worker, including the head host. Each has one Quadro RTX 6000
with 24 GiB VRAM, 48 logical CPUs, and about 187 GiB host RAM. The expansion
host `ray-v100` runs one worker exposing four V100 32 GiB GPUs, with 80 CPUs
and about 125 GiB RAM. Ray advertises **12 GPUs and 464 CPUs** across ten
logical Ray nodes: one head plus nine workers on nine physical hosts.

| Hostname | Private IP | Public IP |
|---|---|---|
| `ray-uc-1` | `10.140.81.200` | `192.5.87.177` |
| `ray-uc-2` | `10.140.82.107` | none |
| `ray-uc-3` | `10.140.82.132` | none |
| `ray-uc-4` | `10.140.82.89` | none |
| `ray-uc-5` | `10.140.82.191` | none |
| `ray-uc-6` | `10.140.83.1` | none |
| `ray-uc-7` | `10.140.81.125` | none |
| `ray-uc-8` | `10.140.83.84` | none |
| `ray-v100` (`ray-uc-extra-1` SSH alias) | `10.140.82.90` | `192.5.87.163` |

## SSH and dashboard

Run client commands from the repository root with the Chameleon key loaded
in the local SSH agent. `uc-ssh.conf` enables agent forwarding and reaches
private-only nodes through `ray-uc-1`:

```bash
ssh -F training/ray/docker/uc-ssh.conf ray-uc-1
ssh -F training/ray/docker/uc-ssh.conf ray-uc-8
ssh -F training/ray/docker/uc-ssh.conf ray-uc-extra-1
```

The scoped public host-key record is the ignored file
`training/ray/docker/secrets/uc_known_hosts`. Private keys remain on the
workstation.

Use local port 8268 for this cluster's dashboard and Jobs API:

```bash
ssh -F training/ray/docker/uc-ssh.conf -N \
  -L 8268:localhost:8265 -L 9004:localhost:9001 ray-uc-1
```

Open `http://localhost:8268` for Ray and `http://localhost:9004` for the MinIO
console.

## Runtime

The hosts run Ubuntu 24.04, Docker, and NVIDIA Container Toolkit with NVIDIA
as Docker's default runtime. The manager advertises `10.140.81.200` for
Swarm control/data traffic and uses overlay pool `10.242.0.0/16`.
`configure-peer-firewall.sh` added the eight exact private peer addresses to
firewalld's runtime and permanent trusted-source rules. Public firewall rules
were left unchanged.

During expansion, we added the V100 host's exact private address to the eight
existing hosts' trusted-source rules and trusted those peers on the new host.

The deployment combines `stack.yaml`, `uc.yaml`, and `uc-extra.yaml`.
Each original `ray.gpu=true` host
runs one `uc_ray-worker`, with one GPU, a 64 GiB `/dev/shm` tmpfs, and a 16 GiB
Ray object store. The separate head advertises zero CPUs/GPUs, with a 1 GiB
logical memory resource, a 1 GiB object store, and 2 GiB `/dev/shm`.

The V100 host has **only `ray.uc-extra=true`, with `ray.gpu` unset**.
`uc_ray-worker-extra` uses that separate placement label and
`RAY_NUM_GPUS=4`, so one worker exposes all four GPUs. It uses the same image,
64 GiB `/dev/shm`, and a 16 GiB object store. Keep the labels separate to avoid
placing the original one-GPU worker service on this host as well.

The extra worker mounts `/home/cc/uc-runtime/pylibs` at `/workspace/pylibs`
and `/home/cc/uc-runtime/hf-cache` at `/workspace/hf-cache`. Its isolated
Transformers version is 4.44.2; the cache includes GPT-2 and BERT. NumPy 2.1.3
and PyTorch 2.5.1+cu124 match the original UC workers. V100s use compute
capability 7.0 and support fp32/fp16, **not native bf16**.

The Quadro GPUs use Turing architecture (compute capability 7.5). They have
24 GiB each, rather than the H100/A100 capacities in the other clusters.
Fractional GPU requests control scheduling, not VRAM partitioning; select
model precision, batch size, and concurrency for this hardware.

We built the Ray image on this cluster's head using `Dockerfile.core`,
BuildKit, and the pinned PyTorch 2.5.1/CUDA 12.4 base. The image contains
Ray 2.54.0. The manager records its immutable tag in `/home/cc/uc-image-tag`.
The exact image is distributed with `docker image save/load` over the private
network. The existing clusters' images and services are not involved.

The deployed tag is `spectrum-ray:2.54.0-uc-bb30223667a7`, image ID
`sha256:bb30223667a7406e9d36e45cdbde0eac93a04c39136aff534cd4f37706e49098`.
All nine hosts have that image ID.

Redeploy on `ray-uc-1` when an image or configuration update is intended:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE="$(</home/cc/uc-image-tag)"
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/uc.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/uc-extra.yaml uc
```

## Source and assets

The initial source snapshot is local committed revision `7550f8d`, including
the unpublished training/recovery updates, plus the UC deployment and
management files. The hosts record the selected commit in
`/home/cc/spectrum-source-revision`. The source tree is a snapshot, not a Git
clone.

A concurrent source sync advanced the hosts to local committed revision
`dd34a80` during setup. The final check verified 668 files per host against
that revision plus the pending UC/management changes, without reverting the
newer TimeRAN and rescoring updates.

Each host has `/home/cc/spectrum-usage`, mounted at
`/workspace/spectrum-usage` inside the Ray containers. Each physical host has
its own local `data/` copy. The original eight hosts also have local
`checkpoints/` copies; the expansion host keeps code and datasets only. The head and worker on node 1
share that host's copy. `data/locations/`, including `powder.json`, comes from
the source snapshot.

To sync only this cluster's code from the workstation:

```bash
python3 training/ray/scripts/sync-cluster-code.py --cluster uc --revision HEAD
```

Choose the revision on purpose: the local branch includes unpublished
commits. Do not downgrade to an older remote tip or push other agents' commits
without the user's request. Source sync does not restart services.

To sync processed CSVs or checkpoint objects:

```bash
RAY_INVENTORY=training/ray/docker/uc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
RAY_INVENTORY=training/ray/docker/uc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

Add `--site powder` for only POWDER or `--dry-run` to preview. The processed
selection includes `600_800` and `2400_2600` CSVs and skips raw capture
subtrees. The wrapper shares one inventory and compares full-file SHA-256
checksums across all selected hosts. If a source CSV changes during a run,
refresh the inventory and retry.

The inventory sets `RAY_CHECKPOINT_SKIP_HOSTS=(ray-uc-extra-1)`. Checkpoint
sync excludes the expansion host; CSV sync includes all nine hosts. Do not
bulk-copy the checkpoint archive to `ray-v100`.

The UC inventory sets `RAY_DATA_SITES=(aerpaw ara cosmos powder)`, which uses
the sync tool's `--sites` option to avoid traversing unrelated experiment,
backup, and checkpoint prefixes during CSV discovery. An explicit
`--site powder` overrides that default selection.

ElectroSense benchmark data uses the existing
`training.ray.electrosense.ensure_dataset` function to prepare `spec4`,
`spec25`, and `spec151` under `data/electrosense/`. It creates `series.csv`
and `stats.json` per dataset; benchmark runners also call it when the cache
is missing. UC setup prepares these caches before handoff and compares their
checksums across all eight hosts.

The November 9, 2026 UTC handoff verified matching manifests on all eight
hosts for:

| Selection | Files/datasets | Bytes per host |
|---|---:|---:|
| Processed measurement CSVs | 211 files | 4,176,700,645 |
| Checkpoint objects | 510 files | 44,599,296,288 |
| ElectroSense caches | 3 datasets, 6 files | 2,485,799 |

Each host has 35 POWDER `2400_2600` CSVs, including seven receiver files for
each of `20261015T1434Z` and `20261025T1355Z`. The final CSV refresh ran after
the larger checkpoint transfer to pick up intervening source updates.

## Storage and jobs

This Swarm has its own spectrum-source secrets, populated from local
`evaluation/.env`. Use `spectrum_s3_client()` and the `SPECTRUM_*` variables
for durable bucket `spectrum` at `https://chi.tacc.chameleoncloud.org:7480`.
The default `AWS_*` environment selects this cluster's fresh MinIO at
`http://minio:9000`, bucket `spectrum-usage`. Its generated credentials and
Docker volume `uc_minio-data` are independent of other clusters. No old
campaign state or training jobs are restored automatically.

On the manager:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=uc_ray-head)
docker exec "$head" ray status
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id uc-probe-$(date +%Y%m%d-%H%M%S) -- \
  python -m training.ray.cluster_probe --fractional-gpu-check 0.25 \
  --check-model-imports --check-minio-cross-worker --check-spectrum-read
```

Use `ray job list`, `ray job logs`, and `ray job status` through this head
container. Use distinct submission IDs and output directories. Keep durable
results in the source object store or back up this MinIO with a UC-specific
backup prefix; do not overwrite another cluster's backup namespace.

## Acceptance records

Jobs API submission `uc-acceptance-20261109` succeeded with all eight expected
GPU UUIDs, core model imports, 32 quarter-GPU tasks, eight source-store reads,
and a cross-worker MinIO round trip. The manager stores the log and result at
`/home/cc/uc-acceptance.log` and `/home/cc/uc-acceptance.json`.

A node-affinity follow-up performed CUDA matrix multiplication and PyTorch
shared-memory allocation on every Quadro, checked compute capability 7.5 and
the 64 GiB `/dev/shm` mounts, verified the saved asset manifests and location
JSON across all nodes, and wrote a 1 MiB MinIO object on worker 8 for all
eight workers to read. The result is
`/home/cc/spectrum-usage/runs/infra/uc-assets-gpu-acceptance.json` on the manager.

### November 10 V100 expansion

We prepared `ray-v100` before joining Swarm. We reused the exact UC image and
copied the head's source snapshot (marker `24a4cb5` at seeding), data,
ElectroSense/POWDER caches, and isolated libraries. A one-off container using
the image and host mounts with `--gpus all` then synced measurement data using
the UC inventory's `RAY_DATA_SITES` selection. It verified full-file hashes
for **211 CSVs, 4,245,388,258 bytes**, and prepared the dataset caches. We
excluded the checkpoint archive.

The one-off checks passed imports, matching NumPy/PyTorch versions, fp32/fp16
matrix multiplication on all four V100s, shared memory, GPT-2/BERT loading,
and a durable spectrum listing. All four GPUs reported compute capability
7.0 with no native bf16 support. For this pre-Swarm S3 test, we passed source
credentials through stdin from an existing UC service's secrets, keeping
credential values out of files, command arguments, and logs.

After validation, we joined the host without labels, added only
`ray.uc-extra=true`, and created `uc_ray-worker-extra` with four GPUs and the
existing Swarm secrets. The original head, MinIO, and eight worker containers
kept their IDs and creation times throughout expansion.

Jobs API submission **`uc-v100-acceptance-20261110` succeeded**. We verified
nine GPU workers, 12 GPUs, and 464 CPUs; the new worker reports four GPUs.
Source listings succeeded on all nine workers. The V100 worker wrote a 1 MiB
MinIO object, and all nine workers read matching contents before deletion.
The check also confirmed the absence of a local checkpoint directory on the
new host.

The manager keeps the one-off result at
`/home/cc/spectrum-usage/runs/infra/uc-extra-oneoff-validation-20261110.json`,
the distributed result at
`/home/cc/spectrum-usage/runs/infra/uc-v100-acceptance-20261110.json`, and the
job log at `/home/cc/uc-v100-acceptance.log`.
