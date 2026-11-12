# Six-host H100 cluster

The head's lease ends **2026-11-14 01:49 UTC**. Finish jobs and copy their outputs to
durable storage before that deadline.

Use stack `final` on its own Swarm. `ray-final-1` runs the manager, Ray head,
MinIO, and one GPU worker. Each of the six hosts has one NVIDIA H100 with
95,830 MiB reported VRAM, 48 logical CPUs, and about 240 GiB RAM. Ray advertises
six GPUs and 288 CPUs across seven logical nodes, including the separate head.

| Host | Private IP | Shared network IP | Floating IP |
|---|---|---|---|
| `ray-final-1` | `192.168.1.226` | `10.56.2.238` | `129.114.25.135` |
| `ray-final-2` | `192.168.1.122` | `10.56.3.185` | none |
| `ray-final-3` | `192.168.1.47` | `10.56.0.249` | none |
| `ray-final-4` | `192.168.1.15` | `10.56.0.242` | none |
| `ray-final-5` | `192.168.1.45` | `10.56.2.218` | none |
| `ray-final-6` | `192.168.1.9` | `10.56.1.209` | none |

## Access

Run these commands from the repository root with the Chameleon key in your
local SSH agent. `final-ssh.conf` enables agent forwarding and uses
`ray-final-1` as the jump host for the other nodes:

```bash
ssh -F training/ray/docker/final-ssh.conf ray-final-1
ssh -F training/ray/docker/final-ssh.conf ray-final-6
```

The ignored `training/ray/docker/secrets/final_known_hosts` file holds this
cluster's public host keys. Keep your private key on your workstation.

Open the dashboard at `http://localhost:8269` and the MinIO console at
`http://localhost:9005` through this tunnel:

```bash
ssh -F training/ray/docker/final-ssh.conf -N \
  -L 8269:localhost:8265 -L 9005:localhost:9001 ray-final-1
```

## Runtime and storage

Each host runs Ubuntu 24.04, Docker, and NVIDIA Container Toolkit. Docker uses
the NVIDIA runtime by default. Swarm uses `192.168.1.226` for the manager and
`10.243.0.0/16` for its overlay pool. Hosts had no active firewalld during setup.

We deploy `stack.yaml` with the resource settings in `h100.yaml` and the
expansion service in `final-extra.yaml`. Each GPU
worker has 64 GiB `/dev/shm` and a 16 GiB Ray object store. The separate head
advertises zero CPUs/GPUs, with 1 GiB logical RAM, a 1 GiB object store, and
2 GiB `/dev/shm`.

We build the image on `ray-final-1` with `Dockerfile.core`, BuildKit, and the
PyTorch 2.5.1/CUDA 12.4 base, then distribute that image over the private network
with `docker image save/load`. Ray uses version 2.54.0. The head records the
deployed tag in `/home/cc/final-image-tag`.

The deployed tag is `spectrum-ray:2.54.0-final-64bd487e533c`. We verified image
ID `sha256:64bd487e533c42ad11d89d959fcf5c5bcffffc8c8373211944559b17c2105ee5`
on all six hosts.

`final_ray-worker` runs on hosts 1–4, labeled `ray.gpu=true`.
`final_ray-worker-extra` runs on hosts 5–6, labeled `ray.final-extra=true`.
Keep those roles separate: adding `ray.gpu=true` to an expansion host would
start a second worker on its sole GPU.

We added the expansion service while the original four GPUs were busy. It
uses the same image and storage secrets, with host mounts for isolated Python
libraries at `/home/cc/final-runtime/pylibs` and an on-demand model cache at
`/home/cc/final-runtime/hf-cache`. The corresponding container paths are
`/workspace/pylibs` and `/workspace/hf-cache`. We initialized the cache empty;
running jobs can download the backbone they need. We copied the existing
Transformers 4.44.2 libraries before starting the workers.

For a deliberate redeploy on the manager:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE="$(</home/cc/final-image-tag)"
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/h100.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/final-extra.yaml final
```

The Swarm holds its own storage secrets, seeded from local `evaluation/.env`.
Use `spectrum_s3_client()` and `SPECTRUM_*` variables for durable source bucket
`spectrum` at `https://chi.tacc.chameleoncloud.org:7480`. Default `AWS_*`
variables select this cluster's MinIO at `http://minio:9000`, bucket
`spectrum-usage`, with volume `final_minio-data`.

`final_minio-init` creates the bucket and exits with code zero. Its completed
`0/1` service count is normal; the head, six workers, and MinIO remain running.

## Code and assets

Each host keeps a source snapshot at `/home/cc/spectrum-usage`; containers mount
it at `/workspace/spectrum-usage`. Read `/home/cc/spectrum-source-revision`
for the commit. Use the workstation helper to update this cluster:

```bash
python3 training/ray/scripts/sync-cluster-code.py --cluster final --revision HEAD
```

We seeded the hosts from local committed revision `62f3a09`, plus the pending
deployment helpers and cluster instructions. Read the current host marker
before selecting a revision for later updates.

The helper's default selection includes `final`, `tacc`, and `uc`. Select
`mixed-gpu` explicitly to address the older H100 inventory, whose four
`ray-new-*` leases expired on November 9 at 13:40 UTC.

Select a revision that includes the current training changes. The local branch
can contain unpublished commits. Source sync checks file hashes and container
identities without restarting services. Remote hosts hold snapshots, so use
the helper rather than `git pull`.

Sync processed measurement CSVs and checkpoints with the cluster inventory:

```bash
RAY_INVENTORY=training/ray/docker/final-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
RAY_INVENTORY=training/ray/docker/final-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

The inventory scopes CSV discovery to `aerpaw`, `ara`, `cosmos`, and `powder`.
Use `--site powder` to narrow the selection or `--dry-run` to inspect it.
The helper compares full-file SHA-256 checksums across the four hosts. If a
source CSV changes during transfer, rerun with a fresh inventory.

For hosts 1–4, `--checkpoints` selects `checkpoints/1d/`, `checkpoints/2d/`,
`checkpoints/4d/`, and `checkpoints/MANIFEST.tsv`: **68 files, 173,142,276 bytes**
per host. The inventory sets `RAY_CHECKPOINT_PREFIXES` to enforce that scope.
We removed reproduction files downloaded during setup at the user's request.
Fetch specific `checkpoints/repro/` objects from durable storage when a job
needs them.

**Hosts 5–6 keep code and datasets only, with no local `checkpoints/` copy.**
The user requested this scope during expansion. The inventory sets
`RAY_CHECKPOINT_SKIP_HOSTS=(ray-final-5 ray-final-6)`, so the checkpoint command
above targets hosts 1–4. CSV sync still includes all six hosts. Do not bulk-copy
checkpoints to the expansion workers.

Prepare ElectroSense `spec4`, `spec25`, and `spec151` caches with
`training.ray.electrosense.ensure_dataset`, using `spectrum_s3_client()` for
the raw source objects. Each dataset has `series.csv` and `stats.json` under
`data/electrosense/`.

## Submit and monitor jobs

On the manager:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=final_ray-head)
docker exec "$head" ray status
docker exec "$head" ray job list
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id final-probe-$(date +%Y%m%d-%H%M%S) -- \
  python -m training.ray.cluster_probe --fractional-gpu-check 0.25 \
  --check-model-imports --check-minio-cross-worker --check-spectrum-read
```

Use distinct submission IDs and output prefixes. Copy results to durable
storage before the lease ends; MinIO on this head shares the instance lease.

## Verified setup

For the initial four-host setup on November 9, 2026, we verified 695 source and deployment files per host
against revision `62f3a09` plus the pending management changes. Each host has:

| Selection | Files | Bytes per host |
|---|---:|---:|
| Processed measurement CSVs | 211 | 4,197,938,335 |
| Core checkpoint objects | 68 | 173,142,276 |
| ElectroSense caches | 6 across 3 datasets | 2,485,799 |

We compared full-file SHA-256 checksums across the four hosts. Each host has
35 POWDER `2400_2600` CSVs and matching location JSON.

Jobs API submission **`final-acceptance-20261109-v2` succeeded**. We checked
the four expected GPU UUIDs, model imports, 16 quarter-GPU tasks, and source
storage reads from each worker. A follow-up in the same job performed CUDA
matrix multiplication and PyTorch shared-memory allocation on each GPU,
checked compute capability 9.0 and 64 GiB `/dev/shm`, and compared asset
manifests and location files. One worker wrote a 1 MiB MinIO object; all four
read matching contents before we deleted the test object.

The manager keeps the log at `/home/cc/final-acceptance.log` and the result at
`/home/cc/spectrum-usage/runs/infra/final-acceptance.json`. The first submission
failed before GPU checks because the standalone probe script lacked the
repository import path. We supplied `PYTHONPATH=/workspace/spectrum-usage` in
the successful job's runtime environment. Each host had about 106–108 GiB
free after checkpoint cleanup.

### November 10 expansion

We joined hosts 5–6 to the existing Swarm and reused the image above. We copied
the head's current source snapshot rather than the original November 9
revision, then copied its measurement data, ElectroSense caches, and prepared
15-minute POWDER data. The snapshot marker at seeding was `3d77d4f`.

Before joining Ray, both new GPUs passed CUDA matrix multiplication and
PyTorch shared-memory checks. We compared all 211 processed CSV hashes with
the saved manifest and checked the isolated Transformers import. At the
user's request, we removed the 68 staged core checkpoints and the copied model
cache before starting these workers.

Jobs API submission **`final-expansion-acceptance-20261110` succeeded**. We
verified six expected GPU UUIDs, 288 CPUs, matching CSV/ElectroSense manifests,
prepared POWDER data, and source reads on all six workers. Worker 6 wrote a
1 MiB MinIO object, and all six workers read matching contents. The check
confirmed that hosts 5–6 have no checkpoint directory. We compared container
IDs and creation times before and after expansion: the head, MinIO, and all
four original workers remained unchanged.

The manager keeps `/home/cc/final-expansion-acceptance.log` and
`/home/cc/spectrum-usage/runs/infra/final-expansion-acceptance-20261110.json`.

### One-off container validation and future preparation

We validated both expansion hosts with `docker run --rm -i --gpus all`, using
the deployed image, the same code/library/cache mounts as `final-extra.yaml`,
and a 64 GiB `/dev/shm`. These checks ran outside Ray. Both hosts passed:

- Core model imports and Transformers 4.44.2 imports from the host-mounted
  `/home/cc/final-runtime/pylibs` directory.
- A small CUDA matrix multiplication and a bf16 matrix multiplication. These
  hosts have H100s and report bf16 support; V100s need a precision choice that
  fits their capabilities.
- NumPy **2.1.3** and PyTorch **2.5.1+cu124**, matching the existing workers.
- Presence and expected sizes of all 211 measurement CSVs, with no local
  checkpoint directory.
- Loading GPT-2 (`openai-community/gpt2`) and BERT
  (`google-bert/bert-base-uncased`) from the host-mounted model cache.
- An object listing in the durable `spectrum` bucket.

For the one-off S3 test, we read source credentials from an existing service's
secrets and passed them to the validation container through stdin. We kept
them out of command arguments, files, and logs. Standalone containers cannot
mount Swarm secrets. The manager keeps the credential-free result at
`/home/cc/spectrum-usage/runs/infra/final-oneoff-validation-20261110.json`.

For future hosts, finish preparation before adding a worker-placement label:
join Swarm without that label, seed the code and data directories, prepare
the isolated libraries and requested model caches in their host directories,
then validate with a one-off container using the deployed image and mounts.
Use that image's Python interpreter when installing packages into the host
directory so compiled dependencies match the worker's Python ABI. Add the
placement label for the intended service after validation, then check Ray
membership and run a smoke test.

**Keep the current hosts 5–6 on `ray.final-extra=true`, with `ray.gpu` unset.**
The user confirmed that we should retain their working extra-worker service.
Adding `ray.gpu=true` would admit the original service to those hosts and could
duplicate workers. We made no label or placement changes during validation.
