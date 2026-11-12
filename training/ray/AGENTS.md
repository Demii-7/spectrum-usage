# Ray cluster access for agents

Use this page to reach the active clusters, update their source, sync data, and
submit Ray jobs. Cluster-specific build, redeploy, and expansion details remain
in [FINAL.md](docker/FINAL.md), [H100.md](docker/H100.md),
[TACC.md](docker/TACC.md), [UC.md](docker/UC.md), [NCAR.md](docker/NCAR.md),
and [H100-THREE.md](docker/H100-THREE.md).
`STANDALONE.md` and `CHAMELEON.md` document retired topologies; do not use their
old head addresses or deployment overlays.

## Active clusters

| Cluster | Ray stack | Head host | Workers | Capacity |
|---|---|---|---|---|
| New H100 (`final` sync selector) | `final` | `129.114.25.135` | 6 hosts | 6 H100s, 288 CPUs |
| Older H100 (`mixed-gpu`, explicit sync only) | `standalone` | `129.114.27.86` | 6 configured hosts; 4 leases expired | See lease dates below |
| TACC (historical inventory, explicit only) | `tacc` | `ray-tacc-1` | Gateway IP reassigned | See historical notes below |
| RTX/V100 (`uc` sync selector) | `uc` | `192.5.87.177` | 9 hosts | 8 RTX 6000 + 4 V100, 464 CPUs |
| NCAR (`ncar` sync selector) | `ncar` | `128.117.250.76` | 2 hosts | 2 GH200s, 144 ARM CPUs |
| Three-GPU H100 (`h100-three` sync selector) | `h100three` | `129.114.109.32` | 1 host | 3 H100s; 140 Ray CPUs + 4 control/storage CPUs |

### Three-GPU H100 allocation

`ray-h100-three-1` reaches `cc@129.114.109.32` (`h4-lm`, private `10.52.2.210`).
It has a head and one three-GPU worker in its own `h100three` Swarm.
**Physical GPU 0 belongs to the collaborator: use only physical GPUs 1–3.**
The overlay exposes those GPUs by UUID; CUDA renumbers them 0–2 inside the
worker. Do not replace that restriction with `NVIDIA_VISIBLE_DEVICES=all`.

The total CPU budget is 144/192 logical CPUs: Ray worker 140, head 2, MinIO 2.
All stack containers use cpuset `0-71,96-167`, leaving `72-95,168-191` outside
the allocation. The enabled `h100-three-cpuset.service` applies this affinity
to Swarm replacements; the Ray entrypoint waits for it before starting.
Keep the guard enabled. See `H100-THREE.md` for the UUIDs and verified limits.

```bash
ssh -F training/ray/docker/h100-three-ssh.conf ray-h100-three-1
```

Always include `h100-three.yaml` alongside `stack.yaml` for this deployment.
The worker has its own `pylibs`, `hf-cache`, and `pretrained` host mounts under
`/home/cc/h100-three-runtime`. Data and model preparation passed before
admission; no bulk checkpoint archive is present. The user has not supplied
a lease end.

### NCAR aarch64 cluster

`ray-ncar-1` (`cc@128.117.250.76`, private `172.20.0.252`) runs the head and
one worker in the independent `ncar` Swarm. The host has 72 ARM CPUs and one
GH200 GPU reporting 97,871 MiB. Linux reports about 565 GiB host memory.

```bash
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-1
ssh -F training/ray/docker/ncar-ssh.conf ray-ncar-2
```

Use **native aarch64 binaries**. `Dockerfile.ncar` uses Python 3.12, official
ARM PyTorch 2.6.0/CUDA 12.6 and torchvision 0.21.0 wheels, Ray 2.54.0, and
NumPy 2.1.3. The x86 image and its PyTorch 2.5.1/CUDA 12.4 libraries do not
apply to this host. Build the isolated Transformers 4.44.2 install with the
ARM image's Python; do not copy x86 `pylibs` directories. See `NCAR.md`.

The worker mounts source/data at `/workspace/spectrum-usage`, and
`/home/cc/ncar-runtime/{pylibs,hf-cache}` at `/workspace/{pylibs,hf-cache}`.
It has 64 GiB `/dev/shm` and a 16 GiB object store. NCAR keeps code and data
only, with no checkpoint archive. Its head and worker use the same ARM image.
The user has not supplied a lease end.

Node 2 is `172.20.0.132`, reached through the head with SSH agent forwarding.
It runs `ncar_ray-worker-extra` with **`ray.ncar-extra=true`, and no
`ray.gpu` label**. Do not add the generic GPU label there. Use `ncar-extra.yaml`
alongside `stack.yaml` and `ncar.yaml` for an intended redeploy.

We completed node-local `ncar_prep.setup_on_node` before joining it: isolated
Transformers 4.44.2, momentfm 0.1.4, GPT-2/BERT/MOMENT caches, TimeRAN weights,
and 81 converted data files. Fresh offline processes and a network-disabled
replacement container loaded all backbones before admission. Node 2's
`/home/cc/ncar-runtime/pretrained` also mounts at `/workspace/pretrained`, so
its selected TimeRAN weights survive worker recreation. Its `pylibs` and
`hf-cache` belong to that host. NCAR now advertises 2 GPUs and 144 CPUs.

### New six-host H100 cluster

The lease ends **2026-11-14 01:49 UTC**. `ray-final-1` is the head and SSH
gateway at `129.114.25.135`, private `192.168.1.226`. Each host has one H100
with 95,830 MiB reported VRAM, 48 CPUs, and about 240 GiB RAM. The head also
runs a GPU worker.

| Host | Private IP |
|---|---|
| `ray-final-1` | `192.168.1.226` |
| `ray-final-2` | `192.168.1.122` |
| `ray-final-3` | `192.168.1.47` |
| `ray-final-4` | `192.168.1.15` |
| `ray-final-5` | `192.168.1.45` |
| `ray-final-6` | `192.168.1.9` |

```bash
ssh -F training/ray/docker/final-ssh.conf ray-final-1
ssh -F training/ray/docker/final-ssh.conf ray-final-6
```

The config enables agent forwarding and private-node access through the head.
It keeps this cluster's keys in ignored `docker/secrets/final_known_hosts`.
Services use the `final_` prefix. Each GPU worker has 64 GiB `/dev/shm` and a
16 GiB object store. Use [FINAL.md](docker/FINAL.md) for deployment, job, and
asset commands. Copy outputs to durable storage before the lease deadline.

Hosts 1–4 keep the core checkpoint selection: `1d/`, `2d/`, `4d/`, and
`MANIFEST.tsv` under `checkpoints/` (68 files, 173 MB per host). **Hosts 5–6
keep code and datasets only, with no local checkpoint copy.** The inventory
excludes hosts 5–6 from `--checkpoints` syncs. The user requested both this
exclusion and the exclusion of the bulk `checkpoints/repro/` archive.

The original worker service runs on hosts 1–4 (`ray.gpu=true`). The expansion
service `final_ray-worker-extra` runs on hosts 5–6 (`ray.final-extra=true`),
with the same image, 64 GiB shared memory, and 16 GiB object store. Keep these
labels separate to avoid placing two workers on one GPU. Include
`docker/final-extra.yaml` for an intended redeploy; routine source sync needs
no redeploy. We added the two workers without restarting the original services.

The user confirmed that hosts 5–6 must retain `ray.final-extra=true`, with
`ray.gpu` unset. Do not switch their labels. One-off containers using the same
image and host mounts passed CUDA/bf16, import/version, data, cached GPT-2/BERT,
and source-S3 checks; see `FINAL.md`. For future hosts, prepare and validate
before adding the placement label for the intended worker service.

### Rebuilt H100 cluster

| Hostname | SSH address | Private IP | GPUs | Worker service |
|---|---|---|---:|---|
| **`ray-worker-2` (head)** | `h100-head` | `192.168.1.185` | 1 | `standalone_ray-worker` |
| `ray-worker-1` | `ray-worker-1` | `192.168.1.39` | 1 | `standalone_ray-worker` |
| `ray-new-1` | `ray-new-1` | `192.168.1.57` | 1 | `standalone_ray-worker` |
| `ray-new-2` | `ray-new-2` | `192.168.1.219` | 1 | `standalone_ray-worker` |
| `ray-new-3` | `ray-new-3` | `192.168.1.104` | 1 | `standalone_ray-worker` |
| `ray-new-4` | `ray-new-4` | `192.168.1.82` | 1 | `standalone_ray-worker` |

The old four-GPU head expired. `129.114.25.96` now belongs to `ray-new-1`,
not the head. Use SSH aliases with agent forwarding through the longer-lived
head for private-only hosts:

```bash
ssh -F training/ray/docker/h100-ssh.conf h100-head
ssh -F training/ray/docker/h100-ssh.conf ray-new-3
```

The config uses the ignored `secrets/h100_known_hosts` record and accepts new
keys on first connection. The two old instances (`ray-worker-1` and the head
`ray-worker-2`) end **2026-11-10 18:00 UTC**. The four `ray-new-*` instances
end **2026-11-09 13:40 UTC**. Finish/checkpoint work before draining expiring
workers. See `H100.md` for recovery directories and the MinIO backup restore.

### TACC cluster

**Historical access:** `129.114.109.32` now belongs to `h4-lm`, the independent
three-GPU H100 allocation above. Do not use the old TACC aliases against that
host or include TACC in default source syncs. The addresses below preserve
the former topology; they do not describe the new H100 host.

| Hostname | Private IP | Public IP | GPUs |
|---|---|---|---:|
| `ray-tacc-1` | `10.52.2.186` | `129.114.108.20` (unreachable) | 1 |
| `ray-tacc-2` | `10.52.0.30` | `129.114.109.32` (SSH gateway) | 1 |
| `ray-tacc-3` | `10.52.0.165` | none | 1 |
| `ray-tacc-4` | `10.52.2.197` | none | 1 |

Access nodes 1, 3, and 4 through `ray-tacc-2` using the checked-in SSH config:

```bash
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-3
```

The config enables agent forwarding, so your local `ssh-agent` must hold the
Chameleon key. The scoped known-hosts file is
`training/ray/docker/secrets/tacc_known_hosts`; the config accepts new private
host keys on first connection.

### UC RTX/V100 cluster

`ray-uc-1` is the head and sole SSH gateway at `cc@192.5.87.177`, private
`10.140.81.200` for the private-only RTX nodes. The original eight hosts each
have one Quadro RTX 6000 with 24 GiB VRAM.
The head also runs a GPU worker. Use the configured SSH aliases and agent
forwarding for private-only nodes:

```bash
ssh -F training/ray/docker/uc-ssh.conf ray-uc-1
ssh -F training/ray/docker/uc-ssh.conf ray-uc-8
```

| Host | Private IP |
|---|---|
| `ray-uc-1` | `10.140.81.200` |
| `ray-uc-2` | `10.140.82.107` |
| `ray-uc-3` | `10.140.82.132` |
| `ray-uc-4` | `10.140.82.89` |
| `ray-uc-5` | `10.140.82.191` |
| `ray-uc-6` | `10.140.83.1` |
| `ray-uc-7` | `10.140.81.125` |
| `ray-uc-8` | `10.140.83.84` |
| `ray-v100` (`ray-uc-extra-1` SSH alias) | `10.140.82.90` |

The scoped record is `docker/secrets/uc_known_hosts`. Services are
`uc_ray-head`, `uc_ray-worker` (global on all eight GPU hosts), and `uc_minio`.
Each worker has 64 GiB `/dev/shm` and a 16 GiB Ray object store. These GPUs
are Turing/compute capability 7.5; select precision, batch size, and trial
concurrency for 24 GiB per GPU rather than H100/A100 assumptions.

The expansion host is `cc@192.5.87.163`, reached with:

```bash
ssh -F training/ray/docker/uc-ssh.conf ray-uc-extra-1
```

It runs **one `uc_ray-worker-extra` with `RAY_NUM_GPUS=4`**, exposing four V100
32 GiB GPUs. Its only worker-placement label is `ray.uc-extra=true`; **do not
add `ray.gpu=true`**. UC now has 12 GPUs and 464 CPUs across nine workers.
The V100 host keeps code and datasets only and is excluded from checkpoint
syncs. Its host-mounted libraries/cache are under `/home/cc/uc-runtime/`, with
Transformers 4.44.2 and cached GPT-2/BERT. NumPy 2.1.3 and PyTorch 2.5.1+cu124
match the existing workers. V100s support fp32/fp16, not native bf16.
Use `docker/uc-extra.yaml` alongside `stack.yaml` and `uc.yaml` for an intended
redeploy; routine source sync needs no redeploy.

### Dashboards

**Never expose 8265, 10001, or 9000/9001 to the internet.** The Ray Jobs API
has no authentication, and anyone who can reach 8265 can run code as root in
the head container. On 2026-11-12 an attacker did exactly that on `ray-uc-1`
(cryptominer plus a scanner, which led to an abuse report). Every head with a
floating IP must run `scripts/host-firewall.sh`, installed as
`/usr/local/sbin/spectrum-firewall` with `scripts/spectrum-firewall.service`.
It allows SSH from anywhere and everything from private addresses, and drops
other inbound traffic, including Docker-published ports (through
`DOCKER-USER`). This is installed on every public host: `ray-uc-1`, `ray-uc-extra-1` (the V100 worker has its own floating IP, 192.5.87.163) and `ray-ncar-1`. Check with
`systemctl is-active spectrum-firewall`. Reach the dashboards only through
the tunnels below.

Use different local ports for each cluster:

```bash
# Three-GPU H100: http://localhost:8271, MinIO console http://localhost:9007
ssh -F training/ray/docker/h100-three-ssh.conf -N \
  -L 8271:localhost:8265 -L 9007:localhost:9001 ray-h100-three-1

# NCAR: http://localhost:8270 and MinIO console http://localhost:9006
ssh -F training/ray/docker/ncar-ssh.conf -N \
  -L 8270:localhost:8265 -L 9006:localhost:9001 ray-ncar-1

# New H100: http://localhost:8269 and MinIO console http://localhost:9005
ssh -F training/ray/docker/final-ssh.conf -N \
  -L 8269:localhost:8265 -L 9005:localhost:9001 ray-final-1

# H100: http://localhost:8266 and MinIO console http://localhost:9002
ssh -F training/ray/docker/h100-ssh.conf -N \
  -L 8266:localhost:8265 -L 9002:localhost:9001 h100-head

# TACC: http://localhost:8267 and MinIO console http://localhost:9003
ssh -F training/ray/docker/tacc-ssh.conf -N \
  -L 8267:localhost:8265 -L 9003:localhost:9001 ray-tacc-1

# UC: http://localhost:8268 and MinIO console http://localhost:9004
ssh -F training/ray/docker/uc-ssh.conf -N \
  -L 8268:localhost:8265 -L 9004:localhost:9001 ray-uc-1
```

Find the head container on each manager:

```bash
# New H100
ssh -F training/ray/docker/final-ssh.conf ray-final-1 \
  'docker ps -q --filter label=com.docker.swarm.service.name=final_ray-head'

# Mixed GPU
ssh -F training/ray/docker/h100-ssh.conf h100-head \
  'docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-head'

# TACC
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1 \
  'docker ps -q --filter label=com.docker.swarm.service.name=tacc_ray-head'

# UC
ssh -F training/ray/docker/uc-ssh.conf ray-uc-1 \
  'docker ps -q --filter label=com.docker.swarm.service.name=uc_ray-head'
```

## Sync source code

Run this from the repository root. It fetches `origin` and syncs committed
source to eighteen hosts across `final`, `uc`, `ncar`, and `h100-three`. The
default excludes the older H100 inventory because four of its leases expired
at 2026-11-09 13:40 UTC. It also excludes TACC because its gateway IP now
belongs to the three-GPU allocation:

```bash
python3 training/ray/scripts/sync-cluster-code.py
```

Options:

```bash
python3 training/ray/scripts/sync-cluster-code.py --cluster h100-three --revision HEAD
python3 training/ray/scripts/sync-cluster-code.py --cluster ncar --revision HEAD
python3 training/ray/scripts/sync-cluster-code.py --cluster final --revision HEAD
python3 training/ray/scripts/sync-cluster-code.py --cluster mixed-gpu
python3 training/ray/scripts/sync-cluster-code.py --cluster tacc
python3 training/ray/scripts/sync-cluster-code.py --cluster uc
python3 training/ray/scripts/sync-cluster-code.py --revision origin/integrate
python3 training/ray/scripts/sync-cluster-code.py --revision <commit>
```

The rebuilt H100 hosts were seeded from local commit `5127140`, which contains
unpublished recovery/training changes. Compare `HEAD` and `origin/integrate`
before syncing: use `--revision HEAD` when the remote would downgrade those
changes. Do not push other agents' local commits without the user's request.
The UC cluster is seeded from local committed revision `7550f8d`, which also
includes unpublished updates; retain the same revision-selection precaution.
Its handoff source check verified `dd34a80` after newer code was synced during
setup. Current host markers remain authoritative for later syncs.

The script:

- syncs `training/`, `models/`, `tests/`, `data/locations/`,
  `data/sync_object_storage.py`, evaluation code outside `results/`, and
  top-level scripts and docs;
- keeps the local deployment helpers in `training/ray/scripts/` and
  `training/ray/docker/README.md` when they are newer than the commit;
- writes each host's revision to `/home/cc/spectrum-source-revision`;
- verifies SHA-256 checksums for every synced file on every host;
- verifies container IDs and creation times did not change;
- does **not** restart any service.

Ray containers bind-mount `/home/cc/spectrum-usage`, so new job submissions
read the updated files. Running jobs and long-lived actors keep the modules they
already imported. Start a new job to pick up changed code, or recreate the
actor. A source sync is safe between jobs and does not require a Ray restart.

Rebuild and redistribute the image when `training/ray/requirements/core.txt`,
`Dockerfile.core`, or an entrypoint changes. Image builds and stack redeploys
are cluster-specific; follow `FINAL.md`, `H100.md`, `TACC.md`, or `UC.md`.

Do not `git pull` on the remote hosts. They hold source snapshots, not clones.

## Sync data and checkpoints

Each physical host keeps its own copy under `/home/cc/spectrum-usage/`. Ray
containers see it at `/workspace/spectrum-usage/`. The head and GPU worker on
the H100 manager share that host's copy.

Processed power CSVs:

```bash
# Three-GPU H100 allocation
RAY_INVENTORY=training/ray/docker/h100-three-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh

# NCAR ARM cluster
RAY_INVENTORY=training/ray/docker/ncar-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh

# New H100 cluster
RAY_INVENTORY=training/ray/docker/final-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh

# Mixed GPU cluster
RAY_INVENTORY=training/ray/docker/standalone-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh

# TACC cluster
RAY_INVENTORY=training/ray/docker/tacc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh

# UC cluster
RAY_INVENTORY=training/ray/docker/uc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
```

Checkpoints:

```bash
RAY_INVENTORY=training/ray/docker/final-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
RAY_INVENTORY=training/ray/docker/standalone-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
RAY_INVENTORY=training/ray/docker/tacc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
RAY_INVENTORY=training/ray/docker/uc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

Select one testbed or preview first:

```bash
RAY_INVENTORY=training/ray/docker/tacc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --dry-run --site powder
```

The wrapper downloads processed `power_1mhz_avg_per_minute.csv` files, skips
raw captures, shares one inventory across the selected hosts, and compares
full-file SHA-256 checksums. Data sync does not restart Ray. If the source
bucket changes during a run, the size check fails; rerun the command. Do not
run a checkpoint sync on a full disk; use the dry-run inventory's byte totals
to check the current corpus size and required updates.

## Submit jobs

Submit through the Ray Jobs API on the head container. Use a unique
`--submission-id` so retries and logs are easy to find.

Non-interactive template:

```bash
# New H100 cluster
ssh -F training/ray/docker/final-ssh.conf ray-final-1 \
  'head=$(docker ps -q --filter label=com.docker.swarm.service.name=final_ray-head); \
   docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
     --submission-id my-job -- python -m training.ray.cluster_probe'

# Mixed GPU cluster
ssh -F training/ray/docker/h100-ssh.conf h100-head \
  'head=$(docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-head); \
   docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
     --submission-id my-job -- python -m training.ray.cluster_probe'

# TACC cluster
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1 \
  'head=$(docker ps -q --filter label=com.docker.swarm.service.name=tacc_ray-head); \
   docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
     --submission-id my-job -- python -m training.ray.cluster_probe'

# UC cluster
ssh -F training/ray/docker/uc-ssh.conf ray-uc-1 \
  'head=$(docker ps -q --filter label=com.docker.swarm.service.name=uc_ray-head); \
   docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
     --submission-id my-job -- python -m training.ray.cluster_probe'
```

Interactive equivalent on the manager:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-head)
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id my-job -- python -m training.ray.cluster_probe
```

### Examples

Cluster acceptance probe:

```bash
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id acceptance-$(date +%Y%m%d-%H%M%S) -- \
  python -m training.ray.cluster_probe \
  --fractional-gpu-check 0.25 --check-model-imports \
  --check-minio-cross-worker --check-spectrum-read
```

Single-model training:

```bash
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id train-vanillalstm1d-$(date +%Y%m%d-%H%M%S) -- \
  python training/common/train_integrated.py \
  --config training/common/config.yaml --name vanillalstm1d-agent
```

Ray tuning plan or search:

```bash
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id tuning-agent-$(date +%Y%m%d-%H%M%S) -- \
  python -m training.ray.run --config training/common/config.yaml \
  --models vanillalstm1d --dry-run
```

Pass CPU thread limits to concurrent trials with a runtime environment:

```bash
docker exec "$head" ray job submit --address http://127.0.0.1:8265 \
  --submission-id my-job \
  --runtime-env-json '{"env_vars":{"RAY_LOG_TO_STDERR":"0","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"}}' \
  -- python -m training.ray.cluster_probe
```

### Monitor and stop

```bash
docker exec "$head" ray status
docker exec "$head" ray job list
docker exec "$head" ray job status <submission-id>
docker exec "$head" ray job logs <submission-id> --follow
docker exec "$head" ray job stop <submission-id>
```

The existing script `training/ray/scripts/submit-job.sh` can reach the Jobs API
from the manager host, but its default address points at the retired cluster.
Pass `--address http://127.0.0.1:8265` explicitly.

## Storage inside jobs

| Variable group | Store | Bucket | Endpoint |
|---|---|---|---|
| `AWS_*`, `S3_BUCKET` | Cluster MinIO (outputs) | `spectrum-usage` | `http://minio:9000` |
| `SPECTRUM_AWS_*`, `SPECTRUM_S3_*` | Chameleon source data | `spectrum` | Chameleon object storage |

Use the helper for source data:

```python
import os
from training.ray.storage import spectrum_s3_client

client = spectrum_s3_client()
bucket = os.environ["SPECTRUM_S3_BUCKET"]
```

Use `boto3.client("s3")` with the default `AWS_*` environment for Ray outputs.
Credentials are mounted as files under `/run/secrets/`; never print or copy
them. The endpoint and bucket values differ between clusters only in their
MinIO instance, not in the environment variable names.

### H100 head-loss recovery and the old MinIO backup

The expired four-GPU head's MinIO was backed up to the **durable source
bucket**, at **`s3://spectrum/oct2026-minio-backup-newcluster/`**. This is
separate from the model files under `checkpoints/`. The backup contains
**4,087 objects (551,205,275 bytes)**; the latest object modification is
**2026-11-06 14:48:46 UTC**.

The rebuild restored all 4,087 objects into the new MinIO on `ray-worker-2`
under their original keys. Job **`h100-minio-restore-fast-20261107`**
**succeeded**, with no size mismatches and ten JSON objects checked against
the durable backup's contents. To inspect its status:

```bash
ssh -F training/ray/docker/h100-ssh.conf h100-head \
  'head=$(docker ps -q --filter label=com.docker.swarm.service.name=standalone_ray-head); \
   docker exec "$head" ray job status --address http://127.0.0.1:8265 h100-minio-restore-fast-20261107'
```

The final verified restore record is
`/home/cc/spectrum-usage/runs/infra/minio-restoration-20261107.json` on the
manager. The earlier restore job `h100-minio-restore-20261107` was replaced
by the faster idempotent copy job. Do not treat its stopped status as a
failed recovery. Old Ray drivers were lost with the head and are not
automatically restarted; submit/resume them through their runners. Preserve
the durable backup prefix.

Root-only `/home/cc/ray-recovery/` directories on the two surviving workers
also preserve their exited containers' writable files. They include file
manifests and may contain recoverable logs or checkpoints. Do not prune or
delete them as routine cleanup. See [H100.md](docker/H100.md).

After recovery on November 7, `ray-worker-1` had about 11 GiB free and the
head about 20 GiB; each new node had about 67 GiB free. Check current `df -h`
before large model downloads or cache builds. Recovery files account for a
substantial part of the older hosts' disk use and need review before cleanup.

## Operational rules

- Sync code with `sync-cluster-code.py`. Do not restart Ray to update source.
- Do not run `docker service update`, `docker stack deploy`, or `--prune` unless
  the task specifically requires an image or configuration change.
- Do not join hosts between clusters during routine work. `final`, `standalone`, `tacc`, `uc`, `ncar`, and `h100three` are independent
  Swarms with separate MinIO stores and secrets.
- Do not modify firewalld, Swarm membership, or node labels for routine work.
- Use unique submission IDs and output directories. Jobs from multiple agents
  share the same cluster.
- Keep raw inputs immutable. Download outputs into `runs/<your-name>/` or a
  MinIO prefix; do not overwrite `data/` or `checkpoints/`.
- Files written by job containers appear root-owned on the host. Clean them up
  on the manager with `sudo` if needed.
- Check `ray status` and `ray job list` before resource-heavy work. `final` has
  six GPU workers, UC has nine (12 GPUs), NCAR has two, and `h100three` has one
  three-GPU worker. TACC and the older H100 inventories require separate
  current-access checks; their historical addresses are not active defaults.

## Troubleshooting

| Symptom | Action |
|---|---|
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Use the scoped known-hosts file for that cluster. Do not delete entries from the global file. |
| `No route to host` for `129.114.108.20` | Use `ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1`. |
| `Permission denied` on TACC private nodes | Load the Chameleon key in your local `ssh-agent`; the config forwards it through `ray-tacc-2`. |
| Data sync reports `Size mismatch` | The source bucket changed mid-download. Rerun; the next inventory is fresh. |
| A job fails right after a source change | Start a new submission. Running drivers and actors keep old imports. |
| Worker missing from `ray status` | Check `docker stack services` and `docker service ps` on the manager; inspect the worker service logs. |
| Job cannot import a new dependency | The image lacks it. Add it to `requirements/core.txt`, rebuild, distribute the new image, and redeploy. |
| GPU out of memory | Reduce batch size or `--max-concurrent`, or reserve a larger GPU fraction to reduce colocation; inspect `nvidia-smi` on the host. |
