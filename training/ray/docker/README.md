# Ray Docker and Swarm deployment

Ray `2.54.0` runs from one source-hashed image containing explicit `ray-head`
and `ray-worker` entrypoints. The four one-H100 hosts are:

| Address | Swarm hostname | Roles |
|---|---|---|
| `192.168.1.201` | current manager hostname | manager, Ray head, MinIO, GPU worker |
| `192.168.1.120` | `ray-worker-2` | GPU worker |
| `192.168.1.130` | `ray-worker-1` | GPU worker |
| `192.168.1.189` | `ray-worker-3` | GPU worker |

Every host must have `/home/cc/spectrum-usage`; Stack bind-mounts it at
`/workspace/spectrum-usage`. The global worker service places exactly one task
on each `ray.gpu=true` node and exposes its sole H100 with
`NVIDIA_VISIBLE_DEVICES=all`. There is no generic-resource reservation.

## NVIDIA runtime

Swarm does not honor Compose's `gpus` key and cannot reliably select a service
runtime, so NVIDIA must be Docker's default runtime on each host. Check without
making changes:

```bash
training/ray/scripts/preflight.sh
```

If preflight reports that `nvidia` is installed but not the default, review the
dry-run and separately configure all daemon files:

```bash
training/ray/scripts/prepare-nvidia-runtime.sh
training/ray/scripts/prepare-nvidia-runtime.sh --apply
```

That script runs `nvidia-ctk runtime configure --runtime=docker
--set-as-default`. It never restarts Docker. An operator must schedule any
required Docker restarts and rerun preflight afterward. Use
`--allow-busy-gpus` only as an explicit exception to the idle guard.

## No-registry image

Docker Stack cannot build. No private registry is required: the distribution
script computes a tag from the Dockerfile, requirements, and entrypoints, builds
once on `.201`, and streams that exact image with `docker image save/load` to
`.120` and `.130`. Existing source-hashed tags are not rebuilt.

```bash
training/ray/scripts/build-distribute-image.sh
training/ray/scripts/build-distribute-image.sh --apply
# Export the RAY_IMAGE value printed by the script.
```

Deployment verifies that all four local image IDs are identical and uses
`docker stack deploy --resolve-image never`. Public pinned MinIO images still
come from their normal public registry.

## Swarm and secrets

All nodes are initially Swarm-inactive. Initialization joins the two known
workers over SSH, labels the manager as `ray.head=true` and all hosts as
`ray.gpu=true`, and interactively creates external secrets
`ray_minio_root_user` and `ray_minio_root_password`. No credential values are in
the repository. Local Compose instead reads ignored files from
`training/ray/docker/secrets/`.

```bash
training/ray/scripts/init-swarm.sh
training/ray/scripts/init-swarm.sh --apply
training/ray/scripts/deploy.sh
training/ray/scripts/deploy.sh --apply
```

The Ray head, MinIO, and MinIO's persistent local volume are pinned to the
manager. The global worker also places a GPU task there. The encrypted overlay
connects all services. Dashboard, Jobs API, Ray Client, MinIO S3, and MinIO
console are published on `.201` at ports `8265`, `10001`, `9000`, and `9001`.

## Acceptance and jobs

```bash
training/ray/scripts/acceptance.sh
training/ray/scripts/acceptance.sh --run
training/ray/scripts/submit-job.sh --submit -- python training/common/train_integrated.py --config training/common/config.yaml
```

Acceptance first checks `/api/version`, then submits
`python -m training.ray.cluster_probe` through the Jobs API. It passes the three
live host-to-GPU UUID mappings and requests a `0.25` fractional-GPU synthetic
task, model import checks, and MinIO write/read/delete across workers. The probe
module supplies the corresponding `--expected-gpu-node`,
`--fractional-gpu-check`, `--check-model-imports`, and
`--check-minio-cross-worker` options.

All mutating scripts default to dry-run, require a typed confirmation, and fail
when any H100 utilization exceeds `${RAY_GPU_IDLE_THRESHOLD:-5}` percent or any
GPU compute process exists. Teardown preserves MinIO data and Swarm state:

```bash
training/ray/scripts/teardown.sh --apply
```
