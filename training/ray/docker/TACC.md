# Independent TACC Ray cluster

Stack `tacc` runs in its own Swarm. It has a CPU-only Ray head and one A100
worker on each of four hosts, including the manager. Ray advertises four
A100-PCIE-40GB GPUs and 1,024 logical CPUs across five Ray nodes.

| Host | Private IP | Public IP | Role |
|---|---|---|---|
| `ray-tacc-1` | `10.52.2.186` | `129.114.108.20` | Swarm manager, Ray head, GPU worker, MinIO |
| `ray-tacc-2` | `10.52.0.30` | `129.114.109.32` | GPU worker and SSH gateway |
| `ray-tacc-3` | `10.52.0.165` | none | GPU worker |
| `ray-tacc-4` | `10.52.2.197` | none | GPU worker |

The manager's public address returned `No route to host` during setup.
`tacc-ssh.conf` routes access to nodes 1, 3, and 4 through node 2. Run the
client commands below from the repository root. The SSH configuration enables
agent forwarding; it does not copy private keys to the hosts.

```bash
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1
ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-3
```

The scoped SSH host-key record lives in the ignored file
`training/ray/docker/secrets/tacc_known_hosts`. It keeps keys for these new
instances separate from stale entries for reused public IPs.

## Runtime

We installed Docker and NVIDIA Container Toolkit on the four Ubuntu 24.04
hosts and made NVIDIA Docker's default runtime. Node 2 had MIG enabled with
no GPU instances. We disabled MIG and rebooted that idle host before joining
it to the Swarm; its full 40 GB GPU then passed the CUDA checks.

The manager advertises `10.52.2.186` for Swarm control and data traffic. Its
overlay address pool is `10.240.0.0/16`. The four private host addresses have
runtime and permanent source entries in firewalld's `trusted` zone. We used
`training/ray/scripts/configure-peer-firewall.sh` to add those exact peers,
covering the control, gossip, VXLAN, and ESP traffic without opening their
ports to arbitrary public sources.

We built `spectrum-ray:2.54.0-tacc-844647d4fa55` once on node 1 from
`Dockerfile.core` and distributed that image to the other nodes with
`docker image save/load` over the private network. It contains Ray 2.54.0
and PyTorch 2.5.1 with CUDA 12.4. The manager records the image tag in
`/home/cc/tacc-image-tag`.

The deployment combines `stack.yaml` and `tacc.yaml`. Each `ray.gpu=true`
host runs one worker with one GPU, 64 GiB `/dev/shm`, and a 16 GiB Ray object
store. The head advertises zero CPUs and GPUs, with a 1 GiB logical memory
resource, a 1 GiB object store, and 2 GiB `/dev/shm`.

To redeploy from the manager:

```bash
source /home/cc/spectrum-storage.sh
export RAY_IMAGE="$(</home/cc/tacc-image-tag)"
docker stack deploy --resolve-image never \
  -c /home/cc/spectrum-usage/training/ray/docker/stack.yaml \
  -c /home/cc/spectrum-usage/training/ray/docker/tacc.yaml tacc
```

## Source, data, and storage

Each host has the committed source snapshot from `integrate` at `399c8c0`,
plus the TACC deployment files and SSH-aware management helpers. The snapshot
includes training and evaluation code, model implementations, tests,
`data/sync_object_storage.py`, and `data/locations/`. Hosts record the source
commit in `/home/cc/spectrum-source-revision`.

The containers bind-mount `/home/cc/spectrum-usage` at
`/workspace/spectrum-usage`. Each physical host keeps its own local datasets
and checkpoints. The head and GPU worker on node 1 share that host's copy.

The cluster has independent Swarm secrets populated from the local
`evaluation/.env`. Use `spectrum_s3_client()` and the `SPECTRUM_*` variables
for bucket `spectrum` at `https://chi.tacc.chameleoncloud.org:7480`.
The standard `AWS_*` environment selects this cluster's MinIO at
`http://minio:9000`, bucket `spectrum-usage`. MinIO uses its own generated
credentials and persistent Docker volume `tacc_minio-data` on node 1.

To sync processed CSVs and checkpoints from your workstation:

```bash
RAY_INVENTORY=training/ray/docker/tacc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh
RAY_INVENTORY=training/ray/docker/tacc-nodes.sh \
  bash training/ray/scripts/sync-cluster-data.sh --checkpoints
```

The inventory selects only the TACC hosts. The wrapper uses the SSH aliases
and compares full-file SHA-256 checksums across the selected nodes. It skips
raw captures. The processed datasets include POWDER's `600_800` and
`2400_2600` bands, including the October runs. Location metadata lives at
`/workspace/spectrum-usage/data/locations/powder.json` inside containers.

The November 3, 2026 sync verified matching SHA-256 checksums on all four hosts
for 208 processed CSVs (3,939,272,193 bytes per host) and 297 checkpoint objects
(27,591,393,491 bytes per host). Each host has 35 POWDER `2400_2600` files,
including seven receiver files for each of `20261015T1434Z` and
`20261025T1355Z` in that band. The sync retried after a source CSV changed
during its initial download; the final saved inventories match across hosts.

## Dashboard and jobs

Use local port 8267 to keep this cluster separate from existing tunnels:

```bash
ssh -F training/ray/docker/tacc-ssh.conf -N \
  -L 8267:localhost:8265 -L 9003:localhost:9001 ray-tacc-1
```

Open `http://localhost:8267` for Ray and `http://localhost:9003` for this
cluster's MinIO console. On the manager:

```bash
head=$(docker ps -q --filter label=com.docker.swarm.service.name=tacc_ray-head)
docker exec "$head" ray status
docker exec "$head" ray job submit --address http://127.0.0.1:8265 -- \
  python -m training.ray.cluster_probe --fractional-gpu-check 0.25 \
  --check-model-imports --check-minio-cross-worker --check-spectrum-read
```

The initial Jobs API acceptance run, `tacc-acceptance-v1`, passed with all
four expected GPU UUIDs, sixteen fractional GPU tasks, model imports,
source-bucket reads on all workers, and a cross-worker MinIO round trip.
The manager keeps `/home/cc/tacc-acceptance.log` and
`/home/cc/tacc-acceptance.json`.

A follow-up node-affinity check performed CUDA matrix multiplication and
PyTorch shared-memory allocation on every A100, checked both asset manifests,
verified the 64 GiB `/dev/shm` mounts and location JSON, and read one MinIO
object across all four workers. Its result is at
`/home/cc/spectrum-usage/runs/infra/tacc-assets-gpu-acceptance.json`.

## Add the fifth node

Keep the existing services running during expansion:

1. Add a `ray-tacc-5` SSH stanza to `tacc-ssh.conf`, using its private address
   and `ProxyJump ray-tacc-2`. Append its private IP, hostname, SSH alias, and
   `tacc_ray-worker` service to the arrays in `tacc-nodes.sh`.
2. Inspect its GPU and idle state, then run `install-host.sh` on that host.
   Check MIG mode and make the full GPU available before starting workloads.
3. Run `configure-peer-firewall.sh` on all five hosts with the updated private
   address list. Adding the new source entries does not restart Ray or Docker.
4. Copy the current source snapshot, locations, processed data, and checkpoints
   to `/home/cc/spectrum-usage` on the new host. Use `rsync -a` to preserve
   data timestamps. Exclude `runs/`, raw captures, `.part` files, and generated
   map caches when copying from an active host. Verify the copied files
   against the saved sync manifests before enabling scheduling.
5. Distribute the exact image from the manager over the private network:

   ```bash
   ssh -F training/ray/docker/tacc-ssh.conf ray-tacc-1 \
     'set -o pipefail; docker image save "$(</home/cc/tacc-image-tag)" | \
       ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new cc@NEW_PRIVATE_IP docker image load'
   ```

6. Join the new host to `10.52.2.186:2377` using a worker token, its private
   advertise address, and the same data-path address. Keep `ray.gpu` unset
   until its files and image are ready. Then run
   `docker node update --label-add ray.gpu=true ray-tacc-5` on the manager.
   The global service starts a worker on that node without replacing the
   existing tasks.
7. Refresh both asset selections with the updated inventory. Run acceptance
   with the fifth GPU UUID and confirm five GPU workers. Adding this host
   does not require rebuilding the image or restarting the head or MinIO.
