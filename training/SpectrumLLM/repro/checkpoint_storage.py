"""Manage the SpectrumLLM repro checkpoints in object storage (Swift/Keystone).

The remote layout (container ``spectrum`` on Chameleon Chi@TACC, credentials in
``data/.env`` -- OpenStack application credentials; the S3 keys in that file are
not accepted by this RGW):

    checkpoints/repro/spectrumllm/
      literal/seed-N/...   the 150 first-campaign weights (literal split, old prompt)
      tslib/seed-N/...     the current weights (tslib split; the _tslib filename
                           tag marks the verbatim-prompt vintage)

Run inside a container that has the repo mounted at /w and swiftclient installed:

  docker exec -w /w tslib python training/SpectrumLLM/repro/checkpoint_storage.py relocate
  docker exec -w /w tslib python training/SpectrumLLM/repro/checkpoint_storage.py upload
  docker exec -w /w tslib python training/SpectrumLLM/repro/checkpoint_storage.py verify

Only the standard library, swiftclient and keystoneauth1 are used. Credentials
are read from data/.env and never printed.
"""
import argparse
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONTAINER = "spectrum"
REMOTE_ROOT = "checkpoints/repro/spectrumllm"
LOCAL_ROOT = os.path.join(REPO, "checkpoints", "repro", "2d", "spectrumllm")


def connect():
    env = os.path.join(REPO, "data", ".env")
    for line in open(env):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    from keystoneauth1.identity import v3
    from keystoneauth1 import session as ks_session
    import swiftclient
    sess = ks_session.Session(auth=v3.ApplicationCredential(
        auth_url=os.environ["OS_AUTH_URL"],
        application_credential_id=os.environ["OS_APPLICATION_CREDENTIAL_ID"],
        application_credential_secret=os.environ["OS_APPLICATION_CREDENTIAL_SECRET"]))
    return swiftclient.Connection(session=sess)


def sha16_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def list_prefix(conn, prefix):
    objs, marker = [], None
    while True:
        _, data = conn.get_container(CONTAINER, prefix=prefix, marker=marker)
        if not data:
            return objs
        objs.extend(data)
        marker = data[-1]["name"]


def cmd_relocate(conn):
    """Move the 150 first-campaign objects from seed-N/ to literal/seed-N/."""
    src = [o for o in list_prefix(conn, f"{REMOTE_ROOT}/") if "/seed-" in o["name"]]
    print(f"objects at the flat layout: {len(src)}", flush=True)
    moved, failed = 0, []
    for i, o in enumerate(src):
        rel = o["name"][len(REMOTE_ROOT) + 1:]           # seed-N/file.pt
        dst = f"{REMOTE_ROOT}/literal/{rel}"
        # server-side copy: PUT with X-Copy-From, no data through the client
        conn.put_object(CONTAINER, dst, contents=b"",
                        headers={"X-Copy-From": f"/{CONTAINER}/{o['name']}",
                                 "Content-Length": "0"})
        head = conn.head_object(CONTAINER, dst)
        if int(head.get("content-length", 0)) == o["bytes"]:
            conn.delete_object(CONTAINER, o["name"])
            moved += 1
        else:
            failed.append(o["name"])
        if (i + 1) % 25 == 0:
            print(f"  relocated {i+1}/{len(src)}", flush=True)
    print(f"relocated {moved}/{len(src)}; failures: {failed}", flush=True)


def cmd_upload(conn):
    """Upload every local repro checkpoint to tslib/seed-N/."""
    local = []
    for seed_dir in sorted(os.listdir(LOCAL_ROOT)):
        d = os.path.join(LOCAL_ROOT, seed_dir)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".pt"):
                local.append((seed_dir, f, os.path.join(d, f)))
    print(f"local checkpoints: {len(local)}", flush=True)
    done = skipped = 0
    for seed_dir, name, path in local:
        dst = f"{REMOTE_ROOT}/tslib/{seed_dir}/{name}"
        try:
            head = conn.head_object(CONTAINER, dst)
            if int(head.get("content-length", 0)) == os.path.getsize(path):
                skipped += 1
                continue
        except Exception:
            pass
        with open(path, "rb") as f:
            conn.put_object(CONTAINER, dst, contents=f,
                            content_length=os.path.getsize(path))
        done += 1
        print(f"  uploaded {dst} ({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
    print(f"uploaded {done}, skipped (already present) {skipped}", flush=True)


def cmd_verify(conn, sample=3):
    for sub in ("literal", "tslib"):
        objs = list_prefix(conn, f"{REMOTE_ROOT}/{sub}/")
        gb = sum(o["bytes"] for o in objs) / 1e9
        print(f"{sub}: {len(objs)} objects, {gb:.2f} GB")
        for o in objs[:sample]:
            hdr, data = conn.get_object(CONTAINER, o["name"])
            print(f"  sample {o['name'].rsplit('/', 1)[-1]}: "
                  f"sha256_16={hashlib.sha256(data).hexdigest()[:16]} "
                  f"({len(data)/1e6:.0f} MB)")
    flat = [o for o in list_prefix(conn, f"{REMOTE_ROOT}/") if "/seed-" in o["name"]]
    print(f"remaining at the old flat layout: {len(flat)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["relocate", "upload", "verify"])
    args = ap.parse_args()
    conn = connect()
    print("keystone auth OK", flush=True)
    {"relocate": cmd_relocate, "upload": cmd_upload, "verify": cmd_verify}[args.command](conn)


if __name__ == "__main__":
    main()
