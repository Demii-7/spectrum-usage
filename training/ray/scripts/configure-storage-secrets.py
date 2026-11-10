"""Create Swarm storage secrets over SSH without logging credential values."""
import argparse
import secrets
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--head", required=True)
parser.add_argument("--env-file", required=True)
parser.add_argument("--ssh-config", type=Path, help="SSH config for aliases or jump hosts")
args = parser.parse_args()
ssh_command = ["ssh", "-A"]
if args.ssh_config:
    ssh_command += ["-F", str(args.ssh_config)]
ssh_command.append(args.head)
values = {}
for line in Path(args.env_file).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    key, sep, value = line.removeprefix("export ").partition("=")
    if sep:
        parsed = shlex.split(value, comments=True)
        values[key.strip()] = parsed[0] if parsed else ""
credentials = {
    "ray_spectrum_access_key": values["AWS_ACCESS_KEY_ID"],
    "ray_spectrum_secret_key": values["AWS_SECRET_ACCESS_KEY"],
    "ray_minio_root_user": "spectrum-admin",
    "ray_minio_root_password": secrets.token_urlsafe(36),
}
for name, value in credentials.items():
    if not value:
        raise ValueError(f"Empty credential for {name}")
    subprocess.run(
        [*ssh_command, f"docker secret inspect {name} >/dev/null 2>&1 || docker secret create {name} - >/dev/null"],
        input=value.encode(), check=True,
    )
endpoint = values.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
bucket = values.get("S3_BUCKET", "spectrum")
config = f"export SPECTRUM_S3_ENDPOINT_URL={shlex.quote(endpoint)}\nexport SPECTRUM_S3_BUCKET={shlex.quote(bucket)}\n"
subprocess.run([*ssh_command, "umask 077; tee /home/cc/spectrum-storage.sh >/dev/null"], input=config.encode(), check=True)
print(f"Storage secrets ready; spectrum bucket={bucket}, endpoint={endpoint}")
