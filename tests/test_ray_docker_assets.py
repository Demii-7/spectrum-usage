from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
RAY = ROOT / "training" / "ray"


def _load_yaml(name: str) -> dict:
    text = (RAY / "docker" / name).read_text()
    text = re.sub(r"\$\{RAY_IMAGE:\?[^}]+\}", "spectrum-ray:2.54.0-test", text)
    text = re.sub(r"\$\{([^}:]+)(?::-[^}]*)?\}", r"placeholder", text)
    return yaml.safe_load(text)


def test_compose_has_head_worker_minio_and_repository_mount():
    compose = _load_yaml("compose.yaml")
    assert {"ray-head", "ray-worker", "minio", "minio-init"} <= compose["services"].keys()
    assert any("/workspace/spectrum-usage" in str(v) for v in compose["services"]["ray-head"]["volumes"])
    assert compose["services"]["ray-worker"]["gpus"] == 1
    assert compose["services"]["ray-head"]["entrypoint"] == ["/usr/local/bin/ray-head"]
    assert compose["services"]["ray-worker"]["entrypoint"] == ["/usr/local/bin/ray-worker"]
    assert compose["services"]["ray-head"]["image"] == compose["services"]["ray-worker"]["image"]
    assert "healthcheck" in compose["services"]["ray-head"]


def test_stack_uses_local_image_encrypted_overlay_and_global_workers():
    stack = _load_yaml("stack.yaml")
    worker = stack["services"]["ray-worker"]
    assert worker["deploy"]["mode"] == "global"
    assert "node.labels.ray.gpu == true" in worker["deploy"]["placement"]["constraints"]
    assert stack["services"]["ray-head"]["entrypoint"] == ["/usr/local/bin/ray-head"]
    assert worker["entrypoint"] == ["/usr/local/bin/ray-worker"]
    assert stack["services"]["ray-head"]["image"] == worker["image"]
    assert "generic_resources" not in str(stack)
    assert "resources" not in worker["deploy"]
    assert worker["environment"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert "build" not in str(stack)
    assert stack["networks"]["ray-overlay"]["driver"] == "overlay"
    assert stack["networks"]["ray-overlay"]["driver_opts"]["encrypted"] == "true"


def test_stack_pins_head_and_minio_volume_and_uses_external_secrets():
    stack = _load_yaml("stack.yaml")
    head_constraints = stack["services"]["ray-head"]["deploy"]["placement"]["constraints"]
    minio_constraints = stack["services"]["minio"]["deploy"]["placement"]["constraints"]
    assert "node.labels.ray.head == true" in head_constraints
    assert "node.labels.ray.head == true" in minio_constraints
    assert stack["secrets"]["minio_root_password"]["external"] is True
    assert stack["services"]["minio"]["volumes"] == ["minio-data:/data"]


def test_ray_version_is_pinned_and_scripts_default_to_safe_mode():
    requirements = (RAY / "requirements" / "core.txt").read_text()
    for dependency in ("ray[default]", "scipy", "timm", "einops", "s3fs", "pyarrow"):
        assert re.search(rf"^{re.escape(dependency)}==", requirements, re.MULTILINE)
    assert "gdown" not in requirements.lower()
    dockerfile = (RAY / "docker" / "Dockerfile.core").read_text()
    assert "entrypoint-head.sh" in dockerfile
    assert "entrypoint-worker.sh" in dockerfile
    assert not (RAY / "docker" / "Dockerfile.head").exists()
    for name in (
        "init-swarm.sh",
        "prepare-nvidia-runtime.sh",
        "build-distribute-image.sh",
        "deploy.sh",
        "teardown.sh",
        "submit-job.sh",
        "acceptance.sh",
    ):
        script = (RAY / "scripts" / name).read_text()
        assert "DRY RUN" in script
        assert "read -r -p" in script


def test_no_registry_deployment_and_gpu_idle_guards_are_wired():
    deploy = (RAY / "scripts" / "deploy.sh").read_text()
    init = (RAY / "scripts" / "init-swarm.sh").read_text()
    distribute = (RAY / "scripts" / "build-distribute-image.sh").read_text()
    common = (RAY / "scripts" / "common.sh").read_text()
    assert "--resolve-image never" in deploy
    assert "docker image save" in distribute
    assert "check_all_gpus_idle" in deploy
    assert "verify_all_repo_commits" in deploy
    assert "status --porcelain --untracked-files=no" in common
    assert "check_all_gpus_idle" in init
    assert "check_all_gpus_idle" in distribute
    assert "utilization.gpu" in common
    assert "query-compute-apps" in common
    assert "192.168.1.120" in common and "ray-worker-2" in common
    assert "192.168.1.130" in common and "ray-worker-1" in common
    assert "docker restart" not in (RAY / "scripts" / "prepare-nvidia-runtime.sh").read_text()
