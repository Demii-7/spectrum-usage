import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "h100_three_cpuset",
    Path(__file__).resolve().parents[1] / "training/ray/scripts/h100-three-cpuset.py",
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class StackCpuSetTests(unittest.TestCase):
    def test_only_unpinned_stack_containers_are_updated(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[1] == "ps":
                self.assertEqual(command[-1], "label=com.docker.stack.namespace=h100three")
                return subprocess.CompletedProcess(command, 0, "replacement pinned removed\n")
            if command[1] == "inspect":
                if command[-1] == "removed":
                    return subprocess.CompletedProcess(command, 1, "")
                cpuset = guard.CPUSET if command[-1] == "pinned" else ""
                return subprocess.CompletedProcess(command, 0, json.dumps([{"HostConfig": {"CpusetCpus": cpuset}}]))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(guard.subprocess, "run", side_effect=run):
            guard.enforce()
        self.assertEqual([c for c in commands if c[1] == "update"],
                         [["docker", "update", "--cpuset-cpus", "0-71,96-167", "replacement"]])


if __name__ == "__main__":
    unittest.main()
