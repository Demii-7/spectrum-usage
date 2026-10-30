import argparse
import unittest
from types import SimpleNamespace

from training.ray import campaign_all as supervisor


class FakeClient:
    def __init__(self, existing=()):
        self.submitted, self.statuses = [], {}
        self._existing = list(existing)

    def submit_job(self, entrypoint, submission_id, metadata=None):
        self.submitted.append((submission_id, entrypoint))
        self.statuses[submission_id] = "RUNNING"

    def get_job_status(self, submission_id):
        return self.statuses[submission_id]

    def list_jobs(self):
        return [SimpleNamespace(submission_id=i, status=s) for i, s in self._existing]


def _args(**overrides):
    args = supervisor.parse_args(["--retry-delay", "0"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SupervisorTests(unittest.TestCase):
    def test_never_exceeds_the_concurrent_job_limit(self):
        jobs, client = supervisor.make_jobs(None, {}), FakeClient()
        supervisor.step(jobs, client, _args(max_concurrent_jobs=3), now=1.0)
        self.assertEqual(len(client.submitted), 3)
        supervisor.step(jobs, client, _args(max_concurrent_jobs=3), now=2.0)
        self.assertEqual(len(client.submitted), 3)

    def test_finished_job_frees_a_slot_and_failed_job_is_resubmitted(self):
        jobs, client = supervisor.make_jobs(["linearar2d", "vanillalstm"], {}), FakeClient()
        args = _args(max_concurrent_jobs=2)
        supervisor.step(jobs, client, args, now=1.0)
        first, second = jobs
        client.statuses[first.submission_id] = "SUCCEEDED"
        client.statuses[second.submission_id] = "FAILED"
        supervisor.step(jobs, client, args, now=2.0)
        self.assertEqual(first.state, "done")
        self.assertEqual((second.state, second.attempts), ("running", 2))
        self.assertTrue(second.submission_id.endswith("-1"))

    def test_gives_up_after_the_attempt_limit(self):
        jobs, client = supervisor.make_jobs(["linearar2d"], {}), FakeClient()
        args = _args(max_attempts=2)
        for now in (1.0, 2.0, 3.0, 4.0):
            supervisor.step(jobs, client, args, now=now)
            if jobs[0].submission_id:
                client.statuses[jobs[0].submission_id] = "FAILED"
        self.assertEqual((jobs[0].state, jobs[0].attempts), ("gave_up", 2))

    def test_restart_adopts_running_and_finished_jobs_instead_of_resubmitting(self):
        client = FakeClient(existing=[("campaign-linearar2d-2d-0", "SUCCEEDED"),
                                      ("campaign-vanillalstm-2d-1", "RUNNING")])
        client.statuses["campaign-vanillalstm-2d-1"] = "RUNNING"
        jobs = supervisor.make_jobs(["linearar2d", "vanillalstm"], {})
        supervisor.reconcile(jobs, client)
        self.assertEqual([j.state for j in jobs], ["done", "running"])
        supervisor.step(jobs, client, _args(), now=1.0)
        self.assertEqual(client.submitted, [])

    def test_same_model_in_two_representations_gets_separate_prefixes(self):
        jobs = [j for j in supervisor.make_jobs(["residualvanillalstm"], {})]
        self.assertEqual({j.representation for j in jobs}, {"1d", "2d"})
        args = _args()
        entries = {j.representation: supervisor.build_entrypoint(j, args) for j in jobs}
        self.assertIn("oct2026-core4/1d", entries["1d"])
        self.assertIn("oct2026-core4/2d", entries["2d"])
        self.assertIn("ray_tuning_residualvanillalstm_1d.yaml", entries["1d"])
        self.assertIn("ray_tuning_2d.yaml", entries["2d"])

    def test_overrides_reach_the_entrypoint(self):
        jobs = supervisor.make_jobs(["convlstm"], {"convlstm": {"gpu_fraction": 0.5}})
        self.assertIn("--gpu-fraction 0.5", supervisor.build_entrypoint(jobs[0], _args()))

    def test_every_planned_model_has_a_config_that_exists(self):
        from pathlib import Path

        for model, _, config in supervisor.PLAN:
            self.assertTrue(Path(f"{supervisor.CONFIG_DIR}/{config}.yaml").exists(), (model, config))
        self.assertEqual(len(supervisor.PLAN), 19)


if __name__ == "__main__":
    unittest.main()
