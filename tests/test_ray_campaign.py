import json
import tempfile
import unittest

try:
    import pyarrow.fs as pafs
except ImportError:  # pyarrow is installed on the cluster, not on every dev machine
    pafs = None

from training.ray import campaign


def _args(*extra):
    return campaign.parse_args(
        ["--config", "c.yaml", "--model", "vanillalstm", "--prefix", "oct", "--output-dir", "out", *extra]
    )


class CampaignArgvTests(unittest.TestCase):
    def test_defaults_match_the_agreed_search_protocol(self):
        args = _args()
        self.assertEqual(
            (args.candidates, args.grace_period, args.reduction_factor, args.brackets),
            (16, 8, 2, 2),
        )
        self.assertEqual(args.seeds, [40, 41, 42, 43, 44])

    def test_stage_commands_always_resume_and_pass_gpu_fraction(self):
        args = _args("--gpu-fraction", "0.25")
        for build in (campaign._search_argv, campaign._replicate_argv):
            argv = build(args)
            self.assertIn("--resume", argv)
            self.assertEqual(argv[argv.index("--gpu-fraction") + 1], "0.25")
        self.assertEqual(
            campaign._replicate_argv(args)[campaign._replicate_argv(args).index("--search-path") + 1],
            "spectrum-usage/oct/vanillalstm-search",
        )

    def test_gpu_fraction_is_omitted_when_not_set(self):
        self.assertNotIn("--gpu-fraction", campaign._search_argv(_args()))


@unittest.skipUnless(pafs, "pyarrow not installed")
class RunStageTests(unittest.TestCase):
    def test_stage_runs_once_then_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            fs, calls = pafs.LocalFileSystem(), []
            self.assertTrue(campaign.run_stage(fs, root, "p", "m", "search", lambda: calls.append(1)))
            self.assertFalse(campaign.run_stage(fs, root, "p", "m", "search", lambda: calls.append(1)))
            self.assertEqual(len(calls), 1)

    def test_failure_is_recorded_and_the_stage_reruns_next_time(self):
        with tempfile.TemporaryDirectory() as root:
            fs, calls = pafs.LocalFileSystem(), []

            def flaky():
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("node lost")

            with self.assertRaises(RuntimeError):
                campaign.run_stage(fs, root, "p", "m", "replicate", flaky)
            with fs.open_input_stream(campaign._marker_path(root, "p", "m", "status.json")) as stream:
                status = json.loads(stream.read().decode())
            self.assertEqual((status["state"], status["stage"]), ("failed", "replicate"))
            self.assertIn("node lost", status["error"])
            self.assertTrue(campaign.run_stage(fs, root, "p", "m", "replicate", flaky))
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
