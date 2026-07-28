import unittest

from training.ray.registry import MODEL_REGISTRY, get_model_spec


EXPECTED = {
    "arima", "vanillalstm", "convlstm", "convlstmfm", "dswinlstm_i",
    "lookbackmean1d", "lookbackmean2d", "lookbackmean4d", "linearar1d",
    "linearar2d", "linearar4d", "residuallinearar1d", "residuallinearar2d",
    "residuallinearar4d", "residualvanillalstm", "residualconvlstm",
    "temporalconvnet", "lstmattn", "autoformer_csa", "stsprednet", "tss_lcd",
    "deepspred",
}


class RayRegistryTests(unittest.TestCase):
    def test_covers_integrated_models_except_timeran(self):
        self.assertEqual(set(MODEL_REGISTRY), EXPECTED)
        self.assertNotIn("timeran", MODEL_REGISTRY)

    def test_all_models_have_policy_resources_and_three_anchors(self):
        for spec in MODEL_REGISTRY.values():
            self.assertTrue(spec.policy)
            self.assertGreater(spec.resources.cpu, 0)
            self.assertGreaterEqual(spec.resources.gpu, 0)
            self.assertEqual(set(spec.anchors), {"tiny", "small", "reference"})
            for domain in spec.space.values():
                if domain.kind != "choice":
                    self.assertLess(domain.lower, domain.upper)

    def test_publication_block_is_enforced(self):
        spec = MODEL_REGISTRY["deepspred"]
        self.assertEqual(spec.status, "not_publication_ready")
        self.assertEqual(spec.policy, "blocked_exact_minute")
        self.assertIn("exact-minute", spec.reason)
        with self.assertRaisesRegex(RuntimeError, "exact-minute"):
            get_model_spec("deepspred", require_executable=True)

    def test_resource_defaults_match_model_class(self):
        for name in ("vanillalstm", "residualvanillalstm", "temporalconvnet",
                     "lstmattn", "autoformer_csa"):
            self.assertEqual(MODEL_REGISTRY[name].resources.gpu, 0.5)
        for name in ("convlstm", "residualconvlstm", "convlstmfm", "dswinlstm_i",
                     "stsprednet", "tss_lcd", "deepspred"):
            expected = 0.5 if name in {"convlstm", "residualconvlstm", "convlstmfm", "dswinlstm_i"} else 1.0
            self.assertEqual(MODEL_REGISTRY[name].resources.gpu, expected)
        for name, spec in MODEL_REGISTRY.items():
            if name.startswith(("lookbackmean", "linearar", "residuallinearar")) or name == "arima":
                self.assertEqual(spec.resources.gpu, 0.0)

    def test_tunable_architectures_are_coupled_bundles(self):
        for spec in MODEL_REGISTRY.values():
            if not spec.hpo_executable:
                self.assertEqual(dict(spec.space), {})
                continue
            self.assertIn("architecture", spec.space)
            architecture_keys = [key for key in spec.space if key.startswith("model.")]
            self.assertEqual(architecture_keys, [])
            self.assertGreaterEqual(len(spec.space["architecture"].choices), 3)
            reference = spec.anchor("reference")["architecture"]
            self.assertIn(reference, spec.space["architecture"].choices)

    def test_historical_presets_are_valid_and_guaranteed_for_known_results(self):
        expected = {
            "vanillalstm", "residualvanillalstm", "temporalconvnet",
            "lstmattn", "autoformer_csa", "convlstm", "residualconvlstm",
        }
        self.assertEqual(
            {name for name, spec in MODEL_REGISTRY.items() if spec.historical},
            expected,
        )
        for name in expected:
            spec = MODEL_REGISTRY[name]
            for parameters in spec.historical.values():
                self.assertIn(parameters["architecture"], spec.space["architecture"].choices)
                for key, value in parameters.items():
                    self.assertTrue(spec.space[key].contains(value), f"{name}: {key}={value}")

    def test_specialized_physical_metric_status_is_explicit(self):
        for name in ("stsprednet", "tss_lcd"):
            self.assertEqual(MODEL_REGISTRY[name].status, "not_publication_ready")
            self.assertIn("physical", MODEL_REGISTRY[name].reason)


if __name__ == "__main__":
    unittest.main()
