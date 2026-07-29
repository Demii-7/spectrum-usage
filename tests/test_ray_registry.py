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
            expected = 0.5 if name in {"convlstm", "residualconvlstm", "convlstmfm", "dswinlstm_i", "tss_lcd", "stsprednet"} else 1.0
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

    def test_temporal_conv_net_search_excludes_joint_feature_mode(self):
        spec = MODEL_REGISTRY["temporalconvnet"]
        modes = {
            bundle["model.feature_mode"]
            for bundle in spec.space["architecture"].choices
        }
        self.assertEqual(modes, {"independent"})
        for parameters in spec.historical.values():
            self.assertEqual(
                parameters["architecture"]["model.feature_mode"],
                "independent",
            )

    def test_linear_ar_hpo_is_cpu_only_and_tunes_ridge_and_learning_rate(self):
        for family in ("linearar", "residuallinearar"):
            for dimension in ("1d", "2d", "4d"):
                spec = MODEL_REGISTRY[f"{family}{dimension}"]
                self.assertTrue(spec.hpo_executable)
                self.assertEqual(spec.resources.gpu, 0.0)
                self.assertIn("train.learning_rate", spec.space)
                ridge_values = {
                    bundle["model.ridge_alpha"]
                    for bundle in spec.space["architecture"].choices
                }
                self.assertEqual(ridge_values, {1e-4, 1e-2, 1.0})

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

    def test_tss_lcd_uses_the_common_physical_objective(self):
        spec = MODEL_REGISTRY["tss_lcd"]
        self.assertTrue(spec.hpo_executable)
        self.assertEqual(spec.representation, "2d")
        self.assertEqual(spec.objective, "val_mean_horizon_mae_db")

    def test_stsprednet_tunes_with_its_within_model_fallback_metric(self):
        spec = MODEL_REGISTRY["stsprednet"]
        self.assertTrue(spec.hpo_executable)
        self.assertEqual(spec.representation, "4d")
        self.assertEqual(spec.fallback_objective, "val_loss")


if __name__ == "__main__":
    unittest.main()
