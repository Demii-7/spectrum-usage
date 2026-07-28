import unittest

from training.ray.candidate import candidate_id
from training.ray.selection import select_candidates


class RaySelectionTests(unittest.TestCase):
    def test_candidate_ids_are_order_independent_and_seed_specific(self):
        first = candidate_id("VanillaLSTM", {"b": 2, "a": 1}, 41)
        second = candidate_id("vanillalstm", {"a": 1, "b": 2}, 41)
        self.assertEqual(first, second)
        self.assertNotEqual(first, candidate_id("vanillalstm", {"a": 1, "b": 2}, 42))
        self.assertEqual(
            candidate_id("vanillalstm", {"a": 1}, None),
            candidate_id("vanillalstm", {"a": 1}),
        )

    def test_best_simple_uses_one_percent_tie_not_one_se(self):
        rows = []
        for seed in (41, 42, 43):
            rows.extend([
                {"candidate_id": "complex", "seed": seed, "objective": 1.0,
                  "complexity": 100, "capacity": "small"},
                {"candidate_id": "simple", "seed": seed, "objective": 1.009,
                  "complexity": 10, "capacity": "tiny"},
                {"candidate_id": "reference", "seed": seed, "objective": 1.2,
                  "complexity": 50, "capacity": "reference"},
            ])
        selected = select_candidates(rows)
        self.assertEqual(selected.best, "complex")
        self.assertEqual(selected.best_simple, "simple")
        self.assertEqual(selected.reference, "reference")

    def test_requires_all_three_seeds(self):
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            select_candidates([
                {"candidate_id": "reference", "seed": 41, "objective": 1,
                 "capacity": "reference"},
            ])


if __name__ == "__main__":
    unittest.main()
