import unittest

from training.ray.backup_minio import plan_copy

P, B = "spectrum-usage/ray/oct2026-core4", "oct2026-minio-backup"


class PlanCopyTests(unittest.TestCase):
    def test_backup_prefixes_every_key(self):
        plan = plan_copy("backup", [(f"{P}/4d/x/a.json", 5)], minio_prefix=P, backup_prefix=B)
        self.assertEqual(plan, [(f"{P}/4d/x/a.json", f"{B}/{P}/4d/x/a.json", 5)])

    def test_restore_returns_keys_to_their_original_place(self):
        plan = plan_copy("restore", [(f"{B}/{P}/4d/x/a.json", 5)], minio_prefix=P, backup_prefix=B)
        self.assertEqual(plan, [(f"{B}/{P}/4d/x/a.json", f"{P}/4d/x/a.json", 5)])

    def test_restore_can_target_a_test_prefix_and_filter(self):
        keys = [(f"{B}/{P}/4d/convlstm-search/s.json", 1), (f"{B}/{P}/2d/other/o.json", 2),
                (f"{B}/{P}-r2/2d/z/z.json", 3), ("unrelated/key", 4)]
        plan = plan_copy("restore", keys, minio_prefix=P, backup_prefix=B,
                         dest_prefix="spectrum-usage/ray/restore-test", only="convlstm-search")
        self.assertEqual([t for _, t, _ in plan], ["spectrum-usage/ray/restore-test/4d/convlstm-search/s.json"])

    def test_restore_keeps_the_second_round_prefix_family(self):
        plan = plan_copy("restore", [(f"{B}/{P}-r2/2d/z/z.json", 3)], minio_prefix=P, backup_prefix=B)
        self.assertEqual(plan[0][1], f"{P}-r2/2d/z/z.json")


if __name__ == "__main__":
    unittest.main()
