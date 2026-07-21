import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.common.data_loader import data_loader_kwargs


class DataLoaderKwargsTest(unittest.TestCase):
    def test_defaults_preserve_synchronous_loading(self):
        with patch("training.common.data_loader.torch.cuda.is_available", return_value=False):
            self.assertEqual(data_loader_kwargs(), {"num_workers": 0, "pin_memory": False})

    def test_worker_only_options_are_omitted_without_workers(self):
        kwargs = data_loader_kwargs(
            {
                "num_workers": 0,
                "prefetch_factor": 3,
                "multiprocessing_context": "spawn",
                "pin_memory": False,
            }
        )
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)
        self.assertNotIn("multiprocessing_context", kwargs)

    def test_validation(self):
        invalid_configs = [
            {"num_workers": -1},
            {"num_workers": True},
            {"persistent_workers": 1},
            {"persistent_workers": True},
            {"prefetch_factor": 0},
            {"pin_memory": "yes"},
            {"multiprocessing_context": 1},
            {"multiprocessing_context": "invalid"},
            {"unknown": True},
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                data_loader_kwargs(config)

    def test_two_worker_smoke(self):
        dataset = TensorDataset(torch.arange(24))
        loader = DataLoader(
            dataset,
            batch_size=5,
            **data_loader_kwargs({"num_workers": 2, "pin_memory": False}),
        )
        actual = torch.cat([batch[0] for batch in loader])
        self.assertTrue(torch.equal(actual, torch.arange(24)))


if __name__ == "__main__":
    unittest.main()
