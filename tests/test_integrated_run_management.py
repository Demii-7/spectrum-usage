from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.common.config import model_names, unique_run_dir
from training.common.data import ChunkSpec, _select_chunk
from training.common.data_sources import LoadedSource
from training.common.preprocessing import SequenceSegment


class IntegratedRunManagementTests(unittest.TestCase):
    def test_chunk_selection_slices_frequency_axis_and_metadata(self) -> None:
        source = LoadedSource(
            data=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            frequencies=np.asarray([599.0, 600.0, 700.0, 801.0]),
            timestamps=None,
            files=[Path("input.csv")],
            feature_labels=["599", "600", "700", "801"],
            segments=(SequenceSegment(0, 2, "input"),),
        )

        selected = _select_chunk(source, ChunkSpec("band", 600.0, 800.0))

        np.testing.assert_array_equal(selected.frequencies, [600.0, 700.0])
        np.testing.assert_array_equal(selected.data, source.data[..., 1:3])
        self.assertEqual(selected.feature_labels, ["600", "700"])

    def test_chunk_selection_rejects_empty_ranges(self) -> None:
        source = LoadedSource(
            data=np.ones((2, 1), dtype=np.float32),
            frequencies=np.asarray([700.0]),
            timestamps=None,
            files=[],
            feature_labels=["700"],
            segments=(),
        )
        with self.assertRaisesRegex(ValueError, "contains no frequencies"):
            _select_chunk(source, ChunkSpec("empty", 800.0, 900.0))

    def test_model_names_supports_single_and_multiple_models(self) -> None:
        self.assertEqual(
            model_names({"training": {"model_name": "VanillaLSTM"}, "vanillalstm": {}}),
            ["vanillalstm"],
        )
        self.assertEqual(
            model_names(
                {
                    "training": {"models": ["VanillaLSTM", "LinearAR2D"]},
                    "vanillalstm": {},
                    "linearar2d": {},
                }
            ),
            ["vanillalstm", "linearar2d"],
        )

    def test_unique_run_dir_adds_next_available_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            requested = Path(temporary_directory) / "experiment"
            requested.mkdir()
            requested.with_name("experiment_1").mkdir()
            allocated = unique_run_dir(requested)
            self.assertEqual(allocated, requested.with_name("experiment_2"))
            self.assertTrue(allocated.is_dir())


if __name__ == "__main__":
    unittest.main()
