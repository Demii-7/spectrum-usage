import numpy as np

from training.common.evaluation_integrated import _flattened_1d_normalization
from training.common.preprocessing import SequenceSegment


def test_flattened_1d_evaluation_uses_frequency_specific_normalization():
    segments = (
        SequenceSegment(0, 3, "site:frequency_0"),
        SequenceSegment(3, 6, "site:frequency_1"),
    )
    normalization, indices = _flattened_1d_normalization(
        {"mean_dbm": np.array([[10.0, 20.0]]), "std_dbm": np.array([[2.0, 3.0]])},
        segments,
        np.array([0, 4, 5]),
    )

    np.testing.assert_array_equal(indices, [0, 1, 1])
    np.testing.assert_array_equal(normalization["mean_dbm"].ravel(), [10.0, 20.0, 20.0])
    np.testing.assert_array_equal(normalization["std_dbm"].ravel(), [2.0, 3.0, 3.0])
