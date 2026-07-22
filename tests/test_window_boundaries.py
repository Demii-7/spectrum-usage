import numpy as np
import pandas as pd

from training.common.data import _flatten_segments
from training.common.data_sources import load_csv_sources
from training.common.preprocessing import SequenceSegment
from training.common.windowing import make_window_starts


def test_window_starts_do_not_cross_temporal_segments():
    starts = make_window_starts(
        n_timesteps=10,
        lookback=3,
        rollout_horizon=2,
        stride=1,
        segments=(SequenceSegment(0, 5, "run-a"), SequenceSegment(5, 10, "run-b")),
    )

    assert starts.tolist() == [0, 5]


def test_frequency_series_segments_prevent_cross_bin_windows(tmp_path):
    paths = []
    for file_index in range(2):
        path = tmp_path / f"run-{file_index}.csv"
        pd.DataFrame({"100.0": [1, 2, 3], "101.0": [4, 5, 6]}).to_csv(path, index=False)
        paths.append(path)

    source = load_csv_sources(paths, concat="rows")
    flattened, segments = _flatten_segments(source.data, source.segments)
    starts = make_window_starts(
        n_timesteps=len(flattened),
        lookback=2,
        rollout_horizon=1,
        stride=1,
        segments=segments,
    )

    assert [(segment.start, segment.end) for segment in segments] == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
    ]
    assert starts.tolist() == [0, 3, 6, 9]
    assert np.all(
        [
            any(
                start >= segment.start
                and start + 2 + 1 <= segment.end
                for segment in segments
            )
            for start in starts
        ]
    )
