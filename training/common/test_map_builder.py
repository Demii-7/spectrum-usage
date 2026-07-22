import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.common.map_builder import load_4d, load_map_layout, prepare_4d_partitions


def _sources(tmp_path: Path, names=("alpha", "beta"), coordinates=None):
    coordinates = coordinates or ((-73.0, 40.0), (-73.001, 40.001))
    files = []
    locations = []
    for index, (name, (lon, lat)) in enumerate(zip(names, coordinates)):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "power.csv"
        pd.DataFrame({"100.0": [index + 1.0, index + 2.0], "101.0": [3.0, 4.0]}).to_csv(path, index=False)
        files.append(path)
        locations.append({"name": name, "longitude": lon, "latitude": lat})
    location_path = tmp_path / "locations.json"
    location_path.write_text(json.dumps({"endpoints": locations}), encoding="utf-8")
    return files, location_path


def _load(tmp_path, files, locations, name="sample", **kwargs):
    return load_4d(
        files, name, tmp_path / "maps", locations, "endpoints",
        {"height": 2, "width": 3}, **kwargs,
    )


def _write(path, values, timestamps=None):
    frame = pd.DataFrame({"100.0": values})
    if timestamps is not None:
        frame.insert(0, "timestamp_utc", timestamps)
    frame.to_csv(path, index=False)


def test_incompatible_cache_rebuilds_from_available_csvs(tmp_path):
    files, locations = _sources(tmp_path)
    first = _load(tmp_path, files, locations)
    pd.DataFrame({"100.0": [20.0, 21.0], "101.0": [22.0, 23.0]}).to_csv(files[0], index=False)

    rebuilt = _load(tmp_path, files, locations)

    assert not np.array_equal(first.data, rebuilt.data)


@pytest.mark.parametrize(
    "names, coordinates, match",
    [
        (("beta", "alpha"), None, "coordinates"),
        (("alpha",), ((-73.0, 40.0),), "names/order"),
        (("alpha", "beta", "gamma"), ((-73.0, 40.0), (-73.001, 40.001), (-73.002, 40.002)), "names/order"),
        (("alpha", "beta"), ((-73.0, 40.0), (-73.01, 40.001)), "coordinates"),
    ],
)
def test_test_map_rejects_site_layout_changes(tmp_path, names, coordinates, match):
    train_files, train_locations = _sources(tmp_path / "train")
    _load(tmp_path, train_files, train_locations, name="train")
    layout = load_map_layout(tmp_path / "maps" / "train.npz")
    test_files, test_locations = _sources(tmp_path / "test", names, coordinates)

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, test_files, test_locations, name="test", expected_layout=layout)


def test_cache_validates_permutation_and_grid_settings(tmp_path):
    files, locations = _sources(tmp_path)
    _load(tmp_path, files, locations)

    _load(tmp_path, files, locations, permute=True, permute_seed=7)
    with np.load(tmp_path / "maps" / "sample.npz", allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["permute"] is True
    assert metadata["permute_seed"] == 7


def test_prepare_partitions_excludes_site_with_long_outage(tmp_path):
    train_files, locations = _sources(tmp_path / "train")
    test_files, _ = _sources(tmp_path / "test")
    pd.DataFrame({"100.0": [1.0] * 4, "101.0": [3.0] * 4}).to_csv(
        train_files[0], index=False
    )
    pd.DataFrame({"100.0": [1.0, np.nan, np.nan, 2.0], "101.0": [3.0] * 4}).to_csv(
        train_files[1], index=False
    )

    partitions, selected, excluded = prepare_4d_partitions(
        {"train": train_files, "test": test_files}, locations, "endpoints", None, None, 1
    )

    assert selected == ["alpha"]
    assert excluded == ["beta"]
    assert {key: [path.parent.name for path in value] for key, value in partitions.items()} == {
        "train": ["alpha"],
        "test": ["alpha"],
    }


def test_prepare_partitions_rejects_different_site_sets(tmp_path):
    train_files, locations = _sources(tmp_path / "train")
    test_files, _ = _sources(tmp_path / "test", names=("alpha",))

    with pytest.raises(ValueError, match="canonical site sets differ"):
        prepare_4d_partitions(
            {"train": train_files, "test": test_files}, locations, "endpoints", None, None, 1
        )


def test_leading_missing_prefix_trims_instead_of_excluding(tmp_path):
    train_files, locations = _sources(tmp_path / "train")
    test_files, _ = _sources(tmp_path / "test")
    _write(train_files[0], [np.nan, np.nan, 1.0, 2.0])
    _write(train_files[1], [3.0, 4.0, 5.0, 6.0])

    partitions, selected, excluded = prepare_4d_partitions(
        {"train": train_files, "test": test_files}, locations, "endpoints", [100.0], None, 1
    )
    source = _load(tmp_path, partitions["train"], locations, frequency_bins=[100.0])

    assert selected == ["alpha", "beta"]
    assert excluded == []
    assert len(source.data) == 2


def test_short_internal_gap_is_forward_filled_without_compression(tmp_path):
    files, locations = _sources(tmp_path, names=("alpha",), coordinates=((-73.0, 40.0),))
    _write(files[0], [1.0, np.nan, 5.0])

    source = _load(tmp_path, files, locations, frequency_bins=[100.0], outage_threshold=1)

    assert source.data[:, 0, 0, 0].tolist() == pytest.approx([1.0, 1.0, 5.0])


@pytest.mark.parametrize("gap, is_excluded", [(2, True), (1, False)])
def test_post_start_outage_threshold_filters_both_partitions(tmp_path, gap, is_excluded):
    train_files, locations = _sources(tmp_path / "train")
    test_files, _ = _sources(tmp_path / "test")
    values = [1.0] + [np.nan] * gap + [2.0]
    _write(train_files[1], values)
    _write(train_files[0], [3.0] * len(values))
    _write(test_files[0], [3.0] * len(values))
    _write(test_files[1], [4.0] * len(values))

    partitions, selected, excluded = prepare_4d_partitions(
        {"train": train_files, "test": test_files}, locations, "endpoints", [100.0], None, 1
    )

    assert ("beta" in excluded) is is_excluded
    assert ("beta" in selected) is not is_excluded
    for partition in ("train", "test"):
        assert (any(path.parent.name == "beta" for path in partitions[partition])) is not is_excluded


def test_common_timestamp_rows_stay_aligned_after_trim_and_fill(tmp_path):
    files, locations = _sources(tmp_path)
    _write(files[0], [np.nan, 10.0, np.nan, 30.0], pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC"))
    _write(files[1], [20.0, 40.0, 50.0], pd.to_datetime([
        "2024-01-01T00:01Z", "2024-01-01T00:02Z", "2024-01-01T00:03Z"
    ]))

    source = _load(tmp_path, files, locations, frequency_bins=[100.0], outage_threshold=1)

    expected = pd.to_datetime([
        "2024-01-01T00:01Z", "2024-01-01T00:02Z", "2024-01-01T00:03Z"
    ])
    assert source.timestamps.tolist() == expected.tolist()
    assert len(source.data) == 3
    with np.load(tmp_path / "maps" / "sample.npz", allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["timeline_semantics"] == "aligned-common-start-causal-ffill-v1"
    assert metadata["timeline_start"].startswith("2024-01-01 00:01:00")
