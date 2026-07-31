from pathlib import Path
import tarfile

import pytest

from training.common.spatial_ablation_eval import _permutation
from training.ray.spatial_ablation import (
    campaign_cells,
    extract_checkpoint,
    unique_permutation_seeds,
)


def test_campaign_cells_share_permutations_across_seeds():
    cells = campaign_cells(["model"], [40, 41], ["geometry_correct", "geometry_permuted"], 3)
    assert len(cells) == 8
    for seed in (40, 41):
        permutations = [
            cell["permutation_seed"]
            for cell in cells
            if cell["seed"] == seed and cell["condition"] == "geometry_permuted"
        ]
        assert permutations == [10_000, 10_001, 10_002]


def test_campaign_rejects_unknown_condition():
    with pytest.raises(ValueError, match="Unknown conditions"):
        campaign_cells(["model"], [42], ["train"], 0)


def test_one_hundred_permutations_are_unique_and_nonidentity():
    seeds = unique_permutation_seeds(100)
    assignments = {tuple(_permutation(6, seed)) for seed in seeds}
    assert len(seeds) == len(assignments) == 100
    assert tuple(range(6)) not in assignments


def test_permutation_shards_do_not_overlap():
    first = unique_permutation_seeds(35)
    second = unique_permutation_seeds(65, offset=35)
    assert first + second == unique_permutation_seeds(100)


def test_extract_checkpoint_uses_expected_archive_member(tmp_path: Path):
    source = tmp_path / "powder_600_800_t4t6_convlstm.pt"
    source.write_bytes(b"checkpoint")
    archive = tmp_path / "checkpoints.tgz"
    member = "checkpoints/4d/convlstm/seed-42/powder_600_800_t4t6_convlstm.pt"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname=member)
    extracted = extract_checkpoint(archive, "convlstm", 42, tmp_path / "out")
    assert extracted.read_bytes() == b"checkpoint"
