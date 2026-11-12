from training.ray.spatial15 import CORE, SIX, variants


def test_solo_variants_one_receiver_each():
    solo = variants(["solo"])
    assert [v["name"] for v in solo] == ["solo-humanities", "solo-madsen", "solo-moran", "solo-sagepoint"]
    for v, node in zip(solo, CORE):
        assert v["nodes"] == [node] and v["grid"] == {} and v["solo"] is True


def test_existing_experiments_unchanged():
    s2 = variants(["s2"])
    assert len(s2) == 8 and s2[0]["name"] == "core4-p2" and all("solo" not in v for v in s2)
    assert len(variants(["s3"])) == 6 and len(variants(["s5"])) == 6
    assert len(variants(["s2", "solo"])) == 12
