from training.ray.upload_local_checkpoints import classify


def test_classify_never_overwrites():
    assert classify(10, None) == "missing"
    assert classify(10, 10) == "present"
    assert classify(10, 11) == "conflict"      # a different object under the same key is reported, not replaced
