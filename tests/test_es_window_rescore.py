import pytest

from training.ray.es_window_rescore import KEY, window_average


def test_window_average_is_the_mean_of_per_step_values():
    assert window_average({1: 1.0, 2: 2.0, 3: 6.0}, 3) == 3.0
    assert window_average({1: 5.0, 2: 9.0}, 1) == 5.0            # horizon 1: the window is the endpoint
    with pytest.raises(ValueError):
        window_average({1: 1.0, 3: 3.0}, 3)


def test_checkpoint_key_pattern():
    m = KEY.match("direct-tuning/electrosense/spec25/revinmlp/weights/h192/seed-41/electrosense_spec25_revinmlp.pt")
    assert m and m["ds"] == "spec25" and m["model"] == "revinmlp" and m["h"] == "192" and m["seed"] == "41"
    assert KEY.match("direct-tuning/electrosense/spec25/revinmlp/weights/h12/seed-40/x.pt.meta.json") is None
