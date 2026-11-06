from training.ray.restart_when_trials_done import count_trials, trials_complete


class FakeClient:
    def __init__(self, keys):
        self.keys = keys

    def get_paginator(self, _):
        keys = self.keys

        class P:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}]

        return P()


def test_trials_complete():
    assert trials_complete({"a": 16, "b": 17}, 16)
    assert not trials_complete({"a": 16, "b": 15}, 16)
    assert not trials_complete({}, 16)


def test_count_ignores_finals_and_other_files():
    keys = ["t/g/trial-a.json", "t/g/trial-b.json", "t/g/final/seed-40.json", "t/g/final/weights/x.pt", "t/g/summary.csv", "t/g2/trial-c.json"]
    assert count_trials(FakeClient(keys), "b", "t/g") == 2
