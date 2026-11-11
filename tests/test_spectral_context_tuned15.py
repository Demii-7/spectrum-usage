import io
import json

import pytest

sf = pytest.importorskip("training.ray.spectral_context_fixed")


class FakeClient:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


def test_tuned15_entries_read_the_winner_from_the_first_finished_final():
    objects = {"tuning/powder15/600_800/2d/vanillalstm/final/seed-40.json": {"ok": False},
               "tuning/powder15/600_800/2d/vanillalstm/final/seed-41.json": {"ok": True, "parameters": {"lr": 0.001}, "candidate_id": "c9", "search_objective": 1.2}}
    entries = sf.tuned15_entries(FakeClient(objects), "b", "600_800", ["vanillalstm"])
    assert entries[0]["parameters"] == {"lr": 0.001} and entries[0]["config_source"] == "tuned15:c9" and entries[0]["rep"] == "2d"


def test_tuned15_entries_fail_loudly_without_a_final():
    with pytest.raises(FileNotFoundError):
        sf.tuned15_entries(FakeClient({}), "b", "600_800", ["lstmattn"])


def test_jobs_carry_the_tuned_parameters_and_record_keys_are_per_cell():
    entries = [{"rep": "2d", "model": "lstmattn", "parameters": {"lr": 0.002}, "config_source": "tuned15:c3"}]
    regions = [{"band_id": "R1", "start_mhz": 600.5, "end_mhz": 607.5}]
    jobs = sf.make_jobs(entries, regions, [40, 41])
    assert len(jobs) == 4 and all(j["parameters"] == {"lr": 0.002} for j in jobs)
    assert sf.record_key("p/", jobs[0]) == "p/lstmattn/R1-full_context-seed-40.json"
