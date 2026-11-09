from training.ray.job_watchdog import chain_ids, decide


def test_chain_and_decisions():
    spec = {"id": "job", "entrypoint": "x", "max_restarts": 2}
    assert chain_ids("job", {"job", "job-r1", "other"}) == ["job", "job-r1"]
    assert decide(spec, {"job": "RUNNING"}) is None
    assert decide(spec, {"job": "SUCCEEDED"}) is None
    assert decide(spec, {"job": "STOPPED"}) is None                      # stopped on purpose: never resubmitted
    assert decide(spec, {"job": "FAILED"}) == "job-r1"
    assert decide(spec, {"job": "FAILED", "job-r1": "RUNNING"}) is None
    assert decide(spec, {"job": "FAILED", "job-r1": "FAILED"}) == "job-r2"
    assert decide(spec, {"job": "FAILED", "job-r1": "FAILED", "job-r2": "FAILED"}) is None   # restarts used up
    assert decide(spec, {}) is None                                      # not started: left alone
