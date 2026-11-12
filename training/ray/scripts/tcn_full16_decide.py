"""After the TCN failed search trials were rerun into ``tuning-tcn-full16``: pick the winner from all 16 trials, compare with the original winner
under ``tuning`` and, only if it differs, train the new winner's finals (seeds 40-44) under ``tuning-tcn-full16``. Nothing under ``tuning`` is touched.
Writes analysis-outputs/tcn_full16_trials.csv (every trial of both prefixes side by side) and tcn_full16_decision.json."""
import io
import json
import subprocess
import sys

import pandas as pd

sys.path.insert(0, "/workspace/spectrum-usage")
from training.ray.storage import spectrum_s3_client  # noqa: E402

BANDS = ("600_800", "2400_2600")
OLD, NEW = "tuning", "tuning-tcn-full16"


def trials(client, prefix):
    out = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            name = o["Key"].rsplit("/", 1)[-1]
            if name.startswith("trial-"):
                d = json.loads(client.get_object(Bucket="spectrum", Key=o["Key"])["Body"].read())
                out[d["candidate_id"]] = d
    return out


def main():
    client = spectrum_s3_client()
    rows, decision = [], {}
    for band in BANDS:
        grp = f"powder15/{band}/2d/temporalconvnet"
        old, new = trials(client, f"{OLD}/{grp}"), trials(client, f"{NEW}/{grp}")
        for cid, d in sorted(new.items(), key=lambda kv: kv[1]["index"]):
            rows.append(dict(band=band, index=d["index"], candidate_id=cid, ok=d["ok"], objective=d.get("objective"),
                             in_original_search_ok=bool(old.get(cid, {}).get("ok")), error=(d.get("error") or "")[:60],
                             parameters=json.dumps(d["parameters"])))
        ok_old = [d for d in old.values() if d.get("ok")]
        ok_new = [d for d in new.values() if d.get("ok")]
        w_old = min(ok_old, key=lambda d: (d["objective"], d["candidate_id"]))
        w_new = min(ok_new, key=lambda d: (d["objective"], d["candidate_id"]))
        decision[band] = dict(original_winner=w_old["candidate_id"], original_objective=w_old["objective"], trials_ok_original=len(ok_old),
                              full_winner=w_new["candidate_id"], full_objective=w_new["objective"], trials_ok_full=len(ok_new),
                              changed=w_old["candidate_id"] != w_new["candidate_id"])
        print(band, decision[band], flush=True)
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    client.put_object(Bucket="spectrum", Key="analysis-outputs/tcn_full16_trials.csv", Body=buf.getvalue().encode())
    client.put_object(Bucket="spectrum", Key="analysis-outputs/tcn_full16_decision.json", Body=json.dumps(decision, indent=1).encode())
    changed = [b for b, d in decision.items() if d["changed"]]
    if not changed:
        print("[decide] winner unchanged in both bands: no extra finals", flush=True)
        return
    cmd = ["python", "-u", "-m", "training.ray.tune_fixed", "--source", "powder15", "--datasets", *changed, "--models", "temporalconvnet:2d",
           "--output-prefix", NEW, "--gpu-fraction", "0.6", "--max-parallel", "5"]
    print("[decide] winner changed in", changed, "->", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd="/workspace/spectrum-usage"))


if __name__ == "__main__":
    main()
