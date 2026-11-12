"""Does the TSS-LCD search winner still match the hyperparameters of the finals? Writes a JSON verdict to analysis-outputs.

    python evaluation/analysis/tsslcd_winner_check.py --resolution 15min --band 2400_2600

Reads every ``trial-*.json`` under the tuning prefix, picks the winner exactly as ``tune_tsslcd.pick_winner`` does, and compares it with the
``hparams`` of the five savgol-off finals under the output prefix. Verdict ``unchanged`` (all finals use the winner), ``changed`` (some do not; the
new winner and its validation objective are listed, finals are NOT rerun here) or ``incomplete`` (fewer than --budget trials ok).
"""
import argparse
import json
import sys

sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", default="15min")
    p.add_argument("--band", default="2400_2600")
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--out-key", default=None)
    a = p.parse_args()
    from training.ray.storage import spectrum_s3_client
    from training.ray.tune_tsslcd import pick_winner, read_trials

    c = spectrum_s3_client()
    t_prefix = f"tuning/tsslcd_authors/{a.resolution}/{a.band}" if a.resolution == "15min" else f"tuning/tsslcd_authors/{a.band}"
    suffix = "" if a.band == "600_800" else f"-{a.band.split('_')[0]}"
    o_prefix = (f"tsslcd-authors-15min-tuned{suffix}" if a.resolution == "15min" else f"tsslcd-authors-tuned{suffix}")
    trials = read_trials(c, "spectrum", t_prefix)
    ok = [r for r in trials if r.get("ok")]
    w = pick_winner(trials)
    finals = []
    for s in range(40, 45):
        try:
            r = json.loads(c.get_object(Bucket="spectrum", Key=f"{o_prefix}/savgol-off/seed-{s}.json")["Body"].read())
            h = r.get("hparams", {})
            same = (r.get("ok") and h.get("latent_dim") == w["parameters"]["latent_dim"] and abs(h["learning_rate"] - w["parameters"]["learning_rate"]) < 1e-12
                    and h.get("batch_size") == w["parameters"]["batch_size"] and abs(h["train_mask_ratio"] - w["parameters"]["train_mask_ratio"]) < 1e-12)
            finals.append({"seed": s, "ok": bool(r.get("ok")), "uses_winner": bool(same)})
        except Exception as e:
            finals.append({"seed": s, "ok": False, "uses_winner": False, "error": str(e)[:80]})
    verdict = "incomplete" if len(ok) < a.budget else ("unchanged" if all(f["uses_winner"] for f in finals) else "changed")
    out = {"resolution": a.resolution, "band": a.band, "trials_ok": len(ok), "budget": a.budget, "winner_id": w["candidate_id"],
           "winner_objective": w["objective"], "winner_parameters": w["parameters"], "finals": finals, "verdict": verdict,
           "trial_seconds_mean": sum(r["wall_seconds"] for r in ok) / max(1, len(ok))}
    print(json.dumps(out, indent=1))
    key = a.out_key or f"analysis-outputs/tsslcd{a.resolution.replace('min', '')}_{a.band.split('_')[0]}_winner_check.json"
    c.put_object(Bucket="spectrum", Key=key, Body=json.dumps(out, indent=1).encode())
    print("uploaded", key)


if __name__ == "__main__":
    main()
