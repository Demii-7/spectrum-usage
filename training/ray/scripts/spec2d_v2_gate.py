"""Smoke gate for the v2 spectral ablation of the 2D models: exit 0 only if the smoke cells exist, are ok, and the trained model's out-of-region outputs sit at the
noise-floor level: for every horizon |mean out-of-region prediction - --noise-floor-db| <= --max-floor-diff dB (the evaluation scores against the RAW series, so the recorded target mean is the
real power of the masked bins and is NOT the training target; the training target is the noise floor, about -132 dBm at 600-800) and the in-region MAE is finite.

    python training/ray/scripts/spec2d_v2_gate.py --band 600_800 --models vanillalstm lstmattn --region R3
"""
import argparse
import json
import sys

sys.path.insert(0, "/workspace/spectrum-usage")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--region", default="R3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="oct2026-spectral-context-15min-v2")
    p.add_argument("--noise-floor-db", type=float, default=-132.0)
    p.add_argument("--max-floor-diff", type=float, default=3.0)
    a = p.parse_args()
    from training.ray.storage import spectrum_s3_client

    client = spectrum_s3_client()
    bad = 0
    for model in a.models:
        key = f"{a.prefix}/{a.band}/{model}/{a.region}-region_only_fullloss-seed-{a.seed}.json"
        try:
            rec = json.loads(client.get_object(Bucket="spectrum", Key=key)["Body"].read())
        except Exception:
            print(f"[gate] {model}: no smoke record at {key}", flush=True)
            bad += 1
            continue
        if not rec.get("ok"):
            print(f"[gate] {model}: smoke cell failed: {rec.get('error')}", flush=True)
            bad += 1
            continue
        for h in (4, 12, 24, 48):
            diff = abs(rec[f"oor_pred_mean_db_t{h}"] - a.noise_floor_db)
            finite = rec[f"test_mae_db_t{h}"] == rec[f"test_mae_db_t{h}"]
            status = "ok" if diff <= a.max_floor_diff and finite else "BAD"
            bad += status == "BAD"
            print(f"[gate] {model} {a.band} {a.region} t+{h}: out-of-region pred mean {rec[f'oor_pred_mean_db_t{h}']:.2f} dB vs noise floor {a.noise_floor_db:.1f} dB "
                  f"(|diff| {diff:.2f}) -> {status}; in-region MAE {rec[f'test_mae_db_t{h}']:.3f}; raw out-of-region target mean {rec[f'oor_target_mean_db_t{h}']:.2f}", flush=True)
    print(f"[gate] {'PASSED' if not bad else 'FAILED'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
