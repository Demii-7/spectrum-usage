"""TSS-LCD (the authors' released model) for the paper's configuration and timing tables: hyperparameters, search results, parameter counts and timings.

Per result set (1 minute 600-800 authors' defaults and tuned winner; 1 minute 2400-2600 defaults; 15 minutes 600-800 and 2400-2600 defaults and tuned winners) the mean over seeds
40-44 of the stored records (savgol off): the constructor/optimiser hyperparameters, epochs per stage, the context and future length, the diffusion steps, the
parameter counts of the three networks, the training seconds per stage and per epoch, the total wall seconds of a run, the inference seconds for all test windows
and per window, and the numbers of training and test windows. Also the search: the 16 candidates (the authors' defaults first), the winner and the validation
objective of the defaults and of the winner (``tuning/tsslcd_authors/...``). Hardware is not recorded in the records (the runs shared GPUs): do not quote a GPU model.
Writes ``evaluation/results/tsslcd_config_timings.csv`` and ``tsslcd_search_winners.csv``.

    python evaluation/analysis/tsslcd_config_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.ray.storage import spectrum_s3_client  # noqa: E402

SETS = [("1min", "600_800", "authors defaults", "tsslcd-authors-full"), ("1min", "600_800", "tuned winner", "tsslcd-authors-tuned"),
        ("1min", "2400_2600", "authors defaults", "tsslcd-authors-full-2400"),
        ("15min", "600_800", "authors defaults", "tsslcd-authors-15min-default"), ("15min", "600_800", "tuned winner", "tsslcd-authors-15min-tuned"),
        ("15min", "2400_2600", "authors defaults", "tsslcd-authors-15min-default-2400"), ("15min", "2400_2600", "tuned winner", "tsslcd-authors-15min-tuned-2400")]
SEARCHES = [("1min", "600_800", "tuning/tsslcd_authors/600_800"), ("15min", "600_800", "tuning/tsslcd_authors/15min/600_800"),
            ("15min", "2400_2600", "tuning/tsslcd_authors/15min/2400_2600")]


def main() -> None:
    c = spectrum_s3_client()
    rows = []
    for res, band, setting, prefix in SETS:
        recs = []
        for seed in range(40, 45):
            try:
                d = json.loads(c.get_object(Bucket="spectrum", Key=f"{prefix}/savgol-off/seed-{seed}.json")["Body"].read())
            except Exception:
                continue
            if d.get("ok"):
                recs.append(d)
        if not recs:
            continue
        m = lambda f: float(np.mean([f(r) for r in recs]))
        r0 = recs[0]
        rows.append({"resolution": res, "band": band, "setting": setting, "seeds": len(recs), "context_steps": r0.get("context"), "future_steps": r0.get("future"),
                     "latent_dim": r0["latent_dim"], "learning_rate": r0["learning_rate"], "batch_size": r0["batch_size"], "train_mask_ratio": r0["train_mask_ratio"],
                     "diffusion_steps": r0["n_timesteps"], "epochs_context": r0["epochs"]["ctx"], "epochs_future": r0["epochs"]["fut"], "epochs_diffusion": r0["epochs"]["diff"],
                     "params_context_ae": r0["parameters"]["ae_ctx"], "params_future_ae": r0["parameters"]["ae_fut"], "params_diffusion": r0["parameters"]["diffusion"],
                     "params_total": sum(r0["parameters"].values()), "train_windows": r0.get("train_windows"), "test_windows": r0.get("test_windows"),
                     "seconds_context_ae": m(lambda r: r["stage_ctx"]["seconds"]), "seconds_future_ae": m(lambda r: r["stage_fut"]["seconds"]),
                     "seconds_diffusion": m(lambda r: r["stage_diff"]["seconds"]),
                     "seconds_per_epoch_context": m(lambda r: r["stage_ctx"]["seconds_per_epoch"]), "seconds_per_epoch_diffusion": m(lambda r: r["stage_diff"]["seconds_per_epoch"]),
                     "training_seconds_total": m(lambda r: r["stage_ctx"]["seconds"] + r["stage_fut"]["seconds"] + r["stage_diff"]["seconds"]),
                     "inference_seconds_all_test_windows": m(lambda r: r["inference_seconds"]),
                     "inference_ms_per_window": 1000 * m(lambda r: r["inference_seconds"] / max(r["test_windows"], 1)),
                     "wall_seconds_run": m(lambda r: r["wall_seconds"]), "device": r0.get("device")})
    pd.DataFrame(rows).to_csv("evaluation/results/tsslcd_config_timings.csv", index=False)
    srows = []
    for res, band, prefix in SEARCHES:
        trials = []
        for pg in c.get_paginator("list_objects_v2").paginate(Bucket="spectrum", Prefix=prefix + "/trial-"):
            for o in pg.get("Contents", []):
                trials.append(json.loads(c.get_object(Bucket="spectrum", Key=o["Key"])["Body"].read()))
        ok = [t for t in trials if t.get("ok") and t.get("objective") is not None]
        if not ok:
            continue
        defaults = next((t for t in ok if t.get("index") == 0), None)
        win = min(ok, key=lambda t: (t["objective"], t["candidate_id"]))
        srows.append({"resolution": res, "band": band, "trials_ok": len(ok), "objective_defaults": defaults["objective"] if defaults else None,
                      "objective_winner": win["objective"], "winner_id": win["candidate_id"], **{f"winner_{k}": v for k, v in win["parameters"].items()},
                      "objective_worst": max(t["objective"] for t in ok), "trial_seconds_mean": float(np.mean([t["wall_seconds"] for t in ok]))})
    pd.DataFrame(srows).to_csv("evaluation/results/tsslcd_search_winners.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame(rows).T.to_string()); print(); print(pd.DataFrame(srows).T.to_string())


if __name__ == "__main__":
    main()
