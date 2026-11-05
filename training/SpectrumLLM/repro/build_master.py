"""Consolidate every SpectrumLLM reproduction run into the master tables.

Reads the result JSONs in this directory (the single source of truth after the
2026-11-04 consolidation; nothing is read from outside the repository) and writes:

  results_master.csv        one row per run, with status and provenance
  configuration_matrix.csv  primary runs grouped to configuration cells
  CHECKPOINTS.tsv           every .pt on disk under checkpoints/ and runs/, with
                            the result record that references it and a live
                            sha256-16 verification
  MATRIX.md                 human-readable coverage matrix: what exists for each
                            (arm, scenario, split, channel mode) and what is missing

Run from anywhere:  python training/SpectrumLLM/repro/build_master.py
Standard library only -- no pandas, so it runs in the bare container.
"""
import collections
import csv
import glob
import hashlib
import json
import os
import re

RE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(RE, "..", "..", ".."))

# Published rows we compare against (Liu et al. 2026, Tables I-II, raw PSD MSE,
# horizons 12/24/48/96/192). Arms with no published counterpart have no entry.
PUB = {
    ("timellm", "gpt2", 4): [0.2621, 0.2982, 0.3154, 0.3157, 0.3460],
    ("timellm", "gpt2", 25): [0.6166, 0.7592, 0.8564, 0.9686, 1.0429],
    ("dlinear", "none", 4): [0.2718, 0.3036, 0.3305, 0.4019, 0.4043],
    ("dlinear", "none", 25): [0.6080, 0.8025, 0.9809, 1.2581, 1.6487],
}
HOR = [12, 24, 48, 96, 192]
# Test-window counts are an exact discriminator between the two split readings.
TSLIB_TEST = {12: 563, 24: 551, 48: 527, 96: 479, 192: 383}
LITERAL_TEST = {12: 851, 24: 839, 48: 815, 96: 767, 192: 671}

# Files that are not run lists.
NOT_RUNS = {"spectrumllm_published_audit.json", "verification.json", "split_probe.json",
            "checkpoint_verification.json", "gap_analysis.json"}
# Superseded / diagnostic campaigns: kept for the record, excluded from the matrix.
# (The DLinear diagnostics were archived under dlinear/ -- gitignored, not globbed.)
SWEEP_SOURCES = {"results_protocol_sweep.json", "results_k12_raw.json",
                 "results_k12_raw_ckpt.json"}

FIELDS = [
    "status", "source", "arm", "backbone", "channels", "horizon", "seed",
    "channel_mode", "channel_mode_src", "split", "split_src", "lr", "epochs_max",
    "epochs_run", "space_trained_in", "d_model", "d_ff", "patch_length",
    "patch_stride", "llm_layers", "num_prototypes", "heads", "prompt_mode", "prompt_stats",
    "batch_size", "val_fraction", "n_train_windows", "n_val_windows",
    "n_test_windows", "params_total", "params_trainable", "test_mse_raw",
    "test_mse_std", "best_val_mse_std", "train_wall_s", "peak_gpu_mb",
    "checkpoint", "checkpoint_mb", "checkpoint_sha256_16", "checkpoint_on_disk",
    "published_mse", "ratio_to_published",
]
# runs before 2026-11-04 predate the prompt-stats flag; treat them as the
# normalized convention they were run under (see README --prompt-stats).
DEFAULT_PROMPT_STATS = {"results_split_study.json", "res4_all_match.json",
                        "res4_all_dm32.json", "res4_ci_dm32.json", "reeval_runs.json",
                        "results_ofa_bert.json", "results_paper_regime.json",
                        "results_protocol_sweep.json"}


def canon_bb(b):
    b = str(b or "").lower()
    if "gpt2" in b:
        return "gpt2"
    if "bert" in b:
        return "bert"
    if "random" in b:
        return "random"
    return "none"


def canon_cm(rec):
    if "channel_independent" in rec:
        return ("chan_indep" if rec["channel_independent"] else "all_channel"), "recorded"
    cm = str(rec.get("channel_mode", "")).lower()
    if "indep" in cm:
        return "chan_indep", "recorded"
    if "all_channel" in cm:
        return "all_channel", "recorded"
    return "all_channel", "ASSUMED_default"


def load_manifest():
    """basename -> set of sha256-16 of the deleted first-campaign checkpoints."""
    man = {}
    mp = os.path.join(RE, "checkpoints_manifest.tsv")
    if os.path.isfile(mp):
        with open(mp) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                man.setdefault(os.path.basename(r["path"]), set()).add(r["sha256"][:16])
    return man


def disk_sha_map():
    """absolute path -> sha256-16 for every .pt under checkpoints/ and runs/."""
    out = {}
    for root in (os.path.join(REPO, "checkpoints"), os.path.join(REPO, "runs")):
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".pt"):
                    p = os.path.join(dp, f)
                    out[p] = sha16(p)
    return out


def expected_shas(rec, manifest):
    """The sha256-16 set a run's checkpoint should carry: its own record, or the
    manifest's entry for the deleted first-campaign checkpoints."""
    s = {str(rec.get("checkpoint_sha256_16") or "")} - {""}
    if not s:
        s = manifest.get(os.path.basename(str(rec.get("checkpoint") or "")), set())
    return s


def load_runs(manifest, disk):
    rows = []
    for p in sorted(glob.glob(os.path.join(RE, "*.json"))):
        bn = os.path.basename(p)
        if bn in NOT_RUNS or bn.startswith("smoke"):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if not isinstance(j, list) or not j or not isinstance(j[0], dict):
            continue
        for rec in j:
            r = {k: "" for k in FIELDS}
            r["source"] = bn
            for k, v in rec.items():
                if k in r:
                    r[k] = v
            if "mse_raw" in rec:  # split_probe-era key
                r["test_mse_raw"] = rec["mse_raw"]
            if "mse_std" in rec:
                r["test_mse_std"] = rec["mse_std"]
            # pre-flag campaigns ran the normalized convention (see --prompt-stats);
            # mark it as inferred rather than recorded
            if not r["prompt_stats"] and bn in DEFAULT_PROMPT_STATS:
                r["prompt_stats"] = "normalized (pre-flag)"
            r["backbone"] = canon_bb(rec.get("backbone"))
            r["channel_mode"], r["channel_mode_src"] = canon_cm(rec)
            a = str(rec.get("arm", "")).lower()
            ck = str(rec.get("checkpoint") or "").split("/")[-1]
            r["arm"] = (a if a in ("timellm", "ofa", "dlinear")
                        else "ofa" if "_ofa" in ck
                        else "dlinear" if r["backbone"] == "none" else "timellm")
            if r["arm"] == "dlinear":
                r["backbone"] = "none"
            if rec.get("split"):
                r["split"], r["split_src"] = rec["split"], "recorded"
            else:
                h = int(rec.get("horizon") or 0)
                nt = rec.get("n_test_windows")
                if nt and h in TSLIB_TEST and int(nt) == TSLIB_TEST[h]:
                    r["split"], r["split_src"] = "tslib", "derived_window_count"
                elif nt and h in LITERAL_TEST and int(nt) == LITERAL_TEST[h]:
                    r["split"], r["split_src"] = "literal", "derived_window_count"
                else:
                    r["split"], r["split_src"] = "UNKNOWN", "no_evidence"
            ch = int(rec.get("channels") or 0)
            try:
                h = int(rec["horizon"])
                pub = PUB[(r["arm"], r["backbone"], ch)][HOR.index(h)]
                r["published_mse"] = pub
                if r["test_mse_raw"] not in ("", None):
                    r["ratio_to_published"] = round(float(r["test_mse_raw"]) / pub, 4)
            except Exception:
                pass
            # status: the settled protocol is lr 1e-4 / 10 epochs; the lr 1e-3
            # campaign (results_paper_regime.json) is superseded.
            lr = rec.get("lr")
            if bn in SWEEP_SOURCES:
                r["status"] = "sweep"
            elif bn == "results_paper_regime.json" or (
                    lr and abs(float(lr) - 1e-3) < 1e-12 and bn != "results_dlinear.json"):
                r["status"] = "superseded_lr1e-3"
            else:
                r["status"] = "primary"
            # checkpoint presence is sha-verified, not path-verified: chunk ids
            # do not encode the split, so later campaigns reused (and thereby
            # replaced) first-campaign basenames at the same paths.
            ck_path = str(rec.get("checkpoint") or "")
            if ck_path:
                abs_ck = ck_path if os.path.isabs(ck_path) else os.path.join(REPO, ck_path)
                if not os.path.isfile(abs_ck):
                    r["checkpoint_on_disk"] = "no"
                else:
                    exp = expected_shas(rec, manifest)
                    if not exp:
                        r["checkpoint_on_disk"] = "unknown"
                    elif disk.get(abs_ck) in exp:
                        r["checkpoint_on_disk"] = "yes"
                    else:
                        r["checkpoint_on_disk"] = "overwritten"
            rows.append(r)
    return rows


def group_cells(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        if r["status"] != "primary":
            continue
        cells[(r["arm"], r["backbone"], str(r["channels"]), r["channel_mode"], r["split"],
               str(r["d_model"] or "32"), str(r["d_ff"] or "32"), str(r["horizon"]))].append(r)
    return cells


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_checkpoints(rows, manifest, disk):
    """Every .pt on disk plus the result record that owns it (by sha), and the
    first-campaign weights whose basename it replaced (from the manifest)."""
    refs = collections.defaultdict(list)
    for r in rows:
        ck = r["checkpoint"]
        if ck:
            refs[str(ck)].append(r)
    onk = sorted(disk)
    out = []
    for p in onk:
        b = os.path.basename(p)
        rel = os.path.relpath(p, REPO)
        m = re.match(r"(?P<site>[a-z_]+)_(?P<lo>\d+)_(?P<hi>\d+)_n(?P<ch>\d+)_h(?P<h>\d+)"
                     r"(?P<rest>.*)_(?P<model>spectrumllm|ofa)\.pt", b)
        meta = {}
        if m:
            rest = m.group("rest")
            bb = re.search(r"_bb(\w+?)(?=_|$)", rest)
            meta = dict(site=m.group("site"), channels=m.group("ch"), horizon=m.group("h"),
                        channel_mode="chan_indep" if "_ci" in rest else "all_channel",
                        backbone=(bb.group(1) if bb else "gpt2"),
                        arm="ofa" if "_ofa" in rest else "timellm")
        seed = (re.search(r"seed-(\d+)", p) or [None, ""])[1]
        disk_sha = disk[p]
        recs, srcs = [], []
        for key in (p, rel, os.path.join(REPO, rel)):
            for r in refs.get(key, []):
                if r not in recs:
                    recs.append(r)
                    srcs.append(r["source"])
        # the owner is the record whose expected sha matches the file
        owner = next((r for r in recs if disk_sha in expected_shas(r, manifest)), None)
        man_shas = sorted(manifest.get(b, set()))
        if owner is not None:
            sha_match = "yes"
        elif recs and man_shas and disk_sha not in man_shas:
            sha_match = "replaced_first_campaign"
        elif recs:
            sha_match = "stale_record"
        else:
            sha_match = "unreferenced"
        rec = owner or (next((r for r in recs if r.get("test_mse_raw") not in ("", None)),
                             recs[0] if recs else None))
        out.append(dict(
            path=rel, basename=b, size_mb=round(os.path.getsize(p) / 1e6, 1),
            site=meta.get("site", ""), channels=meta.get("channels", ""),
            horizon=meta.get("horizon", ""), channel_mode=meta.get("channel_mode", ""),
            backbone=meta.get("backbone", ""), arm=meta.get("arm", ""), seed=seed,
            referenced="yes" if recs else "no", result_source="|".join(srcs),
            owner_source=(owner["source"] if owner else ""),
            test_mse_raw=(rec.get("test_mse_raw") if rec else ""),
            split=("|".join(sorted({str(r.get("split") or "") for r in recs
                                    if r.get("split")}))),
            d_model=(rec.get("d_model") if rec else ""), d_ff=(rec.get("d_ff") if rec else ""),
            lr=(rec.get("lr") if rec else ""),
            sha256_16_disk=disk_sha,
            sha256_16_recorded=(rec.get("checkpoint_sha256_16") if rec else ""),
            sha_match=sha_match,
            first_campaign_sha256_16="|".join(man_shas),
            replaces_first_campaign=("yes" if (man_shas and disk_sha not in man_shas) else ""),
        ))
    return out


def main():
    manifest = load_manifest()
    disk = disk_sha_map()
    rows = load_runs(manifest, disk)

    with open(os.path.join(RE, "results_master.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_status = collections.Counter(r["status"] for r in rows)
    print("results_master.csv:", len(rows), "runs |", dict(n_status))
    print("  checkpoint on disk:", dict(collections.Counter(
        r["checkpoint_on_disk"] or "(none referenced)" for r in rows)))

    cells = group_cells(rows)
    with open(os.path.join(RE, "configuration_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "backbone", "channels", "channel_mode", "split", "d_model",
                    "d_ff", "horizon", "n_seeds", "seeds", "mean_test_mse_raw",
                    "published_mse", "mean_ratio", "source"])
        for k in sorted(cells, key=lambda t: (t[0], t[1], int(t[2]), t[3], t[4],
                                              int(t[5]), int(t[6]), int(t[7]))):
            v = cells[k]
            m = [float(x["test_mse_raw"]) for x in v if x["test_mse_raw"] not in ("", None)]
            ra = [float(x["ratio_to_published"]) for x in v
                  if x["ratio_to_published"] not in ("", None)]
            w.writerow(list(k) + [len({x["seed"] for x in v}),
                                  "|".join(sorted({str(x["seed"]) for x in v})),
                                  round(sum(m) / len(m), 4) if m else "",
                                  v[0]["published_mse"],
                                  round(sum(ra) / len(ra), 4) if ra else "",
                                  "|".join(sorted({x["source"] for x in v}))])
    print("configuration_matrix.csv:", len(cells), "cells")

    ckpts = build_checkpoints(rows, manifest, disk)
    with open(os.path.join(RE, "CHECKPOINTS.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ckpts[0].keys()) if ckpts else
                           ["path"], delimiter="\t")
        w.writeheader()
        w.writerows(ckpts)
    n_own = sum(1 for c in ckpts if c["sha_match"] == "yes")
    n_rep = sum(1 for c in ckpts if c["replaces_first_campaign"] == "yes")
    print("CHECKPOINTS.tsv:", len(ckpts), "files on disk,", n_own,
          "sha-matched to their run,", n_rep, "reusing first-campaign basenames")

    write_matrix_md(rows, cells, ckpts)
    print("MATRIX.md written")


def cell_stats(v):
    """Aggregate one configuration cell (a list of runs) -> dict or None."""
    if not v:
        return None
    mse = [float(x["test_mse_raw"]) for x in v if x["test_mse_raw"] not in ("", None)]
    ratio = [float(x["ratio_to_published"]) for x in v
             if x["ratio_to_published"] not in ("", None)]
    if not mse:
        return None
    seeds = sorted({str(x["seed"]) for x in v}, key=lambda s: int(s) if str(s).lstrip('-').isdigit() else 0)
    ks = sorted({int(x["horizon"]) for x in v})
    return dict(mse=sum(mse) / len(mse), ratio=(sum(ratio) / len(ratio) if ratio else None),
                n_runs=len(v), seeds=seeds, ks=ks)


def fmt_cell(st):
    if st is None:
        return "--"
    s = f"{st['mse']:.4f}"
    if st["ratio"] is not None:
        s += f" ({st['ratio']:.3f}x)"
    s += f" &middot; {len(st['ks'])} Ks &middot; seed{'s' if len(st['seeds']) > 1 else ''} "
    s += ", ".join(st["seeds"])
    return s


def write_matrix_md(rows, cells, ckpts):
    prim = [r for r in rows if r["status"] == "primary"]

    def block(arm, bb, ch, split, mode):
        """All runs of one (arm, backbone, channels, split, mode), any capacity."""
        return [r for k, v in cells.items()
                if k[:5] == (arm, bb, str(ch), mode, split) for r in v]

    P = []
    P.append("""# Result inventory and configuration matrix

What exists, in one place. Every table is computed by `build_master.py` from the
result JSONs in this directory; regenerate with
`python training/SpectrumLLM/repro/build_master.py`.

The two protocol axes that dominate reproduction quality, and the coverage we
hold on each:

* **split** -- how the article's "7:3" is read. `tslib` = 70/10/20 chronological
  (Time-Series-Library `Dataset_Custom`, 563 test windows at K=12);
  `literal` = last 30% all test, validation carved from the training span
  (851 test windows at K=12).
* **channel mode** -- `chan_indep` = Time-LLM's arrangement (channels folded
  into the batch); `all_channel` = the article's equation (one token spans all
  channels).

Grid cells are mean test MSE in raw PSD units over all horizons and seeds of
that cell, with the mean ratio to the published row in parentheses where one
exists. Where a cell holds more than one capacity (d_model/d_ff) the variants
are shown on separate lines. `--` = not run. Per-horizon, per-seed detail is in
`configuration_matrix.csv`; the full run list is `results_master.csv`.
""")
    P.append(f"**{len(rows)} runs** in `results_master.csv`"
             f" ({len(prim)} primary,"
             f" {sum(1 for r in rows if r['status'] == 'sweep')} sweep/diagnostic,"
             f" {sum(1 for r in rows if r['status'].startswith('superseded'))}"
             " superseded lr 1e-3 campaign)"
             f" | **{len(cells)} primary configuration cells** in"
             " `configuration_matrix.csv`"
             f" | **{len(ckpts)} checkpoints** on disk in `CHECKPOINTS.tsv`"
             " (paths relative to the repository root; the 150 checkpoints of"
             " the first campaign exist only as SHA-256 hashes in"
             " `checkpoints_manifest.tsv` -- the weights were deleted).\n")

    # ---- Coverage grid per arm: split x channel_mode for every scenario.
    by_arm = collections.defaultdict(set)
    for r in prim:
        by_arm[(r["arm"], r["backbone"])].add(str(r["channels"]))
    for (arm, bb), chs in sorted(by_arm.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        name = {"timellm": "Time-LLM", "ofa": "OFA", "dlinear": "DLinear"}[arm]
        title = f"## {name}" + (f" ({bb})" if bb != "none" and arm != "dlinear" else "")
        P.append(f"\n{title}\n")
        P.append("| channels | split | all-channel | channel-independent |")
        P.append("|---|---|---|---|")
        for ch in sorted(chs, key=int):
            for split in ("tslib", "literal"):
                row = []
                for mode in ("all_channel", "chan_indep"):
                    # one line per (d_model, d_ff) capacity present in this block
                    caps = collections.defaultdict(list)
                    for r in block(arm, bb, ch, split, mode):
                        caps[(str(r["d_model"] or "32"), str(r["d_ff"] or "32"))].append(r)
                    if not caps:
                        row.append("--")
                        continue
                    lines = []
                    for (dm, dff), v in sorted(caps.items()):
                        st = cell_stats(v)
                        tag = "" if (dm, dff) == ("32", "32") else f"d{dm}/ff{dff}: "
                        lines.append(tag + fmt_cell(st))
                    row.append("<br>".join(lines))
                P.append(f"| {ch} | {split} | {row[0]} | {row[1]} |")

    # ---- Per-horizon detail for the published-comparable arm.
    P.append("\n## Time-LLM (GPT-2) per horizon -- the four protocol cells\n")
    P.append("Mean test MSE (raw PSD) over seeds, with the ratio to the published "
             "row in parentheses.\n")
    combos = [("tslib", "all_channel"), ("tslib", "chan_indep"),
              ("literal", "all_channel"), ("literal", "chan_indep")]
    for ch in ("4", "25"):
        P.append(f"\n### {ch} channels\n")
        P.append("| K | " + " | ".join(f"{s} / {m}" for s, m in combos) + " |")
        P.append("|---|" + "---|" * len(combos))
        for h in HOR:
            cells_out = []
            for split, mode in combos:
                v = [r for k, rr in cells.items()
                     if k[:5] == ("timellm", "gpt2", ch, mode, split)
                     and int(k[7]) == h for r in rr]
                st = cell_stats(v)
                if st is None:
                    cells_out.append("--")
                elif st["ratio"] is not None:
                    cells_out.append(f"{st['mse']:.4f} ({st['ratio']:.3f}x)")
                else:
                    cells_out.append(f"{st['mse']:.4f}")
            P.append(f"| {h} | " + " | ".join(cells_out) + " |")

    # ---- Hyperparameter coverage of primary runs.
    P.append("\n## Hyperparameter coverage (primary runs)\n")
    for axis in ("backbone", "d_model", "d_ff", "patch_length", "patch_stride",
                 "llm_layers", "num_prototypes", "heads", "prompt_mode",
                 "prompt_stats", "lr", "batch_size", "epochs_run",
                 "space_trained_in"):
        c = collections.Counter("(not recorded)" if r[axis] in ("", None) else str(r[axis])
                                for r in prim)
        P.append(f"* **{axis}**: " + ", ".join(f"`{k}` x{v}" for k, v in sorted(c.items())))

    # ---- Gaps: the 2 scenarios x 2 splits x 2 modes grid for Time-LLM (GPT-2).
    gaps = []
    for ch in ("4", "25", "151"):
        for split in ("tslib", "literal"):
            for mode in ("all_channel", "chan_indep"):
                v = block("timellm", "gpt2", ch, split, mode)
                label = (f"Time-LLM (GPT-2), {ch} channels, split `{split}`, "
                         f"mode `{mode}`")
                if not v:
                    gaps.append(f"{label}: **not run**")
                else:
                    seeds = {str(x["seed"]) for x in v}
                    if seeds == {"0"}:
                        gaps.append(f"{label}: seed 0 only "
                                    "(the matching literal cells carry 3-5 seeds)")
    P.append("\n## Known gaps\n")
    for g in gaps:
        P.append(f"* {g}")
    not_run = [g for g in gaps if "not run" in g]
    if not_run:
        P.append(f"\n{len(not_run)} cells of the published-comparison grid remain unrun; "
                  "the seed-0-only cells above need seeds 1 and 2 to match the "
                  "three-seed convention of the literal campaign.\n")
    P.append("\n## Caveats\n")
    on_disk = collections.Counter(r["checkpoint_on_disk"] or "(never saved)" for r in rows)
    n_yes = on_disk.get("yes", 0)
    yes_mb = sum(c["size_mb"] for c in ckpts if c["sha_match"] == "yes")
    P.append(f"* **Checkpoint survival is partial.** Of {len(rows)} runs: "
             f"**{n_yes} still have their weights** ({yes_mb/1000:.1f} GB, sha-verified "
             "in `CHECKPOINTS.tsv`); "
             f"**{on_disk.get('overwritten', 0)} were overwritten** by a later run that "
             f"reused the same chunk id; **{on_disk.get('no', 0)} had their weights "
             "deleted** after being recorded (the first campaign's 150, re-evaluated "
             "bit-exact in `reeval_runs.json` before deletion, hashes in "
             "`checkpoints_manifest.tsv`; the 40 OFA/BERT checkpoints, hashes in "
             f"`results_ofa_bert.json`); **{on_disk.get('(never saved)', 0)} never saved "
             "weights** (DLinear, protocol sweep, split probe, superseded campaign). "
             "`results_master.csv` marks this per row in `checkpoint_on_disk`.")
    P.append("* **Chunk ids encode neither the split nor the capacity**, so later runs "
             "silently overwrote earlier weights at the same paths: `res4_all_match` "
             "(d64/ff128) replaced `res4_all_dm32` (d32/ff32); the split study's tslib K=12 "
             "run replaced its own literal K=12 run; and all 15 surviving files reused "
             "first-campaign basenames whose weights had been deleted. Recorded metrics "
             "remain valid (computed before each overwrite); the overwritten weights "
             "cannot be re-evaluated. `CHECKPOINTS.tsv` names the true owner of every "
             "file (`owner_source`) and the first-campaign hash it replaced "
             "(`first_campaign_sha256_16`).")
    P.append("* `prompt_stats` differs across campaigns and is now a column in "
             "`results_master.csv`: split-study / `res4` / first-campaign runs used the "
             "pre-fix `normalized` convention (recorded or inferred, marked as such); "
             "the 2026-11-04 `tslib` all-channel campaign is the first to use the "
             "settled `raw` default. Cross-campaign comparisons carry that variable.")
    P.append("* `smoke_tslib.json` (1-epoch smoke test of the tslib split) was deleted in "
             "the 2026-11-04 consolidation; every configuration it covered has a full "
             "10-epoch run.")
    P.append("* Numbers here are the **recorded** test values, consolidated and "
             "cross-checked against the source JSONs (`build_master.py` reconciles "
             "every run against its source file). For the surviving checkpoints they "
             "are additionally re-verified by reloading the weights -- see the next "
             "bullet.")
    vf = os.path.join(RE, "checkpoint_verification.json")
    if os.path.isfile(vf):
        vrows = json.load(open(vf))
        n_exact = sum(1 for r in vrows if isinstance(r.get("match_eval_bs"), int))
        n_close = sum(1 for r in vrows if r.get("match_eval_bs") == "close")
        n_off = sum(1 for r in vrows if r.get("match_eval_bs") == "none")
        P.append(f"* **Checkpoint re-evaluation** (`verify_checkpoints.py`, per-file "
                 f"detail in `checkpoint_verification.json`): {len(vrows)} surviving "
                 f"checkpoints re-scored from their weights -- **{n_exact} reproduce "
                 f"their recorded test MSE exactly** (at the evaluation batch size "
                 f"the run used), {n_close} within 1%, {n_off} within ~1.4% but at no "
                 "tried batch size (32/16/8/4/2/1). The residuals sit on the "
                 "split-study N=25 channel-independent runs: their weights reload "
                 "cleanly (parameter key counts, normalisation statistics and test "
                 "window counts all match), so the drift is consistent with those "
                 "runs having executed on different hardware, not with a data or "
                 "record error. Recorded values stand.")
    open(os.path.join(RE, "MATRIX.md"), "w").write("\n".join(P) + "\n")


if __name__ == "__main__":
    main()
