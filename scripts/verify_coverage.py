"""
COVERAGE GATE (hard assertion, not an inspection).

Asserts that every split row of every fold has a processed .npy, and that every nodule of the
principal cohort is fully covered. FAILS LOUDLY (exit 1) on any gap — this gate exists because a
printed WARNING was once hidden by a grep in the training driver while 2.7% of slices were being
silently dropped from train AND test (multi-series resumability bug, 2026-07-18).

    python scripts/verify_coverage.py --config config.yaml [--enhancement none]

Exit 0 = safe to train. Exit 1 = DO NOT TRAIN.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.datasets import load_processed_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--enhancement", default=None)
    ap.add_argument("--phase", choices=["subset", "grid"], default="subset",
                    help="subset = pre-registered 250-patient experiment (asserts splits + in-use "
                         "cohort); grid = FULL 740-patient cohort MUST be 100%% covered. Using "
                         "'subset' at grid stage would make this gate a false green.")
    ap.add_argument("--require-full-cohort", action="store_true",
                    help="alias for --phase grid")
    args = ap.parse_args()
    cfg = load_config(args.config)
    enh = args.enhancement or cfg["preprocess"]["enhancement_principal"]
    root = cfg["project"]["root"]
    pidx = load_processed_index(cfg, enh)
    pidx_keys = set(map(tuple, pidx[["nodule_id", "z_position"]].values))
    failures = []

    # ---- 1. every split row of every fold must have a processed file ----
    spl_dir = os.path.join(root, cfg["paths"]["outputs"], "splits")
    splits = sorted(glob.glob(os.path.join(spl_dir, "*_test.csv")))
    print(f"[gate] PHASE={args.phase.upper()} " + ("(full 740-patient cohort REQUIRED)" if (args.phase=='grid' or args.require_full_cohort) else "(250-patient subset; full cohort informational)"))
    print(f"[gate] checking {len(splits)*3} split files (enhancement={enh})")
    for te in splits:
        tag = os.path.basename(te)[:-len("_test.csv")]
        for part in ("train", "val", "test"):
            p = os.path.join(spl_dir, f"{tag}_{part}.csv")
            if not os.path.exists(p):
                failures.append(f"{tag}_{part}: split file MISSING"); continue
            df = pd.read_csv(p)
            df["z_position"] = df["z_position"].round(2)
            keys = set(map(tuple, df[["nodule_id", "z_position"]].values))
            miss = keys - pidx_keys
            if miss:
                nods = sorted({n for n, _ in miss})
                failures.append(f"{tag}_{part}: {len(miss)}/{len(df)} rows without .npy "
                                f"({len(nods)} nodules, e.g. {nods[:3]})")

    # ---- 2. cohort coverage, SCOPED to the patients actually used by the splits ----
    # (the full 740-patient cohort is only preprocessed for the full grid; use
    #  --require-full-cohort to make that a hard requirement at that stage)
    m = pd.read_csv(os.path.join(root, cfg["paths"]["outputs"], "metadata", "lidc_master.csv"))
    coh = m[m["cohort_main"]].copy()
    coh["z_position"] = coh["z_position"].round(2)
    used = set()
    for te in splits:
        used |= set(pd.read_csv(te)["patient_id"])
        used |= set(pd.read_csv(te.replace("_test.csv", "_train.csv"))["patient_id"])
    scoped = coh[coh["patient_id"].isin(used)] if used else coh
    smiss = set(map(tuple, scoped[["nodule_id", "z_position"]].values)) - pidx_keys
    fmiss = set(map(tuple, coh[["nodule_id", "z_position"]].values)) - pidx_keys
    print(f"[gate] cohort IN USE by splits: {len(scoped)} slices / {scoped.nodule_id.nunique()} "
          f"nodules / {scoped.patient_id.nunique()} patients -> missing .npy: {len(smiss)}")
    print(f"[gate] FULL cohort (grid stage): {len(coh)} slices -> missing .npy: {len(fmiss)} "
          f"({'OK' if not fmiss else 'not yet preprocessed - informational'})")
    if smiss:
        nods = sorted({n for n, _ in smiss})
        failures.append(f"cohort-in-use: {len(smiss)} slices without .npy across {len(nods)} "
                        f"nodules (e.g. {nods[:3]})")
    require_full = args.require_full_cohort or args.phase == "grid"
    if require_full and fmiss:
        failures.append(f"FULL cohort: {len(fmiss)} slices without .npy (required at phase=grid: the grid MUST cover 100% of the cohort)")

    if failures:
        print("\n=== COVERAGE GATE: FAILED - DO NOT TRAIN ===")
        for f in failures:
            print("  [FAIL]", f)
        sys.exit(1)
    print("\n=== COVERAGE GATE: PASSED - every split row and every in-use cohort nodule has a .npy ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
