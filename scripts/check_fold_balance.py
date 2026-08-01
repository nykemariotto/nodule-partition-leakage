"""
FOLD CLASS-BALANCE AUDIT — the datum that decides `split.stratify_key` (DECISIONS D27).

`config.yaml` declares `split.stratify_key: label`, but `src/splits.py` only GROUPS by
patient_id; it never stratifies. Two ways out, and the measurement decides which:

  Option A  implement StratifiedGroupKFold -> splits CHANGE -> the 20 canonical runs are
            invalidated and must be re-run.
  Option B  delete the key, document that grouping alone produced adequately balanced folds ->
            the 20 canonical runs STAY VALID and the config starts telling the truth.

DECISION RULE — fixed HERE, BEFORE looking at the numbers, so the threshold cannot be chosen
to justify the cheaper option. Option B is admissible only if ALL of the following hold for
every test fold of BOTH arms:

  R1  no degenerate fold: >= 20 positives AND >= 20 negatives at NODULE level
      (below that, a fold's AUC is too unstable to average).
  R2  no extreme fold: nodule-level positive rate within [0.25, 0.75].
  R3  bounded drift: |fold positive rate - pooled positive rate| <= 0.15 at nodule level.
  R4  the imbalance must not be driving the result: the per-fold leakage gap must not be
      rank-correlated with the per-fold prevalence deviation at |rho| >= 0.9 with a consistent
      sign in BOTH architectures (reported, judged qualitatively - a hard cut on 5 points would
      be noise).

R1-R3 are pass/fail here. R4 is reported for judgement; it needs the gap values, so it is only
evaluated when outputs/metrics/audit_controls.json exists.

    python scripts/check_fold_balance.py

Exit 0 = Option B admissible (R1-R3 pass). Exit 1 = a rule failed -> Option A indicated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config

MIN_PER_CLASS = 20
RATE_LO, RATE_HI = 0.25, 0.75
MAX_DRIFT = 0.15


# Split-file prefix, set from --dataset (see audit_controls). Checking fold balance on the
# >=3-annotator cohort BEFORE committing GPU hours is the point: that cohort has 535 patients
# against 740, so per-fold class counts are smaller and could fall under MIN_PER_CLASS.
DATASET = "lidc_binary"


def split_path(cfg, arm, fold, which, rep=0):
    return os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "splits",
                        f"{DATASET}_slice_{arm}_rep{rep}_fold{fold}_{which}.csv")


def describe(df):
    """Class counts at slice and nodule level (nodule label = its slices' first label)."""
    n_slice = len(df)
    pos_slice = int(df["label"].sum())
    nod = df.groupby("nodule_id")["label"].first()
    return dict(n_slice=n_slice, pos_slice=pos_slice, neg_slice=n_slice - pos_slice,
                rate_slice=pos_slice / n_slice if n_slice else float("nan"),
                n_nod=int(len(nod)), pos_nod=int(nod.sum()), neg_nod=int(len(nod) - nod.sum()),
                rate_nod=float(nod.mean()) if len(nod) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3")
    args = ap.parse_args()
    global DATASET
    DATASET = args.dataset
    cfg = load_config(args.config)
    folds = [int(x) for x in args.folds.split(",")]

    print(f"decision rule (fixed before measuring): R1 >={MIN_PER_CLASS}/class at nodule level · "
          f"R2 nodule rate in [{RATE_LO},{RATE_HI}] · R3 |drift| <= {MAX_DRIFT}")

    rows, failures = [], []
    for arm in ("patient", "random"):
        # pooled prevalence = union of the 5 test folds for this arm
        pooled = pd.concat([pd.read_csv(split_path(cfg, arm, f, "test")) for f in folds])
        pooled_nod = pooled.groupby("nodule_id")["label"].first()
        pooled_rate = float(pooled_nod.mean())
        print(f"\n=== arm {arm} · pooled test nodule positive rate {pooled_rate:.4f} "
              f"({int(pooled_nod.sum())}+/{int(len(pooled_nod)-pooled_nod.sum())}-) ===")
        print(f"{'fold':6}{'slices':9}{'pos':7}{'neg':7}{'rate_sl':10}"
              f"{'nodules':9}{'pos':6}{'neg':6}{'rate_nod':10}{'drift':9}  flags")
        for f in folds:
            d = describe(pd.read_csv(split_path(cfg, arm, f, "test")))
            drift = d["rate_nod"] - pooled_rate
            flags = []
            if d["pos_nod"] < MIN_PER_CLASS or d["neg_nod"] < MIN_PER_CLASS:
                flags.append("R1-FAIL(degenerate)")
            if not (RATE_LO <= d["rate_nod"] <= RATE_HI):
                flags.append("R2-FAIL(extreme)")
            if abs(drift) > MAX_DRIFT:
                flags.append("R3-FAIL(drift)")
            if flags:
                failures.append((arm, f, flags))
            rows.append(dict(arm=arm, fold=f, drift=drift, **d))
            print(f"{f:<6}{d['n_slice']:<9}{d['pos_slice']:<7}{d['neg_slice']:<7}{d['rate_slice']:<10.4f}"
                  f"{d['n_nod']:<9}{d['pos_nod']:<6}{d['neg_nod']:<6}{d['rate_nod']:<10.4f}"
                  f"{drift:<+9.4f}  {' '.join(flags) if flags else 'ok'}")

    df = pd.DataFrame(rows)
    print("\n--- spread of the nodule-level positive rate ---")
    for arm in ("patient", "random"):
        s = df[df.arm == arm]["rate_nod"]
        print(f"  {arm:8} min {s.min():.4f}  max {s.max():.4f}  range {s.max()-s.min():.4f}  sd {s.std(ddof=1):.4f}")

    # ---- R4: is the gap driven by prevalence deviation? ----
    # The gap values and the fold prevalences MUST come from the same cohort. Until 2026-07-28 this
    # read the principal cohort's gaps unconditionally, so running with --dataset lidc_binary_ge3
    # printed principal-cohort correlations under a >=3-cohort heading -- a cross-cohort comparison
    # that looks like a within-cohort one. It now refuses rather than mixes.
    GAPS = {"lidc_binary": "outputs/_analysis/audit_controls_AFTER.json",
            "lidc_binary_ge3": "outputs/metrics/audit_controls_lidc_binary_ge3.json"}
    rel = GAPS.get(DATASET)
    p = os.path.join(cfg["project"]["root"], rel) if rel else None
    if rel is None:
        print(f"\n(R4 SKIPPED: no gap artifact is registered for dataset '{DATASET}'. R4 needs "
              f"per-fold gaps from THIS cohort; it cannot borrow another cohort's. Re-run this "
              f"check after the '{DATASET}' grid completes and its audit_controls artifact exists.)")
    elif os.path.exists(p):
        ac = json.load(open(p))
        print(f"\n--- R4: per-fold gap vs per-fold prevalence deviation (patient arm, {DATASET}) ---")
        dev = df[df.arm == "patient"].sort_values("fold")["drift"].to_numpy()
        reported = 0
        for key in ac:
            if "CANONICAL" not in key:
                continue
            for metric in ("slice_auc", "nod_auc"):
                g = np.asarray(ac[key][metric].get("per_fold", []), dtype=float)
                if len(g) != len(dev):
                    continue
                r = float(pd.Series(g).corr(pd.Series(dev), method="spearman"))
                reported += 1
                print(f"  {key.split('|')[0]:16} {metric:10} spearman rho = {r:+.3f}"
                      f"{'   <- |rho|>=0.9, inspect' if abs(r) >= 0.9 else ''}")
        if not reported:
            # an empty block under a printed heading reads as "R4 passed"; say what happened
            print(f"  NOT COMPUTED: {rel} carries no per-fold gap series of length {len(dev)}. "
                  f"R4 needs one gap per fold aligned to the {len(dev)} fold prevalences above; "
                  f"the current artifact summarises over all 15 (rep x fold) points instead. "
                  f"R4 is advisory -- R1-R3 are the gate and are unaffected.")
    else:
        print(f"\n(R4 skipped: {rel} not found)")

    print()
    if failures:
        print(f"RESULT: {len(failures)} test fold(s) violate the pre-set rule -> "
              f"stratification MATTERS -> OPTION A (implement StratifiedGroupKFold; the 20 "
              f"canonical runs are invalidated and must be re-run).")
        for arm, f, fl in failures:
            print(f"  {arm} fold{f}: {', '.join(fl)}")
        return 1
    print("RESULT: every test fold passes R1-R3 -> grouping alone produced adequately balanced "
          "folds -> OPTION B admissible (delete `split.stratify_key` with documented "
          "justification; the 20 canonical runs remain valid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
