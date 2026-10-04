"""
Per-model metrics tables from saved probabilities, at the nodule-aggregated level.

Reads ONLY `outputs/probs/*.npz` + splits (no GPU). For every (arch, arm) present, at the given
sample level, over the (rep, fold) points on disk:
  * per-metric table: accuracy, precision, recall, F1, AUC, log-loss, Brier — each computed per
    fold on one prediction per nodule (the mean of its slice probabilities) and averaged across
    folds, never pooled, with the naive and the Nadeau-Bengio interval from src.stats;
  * ASSERTS F1 == 2PR/(P+R) per fold (a published F1 that violates this identity is how the
    check earned its place).
Within-arm comparisons between architectures live in `scripts/nodule_oof.py` (see main()).

Writes `outputs/results/per_model_metrics.json`. Nothing pooled, nothing re-inferred, nothing
fabricated.

    python -m src.evaluate                                 # the reported grid, every arm on disk
    python -m src.evaluate --dataset lidc_binary_ge3

A narrower call (fewer repeats, folds, arms or architectures, or the nodule sample unit) writes
under a suffixed name and never over the canonical file.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss, accuracy_score,
                             precision_score, recall_score, f1_score)

from src.metadata import load_config
from src.artifacts import (outputs_dir, read_split, check_aligned, resolve_archs, name_for,
                           arms_on_disk)
from src.stats import mean_ci, rho_from_splits

# Split-file prefix, set from --dataset. "lidc_binary" = principal cohort; "lidc_binary_ge3" =
# the >=3-annotator sensitivity cohort, distinct prefix so its runs can never be pooled
# with the principal ones.
DATASET = "lidc_binary"


def _nodule_frame(cfg, arch, arm, sample_unit, rep, fold):
    O = outputs_dir(cfg)
    tag = f"{DATASET}_{sample_unit}_{arm}_rep{rep}_fold{fold}"
    npz = os.path.join(O, "probs", f"{tag}_{arch}_none_seed42.npz")
    csv = os.path.join(O, "splits", f"{tag}_test.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None, None, None
    d = np.load(npz)
    m = read_split(csv)
    check_aligned(m, d["y_true"], d["y_prob"], f"{tag} {arch}")
    m["prob"] = d["y_prob"].astype(float)
    g = m.groupby("nodule_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    # Split sizes for the Nadeau-Bengio rho = n_test / n_train, both counted in ROWS of the split
    # files, as in scripts/audit_controls.py and scripts/fold_tests.py. Counting test nodules over
    # training rows mixed two units and gave rho ~0.044 instead of ~0.29, intervals too narrow.
    n_tr = len(pd.read_csv(os.path.join(O, "splits", f"{tag}_train.csv"), usecols=["nodule_id"]))
    return g["label"].to_numpy(int), np.clip(g["prob"].to_numpy(float), 1e-6, 1 - 1e-6), (len(m), n_tr)


def _fold_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    f1 = f1_score(y, yhat, zero_division=0)
    if prec + rec > 0:                          # F1 must equal 2PR/(P+R)
        assert abs(f1 - 2 * prec * rec / (prec + rec)) < 1e-9, "F1 != 2PR/(P+R)"
    return dict(accuracy=accuracy_score(y, yhat), precision=prec, recall=rec, f1=f1,
                auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
                logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None,
                    help="comma-separated. DEFAULT: every architecture with runs on disk for "
                         "this dataset. A hardcoded pair silently dropped the third one.")
    ap.add_argument("--arms", default=None,
                    help="comma-separated. DEFAULT: every arm with runs on disk for this dataset "
                         "and sample unit")
    ap.add_argument("--sample-unit", default="slice", choices=["slice", "nodule"])
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort -- never pooled)")
    args = ap.parse_args()
    global DATASET
    DATASET = args.dataset
    cfg = load_config(args.config)
    _probs = os.path.join(outputs_dir(cfg), "probs")
    all_arms = arms_on_disk(_probs, DATASET, args.sample_unit)
    arms = args.arms.split(",") if args.arms else all_arms
    reps = [int(x) for x in args.reps.split(",")]; folds = [int(x) for x in args.folds.split(",")]
    archs, _present = resolve_archs(_probs, DATASET, args.archs,
                                    cells=([args.sample_unit], arms, reps, folds))

    table = {}
    # per-model fold-averaged metrics
    for arch in archs:
        for arm in arms:
            per_fold, n_test, n_train = [], [], []
            for rep in reps:
                for fold in folds:
                    y, p, sizes = _nodule_frame(cfg, arch, arm, args.sample_unit, rep, fold)
                    if y is None:
                        continue
                    per_fold.append(_fold_metrics(y, p)); n_test.append(sizes[0]); n_train.append(sizes[1])
            if not per_fold:
                continue
            rho = rho_from_splits(n_test, n_train)
            df = pd.DataFrame(per_fold)
            table[f"{arch}|{arm}"] = {m: mean_ci(df[m].to_numpy(), rho=rho)
                                      for m in df.columns}
            print(f"{arch:16} {arm:8} n={len(df)}  "
                  + "  ".join(f"{m} {table[f'{arch}|{arm}'][m]['mean']:.4f}"
                              for m in ("auc", "f1", "accuracy")))

    # Within-arm comparisons between architectures are NOT made here. Summing McNemar's discordant
    # counts over the folds of all repeats counted every nodule once per repeat -- and a random-arm
    # nodule several times per repeat, on part of its slices -- as if each count were an independent
    # case, with the trained runs held fixed. `scripts/nodule_oof.py` makes the comparison on one
    # prediction per nodule per repeat, with an interval that carries run-to-run variation, and
    # keeps McNemar per repeat as a description.

    R = os.path.join(outputs_dir(cfg), "results")
    os.makedirs(R, exist_ok=True)
    out = {"_note": "one prediction per nodule per fold (mean of its slice probabilities), "
                    "fold-averaged, never pooled; Nadeau-Bengio rho from split rows; "
                    "F1==2PR/(P+R) asserted; within-arm architecture comparisons are in "
                    "nodule_oof.json, not here.",
           "sample_unit": args.sample_unit, "reps": reps, "per_model": table}
    # The output name DERIVES from the dataset and the grid, so neither the sensitivity cohort nor
    # a narrower run can overwrite the canonical file, which holds the slice-unit runs of the full
    # grid with every arm.
    name = name_for("per_model_metrics", DATASET, archs, _present, reps=reps, folds=folds,
                    arms=arms, all_arms=all_arms,
                    extra=[] if args.sample_unit == "slice" else [f"{args.sample_unit}-unit"]) + ".json"
    json.dump(out, open(os.path.join(R, name), "w"), indent=1, default=str)
    print(f"wrote {os.path.join(R, name)}")


if __name__ == "__main__":
    main()
