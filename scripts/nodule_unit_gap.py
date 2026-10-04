"""
THE NODULE SAMPLE UNIT -- leakage gap when every nodule contributes a single slice.

The design is a 2 x 2: the partition unit (patient-grouped arm A, random arm B) crossed with the
sample unit (every slice of a nodule, or one representative slice per nodule -- its largest cross
section). At the nodule sample unit there is no within-nodule redundancy to leak: a nodule is one
sample, so a random split can separate only different nodules of the same patient. This script
reports the random-minus-patient gap at that unit, per fold, with the same Nadeau-Bengio interval and
corrected resampled t-test as the slice level (scripts/fold_tests.py), for AUC, log-loss and Brier,
positive meaning the leakier arm looks better.

It answers a question that arm C answers better: at this unit a patient contributes only about two
training samples, so a null here cannot separate "patient context does not inflate" from "too few
samples per patient to learn a patient signature". Arm C (nodule-grouped, every slice) keeps nearly
the density of arm A (12.4 against 14.9 training slices per patient; scripts/verify_leakage.py) and
is the paper's test of the patient route; this is the prespecified second granularity of the 2 x 2,
reported so that it is not omitted.

Reads archived probabilities and the committed split files only (no GPU).

    python scripts/nodule_unit_gap.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.artifacts import REPO, FULL_REPS, FULL_FOLDS, read_split, check_aligned, resolve_archs, name_for
from src.stats import mean_ci, nadeau_bengio_ttest, rho_from_splits
from decompose_routes import HIGHER_IS_BETTER

METRICS = ("auc", "logloss", "brier")
ALPHA = 0.05


def fold_metrics(dataset, arm, rep, fold, arch, with_train):
    tag = f"{dataset}_nodule_{arm}_rep{rep}_fold{fold}"
    d = np.load(os.path.join(REPO, "outputs", "probs", f"{tag}_{arch}_none_seed42.npz"))
    test = read_split(os.path.join(REPO, "outputs", "splits", f"{tag}_test.csv"))
    check_aligned(test, d["y_true"], d["y_prob"], f"{tag} {arch}")
    if test["nodule_id"].duplicated().any():
        raise RuntimeError(f"{tag}: a nodule appears twice in a nodule-unit test fold")
    y = d["y_true"].astype(int)
    p = np.clip(d["y_prob"].astype(float), 1e-6, 1 - 1e-6)
    out = dict(auc=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
               brier=brier_score_loss(y, p), n_test=len(test))
    if with_train:                       # rho uses the patient arm's split sizes, in rows
        out["n_train"] = sum(1 for _ in open(os.path.join(REPO, "outputs", "splits",
                                                          f"{tag}_train.csv"))) - 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Random - patient gap at the nodule sample unit.")
    ap.add_argument("--archs", default=None, help="comma-separated; default: the reported set")
    ap.add_argument("--dataset", default="lidc_binary")
    ap.add_argument("--json", default=os.path.join("outputs", "_analysis", "nodule_unit_gap.json"))
    args = ap.parse_args()
    reps, folds = list(FULL_REPS), list(FULL_FOLDS)
    archs, present = resolve_archs(os.path.join(REPO, "outputs", "probs"), args.dataset, args.archs,
                                   cells=(["nodule"], ["patient", "random"], reps, folds))
    archs = sorted(archs)
    out = {"_about": dict(level="nodule sample unit: one representative slice per nodule",
                          gap="random - patient; sign-flipped for log-loss and Brier",
                          test="Nadeau-Bengio corrected resampled t-test, dual of the NB 95% interval",
                          rho="mean(n_test / n_train) of the patient arm's folds",
                          dataset=args.dataset, reps=reps, folds=folds)}
    for arch in archs:
        rows = []
        for rep in reps:
            for fold in folds:
                a = fold_metrics(args.dataset, "patient", rep, fold, arch, with_train=True)
                b = fold_metrics(args.dataset, "random", rep, fold, arch, with_train=False)
                rows.append((a, b))
        rho = rho_from_splits([a["n_test"] for a, _ in rows], [a["n_train"] for a, _ in rows])
        res = {"n_folds": len(rows), "rho": rho}
        print(f"\n=== {arch} | n = {len(rows)} folds | rho = {rho:.4f} ===")
        for metric in METRICS:
            sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0    # positive = the leakier arm looks better
            vals = np.asarray([sign * (b[metric] - a[metric]) for a, b in rows])
            s = mean_ci(vals, rho=rho, alpha=ALPHA)
            t = nadeau_bengio_ttest(vals, rho)
            if (t["p"] < ALPHA) != s["excludes_zero_nb"]:
                raise AssertionError(f"{arch} {metric}: p={t['p']:.4g} disagrees with {s['ci_nb']}")
            print(f"  {metric:7s} B-A {s['mean']:+.4f}  NB95 [{s['ci_nb'][0]:+.4f},{s['ci_nb'][1]:+.4f}]"
                  f"  p = {t['p']:.1g}  folds+ {s['folds_positive']}/{s['k']}")
            res[f"{metric}_B-A"] = dict(mean=s["mean"], ci_nb=s["ci_nb"], excludes_zero_nb=s["excludes_zero_nb"],
                                        folds_positive=s["folds_positive"], k=s["k"], t=t["t"], df=t["df"],
                                        p=t["p"], per_fold=s["per_fold"])
        out[arch] = res
    d = os.path.dirname(args.json)
    stem = os.path.basename(args.json)[:-5]
    p = os.path.join(d, name_for(stem.split("__")[0], args.dataset, archs, present) + ".json")
    p = p if os.path.isabs(p) else os.path.join(REPO, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
