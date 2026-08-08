"""
STAGE 6 — per-model metrics tables from saved probs (SPEC §6.6; reviewer I3 per-model table,
B9 numeric-consistency, B5 within-arm McNemar).

Reads ONLY `outputs/probs/*.npz` + splits (no GPU). For every (arch, arm) present, at the given
sample level, over the (rep, fold) points on disk:
  * per-metric table: accuracy, precision, recall, F1, AUC, log-loss, Brier — each fold-averaged
    (D24, never pooled) with the naive and Nadeau-Bengio interval (D6) from src.stats;
  * ASSERTS F1 == 2PR/(P+R) per fold (B9 — Reviewer 3 flagged an Xception F1 that violated this);
  * within EACH arm, McNemar between the two architectures on the identical test set (B5) — valid
    because same-arm same-fold test sets match; NEVER across arms (their test sets differ).

Writes `outputs/results/per_model_metrics.json`. Nothing pooled, nothing re-inferred, nothing
fabricated.

    python -m src.evaluate --sample-unit slice --reps 0
    python -m src.evaluate --sample-unit slice --reps 0,1,2      # after the grid
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
from src.datasets import load_processed_index
from src.stats import mean_ci, rho_from_splits, mcnemar, mcnemar_from_counts

# Split-file prefix, set from --dataset. "lidc_binary" = principal cohort; "lidc_binary_ge3" =
# the >=3-annotator sensitivity cohort (D37), distinct prefix so its runs can never be pooled
# with the principal ones. Was hardcoded until 2026-07-26.
DATASET = "lidc_binary"


def _nodule_frame(cfg, arch, arm, sample_unit, rep, fold, pidx):
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    tag = f"{DATASET}_{sample_unit}_{arm}_rep{rep}_fold{fold}"
    npz = os.path.join(O, "probs", f"{tag}_{arch}_none_seed42.npz")
    csv = os.path.join(O, "splits", f"{tag}_test.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None, None, None
    d = np.load(npz)
    s = pd.read_csv(csv); s["z_position"] = s["z_position"].round(2)
    m = s.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
    if len(m) != len(d["y_prob"]):
        raise RuntimeError(f"align {tag} {arch}: {len(m)} vs {len(d['y_prob'])}")
    m["prob"] = d["y_prob"].astype(float)
    g = m.groupby("nodule_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    n_tr = len(pd.read_csv(os.path.join(O, "splits", f"{tag}_train.csv")))
    return g["label"].to_numpy(int), np.clip(g["prob"].to_numpy(float), 1e-6, 1 - 1e-6), (len(g), n_tr)


def _fold_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    f1 = f1_score(y, yhat, zero_division=0)
    if prec + rec > 0:                          # B9: F1 must equal 2PR/(P+R)
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
    ap.add_argument("--arms", default="patient,random")
    ap.add_argument("--sample-unit", default="slice", choices=["slice", "nodule"])
    ap.add_argument("--reps", default="0")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort, D37 -- never pooled)")
    args = ap.parse_args()
    global DATASET
    DATASET = args.dataset
    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    arms = args.arms.split(",")
    reps = [int(x) for x in args.reps.split(",")]; folds = [int(x) for x in args.folds.split(",")]
    from src.artifacts import resolve_archs, name_for
    _probs = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "probs")
    archs, _present = resolve_archs(_probs, DATASET, args.archs,
                                    cells=([args.sample_unit], arms, reps, folds))

    table, mcn = {}, {}
    # per-model fold-averaged metrics
    for arch in archs:
        for arm in arms:
            per_fold, n_test, n_train = [], [], []
            for rep in reps:
                for fold in folds:
                    y, p, sizes = _nodule_frame(cfg, arch, arm, args.sample_unit, rep, fold, pidx)
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

    # Within-arm McNemar between architectures on the identical test set (B5), for EVERY pair.
    # This used to run only when exactly two architectures were present, so adding a third silently
    # emptied the block while the manuscript still quoted the two-architecture counts. Pairs are the
    # natural generalisation, and with three architectures they also strengthen the claim they
    # support: that a leaked protocol flattens the differences BETWEEN models.
    from itertools import combinations
    for a1, a2 in combinations(archs, 2):
        for arm in arms:
            bs = []
            for rep in reps:
                for fold in folds:
                    ya, pa, _ = _nodule_frame(cfg, a1, arm, args.sample_unit, rep, fold, pidx)
                    yb, pb, _ = _nodule_frame(cfg, a2, arm, args.sample_unit, rep, fold, pidx)
                    if ya is None or yb is None or len(ya) != len(yb) or not np.array_equal(ya, yb):
                        continue
                    bs.append(mcnemar(ya, pa, pb))
            if bs:
                # sum discordant counts across folds (identical test set within each fold), then
                # get the aggregate p from the counts directly — no synthetic-vector reconstruction
                # (which swapped b/c; code review 2026-07-21). b,c stay correctly labelled.
                b = sum(x["b"] for x in bs); c = sum(x["c"] for x in bs)
                mcn[f"{a1}|{a2}|{arm}"] = mcnemar_from_counts(b, c)
                print(f"McNemar {a1} vs {a2} in {arm}: b={b} c={c} "
                      f"p={mcn[f'{a1}|{a2}|{arm}']['p']:.4f} "
                      f"({mcn[f'{a1}|{a2}|{arm}']['method']})")

    R = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "results")
    os.makedirs(R, exist_ok=True)
    out = {"_note": "fold-averaged per D24 (never pooled); F1==2PR/(P+R) asserted (B9); McNemar is "
                    "within-arm arch-vs-arch on identical test sets (B5), invalid across arms.",
           "sample_unit": args.sample_unit, "reps": reps, "per_model": table, "mcnemar_within_arm": mcn}
    # The output name DERIVES from the dataset. It used to be a literal, so running this with
    # --dataset lidc_binary_ge3 would have overwritten the principal cohort's file with the
    # sensitivity cohort's numbers under the principal cohort's name -- the D62 defect exactly,
    # fixed on the reading side but not on the writing side. The principal cohort keeps the
    # original filename so nothing that already points at it breaks.
    name = name_for("per_model_metrics", DATASET, archs, _present) + ".json"
    json.dump(out, open(os.path.join(R, name), "w"), indent=1, default=str)
    print(f"wrote {os.path.join(R, name)}")


if __name__ == "__main__":
    main()
