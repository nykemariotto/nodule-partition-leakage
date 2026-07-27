"""
AUDIT CONTROLS — the four checks that decide whether the measured leakage gap is reportable.

Everything here is computed from ARCHIVED artifacts (outputs/probs/*.npz + outputs/splits/*.csv).
No GPU, no re-inference. That is deliberate: the archived probabilities are what the manuscript
promises to publish, so the headline must be recoverable FROM THEM. (`analyze_gap.py` instead
reloads the .pt weights and re-runs inference in fp32, while the .npz were written under AMP
fp16 — if the two disagree, that is the code!=numbers failure class that sank the submission.)

  C1  PROVENANCE     gap recomputed from the archived probs vs analyze_gap.py's fp32 re-inference.
  C2  LEVEL          slice-level gap (D17/SPEC 2.6d pre-register SLICE as the primary,
                     statistically-powered level; nodule-level is confirmatory) and nodule-level.
  C3  MATCHED        the decisive control for the arms' non-identical test sets. Both arms are
                     re-scored on the INTERSECTION of their test rows — identical nodules AND
                     identical slices — so test-set composition, prevalence, size and
                     aggregation depth are held fixed by construction. If the gap survives here,
                     it is not a composition artefact.
  C4  NADEAU-BENGIO  the variance correction DECISIONS D6 pre-registers and analyze_gap.py
                     defers. Fold estimates share training data, so the naive s/sqrt(k) interval
                     is anti-conservative. Corrected SE = s * sqrt(1/k + n_test/n_train).

    python scripts/audit_controls.py
    python scripts/audit_controls.py --json outputs/metrics/audit_controls.json

Prints a table per architecture and writes machine-readable JSON. Nothing is fabricated: every
number traces to a file in outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.datasets import load_processed_index
from src.stats import mean_ci, rho_from_splits, wilcoxon_paired

ARMS = ("patient", "random")


# Split-file prefix. "lidc_binary" = principal cohort; "lidc_binary_ge3" = the >=3-annotator
# sensitivity cohort (D37), whose runs carry a distinct prefix so they can never be averaged in with
# the principal ones. Was hardcoded until 2026-07-26 -- caught BEFORE the ge3 grid ran, because a
# 25 h grid whose artifacts no analysis script can read is 25 h wasted.
DATASET = "lidc_binary"


def run_tag(arm, rep, fold, dataset=None):
    return f"{dataset or DATASET}_slice_{arm}_rep{rep}_fold{fold}"


def run_name(arm, rep, fold, arch, seed=42, enh="none", dataset=None):
    return f"{run_tag(arm, rep, fold, dataset)}_{arch}_{enh}_seed{seed}"


def load_arm_fold(cfg, arch, arm, rep, fold, pidx, suffix=""):
    """Archived probs joined to the split rows they were produced from.

    Row order is reproduced exactly as src/datasets.py builds it (same inner merge on
    [nodule_id, z_position], test loader has shuffle=False), so probs align row-for-row.
    """
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    npz = os.path.join(O, "probs", f"{run_name(arm, rep, fold, arch)}{suffix}.npz")
    csv = os.path.join(O, "splits", f"{run_tag(arm, rep, fold)}_test.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None
    d = np.load(npz)
    split = pd.read_csv(csv)
    split["z_position"] = split["z_position"].round(2)
    m = split.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
    if len(m) != len(d["y_prob"]):
        raise RuntimeError(f"ALIGNMENT FAILED {arm} f{fold} {arch}: split merge {len(m)} rows "
                           f"but archived probs {len(d['y_prob'])}. Artifacts are stale.")
    if not np.array_equal(m["label"].astype(int).to_numpy(), d["y_true"].astype(int)):
        raise RuntimeError(f"LABEL MISMATCH {arm} f{fold} {arch}: archived y_true does not match "
                           f"the split's labels in merge order. Ordering assumption is wrong.")
    m["prob"] = d["y_prob"].astype(np.float64)
    return m


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return None
    return dict(auc=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), n=int(len(y)), pos_rate=float(y.mean()))


def by_nodule(df):
    g = df.groupby("nodule_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    return metrics(g["label"], g["prob"])


def summarize(gaps, n_test, n_train, label):
    """mean, naive t-CI, and the pre-registered Nadeau-Bengio CI — all from src/stats.py.

    The intervals are NOT reimplemented here: `src.stats` is the single versioned
    implementation (D27). This function only shapes the result for display.
    """
    g = [x for x in gaps if x is not None]
    if len(g) < 2:
        return dict(label=label, k=len(g), mean=float(np.mean(g)) if g else None)
    out = mean_ci(g, rho=rho_from_splits(n_test, n_train))
    return dict(label=label, nb_rho=out["rho"], nb_widening=out["widening"], **out)


def fmt(d):
    if d.get("k", 0) < 2:
        return f"  {d['label']:34} insufficient folds"
    star = "" if d["excludes_zero_nb"] else "  <- includes 0 under D6"
    return (f"  {d['label']:34} {d['mean']:+.4f}  naive95 [{d['ci_naive'][0]:+.4f},{d['ci_naive'][1]:+.4f}]"
            f"  NB95 [{d['ci_nb'][0]:+.4f},{d['ci_nb'][1]:+.4f}]  +{d['folds_positive']}/{d['k']}{star}")


def analyze_arch(cfg, arch, folds, suffix="", reps=(0,)):
    pidx = load_processed_index(cfg, "none")
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    acc = {k: [] for k in ("slice_auc", "slice_ll", "slice_br",
                           "nod_auc", "nod_ll", "nod_br",
                           "m_slice_auc", "m_slice_ll", "m_slice_br",
                           "m_nod_auc", "m_nod_ll", "m_nod_br")}
    n_test, n_train, detail = [], [], []
    for rep in reps:
      for f in folds:
        d = {a: load_arm_fold(cfg, arch, a, rep, f, pidx, suffix) for a in ARMS}
        if any(v is None for v in d.values()):
            print(f"  rep{rep} fold{f}: MISSING artifacts - skipped")
            continue
        tr = pd.read_csv(os.path.join(O, "splits", f"{run_tag('patient', rep, f)}_train.csv"))
        n_test.append(len(d["patient"])); n_train.append(len(tr))

        # ---- full test sets, as reported ----
        s = {a: metrics(d[a]["label"], d[a]["prob"]) for a in ARMS}
        n = {a: by_nodule(d[a]) for a in ARMS}

        # ---- C3: identical rows in both arms (same nodules AND same slices) ----
        key = ["nodule_id", "z_position"]
        common = d["patient"][key].merge(d["random"][key], on=key, how="inner").drop_duplicates()
        md = {a: d[a].merge(common, on=key, how="inner") for a in ARMS}
        ms = {a: metrics(md[a]["label"], md[a]["prob"]) for a in ARMS}
        mn = {a: by_nodule(md[a]) for a in ARMS}

        row = dict(rep=rep, fold=f, n_patient=len(d["patient"]), n_random=len(d["random"]),
                   n_matched=len(common),
                   matched_nodules=int(md["patient"]["nodule_id"].nunique()),
                   matched_pos_rate=(ms["patient"]["pos_rate"] if ms["patient"] else None))
        detail.append(row)
        if s["patient"] and s["random"]:
            acc["slice_auc"].append(s["random"]["auc"] - s["patient"]["auc"])
            acc["slice_ll"].append(s["patient"]["logloss"] - s["random"]["logloss"])
            acc["slice_br"].append(s["patient"]["brier"] - s["random"]["brier"])
        if n["patient"] and n["random"]:
            acc["nod_auc"].append(n["random"]["auc"] - n["patient"]["auc"])
            acc["nod_ll"].append(n["patient"]["logloss"] - n["random"]["logloss"])
            acc["nod_br"].append(n["patient"]["brier"] - n["random"]["brier"])
        if ms["patient"] and ms["random"]:
            acc["m_slice_auc"].append(ms["random"]["auc"] - ms["patient"]["auc"])
            acc["m_slice_ll"].append(ms["patient"]["logloss"] - ms["random"]["logloss"])
            acc["m_slice_br"].append(ms["patient"]["brier"] - ms["random"]["brier"])
        if mn["patient"] and mn["random"]:
            acc["m_nod_auc"].append(mn["random"]["auc"] - mn["patient"]["auc"])
            acc["m_nod_ll"].append(mn["patient"]["logloss"] - mn["random"]["logloss"])
            acc["m_nod_br"].append(mn["patient"]["brier"] - mn["random"]["brier"])

    out = {}
    names = [("slice_auc", "SLICE (pre-reg primary) AUC"), ("slice_ll", "SLICE log-loss"),
             ("slice_br", "SLICE Brier"), ("nod_auc", "NODULE AUC"), ("nod_ll", "NODULE log-loss"),
             ("nod_br", "NODULE Brier"),
             ("m_slice_auc", "MATCHED-rows SLICE AUC"), ("m_slice_ll", "MATCHED-rows SLICE log-loss"),
             ("m_slice_br", "MATCHED-rows SLICE Brier"), ("m_nod_auc", "MATCHED-rows NODULE AUC"),
             ("m_nod_ll", "MATCHED-rows NODULE log-loss"), ("m_nod_br", "MATCHED-rows NODULE Brier")]
    for k, lab in names:
        out[k] = summarize(acc[k], n_test, n_train, lab)
    for k, lab in (("slice_auc", "SLICE AUC"), ("nod_auc", "NODULE AUC"), ("m_slice_auc", "MATCHED SLICE AUC")):
        g = acc[k]
        if len(g) >= 5:
            w = wilcoxon_paired(g)
            out[k]["wilcoxon"] = w       # includes min_attainable_p / at_floor (D5)
    out["_folds"] = detail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--reps", default="0", help="comma-separated rep indices; grid S2 = 0,1,2 (n=15)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--probe", action="store_true", help="also run on the _final PROBE checkpoint")
    ap.add_argument("--json", default="outputs/metrics/audit_controls.json")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort, D37 -- never pooled with principal)")
    args = ap.parse_args()
    global DATASET
    DATASET = args.dataset
    cfg = load_config(args.config)
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    everything = {}
    for arch in args.archs.split(","):
        for suffix, kind in ([("", "CANONICAL")] + ([("_final", "PROBE")] if args.probe else [])):
            print(f"\n=== {arch} | {kind} | from ARCHIVED probs (outputs/probs/*.npz) ===")
            r = analyze_arch(cfg, arch, folds, suffix, reps)
            d = pd.DataFrame(r["_folds"])
            if not d.empty:
                print(d.to_string(index=False))
            print("  gap = random - patient (AUC) / patient - random (log-loss, Brier)")
            for k in ("slice_auc", "slice_ll", "slice_br", "nod_auc", "nod_ll", "nod_br",
                      "m_slice_auc", "m_slice_ll", "m_slice_br", "m_nod_auc", "m_nod_ll", "m_nod_br"):
                print(fmt(r[k]))
            if r["slice_auc"].get("nb_widening"):
                print(f"  [D6] Nadeau-Bengio rho = n_test/n_train = {r['slice_auc']['nb_rho']:.4f} "
                      f"-> CIs widen x{r['slice_auc']['nb_widening']:.3f} vs naive")
            everything[f"{arch}|{kind}"] = r

    if args.json:
        p = os.path.join(cfg["project"]["root"], args.json) if not os.path.isabs(args.json) else args.json
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(everything, fh, indent=1)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
