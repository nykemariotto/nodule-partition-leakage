"""
STAGE 3 — seeded splits + leakage assertion (SPEC §6.3).

Repeated Grouped-vs-Random K-Fold (SPEC §2.2): the seeds parameterize the repeats.
For each repeat r (seed) and fold k:
  * arm 'patient' = GroupKFold on patient_id (patients shuffled by seed r)   [Arm A]
  * arm 'random'  = KFold ignoring patient_id (samples shuffled by seed r)   [Arm B]
Test = fold k; validation = a slice carved from the train part (patient-grouped for the
patient arm, random for the random arm); train = the rest.

Exports one CSV per (rep, fold, split) to outputs/splits/. Asserts ZERO patient_id overlap
across train/val/test in the patient arm, and that the random arm DOES overlap (that is the
point). No training here.

    python -m src.splits --config config.yaml --dataset lidc_binary --sample-unit slice \
        --arm patient --repeats 1 --folds 5 --limit-patients 120
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.metadata import load_config


def _cohort(cfg, cohort="principal"):
    """Load the sample table for a cohort.

    'principal' = the min_annotators>=1 cohort (cohort_main flag, D16), the headline experiment.
    'ge3'       = the high-agreement min_annotators>=3 cohort (D16 sensitivity / R4.4 label-noise),
                  reported side by side, NEVER pooled with principal (DECISIONS D37).
    """
    D = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "metadata")
    m = pd.read_csv(os.path.join(D, "lidc_master.csv"))
    if cohort == "ge3":
        return m[m["n_annotators"] >= 3].reset_index(drop=True)
    if cohort == "principal":
        return m[m["cohort_main"]].reset_index(drop=True)
    raise ValueError(f"unknown cohort '{cohort}' (expected 'principal' or 'ge3')")


def _fold_assign(items, k, seed):
    """Return an array assigning each item (index) to one of k folds, shuffled by seed."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    folds = np.empty(len(items), dtype=int)
    for i, idx in enumerate(order):
        folds[idx] = i % k
    return folds


def make_splits(cfg, dataset, sample_unit, arm, repeats, folds, limit_patients, val_frac=0.125,
                cohort="principal"):
    df = _cohort(cfg, cohort)
    # Distinct tag prefix for the non-principal cohort so its splits/runs can NEVER collide with
    # the principal ones or with what the running grid reads (DECISIONS D37 safeguard 1).
    dataset_tag = dataset if cohort == "principal" else f"{dataset}_{cohort}"
    if sample_unit == "nodule":
        df = df[df["is_representative_slice"]].reset_index(drop=True)
    if limit_patients:
        keep = sorted(df["patient_id"].unique())[:limit_patients]
        df = df[df["patient_id"].isin(keep)].reset_index(drop=True)

    out_dir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "splits")
    os.makedirs(out_dir, exist_ok=True)
    seeds = cfg["repetition"]["seed_list"]
    written = []

    for r in range(repeats):
        seed = seeds[r]
        if arm == "patient":
            pats = df["patient_id"].unique()
            pf = _fold_assign(pats, folds, seed)
            pat_fold = dict(zip(pats, pf))
            fold_of = df["patient_id"].map(pat_fold).to_numpy()
        elif arm == "random":
            fold_of = _fold_assign(np.arange(len(df)), folds, seed)
        else:
            raise ValueError(arm)

        for k in range(folds):
            test = df[fold_of == k]
            rest = df[fold_of != k]
            # carve validation from the rest
            if arm == "patient":
                rpats = rest["patient_id"].unique()
                rng = np.random.RandomState(seed * 100 + k)
                n_val = max(1, int(round(len(rpats) * val_frac)))
                val_pats = set(rng.choice(rpats, size=n_val, replace=False))
                val = rest[rest["patient_id"].isin(val_pats)]
                train = rest[~rest["patient_id"].isin(val_pats)]
            else:
                rng = np.random.RandomState(seed * 100 + k)
                val_idx = rng.choice(rest.index.to_numpy(),
                                     size=max(1, int(round(len(rest) * val_frac))), replace=False)
                val = rest.loc[val_idx]
                train = rest.drop(index=val_idx)

            _assert_leakage(train, val, test, arm)
            tag = f"{dataset_tag}_{sample_unit}_{arm}_rep{r}_fold{k}"
            for name, part in (("train", train), ("val", val), ("test", test)):
                p = os.path.join(out_dir, f"{tag}_{name}.csv")
                part.to_csv(p, index=False)
                written.append(p)
            print(f"  {tag}: train {len(train)} / val {len(val)} / test {len(test)} "
                  f"| test patients {test['patient_id'].nunique()} | leakage-check OK")
    print(f"[splits] wrote {len(written)} CSVs to {out_dir}")
    return written


def _assert_leakage(train, val, test, arm):
    tr, va, te = set(train.patient_id), set(val.patient_id), set(test.patient_id)
    if arm == "patient":
        assert tr.isdisjoint(te), "LEAKAGE: patient_id overlap train/test in patient arm"
        assert tr.isdisjoint(va), "LEAKAGE: patient_id overlap train/val in patient arm"
        assert va.isdisjoint(te), "LEAKAGE: patient_id overlap val/test in patient arm"
    else:  # random arm is EXPECTED to overlap patients (that is the point)
        assert (tr & te), "random arm should share patients across train/test (it did not)"


def main():
    ap = argparse.ArgumentParser(description="STAGE 3 — seeded Grouped-vs-Random K-Fold splits.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="lidc_binary")
    ap.add_argument("--sample-unit", default="slice", choices=["slice", "nodule"])
    ap.add_argument("--arm", default="patient", choices=["patient", "random"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit-patients", type=int, default=0, help="0=all; >0 = subset for validation")
    ap.add_argument("--cohort", default="principal", choices=["principal", "ge3"],
                    help="principal = min_annotators>=1 (headline); ge3 = >=3 sensitivity (D37), "
                         "distinct tag lidc_binary_ge3_...")
    args = ap.parse_args()
    cfg = load_config(args.config)
    make_splits(cfg, args.dataset, args.sample_unit, args.arm, args.repeats, args.folds,
                args.limit_patients or 0, cohort=args.cohort)


if __name__ == "__main__":
    main()
