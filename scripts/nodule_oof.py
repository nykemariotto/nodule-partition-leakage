"""
NODULE-LEVEL ESTIMATE: ONE OUT-OF-FOLD PREDICTION PER NODULE PER REPEAT.

Why. Scoring the nodule level fold by fold has a blind spot in the random arm. Its folds are
drawn over slices, so a nodule's slices are spread over several test folds, and a within-fold score
evaluates that nodule several times, each time on a subset of its slices and with a different model
(in this cohort a random-arm nodule appears in about three of the five test folds). Here each
repeat's out-of-fold slice probabilities are collected first -- in every arm each slice sits in
exactly one test fold per repeat, which is asserted against the full split -- and each nodule's
slice probabilities are then averaged. Every nodule is scored exactly once per repeat, in every arm,
on the same set of nodules, so the arms are compared on identical rows by construction.

Inference. Two sources of variation move a contrast, and each is measured where it can be. Which
patients are evaluated: a patient-cluster bootstrap (nodules of one patient are correlated) that
resamples the evaluated patients with the trained models held fixed, 2,000 draws, seed fixed before
anything here was computed, each draw applied to every arm, architecture and repeat so every
contrast is paired. Which partition and training run happened: the spread of each arm's three
repeats, which are evaluated on the same patients and differ only in partition and training. Two
arms share nothing within a repeat but the seed index, applied to different units, so their repeats
are independent and the repeat term of a contrast is (s_X^2 + s_Y^2) / R, whatever the pairing of
repeats; two architectures in one arm share each repeat's partition, so their comparison uses the
spread of the paired differences. The
verdict interval adds the two (`src.stats.run_aware_interval`): the claim is about the partitioning
PROTOCOL, and a difference that holds for one set of trained runs and reverses in the next is
run-to-run variation, not a protocol effect. The bootstrap interval alone holds the trained runs
fixed; it is kept and reported as "runs fixed", because that is what it answers. The statistic is
the mean over the three repeats of the per-repeat contrast.

Trade-off, stated rather than hidden. The estimator is not neutral between arms. Pooling a repeat's
predictions mixes the five fold models' calibrations across nodules, which lowers the AUC of the
grouped arms (A, C); in the random arm a nodule's prediction averages several fold models, which
raises it. So part of any difference from the within-fold estimate comes from the estimator, not
from repairing the blind spot. The within-fold estimate -- computed with `decompose_routes.collect`,
so it is the estimate that script reports -- is reproduced beside this one, and the per-arm shift
between the two is stored, so its size can be read rather than assumed.

Also produced, all on one prediction per nodule per repeat: per-model metrics at the 0.5 operating
point; confusion-matrix counts summed over the three repeats (every nodule exactly three times in
every panel); the within-nodule share of the combined gap per repeat; and within-arm architecture
comparisons (accuracy difference with the same two intervals). McNemar per repeat is stored as a
description only: it treats the nodules as independent and holds the runs fixed. Reads archived
probabilities only.

    python scripts/nodule_oof.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.artifacts import REPO, FULL_REPS, FULL_FOLDS, outputs_dir, read_split
from src.evaluate import _fold_metrics
from src.stats import (mean_ci, rho_from_splits, nadeau_bengio_ttest, mcnemar_from_counts,
                       cluster_bootstrap_draws, auc_rank, run_aware_interval)
import audit_controls
from audit_controls import load_arm_fold, run_tag
from decompose_routes import collect, gap

ARMS = {"A": "patient", "B": "random", "C": "nodule"}
CONTRASTS = (("B-A", "B", "A"), ("C-A", "C", "A"), ("B-C", "B", "C"))
GAP_METRICS = ("auc", "logloss", "brier")
HIGHER_IS_BETTER = {"auc": True, "logloss": False, "brier": False, "accuracy": True}
EPS = 1e-6                                               # same clipping as every other script
SEED, N_BOOT, ALPHA = 20261002, 2000, 0.05               # fixed before any computation
MCNEMAR_NOTE = ("descriptive only: treats the nodules as independent and holds the trained runs "
                "fixed; the verdict is the run-aware interval")


def slice_keys(frame):
    return frozenset(zip(frame["nodule_id"], frame["z_position"]))


def split_universe(cfg, arm, rep):
    """Every slice of one repeat's split (train + val + test of fold 0), as the loader sees it."""
    O = os.path.join(outputs_dir(cfg), "splits")
    parts = []
    for part in ("train", "val", "test"):
        parts.append(read_split(os.path.join(O, f"{run_tag(arm, rep, 0)}_{part}.csv")))
    u = pd.concat(parts, ignore_index=True)
    if u.duplicated(["nodule_id", "z_position"]).any():
        raise RuntimeError(f"{arm} rep{rep}: a slice in two parts of one fold's split")
    return slice_keys(u)


def oof_by_nodule(cfg, arch, arm, rep, folds, universe):
    """One repeat of one arm: every slice's out-of-fold probability, then one row per nodule."""
    parts = []
    for f in folds:
        d = load_arm_fold(cfg, arch, arm, rep, f)   # asserts row alignment and label order
        if d is None:
            raise SystemExit(f"missing run: {arch} {arm} rep{rep} fold{f}")
        parts.append(d[["patient_id", "nodule_id", "z_position", "label", "prob"]].assign(fold=f))
    s = pd.concat(parts, ignore_index=True)
    dup = s.duplicated(["nodule_id", "z_position"])
    if dup.any():
        raise RuntimeError(f"{arch} {arm} rep{rep}: {int(dup.sum())} slices scored in two test folds")
    keys = slice_keys(s)
    if keys != universe:
        raise RuntimeError(f"{arch} {arm} rep{rep}: the test folds cover {len(keys)} slices, the split "
                           f"has {len(universe)}; every slice must be tested exactly once")
    g = (s.groupby("nodule_id")
          .agg(patient_id=("patient_id", "first"), n_patient_ids=("patient_id", "nunique"),
               label=("label", "first"), n_labels=("label", "nunique"), prob=("prob", "mean"),
               n_slices=("prob", "size"), n_folds=("fold", "nunique"))
          .reset_index().sort_values("nodule_id").reset_index(drop=True))
    if (g["n_labels"] != 1).any() or (g["n_patient_ids"] != 1).any():
        raise RuntimeError(f"{arch} {arm} rep{rep}: a nodule with two labels or two patients")
    return g, keys


def point_metrics(y, p):
    """Every metric at the 0.5 operating point, through the same function the per-fold table uses
    (it asserts F1 == 2PR/(P+R)), and the fast AUC checked against sklearn on the way."""
    m = _fold_metrics(y, p)
    if abs(auc_rank(y, p) - roc_auc_score(y, p)) > 1e-12:
        raise AssertionError("rank AUC disagrees with sklearn")
    return {k: float(v) for k, v in m.items()}


def signed(metric, leakier, cleaner):
    """Positive = the leakier condition looks better, for every metric."""
    return leakier - cleaner if HIGHER_IS_BETTER[metric] else cleaner - leakier


def runs_fixed(boot):
    """The bootstrap's percentile interval: patient sampling only, trained runs held fixed."""
    return [float(np.percentile(boot, 100 * ALPHA / 2)),
            float(np.percentile(boot, 100 * (1 - ALPHA / 2)))]


def summary(per_rep, boot, components=None):
    ra, rf = run_aware_interval(per_rep, boot, ALPHA, components=components), runs_fixed(boot)
    if abs(ra["mean"] - float(np.mean(per_rep))) > 1e-12:
        raise AssertionError("run-aware mean is not the mean of the repeats")
    return dict(mean=ra["mean"], per_repeat=[float(v) for v in per_rep],
                ci95_run_aware=ra["ci"], excludes_zero=ra["excludes_zero"], se=ra["se"],
                df=ra["df"], sd_repeats=float(np.sqrt(ra["var_repeats"])),
                sd_boot=float(np.sqrt(ra["var_boot"])),
                ci95_runs_fixed=rf, excludes_zero_runs_fixed=bool(rf[0] > 0 or rf[1] < 0),
                df_satterthwaite=ra["df_satterthwaite"], ci95_satterthwaite=ra["ci_satterthwaite"],
                excludes_zero_satterthwaite=ra["excludes_zero_satterthwaite"],
                # counted on the pairing only where the pairing is real (the same partitions); for
                # two arms the contrast loop records counts that do not depend on it
                repeats_positive=int(sum(v > 0 for v in per_rep)) if components is None else None,
                n_repeats=len(per_rep),
                repeat_term="independent arms: (s_X^2 + s_Y^2) / R" if components is not None
                            else "paired differences: s^2 / R")


def show(d):
    ra, rf = d["ci95_run_aware"], d["ci95_runs_fixed"]
    return (f"{d['mean']:+.4f}  run-aware [{ra[0]:+.4f}, {ra[1]:+.4f}]{' *' if d['excludes_zero'] else '  '}"
            f"  runs fixed [{rf[0]:+.4f}, {rf[1]:+.4f}]{' *' if d['excludes_zero_runs_fixed'] else '  '}"
            f"  per repeat {' '.join(f'{v:+.4f}' for v in d['per_repeat'])}")


def main():
    ap = argparse.ArgumentParser(description="Nodule level with one prediction per nodule per repeat.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None, help="comma-separated; default: every complete one")
    ap.add_argument("--reps", default="0,1,2", help="the reported grid is 0,1,2; at least two")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--json", default=None, help="default: canonical path, full grid only")
    ap.add_argument("--dataset", default="lidc_binary")
    args = ap.parse_args()
    audit_controls.DATASET = args.dataset
    reps = sorted({int(x) for x in args.reps.split(",")})
    folds = [int(x) for x in args.folds.split(",")]
    if folds != list(FULL_FOLDS):
        raise SystemExit("every fold of a repeat is needed: a repeat's out-of-fold set is its five "
                         "test folds together")
    if len(reps) < 2:
        raise SystemExit("at least two repeats are needed: the verdict interval estimates the "
                         "run-to-run variance from the spread of the repeats")

    from src.artifacts import resolve_archs, name_for
    probs = os.path.join(REPO, "outputs", "probs")
    for arm in ARMS.values():
        if not glob.glob(os.path.join(probs, f"{args.dataset}_slice_{arm}_rep*_seed42.npz")):
            raise SystemExit(f"'{args.dataset}' has no '{arm}' arm on disk; this analysis compares "
                             f"all three arms")
    archs, present = resolve_archs(probs, args.dataset, args.archs,
                                   cells=(["slice"], list(ARMS.values()), reps, folds))
    # Sorted, because the order fixes the direction of every architecture comparison: a permuted
    # --archs would otherwise write sign-flipped comparisons under the canonical name.
    archs = sorted(archs)
    if args.json is None:
        if reps != list(FULL_REPS):
            print(f"  note   narrower than the reported grid (reps {reps}): the canonical JSON is "
                  f"left untouched. Pass --json PATH to save it.")
        else:
            args.json = os.path.join("outputs", "results",
                                     name_for("nodule_oof", args.dataset, archs, present) + ".json")

    cfg = load_config(args.config)
    # The draw count and level were fixed with the seed before anything was computed; they are also
    # the configuration's bootstrap settings, and a mismatch means one of the two moved.
    if (int(cfg["evaluate"]["bootstrap_n"]), round(1.0 - float(cfg["evaluate"]["ci"]), 10)) != (N_BOOT, ALPHA):
        raise SystemExit("config evaluate.bootstrap_n / evaluate.ci no longer match the fixed analysis")

    # ---------------------------------------------------------------- one prediction per nodule
    universes = {(arm, rep): split_universe(cfg, arm, rep) for arm in ARMS.values() for rep in reps}
    P, N_FOLDS_B, ref_nodules, ref_slices, ref_y, ref_pat = {}, {}, None, None, None, None
    for arch in archs:
        for key, arm in ARMS.items():
            for rep in reps:
                g, sl = oof_by_nodule(cfg, arch, arm, rep, folds, universes[(arm, rep)])
                if ref_nodules is None:
                    ref_nodules, ref_slices = g["nodule_id"].to_numpy(), sl
                    ref_y, ref_pat = g["label"].to_numpy(int), g["patient_id"].to_numpy()
                # every arm and repeat must score the same slices and the same nodules, same labels
                if sl != ref_slices or not np.array_equal(g["nodule_id"].to_numpy(), ref_nodules):
                    raise RuntimeError(f"{arch} {arm} rep{rep}: a different slice or nodule set")
                if not np.array_equal(g["label"].to_numpy(int), ref_y) or \
                        not np.array_equal(g["patient_id"].to_numpy(), ref_pat):
                    raise RuntimeError(f"{arch} {arm} rep{rep}: labels or patients differ")
                P[(arch, key, rep)] = np.clip(g["prob"].to_numpy(float), EPS, 1 - EPS)
                if key == "B":
                    N_FOLDS_B[(arch, rep)] = float(g["n_folds"].mean())
    y = ref_y
    n_nod, n_pat, n_sl = len(y), len(np.unique(ref_pat)), len(ref_slices)
    print(f"\n=== nodule level, one prediction per nodule per repeat | {len(archs)} architecture(s) x "
          f"{len(ARMS)} arms x {len(reps)} repeats | {n_sl} slices (every one tested once per repeat) "
          f"-> {n_nod} nodules, {n_pat} patients, {int(y.sum())} malignant ===")
    for arch in archs:
        print(f"  random arm, {arch}: a nodule's slices span "
              f"{np.mean([N_FOLDS_B[(arch, r)] for r in reps]):.2f} of the 5 test folds on average")

    # ---------------------------------------------------------------- point estimates
    per_model = {}
    for arch in archs:
        for key, arm in ARMS.items():
            reps_m = [point_metrics(y, P[(arch, key, r)]) for r in reps]
            per_model[f"{arch}|{arm}"] = dict(
                per_repeat=reps_m, mean={m: float(np.mean([d[m] for d in reps_m])) for m in reps_m[0]})

    # ---------------------------------------------------------------- paired patient bootstrap
    # per-nodule pieces precomputed once; a draw is then a re-weighting of them
    ll = {k: -(y * np.log(p) + (1 - y) * np.log(1 - p)) for k, p in P.items()}
    br = {k: (p - y) ** 2 for k, p in P.items()}
    ok = {k: ((p >= 0.5).astype(int) == y).astype(float) for k, p in P.items()}

    def boot_metrics(take):
        yt = y[take]
        return {k: dict(auc=auc_rank(yt, P[k][take]), logloss=float(ll[k][take].mean()),
                        brier=float(br[k][take].mean()), accuracy=float(ok[k][take].mean()))
                for k in P}

    pairs = list(combinations(archs, 2))
    boot_gap = {(a, c, m): [] for a in archs for c, _, _ in CONTRASTS for m in GAP_METRICS}
    boot_arch = {(arm, a1, a2): [] for arm in ARMS for a1, a2 in pairs}
    degenerate = 0
    for take in cluster_bootstrap_draws(ref_pat, N_BOOT, SEED):
        bm = boot_metrics(take)
        if any(np.isnan(v["auc"]) for v in bm.values()):      # a draw with a single class
            degenerate += 1
        for a in archs:
            for c, hi, lo in CONTRASTS:
                for m in GAP_METRICS:
                    boot_gap[(a, c, m)].append(np.mean([signed(m, bm[(a, hi, r)][m], bm[(a, lo, r)][m])
                                                        for r in reps]))
        for arm in ARMS:
            for a1, a2 in pairs:
                boot_arch[(arm, a1, a2)].append(np.mean([bm[(a1, arm, r)]["accuracy"] -
                                                         bm[(a2, arm, r)]["accuracy"] for r in reps]))
    if degenerate:
        raise RuntimeError(f"{degenerate} of {N_BOOT} bootstrap draws held a single class; an AUC "
                           f"interval built on them would be silently undefined")

    # ---------------------------------------------------------------- contrasts
    contrasts, shares = {}, {}
    print("\n  contrasts (positive = leakier arm looks better); 95% intervals, * = excludes zero;"
          "\n  the verdict is the run-aware interval; 'runs fixed' is the bootstrap alone")
    for a in archs:
        contrasts[a] = {}
        for c, hi, lo in CONTRASTS:
            for m in GAP_METRICS:
                hi_r = [per_model[f"{a}|{ARMS[hi]}"]["per_repeat"][i][m] for i in range(len(reps))]
                lo_r = [per_model[f"{a}|{ARMS[lo]}"]["per_repeat"][i][m] for i in range(len(reps))]
                per_rep = [signed(m, x, y) for x, y in zip(hi_r, lo_r)]
                d = contrasts[a][f"{c}_{m}"] = summary(per_rep, boot_gap[(a, c, m)],
                                                       components=[hi_r, lo_r])
                # Every repeat of one arm against every repeat of the other: a "positive in every
                # repeat" statement that does not depend on how the repeats are paired.
                d["cross_pairs_positive"] = int(sum(signed(m, x, y) > 0 for x in hi_r for y in lo_r))
                d["cross_pairs"] = len(hi_r) * len(lo_r)
                # and the repeats-positive count over every one-to-one pairing of the two arms' repeats
                counts = [sum(signed(m, hi_r[i], lo_r[j]) > 0 for i, j in enumerate(perm))
                          for perm in permutations(range(len(hi_r)))]
                d["repeats_positive_any_pairing"] = [int(min(counts)), int(max(counts))]
                print(f"    {a:16s} {c} {m:7s} {show(d)}")
        # within-nodule share of the combined gap, for the metrics that are means of per-sample
        # scores: a ratio of the two mean contrasts. It has no interval of its own, and it cannot
        # be told from 100% whenever the patient route cannot be told from zero. No per-repeat
        # ratios: the arms' repeats share only a seed index, so a ratio per repeat would depend on
        # how the repeats of the three arms were matched.
        shares[a] = {}
        for m in ("logloss", "brier"):
            bc, ba = contrasts[a][f"B-C_{m}"], contrasts[a][f"B-A_{m}"]
            shares[a][m] = dict(of_means=bc["mean"] / ba["mean"])
        print(f"    {a:16s} B-C as a share of B-A: "
              + "; ".join(f"{m} {shares[a][m]['of_means']:.3f}" for m in ("logloss", "brier")))

    # ---------------------------------------------------------------- architectures within an arm
    arch_cmp = {}
    if pairs:
        print("\n  within-arm architecture comparison: accuracy at 0.5, first minus second "
              "(McNemar per repeat: description only)")
    for key, arm in ARMS.items():
        for a1, a2 in pairs:
            per_rep, mcn = [], []
            for r in reps:
                c1, c2 = ok[(a1, key, r)].astype(bool), ok[(a2, key, r)].astype(bool)
                per_rep.append(float(c1.mean() - c2.mean()))
                mcn.append(mcnemar_from_counts(int((c1 & ~c2).sum()), int((~c1 & c2).sum())))
            d = arch_cmp[f"{arm}|{a1}|{a2}"] = dict(summary(per_rep, boot_arch[(key, a1, a2)]),
                                                    mcnemar_per_repeat=mcn, mcnemar_note=MCNEMAR_NOTE)
            mcn_p = " ".join(format(x["p"], ".2g") for x in mcn)
            print(f"    {arm:8s} {a1} - {a2}: {show(d)}  McNemar p {mcn_p}")

    # ---------------------------------------------------------------- confusion matrices
    conf = {}
    for a in archs:
        for key, arm in ARMS.items():
            tn = fp = fn = tp = 0
            for r in reps:
                yhat = (P[(a, key, r)] >= 0.5).astype(int)
                tn += int(((yhat == 0) & (y == 0)).sum()); fp += int(((yhat == 1) & (y == 0)).sum())
                fn += int(((yhat == 0) & (y == 1)).sum()); tp += int(((yhat == 1) & (y == 1)).sum())
            n = tn + fp + fn + tp
            if n != n_nod * len(reps):
                raise AssertionError(f"{a} {arm}: {n} scored nodules, expected {n_nod * len(reps)}")
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            mean = per_model[f"{a}|{arm}"]["mean"]
            # Two F1s exist and differ in the 4th decimal: one from the summed counts, one the mean
            # of the per-repeat F1s. Both are named, so a table and a figure cannot silently disagree.
            conf[f"{a}|{arm}"] = dict(tn=tn, fp=fp, fn=fn, tp=tp, n=n,
                                      accuracy=(tp + tn) / n,
                                      f1_from_summed_counts=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                                      f1_mean_over_repeats=mean["f1"],
                                      accuracy_mean_over_repeats=mean["accuracy"])

    # ---------------------------------------------------------------- within-fold, for comparison
    sens, shift = {}, {}
    for a in archs:
        rows = collect(cfg, a, reps, folds)
        if len(rows) != len(reps) * len(folds):
            raise RuntimeError(f"{a}: {len(rows)} complete folds, expected {len(reps) * len(folds)}")
        rho = rho_from_splits([r["n_test_A"] for r in rows], [r["n_train_A"] for r in rows])
        sens[a] = {"n_folds": len(rows)}
        for c, hi, lo in CONTRASTS:
            for m in GAP_METRICS:
                v = [gap(m, r[f"nod_{hi}"], r[f"nod_{lo}"]) for r in rows]
                v = np.asarray([x for x in v if x is not None], dtype=float)
                if len(v) != len(rows):
                    raise RuntimeError(f"{a} {c} {m}: a fold without a defined nodule-level metric")
                s, t = mean_ci(v, rho=rho), nadeau_bengio_ttest(v, rho)
                sens[a][f"{c}_{m}"] = dict(mean=s["mean"], ci_nb=s["ci_nb"],
                                           folds_positive=s["folds_positive"], k=s["k"], p_nb=t["p"])
        # how far the pooled estimate sits from the within-fold one, arm by arm
        shift[a] = {}
        for key, arm in ARMS.items():
            wf = {m: float(np.mean([r[f"nod_{key}"][m] for r in rows])) for m in GAP_METRICS}
            shift[a][arm] = {m: per_model[f"{a}|{arm}"]["mean"][m] - wf[m] for m in GAP_METRICS}
        print(f"  within-fold, {a}: B-A AUC {sens[a]['B-A_auc']['mean']:+.4f} "
              f"NB95 [{sens[a]['B-A_auc']['ci_nb'][0]:+.4f}, {sens[a]['B-A_auc']['ci_nb'][1]:+.4f}] "
              f"(one prediction per nodule per repeat: {contrasts[a]['B-A_auc']['mean']:+.4f}); "
              f"AUC shift, pooled minus within-fold: "
              + " ".join(f"{arm} {shift[a][arm]['auc']:+.4f}" for arm in ARMS.values()))

    out = dict(
        _about=dict(estimator="one out-of-fold prediction per nodule per repeat: the repeat's five "
                              "test folds together, slice probabilities averaged per nodule",
                    inference="verdict: run-aware interval, mean +/- t(0.975, R-1) * sqrt(repeat term "
                              "+ var of the bootstrap statistic). Repeat term: for a contrast between "
                              "two arms, whose repeats share only a seed index, (s_X^2 + s_Y^2) / R from "
                              "each arm's per-repeat values; for two architectures within an arm (same "
                              "partitions), the variance of the per-repeat differences / R. Beside it, "
                              f"the patient-cluster percentile bootstrap alone ({N_BOOT} draws, seed "
                              f"{SEED}, each draw shared by every arm, architecture and repeat), which "
                              "holds the trained runs fixed, and a Welch-Satterthwaite version of the "
                              "run-aware interval as a sensitivity to the df convention; statistic = "
                              "the mean contrast over repeats",
                    fields="sd_repeats: square root of the repeat variances summed over the two arms, "
                           "or the SD of the per-repeat differences within an arm. per_repeat of a "
                           "contrast between arms pairs the repeats by index and is given for "
                           "reference: the interval, cross_pairs_positive (every repeat of one arm "
                           "against every repeat of the other) and repeats_positive_any_pairing "
                           "(lowest and highest count over the one-to-one pairings) do not depend on "
                           "it; repeats_positive is given only within an arm",
                    sign="positive = the leakier condition looks better",
                    estimator_shift="pooled minus within-fold, per arm; pooling lowers the grouped "
                                    "arms' AUC and raises the random arm's",
                    dataset=args.dataset, reps=reps, archs=archs, n_slices=n_sl, n_nodules=n_nod,
                    n_patients=n_pat, n_malignant=int(y.sum()), bootstrap_degenerate_draws=degenerate),
        random_arm_test_folds_per_nodule={a: float(np.mean([N_FOLDS_B[(a, r)] for r in reps]))
                                          for a in archs},
        per_model=per_model, contrasts=contrasts, within_nodule_share=shares,
        architecture_comparisons=arch_cmp, confusion_matrices=conf, within_fold_sensitivity=sens,
        estimator_shift_pooled_minus_within_fold=shift)
    if args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(REPO, args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(out, fh, indent=1, allow_nan=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
