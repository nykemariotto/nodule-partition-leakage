"""
PAIRED TEST ACROSS THE 15 FOLDS -- the Nadeau-Bengio corrected resampled t-test.

The 15 fold-level differences of a contrast are not 15 independent pairs: within a repeat the
folds share training data, and the three repeats re-use the same patients. The signed-rank test
treats them as independent. This script reports, for every contrast whose interval the paper
prints, the test built on the SAME corrected standard error, s * sqrt(1/k + rho), and the same t
reference (k - 1 df) as that interval. It is therefore the interval's exact dual -- p < 0.05 if and
only if the Nadeau-Bengio interval excludes zero -- and the script asserts that for every row, so
the printed verdict and the printed interval cannot disagree.

Contrasts, per architecture, at the slice level (primary) and at the within-fold nodule level
(kept as a sensitivity analysis; the confirmatory nodule estimate is `nodule_oof.py`):
    B-A  random - patient          (combined effect)
    C-A  nodule-grouped - patient  (patient route)
    B-C  random - nodule-grouped   (within-nodule route)
for AUC, log-loss and Brier, positive meaning the leakier arm looks better. Per-fold values and rho
come from `decompose_routes.collect`, so the means and intervals here are the ones it reports.

The arms are paired by repeat and fold INDEX. Their test sets differ by construction (each arm
forms its folds over a different unit), so the pairing is a convention, not a match of test
content; the matched-rows control in `audit_controls.py` is what holds content fixed. Re-pairing
the folds within a repeat leaves every verdict unchanged but can move a p-value by an order of
magnitude, which is why p is read to one significant figure.

Reads archived probabilities only (no GPU). Defaults to the full reported grid; a narrower run
estimates a different quantity and does not write the canonical file.

    python scripts/fold_tests.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.artifacts import REPO, FULL_REPS, FULL_FOLDS
from src.stats import mean_ci, rho_from_splits, nadeau_bengio_ttest
import audit_controls
from decompose_routes import collect, gap

CONTRASTS = (("B-A", "B", "A"), ("C-A", "C", "A"), ("B-C", "B", "C"))
LEVELS = (("slice", "slice (primary)"), ("nod", "nodule, within-fold (sensitivity)"))
METRICS = ("auc", "logloss", "brier")
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser(description="Nadeau-Bengio corrected t-test for every fold contrast.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None, help="comma-separated; default: every complete one")
    ap.add_argument("--reps", default="0,1,2", help="the reported grid is 0,1,2 (n = 15)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--json", default=None, help="default: canonical path, full grid only")
    ap.add_argument("--dataset", default="lidc_binary")
    args = ap.parse_args()
    audit_controls.DATASET = args.dataset
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    from src.artifacts import resolve_archs, name_for
    probs = os.path.join(REPO, "outputs", "probs")
    for arm in ("patient", "random", "nodule"):
        if not glob.glob(os.path.join(probs, f"{args.dataset}_slice_{arm}_rep*_seed42.npz")):
            raise SystemExit(f"'{args.dataset}' has no '{arm}' arm on disk; this analysis compares "
                             f"all three arms")
    archs, present = resolve_archs(probs, args.dataset, args.archs,
                                   cells=(["slice"], ["patient", "random", "nodule"], reps, folds))
    archs = sorted(archs)                       # one fixed order, whatever --archs says
    if args.json is None:
        if sorted(reps) != list(FULL_REPS) or sorted(folds) != list(FULL_FOLDS):
            print(f"  note   narrower than the reported grid (reps {reps} x folds {folds}): the "
                  f"canonical JSON is left untouched. Pass --json PATH to save it.")
        else:
            args.json = os.path.join("outputs", "_analysis",
                                     name_for("fold_tests", args.dataset, archs, present) + ".json")

    cfg = load_config(args.config)
    out = {"_about": dict(test="Nadeau-Bengio corrected resampled t-test, two-sided",
                          se="s * sqrt(1/k + rho), rho = mean(n_test / n_train) of arm A's folds",
                          df="k - 1", dual_of="the Nadeau-Bengio 95% interval (asserted per row)",
                          dataset=args.dataset, reps=reps, folds=folds, archs=archs)}

    for arch in archs:
        rows = collect(cfg, arch, reps, folds)
        n = len(rows)
        print(f"\n=== {arch} | n = {n} folds (rep x fold) of {len(reps) * len(folds)} requested ===")
        if n < 2:
            print("  not enough complete (A, B, C) folds")
            continue
        rho = rho_from_splits([r["n_test_A"] for r in rows], [r["n_train_A"] for r in rows])
        res = {"n_folds": n, "rho": rho}
        for level, lname in LEVELS:
            print(f"  {lname}")
            for metric in METRICS:
                for cname, hi, lo in CONTRASTS:
                    vals = [gap(metric, r[f"{level}_{hi}"], r[f"{level}_{lo}"]) for r in rows]
                    vals = [v for v in vals if v is not None]
                    if len(vals) < 2:
                        print(f"    {metric:7s} {cname}  fewer than two folds with a defined value")
                        continue
                    s = mean_ci(np.asarray(vals), rho=rho, alpha=ALPHA)
                    t = nadeau_bengio_ttest(np.asarray(vals), rho)
                    if (t["p"] < ALPHA) != s["excludes_zero_nb"]:
                        raise AssertionError(f"{arch} {level} {metric} {cname}: test p={t['p']:.4g} "
                                             f"disagrees with the interval {s['ci_nb']}")
                    t_txt = "inf (no spread)" if t["t"] is None else f"{t['t']:+.3f}"
                    print(f"    {metric:7s} {cname}  {s['mean']:+.4f}  NB95 [{s['ci_nb'][0]:+.4f},"
                          f"{s['ci_nb'][1]:+.4f}]  t({t['df']}) = {t_txt}  p = {t['p']:.1g}"
                          f"  folds+ {s['folds_positive']}/{s['k']}")
                    res[f"{level}_{metric}_{cname}"] = dict(
                        mean=s["mean"], ci_nb=s["ci_nb"], excludes_zero_nb=s["excludes_zero_nb"],
                        folds_positive=s["folds_positive"], k=s["k"], t=t["t"], df=t["df"],
                        p=t["p"], per_fold=s["per_fold"])
        out[arch] = res

    if args.json and len(out) == 1:
        print("\nno architecture produced a result; nothing written")
    elif args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(REPO, args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(out, fh, indent=1, allow_nan=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
