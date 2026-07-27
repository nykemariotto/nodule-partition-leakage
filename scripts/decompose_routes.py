"""
D43 ROUTE DECOMPOSITION — arm A (patient) vs arm B (random) vs ARM C (nodule-grouped).

    A = patient-grouped : route 1 OFF, route 2 OFF   (L_nod 0.000, L_pat 0.000)
    B = random          : route 1 ON,  route 2 ON    (L_nod 0.973, L_pat 0.995)
    C = nodule-grouped  : route 1 OFF, route 2 ON    (L_nod 0.000, L_pat 0.690)  <- full density

    gap(B-A) = total          gap(C-A) = route 2 (patient)     gap(B-C) = route 1 (within-nodule)

READ THE CAVEATS (DECISIONS D43, corrected 2026-07-25) BEFORE WRITING PROSE FROM THIS OUTPUT:
  * The telescoping identity [M(C)-M(A)] + [M(B)-M(C)] = M(B)-M(A) holds for EVERY metric, AUC
    included -- it is arithmetic on three numbers. This script verifies it numerically as a bug
    check, NOT as evidence about any metric's structure.
  * ATTRIBUTION is the real caveat: reading gap(C-A) as "the share of the total produced by route 2"
    needs the metric to decompose over the ROWS a route acts on. log-loss/Brier/accuracy do (means of
    per-sample scores); AUC does NOT (rank statistic over pairs). So AUC is reported side by side and
    no "route X contributed Y of the AUC gap" sentence may be written.
  * NON-INTERACTION is assumed for every metric: A, B, C are three SEPARATELY TRAINED models, so the
    parts are experimental conditions, not an algebraic split of one quantity.

Reads ARCHIVED probs only (no GPU, safe to run while a grid trains). Reuses audit_controls'
loader, which asserts split/probs alignment and label order, and src.stats for the intervals (D27).

    python scripts/decompose_routes.py --reps 0,1,2 --json outputs/_analysis/decomposition.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.datasets import load_processed_index
from src.stats import mean_ci, rho_from_splits, wilcoxon_paired
import audit_controls
from audit_controls import load_arm_fold, metrics, by_nodule

# gap sign convention, identical to audit_controls: positive ALWAYS means "the leakier arm looks
# better", so AUC is (leakier - cleaner) and log-loss/Brier are (cleaner - leakier).
HIGHER_IS_BETTER = {"auc": True, "logloss": False, "brier": False}


def gap(metric, leakier, cleaner):
    if leakier is None or cleaner is None:
        return None
    return (leakier[metric] - cleaner[metric]) if HIGHER_IS_BETTER[metric] \
        else (cleaner[metric] - leakier[metric])


def collect(cfg, arch, pidx, reps, folds):
    """Per-fold metrics for the three arms at both levels."""
    out = []
    for rep in reps:
        for f in folds:
            row = {"rep": rep, "fold": f}
            ok = True
            for arm, key in (("patient", "A"), ("random", "B"), ("nodule", "C")):
                d = load_arm_fold(cfg, arch, arm, rep, f, pidx)
                if d is None:
                    ok = False
                    break
                row[f"slice_{key}"] = metrics(d["label"], d["prob"])
                row[f"nod_{key}"] = by_nodule(d)
                row[f"n_test_{key}"] = len(d)
            if ok:
                out.append(row)
    return out


def report(rows, level, metric, cfg_label, n_test, n_train):
    """Print the three gaps for one (level, metric) and verify the telescoping identity."""
    tot = [gap(metric, r[f"{level}_B"], r[f"{level}_A"]) for r in rows]
    r2 = [gap(metric, r[f"{level}_C"], r[f"{level}_A"]) for r in rows]
    r1 = [gap(metric, r[f"{level}_B"], r[f"{level}_C"]) for r in rows]
    tot, r2, r1 = [[x for x in v if x is not None] for v in (tot, r2, r1)]
    if len(tot) < 2:
        print(f"  {cfg_label}: not enough folds yet ({len(tot)})")
        return None

    rho = rho_from_splits(n_test, n_train)
    res = {}
    for name, vals in (("total  B-A", tot), ("route2 C-A", r2), ("route1 B-C", r1)):
        s = mean_ci(np.asarray(vals), rho=rho)
        w = wilcoxon_paired(np.asarray(vals))
        pos = sum(1 for x in vals if x > 0)
        nb = s["ci_nb"]
        flag = "" if (nb[0] > 0 or nb[1] < 0) else "  <- includes 0"
        print(f"    {name:11s} {s['mean']:+.4f}  NB95 [{nb[0]:+.4f},{nb[1]:+.4f}]  "
              f"{pos:2d}/{len(vals)}  p={w['p']:.4f}"
              f"{' (AT FLOOR)' if w.get('at_floor') else ''}{flag}")
        res[name.split()[0]] = {"mean": float(s["mean"]), "nb95": [float(nb[0]), float(nb[1])],
                                "folds_pos": pos, "n": len(vals), "p": float(w["p"])}
    # bug check: the telescoping must hold per fold, for every metric
    resid = max(abs((a + b) - t) for a, b, t in zip(r2, r1, tot))
    status = "OK" if resid < 1e-9 else f"BROKEN (max residual {resid:.2e})"
    print(f"    telescoping (C-A)+(B-C) == (B-A): {status}")
    res["telescoping_max_residual"] = float(resid)
    return res


def main():
    ap = argparse.ArgumentParser(description="D43 route decomposition (arms A/B/C).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort, D37 -- never pooled)")
    args = ap.parse_args()
    # audit_controls builds the run tags, so its module-level prefix is what must change
    audit_controls.DATASET = args.dataset

    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    out = {}
    for arch in args.archs.split(","):
        rows = collect(cfg, arch, pidx, reps, folds)
        print(f"\n=== {arch} | arms A(patient) / B(random) / C(nodule-grouped) | "
              f"{len(rows)}/{len(reps)*len(folds)} folds available ===")
        if not rows:
            print("  no complete (A,B,C) folds yet — arm C still training?")
            continue
        n_test = int(np.mean([r["n_test_A"] for r in rows]))
        n_train = int(np.mean([r["n_test_A"] for r in rows]) * (len(folds) - 1))
        arch_res = {}
        for level, lname in (("slice", "SLICE level (primary, D17(i))"),
                             ("nod", "NODULE-AGGREGATED (confirmatory, D26)")):
            print(f"  {lname}")
            for metric in ("auc", "logloss", "brier"):
                note = "  [attribution valid]" if metric != "auc" else \
                       "  [AUC: side-by-side only; NO route-share attribution -- D43]"
                print(f"   {metric.upper()}{note}")
                arch_res[f"{level}_{metric}"] = report(rows, level, metric, arch, n_test, n_train)
        out[arch] = arch_res

    if args.json and out:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
