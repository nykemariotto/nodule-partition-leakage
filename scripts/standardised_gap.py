"""
D53 normalisation: the leakage gap expressed in fold-SD units, so AUC, log-loss and Brier can be
compared on one scale.

    z_M = mean_gap_M / SD_patient(M)

where SD_patient(M) is the standard deviation of the PATIENT-arm value of that metric across the 15
folds. The formula, the reading rule (a concentration claim needs both calibration z's at >= 1.5x
z_AUC) and the caveats were all fixed in DECISIONS D53 BEFORE any of these numbers existed; the
outcome recorded there is that the claim is NOT supported and the Discussion says so.

This existed only as a one-off computation whose numbers were transcribed into D53 and into the
Discussion. Adding a third architecture left no way to extend them without re-deriving the
normalisation by hand, which is exactly the situation D35 forbids -- so it is a script, and its
output is an artifact the freshness gate can check.

    python scripts/standardised_gap.py --reps 0,1,2

Slice level only: D53 declares the slice level as the level at which the cross-metric comparison is
made, matching the pre-registered primary analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.audit_controls as ac
from src.metadata import load_config
from src.datasets import load_processed_index
from src.artifacts import resolve_archs, name_for

METRICS = (("auc", "AUC"), ("logloss", "log-loss"), ("brier", "Brier"))


def per_fold(cfg, arch, reps, folds, pidx):
    """Per-fold patient-arm value and random-minus-patient gap, for each metric.

    Sign convention matches the rest of the paper: positive always means the leaky arm looks
    better, so log-loss and Brier are flipped (patient minus random) while AUC is not.
    """
    pat = {k: [] for k, _ in METRICS}
    gap = {k: [] for k, _ in METRICS}
    for rep in reps:
        for f in folds:
            d = {a: ac.load_arm_fold(cfg, arch, a, rep, f, pidx) for a in ("patient", "random")}
            if any(v is None for v in d.values()):
                continue
            m = {a: ac.metrics(d[a]["label"], d[a]["prob"]) for a in d}
            if any(v is None for v in m.values()):
                continue
            for k, _ in METRICS:
                pat[k].append(m["patient"][k])
                gap[k].append(m["random"][k] - m["patient"][k] if k == "auc"
                              else m["patient"][k] - m["random"][k])
    return pat, gap


def main():
    ap = argparse.ArgumentParser(description="D53 fold-SD standardised gap (slice level).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None,
                    help="comma-separated. DEFAULT: every architecture with runs on disk.")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dataset", default="lidc_binary")
    ap.add_argument("--json", default="outputs/metrics/standardised_gap.json")
    args = ap.parse_args()

    ac.DATASET = args.dataset
    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    probs = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "probs")
    archs, present = resolve_archs(probs, args.dataset, args.archs,
                                   cells=(["slice"], ["patient", "random"], reps, folds))

    out, rows = {}, []
    for arch in archs:
        pat, gap = per_fold(cfg, arch, reps, folds, pidx)
        out[arch] = {}
        for k, lab in METRICS:
            g = float(np.mean(gap[k]))
            mu = float(np.mean(pat[k]))
            # ddof=1: the 15 folds are a sample, and the same convention is used for every
            # interval in the paper (src/stats.py).
            sd = float(np.std(pat[k], ddof=1))
            out[arch][k] = dict(gap=g, patient_mean=mu, patient_sd=sd, k=len(gap[k]),
                                z=g / sd, r=g / abs(mu))
            rows.append((arch, lab, g, mu, sd, g / sd, g / abs(mu)))
        z = out[arch]
        out[arch]["_ratios"] = {"z_logloss/z_auc": z["logloss"]["z"] / z["auc"]["z"],
                                "z_brier/z_auc": z["brier"]["z"] / z["auc"]["z"]}

    print(f"{'arch':17}{'metric':10}{'gap':>9}{'pat mean':>10}{'SD_pat':>9}{'z':>7}{'r':>8}")
    for arch, lab, g, mu, sd, z, r in rows:
        print(f"{arch:17}{lab:10}{g:+9.4f}{mu:10.4f}{sd:9.4f}{z:7.2f}{r:8.3f}")
    print("\nD53 reading rule: a 'concentrated in calibration' claim requires BOTH z_logloss/z_auc "
          "and z_brier/z_auc >= 1.5.")
    for arch in archs:
        r = out[arch]["_ratios"]
        verdict = ("SUPPORTED" if min(r.values()) >= 1.5 else
                   "INVERTED (z_AUC largest)" if max(r.values()) < 1.0 else "NOT supported")
        print(f"  {arch:17} z_ll/z_auc {r['z_logloss/z_auc']:.2f}  "
              f"z_br/z_auc {r['z_brier/z_auc']:.2f}  -> {verdict}")

    d = os.path.dirname(args.json)
    stem = os.path.basename(args.json)[:-5]
    p = os.path.join(cfg["project"]["root"], d,
                     name_for(stem.split("__")[0], args.dataset, archs, present) + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"_note": "D53 normalisation, slice level. z = mean_gap / SD of the patient-arm "
                        "value across folds. r is the SECONDARY relative gap, reported but not the "
                        "basis of any claim.",
               "reps": reps, "folds": folds, "per_arch": out}, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
