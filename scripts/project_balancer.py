"""
PROJECT THE GAIN FROM A NEAR-PERFECT FOLD BALANCER — no GPU, no retraining (D28(iii), rerun on 740).

Question: stratification is inert here (D28(ii)) — but that used sklearn's off-the-shelf splitter.
Would a balancer that ACTUALLY achieves near-perfect fold prevalence buy anything? If even a
15x improvement in the quantity stratification optimises leaves the fold-to-fold METRIC variance
essentially unchanged, then fold prevalence is not what drives the variance, and only n can tighten
the interval (D28(iv)).

WHY THIS EXISTS: the D28(iii) measurement was made on the 250-patient SANITY pool. D35 bars
250-stage numbers from the manuscript, and D54 makes cohort provenance a hard filter, so the claim
had to be recomputed on the 740 cohort or dropped. This script recomputes it.

METHOD (identical in spirit to project_stratification.py). Predictions are held FIXED — the archived
per-slice probabilities of the PATIENT arm, aggregated to one vote per nodule — and only the
PARTITION is redrawn:
    current    shuffled round-robin over patients (verbatim src/splits.py::_fold_assign)
    balanced   the same partition, then a swap-based local search over patient pairs that
               greedily minimises the SD of per-fold positive rate
For each draw we recompute the per-fold metric and its fold-to-fold SD under both partitions.

WHAT THIS ISOLATES, and what it does NOT (same caveat as project_stratification.py): it measures
only the component of fold-to-fold variance caused by FOLD COMPOSITION, which is the only component
a balancer can remove. Under a real balanced CV the models would be retrained on different training
sets, so the predictions themselves would change; that component is NOT captured. INDICATIVE, and
labelled as such wherever it is used.

    python scripts/project_balancer.py --draws 200 --json outputs/metrics/balancer_projection_740.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metadata import load_config
from src.datasets import load_processed_index
from project_stratification import (K, fold_assign_current, load_pooled, to_nodule,
                                    evaluate_partition)


def prevalence_sd(fold_of_pat, pats, pos, tot):
    """SD of the per-fold positive rate, given a patient->fold map (vectors aligned to `pats`)."""
    f = np.array([fold_of_pat[p] for p in pats])
    rates = []
    for k in range(K):
        t = tot[f == k].sum()
        if t == 0:
            return np.inf
        rates.append(pos[f == k].sum() / t)
    return float(np.std(rates))


def balance_by_swaps(fold_of_pat, pats, pos, tot, rng, max_iter=4000, patience=400):
    """Greedy pairwise-swap local search minimising per-fold prevalence SD. Deterministic given rng."""
    f = np.array([fold_of_pat[p] for p in pats])

    def sd_of(fv):
        rates = []
        for k in range(K):
            t = tot[fv == k].sum()
            if t == 0:
                return np.inf
            rates.append(pos[fv == k].sum() / t)
        return float(np.std(rates))

    best = sd_of(f)
    since = 0
    n = len(pats)
    for _ in range(max_iter):
        i, j = rng.randint(0, n), rng.randint(0, n)
        if f[i] == f[j]:
            continue
        f[i], f[j] = f[j], f[i]
        s = sd_of(f)
        if s < best - 1e-12:
            best, since = s, 0
        else:
            f[i], f[j] = f[j], f[i]      # revert
            since += 1
            if since >= patience:
                break
    return {p: int(k) for p, k in zip(pats, f)}, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    out = {}

    for arch in args.archs.split(","):
        pooled = load_pooled(cfg, arch, "patient", pidx)
        if pooled is None:
            print(f"{arch}: patient-arm probs incomplete — skipped")
            continue
        unit = to_nodule(pooled)
        pats = np.array(sorted(unit["patient_id"].unique()))
        g = unit.groupby("patient_id")["label"]
        pos = g.sum().reindex(pats).to_numpy().astype(float)
        tot = g.size().reindex(pats).to_numpy().astype(float)

        cur = {m: [] for m in ("auc", "ll", "br")}
        bal = {m: [] for m in ("auc", "ll", "br")}
        sd_cur_rate, sd_bal_rate = [], []

        for d in range(args.draws):
            fa = fold_assign_current(len(pats), K, seed=1000 + d)
            fmap = {p: int(k) for p, k in zip(pats, fa)}
            r_cur = evaluate_partition(unit, fmap)
            if r_cur is None:
                continue
            rng = np.random.RandomState(50000 + d)
            fmap_b, sd_b = balance_by_swaps(fmap, pats, pos, tot, rng)
            r_bal = evaluate_partition(unit, fmap_b)
            if r_bal is None:
                continue
            sd_cur_rate.append(float(np.std(r_cur["rate"])))
            sd_bal_rate.append(sd_b)
            for m in ("auc", "ll", "br"):
                cur[m].append(float(r_cur[m].std(ddof=1)))
                bal[m].append(float(r_bal[m].std(ddof=1)))

        if not sd_cur_rate:
            print(f"{arch}: no usable draws")
            continue

        print(f"\n=== {arch} | patient arm | {len(sd_cur_rate)} draws | 740 cohort "
              f"({len(pats)} patients, {len(unit)} nodules) ===")
        print(f"  fold-prevalence SD : current {np.mean(sd_cur_rate):.4f}  ->  "
              f"balanced {np.mean(sd_bal_rate):.4f}   "
              f"({np.mean(sd_cur_rate)/max(np.mean(sd_bal_rate),1e-12):.1f}x better)")
        res = {"n_patients": int(len(pats)), "n_nodules": int(len(unit)),
               "draws": len(sd_cur_rate),
               "rate_sd_current": float(np.mean(sd_cur_rate)),
               "rate_sd_balanced": float(np.mean(sd_bal_rate))}
        for m, lbl in (("auc", "AUC"), ("ll", "log-loss"), ("br", "Brier")):
            c, b = float(np.mean(cur[m])), float(np.mean(bal[m]))
            pct = 100.0 * (b - c) / c
            print(f"  fold-to-fold {lbl:9s} SD: current {c:.4f}  ->  balanced {b:.4f}   "
                  f"({pct:+.1f}%)")
            res[f"metric_sd_current_{m}"] = c
            res[f"metric_sd_balanced_{m}"] = b
            res[f"metric_sd_pct_change_{m}"] = pct
        out[arch] = res

    if args.json and out:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
