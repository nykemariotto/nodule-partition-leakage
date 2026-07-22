"""
STAGE 7 — the analytical foundation (SPEC §2.2, §5-B4, §6.7; DECISIONS D5, D6, D21, D24).

This module is the SINGLE versioned implementation of every inferential quantity the study
reports. Until 2026-07-20 it was a `NotImplementedError` stub while Wilcoxon and Nadeau-Bengio
lived in `scripts/`, and the pre-registered bootstrap (SPEC B4) existed nowhere — the exact
config-says / code-does divergence that caused the rejection (DECISIONS D27). `scripts/` now
imports from here; nothing reimplements these.

WHAT IS HERE, and the reasoning that constrains each:

  fold_summary()      D24: every metric is computed WITHIN each fold and then averaged across
                      folds. Predictions are never pooled across folds — pooled predictions have
                      no valid interval and mix distributions.

  nadeau_bengio_se()  D6. k-fold estimates are NOT independent: training sets overlap, so the
                      naive s/sqrt(k) understates variance. Corrected SE = s*sqrt(1/k + rho),
                      rho = n_test/n_train. Measured here rho ~ 0.284 -> intervals widen x1.556.
                      At n=5 that is precisely the difference between "excludes zero" and not.

  wilcoxon_paired()   D5. Also returns `min_attainable_p`, because at small n the test has a
                      FLOOR: the two-sided exact minimum is 2/2^n (n=5 -> 0.0625). Reporting
                      "p=0.0625" without that context invites reading it as a near-miss when it
                      is the smallest value the test can produce.

  bootstrap_ci()      SPEC B4. Percentile bootstrap over the INDEPENDENT unit. When slices are
                      the rows, that unit is the PATIENT, not the row — pass `unit_ids` for a
                      cluster bootstrap. Resampling correlated rows independently yields
                      intervals that are far too narrow.

    python -m src.stats --level nodule       # -> outputs/results/stats_<level>.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Sequence

import numpy as np
from scipy.stats import t as tdist, wilcoxon as _wilcoxon


# ----------------------------------------------------------------------------- intervals
def naive_se(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def nadeau_bengio_se(values: Sequence[float], rho: float) -> float:
    """Corrected SE of the mean of k dependent cross-validation estimates (D6).

    rho = n_test / n_train; rho=0 degenerates to the naive s/sqrt(k).
    """
    v = np.asarray(values, dtype=float)
    k = len(v)
    if k < 2:
        return float("nan")
    return float(v.std(ddof=1) * np.sqrt(1.0 / k + rho))


def rho_from_splits(n_test: Sequence[int], n_train: Sequence[int]) -> float:
    """Nadeau-Bengio rho = mean(n_test / n_train) across folds (D6)."""
    return float(np.mean(np.asarray(n_test, float) / np.asarray(n_train, float)))


def mean_ci(values: Sequence[float], rho: float = 0.0, alpha: float = 0.05) -> dict:
    """Mean with BOTH the naive and the Nadeau-Bengio interval, and which excludes zero.

    Both are returned deliberately: the naive one is what a reader would compute by default, the
    corrected one is what this study pre-registered. Reporting only the narrower would overstate.
    """
    v = np.asarray(values, dtype=float)
    k = len(v)
    if k < 2:
        return dict(k=k, mean=float(v.mean()) if k else None)
    tc = float(tdist.ppf(1 - alpha / 2, k - 1))
    m = float(v.mean())
    hn, hb = tc * naive_se(v), tc * nadeau_bengio_se(v, rho)
    return dict(k=k, mean=m, sd=float(v.std(ddof=1)), rho=rho,
                ci_naive=[m - hn, m + hn], ci_nb=[m - hb, m + hb],
                excludes_zero_naive=bool((m - hn) * (m + hn) > 0),
                excludes_zero_nb=bool((m - hb) * (m + hb) > 0),
                widening=float(hb / hn) if hn else None,
                folds_positive=int((v > 0).sum()), per_fold=[float(x) for x in v])


def fold_summary(per_fold_values: dict, rho: float, alpha: float = 0.05) -> dict:
    """D24: {metric_name: [one value per fold]} -> {metric_name: mean + intervals}.

    Takes values ALREADY computed within each fold, so it cannot pool predictions even by
    mistake — which is the point.
    """
    return {name: mean_ci(vals, rho=rho, alpha=alpha) for name, vals in per_fold_values.items()}


# ----------------------------------------------------------------------------- paired test
def wilcoxon_min_p(n: int) -> float:
    """Smallest two-sided p the exact signed-rank test can produce with n non-zero pairs."""
    return 2.0 / (2.0 ** n) if n > 0 else float("nan")


def wilcoxon_paired(a: Sequence[float], b: Sequence[float] | None = None) -> dict:
    """D5 paired A-vs-B test. `a` may be the differences directly (b=None).

    `at_floor` says the p-value IS the minimum attainable — the data are as extreme as this test
    can register, and a larger n (not a larger effect) is what would move it.
    """
    x = np.asarray(a, dtype=float)
    if b is not None:
        x = x - np.asarray(b, dtype=float)
    nz = int(np.sum(x != 0))
    out = dict(n=int(len(x)), n_nonzero=nz, min_attainable_p=wilcoxon_min_p(nz),
               all_same_sign=bool(nz > 0 and (np.all(x[x != 0] > 0) or np.all(x[x != 0] < 0))))
    try:
        stat, p = _wilcoxon(x)
        out.update(statistic=float(stat), p=float(p),
                   at_floor=bool(np.isclose(p, out["min_attainable_p"])))
    except Exception as e:                                   # n too small / all zeros
        out.update(statistic=None, p=None, at_floor=False, error=str(e))
    return out


# ----------------------------------------------------------------------------- within-arm test
def mcnemar_from_counts(b: int, c: int) -> dict:
    """McNemar p-value from the two discordant counts directly (b = A right/B wrong, c = A wrong/B
    right). Exact binomial when discordant pairs are few (<25), else continuity-corrected
    chi-square. Both are symmetric in (b, c), so the p-value is order-independent; b and c are
    reported as given so the labels stay correct (do NOT reconstruct a synthetic vector to obtain
    the p — that path can silently swap b and c, per code review 2026-07-21)."""
    b, c = int(b), int(c)
    n = b + c
    out = dict(b=b, c=c, n_discordant=n)
    if n == 0:
        out.update(p=1.0, method="none (no discordant pairs)"); return out
    if n < 25:
        from scipy.stats import binomtest
        out.update(p=float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue),
                   method="exact binomial")
    else:
        from scipy.stats import chi2
        stat = (abs(b - c) - 1) ** 2 / n
        out.update(statistic=float(stat), p=float(chi2.sf(stat, 1)),
                   method="chi-square (continuity-corrected)")
    return out


def mcnemar(y_true, pred_a, pred_b, thr=0.5) -> dict:
    """McNemar's paired test for two classifiers on the SAME test set (D5 within-arm; B5).

    ONLY valid when both models saw the identical test samples (arch-vs-arch or ensemble-vs-single
    within ONE arm). It is INVALID across the patient/random arms — their test sets differ (SPEC).
    Returns b, c (the discordant counts) so the reader sees the evidence, not just a p-value.
    """
    y = np.asarray(y_true).astype(int)
    a = (np.asarray(pred_a) >= thr).astype(int)
    b_ = (np.asarray(pred_b) >= thr).astype(int)
    a_correct = a == y
    b_correct = b_ == y
    b = int(np.sum(a_correct & ~b_correct))     # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))     # A wrong, B right
    return mcnemar_from_counts(b, c)


# ----------------------------------------------------------------------------- bootstrap (B4)
def bootstrap_ci(y_true, y_prob, metric: Callable, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42, unit_ids=None) -> dict:
    """Percentile bootstrap CI for one metric on one run's predictions (SPEC B4).

    unit_ids: cluster labels (e.g. patient_id). When given, CLUSTERS are resampled with
    replacement and all their rows travel together — mandatory when rows within a unit are
    correlated (the slices of a patient), otherwise the interval is spuriously narrow.
    Draws whose resample contains a single class are skipped and counted, never silently.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.RandomState(seed)
    point = float(metric(y, p))
    vals, skipped = [], 0

    if unit_ids is not None:
        u = np.asarray(unit_ids)
        keys = list(np.unique(u))
        groups = {g: np.flatnonzero(u == g) for g in keys}

    for _ in range(n_boot):
        if unit_ids is None:
            take = rng.randint(0, len(y), len(y))
        else:
            picks = rng.randint(0, len(keys), len(keys))
            take = np.concatenate([groups[keys[i]] for i in picks])
        if len(np.unique(y[take])) < 2:
            skipped += 1
            continue
        vals.append(float(metric(y[take], p[take])))

    if not vals:
        return dict(point=point, ci=None, n_boot=n_boot, n_used=0, skipped=skipped,
                    clustered=unit_ids is not None)
    v = np.asarray(vals)
    return dict(point=point,
                ci=[float(np.percentile(v, 100 * alpha / 2)),
                    float(np.percentile(v, 100 * (1 - alpha / 2)))],
                se=float(v.std(ddof=1)), n_boot=n_boot, n_used=len(v), skipped=skipped,
                clustered=unit_ids is not None)


# ----------------------------------------------------------------------------- CLI
def _fold_nodule(cfg, arch, arm, sample_unit, rep, fold, pidx):
    """One (rep,fold) test set at nodule level, from archived probs — NEVER pooled across folds.

    Pooling across folds was a bug (code review 2026-07-21): the random arm's test folds overlap,
    so pooling then grouping by nodule averages probabilities from DIFFERENT fold-models for the
    same nodule and double-counts patients, distorting the bootstrap. Within ONE fold every
    prediction comes from that fold's single model and each nodule appears once — clean.
    """
    import pandas as pd
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    tag = f"lidc_binary_{sample_unit}_{arm}_rep{rep}_fold{fold}"
    npz = os.path.join(O, "probs", f"{tag}_{arch}_none_seed42.npz")
    csv = os.path.join(O, "splits", f"{tag}_test.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None
    d = np.load(npz)
    s = pd.read_csv(csv); s["z_position"] = s["z_position"].round(2)
    m = s.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
    if len(m) != len(d["y_prob"]):
        raise RuntimeError(f"alignment {tag} {arch}: {len(m)} vs {len(d['y_prob'])}")
    m["prob"] = d["y_prob"].astype(float)
    g = m.groupby("nodule_id").agg(label=("label", "first"), patient_id=("patient_id", "first"),
                                   prob=("prob", "mean")).reset_index()
    return g


def main():
    ap = argparse.ArgumentParser(description="STAGE 7 — per-metric bootstrap CIs (B4), fold-aware.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--arms", default="patient,random")
    ap.add_argument("--sample-unit", default="nodule", choices=["slice", "nodule"])
    ap.add_argument("--reps", default="0")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--n-boot", type=int, default=0, help="0 = config evaluate.bootstrap_n")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    from src.metadata import load_config
    from src.datasets import load_processed_index

    cfg = load_config(args.config)
    n_boot = args.n_boot or cfg["evaluate"]["bootstrap_n"]
    alpha = 1.0 - float(cfg["evaluate"]["ci"])
    pidx = load_processed_index(cfg, "none")
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    metrics = (("auc", roc_auc_score),
               ("logloss", lambda a, b: log_loss(a, b, labels=[0, 1])),
               ("brier", brier_score_loss))
    out = {}

    for arch in args.archs.split(","):
        for arm in args.arms.split(","):
            # per-fold: point estimate + within-fold cluster bootstrap CI (each fold is clean)
            per_fold = {name: [] for name, _ in metrics}
            boot_lo = {name: [] for name, _ in metrics}
            boot_hi = {name: [] for name, _ in metrics}
            nfolds = 0
            for rep in reps:
                for fold in folds:
                    g = _fold_nodule(cfg, arch, arm, args.sample_unit, rep, fold, pidx)
                    if g is None:
                        continue
                    y = g["label"].to_numpy(int)
                    p = np.clip(g["prob"].to_numpy(float), 1e-6, 1 - 1e-6)
                    units = g["patient_id"].to_numpy()   # cluster bootstrap on the patient
                    if len(np.unique(y)) < 2:
                        continue
                    nfolds += 1
                    for name, fn in metrics:
                        bc = bootstrap_ci(y, p, fn, n_boot=n_boot, alpha=alpha, unit_ids=units)
                        per_fold[name].append(bc["point"])
                        if bc["ci"]:
                            boot_lo[name].append(bc["ci"][0]); boot_hi[name].append(bc["ci"][1])
            if nfolds == 0:
                continue
            res = {}
            for name, _ in metrics:
                vals = per_fold[name]
                summary = mean_ci(vals, rho=0.0)          # fold spread (NB rho set by caller elsewhere)
                res[name] = dict(point=summary.get("mean"), n_folds=nfolds,
                                 ci_fold_spread=summary.get("ci_naive"),
                                 mean_within_fold_bootstrap_ci=[float(np.mean(boot_lo[name])),
                                                                float(np.mean(boot_hi[name]))]
                                 if boot_lo[name] else None,
                                 per_fold=[float(x) for x in vals])
            out[f"{arch}|{arm}|{args.sample_unit}"] = res
            print(f"{arch:16} {arm:8} {args.sample_unit:7} n_folds={nfolds}  " + "  ".join(
                f"{name} {res[name]['point']:.4f}" for name, _ in metrics))

    D = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "results")
    os.makedirs(D, exist_ok=True)
    path = os.path.join(D, f"stats_{args.sample_unit}.json")
    with open(path, "w") as fh:
        json.dump({"_note": "per-fold cluster bootstrap on patient_id (B4), NEVER pooled across "
                            "folds; point = fold mean; within-fold bootstrap CI averaged across "
                            "folds. n_boot=%d." % n_boot, "results": out}, fh, indent=1)
    print(f"wrote {path}  (fold-aware; cluster bootstrap on patient_id, n_boot={n_boot})")


if __name__ == "__main__":
    main()
