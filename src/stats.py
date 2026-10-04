"""
The statistical foundation: every inferential quantity the study reports is implemented here.

This module is the SINGLE implementation of those quantities; `scripts/` imports from here and
nothing reimplements them, so the configuration cannot describe one method while a script
computes another.

WHAT IS HERE, and the reasoning that constrains each:

  mean_ci()           Every metric is computed WITHIN each fold and then averaged across folds.
                      Predictions are never pooled across folds — pooled predictions have no
                      valid interval and mix distributions. One exception: at the NODULE level
                      the random arm's folds are drawn over slices, so a nodule's slices sit in
                      several test folds and a within-fold nodule score counts that nodule
                      several times on partial slices. There, `scripts/nodule_oof.py` collects
                      each repeat's out-of-fold predictions first, scores every nodule once per
                      repeat, and takes its interval from `run_aware_interval()`, which adds the
                      run-to-run variance across repeats to the patient-bootstrap variance from
                      `cluster_bootstrap_draws()`.

  nadeau_bengio_ttest() The paired test across folds, built on the same corrected SE and t
                      quantile as the interval, so it is that interval's exact dual. It is the
                      test the paper reports; the signed-rank test treats the folds as
                      independent pairs and is kept only as a description of sign agreement.
                      Arms are paired by repeat and fold INDEX: their test sets differ by
                      construction, so a p-value is reported to one significant figure.

  nadeau_bengio_se()  k-fold estimates are NOT independent: training sets overlap, so the
                      naive s/sqrt(k) understates variance. Corrected SE = s*sqrt(1/k + rho),
                      rho = n_test/n_train (about 0.29 in this design). With few folds that is
                      precisely the difference between "excludes zero" and not.

  wilcoxon_paired()   Descriptive only (see nadeau_bengio_ttest). It stays because its floor is
                      informative: it also returns `min_attainable_p`, because at small n the
                      test has a FLOOR: the two-sided exact minimum is 2/2^n (n=5 -> 0.0625).
                      Reporting "p=0.0625" without that context invites reading it as a
                      near-miss when it is the smallest value the test can produce.

  cluster_bootstrap_draws()  Percentile bootstrap over the INDEPENDENT unit. When slices are the
                      rows, that unit is the PATIENT, not the row: resampling correlated rows
                      independently yields intervals that are far too narrow.

  mcnemar_from_counts()  McNemar's test from the two discordant counts, reported only as a
                      description (scripts/nodule_oof.py).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import t as tdist, wilcoxon as _wilcoxon


# ----------------------------------------------------------------------------- intervals
def naive_se(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def nadeau_bengio_se(values: Sequence[float], rho: float) -> float:
    """Corrected SE of the mean of k dependent cross-validation estimates.

    rho = n_test / n_train; rho=0 degenerates to the naive s/sqrt(k).
    """
    v = np.asarray(values, dtype=float)
    k = len(v)
    if k < 2:
        return float("nan")
    return float(v.std(ddof=1) * np.sqrt(1.0 / k + rho))


def rho_from_splits(n_test: Sequence[int], n_train: Sequence[int]) -> float:
    """Nadeau-Bengio rho = mean(n_test / n_train) across folds."""
    return float(np.mean(np.asarray(n_test, float) / np.asarray(n_train, float)))


def mean_ci(values: Sequence[float], rho: float = 0.0, alpha: float = 0.05) -> dict:
    """Mean with BOTH the naive and the Nadeau-Bengio interval, and which excludes zero.

    Both are returned deliberately: the naive one is what a reader would compute by default, the
    corrected one is what this study prespecified. Reporting only the narrower would overstate.
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


def nadeau_bengio_ttest(values: Sequence[float], rho: float) -> dict:
    """Corrected resampled t-test for the mean of k dependent cross-validation differences.

    Same standard error as the Nadeau-Bengio interval, s*sqrt(1/k + rho), and the same t reference
    with k-1 degrees of freedom, so p < alpha exactly when `mean_ci(values, rho)["ci_nb"]` excludes
    zero. A test whose verdict could disagree with the interval printed beside it would be a second,
    weaker claim about the same numbers.
    """
    v = np.asarray(values, dtype=float)
    k = len(v)
    if k < 2:
        return dict(k=k, mean=float(v.mean()) if k else None, se=None, rho=rho, t=None, df=None,
                    p=None)
    m = float(v.mean())
    se = nadeau_bengio_se(v, rho)
    # Only an exactly zero spread is special: the interval is then the point itself, so p is 0 or 1
    # by whether the mean is zero, exactly as `mean_ci` decides. A tolerance here would let the test
    # and the interval disagree on rounding noise. t is None rather than infinite, so the result
    # can be written to JSON.
    if se == 0.0:
        t, p = (None, 0.0) if m != 0.0 else (0.0, 1.0)
    else:
        t = float(m / se)
        p = float(2.0 * tdist.sf(abs(t), k - 1))
    return dict(k=k, mean=m, se=se, rho=rho, t=t, df=k - 1, p=p)


# ----------------------------------------------------------------------------- paired test
def wilcoxon_min_p(n: int) -> float:
    """Smallest two-sided p the exact signed-rank test can produce with n non-zero pairs."""
    return 2.0 / (2.0 ** n) if n > 0 else float("nan")


def wilcoxon_paired(a: Sequence[float], b: Sequence[float] | None = None) -> dict:
    """Paired A-vs-B signed-rank test. `a` may be the differences directly (b=None).

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
        stat = max(abs(b - c) - 1, 0) ** 2 / n
        out.update(statistic=float(stat), p=float(chi2.sf(stat, 1)),
                   method="chi-square (continuity-corrected)")
    return out


# ----------------------------------------------------------------------------- bootstrap
def cluster_bootstrap_draws(unit_of_row, n_boot: int, seed: int):
    """Row-index arrays for a percentile bootstrap over clusters, one array per draw.

    Units (patients) are drawn with replacement and all their rows travel together; a unit drawn
    twice contributes its rows twice. The caller applies EACH draw to every quantity it compares --
    every arm, architecture and repeat -- so the comparison is paired: computing the contrast
    inside one draw is what lets the shared patient-sampling variability cancel.
    """
    _, inv = np.unique(np.asarray(unit_of_row), return_inverse=True)
    n_units = int(inv.max()) + 1
    rows = np.arange(len(inv))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
        yield np.repeat(rows, mult[inv])


def run_aware_interval(per_repeat: Sequence[float], boot_stats: Sequence[float],
                       alpha: float = 0.05, components=None) -> dict:
    """Interval for a mean over R repeats that carries BOTH sources of variation.

    `per_repeat` holds the statistic of each repeat; the repeats differ only in partition and
    training, all evaluated on the same patients. `boot_stats` holds the cluster-bootstrap
    distribution of the mean over repeats: the evaluated patients are resampled and the trained
    models held fixed. SE^2 = s_rep^2 / R + var(boot_stats). The two terms are not perfectly
    separate -- a model-by-patient interaction enters both -- which makes the sum conservative. The
    verdict uses t with R - 1 degrees of freedom, the conservative convention, because the repeat
    term is estimated from only R values. A Welch-Satterthwaite interval is returned beside it as a
    sensitivity: when the bootstrap term dominates, the df convention, not the data, can decide a
    borderline verdict, and a reader should be able to see whether it does. Neither term carries
    the variance of drawing a different training cohort. Either term alone answers a narrower
    question; the bootstrap alone cannot tell an effect of the protocol from run-to-run variation,
    which is why it does not carry the verdict.

    `components`: for a contrast between two INDEPENDENT sets of runs (two arms, whose repeats share
    nothing but a seed index), pass the two sets of per-repeat values. The repeat term is then
    (s_X^2 + s_Y^2) / R, which does not depend on how the repeats are paired and equals the average
    of the paired term over every pairing. Without it, the spread of `per_repeat` is used, which is
    right when the pairing is real (two architectures trained on the same partitions).
    """
    r = np.asarray(per_repeat, dtype=float)
    b = np.asarray(boot_stats, dtype=float)
    k = len(r)
    if k < 2:
        return dict(k=k, mean=float(r.mean()) if k else None, se=None, df=None, var_repeats=None,
                    var_boot=None, ci=None, excludes_zero=None, df_satterthwaite=None,
                    ci_satterthwaite=None, excludes_zero_satterthwaite=None)
    m = float(r.mean())
    var_boot = float(b.var(ddof=1))
    if components is None:
        parts = [float(r.var(ddof=1))]
    else:
        cs = [np.asarray(c, dtype=float) for c in components]
        if len(cs) < 2:
            raise ValueError("components: one set of per-repeat values per arm, at least two")
        if any(len(c) != k for c in cs):
            raise ValueError("every component needs one value per repeat")
        parts = [float(c.var(ddof=1)) for c in cs]
    var_rep = float(sum(parts))
    a = var_rep / k
    se = float(np.sqrt(a + var_boot))
    h = float(tdist.ppf(1 - alpha / 2, k - 1)) * se
    # bootstrap term treated as known (thousands of draws): its df contribution is negligible
    den = sum((v / k) ** 2 / (k - 1) for v in parts)
    df_s = float(se ** 4 / den) if den > 0 else float("inf")
    hs = float(tdist.ppf(1 - alpha / 2, df_s)) * se          # scipy accepts df = inf (normal)
    return dict(k=k, mean=m, se=se, df=k - 1, var_repeats=var_rep, var_boot=var_boot,
                ci=[m - h, m + h], excludes_zero=bool((m - h) > 0 or (m + h) < 0),
                df_satterthwaite=df_s if np.isfinite(df_s) else None,
                ci_satterthwaite=[m - hs, m + hs],
                excludes_zero_satterthwaite=bool((m - hs) > 0 or (m + hs) < 0))


def auc_rank(y_true, y_score) -> float:
    """ROC AUC as the Mann-Whitney statistic with average ranks (ties count one half), which is
    what `sklearn.metrics.roc_auc_score` computes. Kept here because a bootstrap calls it tens of
    thousands of times; callers verify it against sklearn on the point estimate."""
    from scipy.stats import rankdata
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(y_score, dtype=float))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
