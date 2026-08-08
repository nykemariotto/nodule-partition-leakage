"""
PROJECT THE GAIN FROM STRATIFIED FOLDS — no GPU, no retraining (DECISIONS D27 / open item (a)).

Question: would StratifiedGroupKFold buy enough statistical power to justify re-running the 20
canonical runs (Option A), or did plain grouping already produce folds good enough to keep them
and simply delete the untrue `split.stratify_key` (Option B)?

METHOD. Predictions are held FIXED (the archived per-nodule / per-slice probabilities from the
runs we already have) and only the PARTITION used to aggregate them is re-drawn, S times, under
two schemes on the identical patient pool:
    current      shuffled round-robin over patients  (exactly `src/splits.py::_fold_assign`)
    stratified   sklearn StratifiedGroupKFold(groups=patient_id, y=label)
For each draw we recompute the per-fold patient-arm metric, the fold-to-fold SD, and the naive
and Nadeau-Bengio (D6) interval half-widths.

WHAT THIS ISOLATES, and what it does NOT. It measures the component of fold-to-fold variance
caused by FOLD COMPOSITION - which is the only component stratification can remove. Under real
stratified CV the models would also be retrained on different training sets, so the predictions
themselves would change; that component is NOT captured here and could move the answer either
way. This is therefore INDICATIVE, not a measurement of the post-A result. Labelled as such.

The random arm is treated as a constant comparator: its fold-to-fold metric SD is ~0.006-0.007
versus 0.023-0.052 in the patient arm, so essentially all of the gap's fold variance is the
patient arm's. Stated rather than hidden.

    python scripts/project_stratification.py --draws 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.datasets import load_processed_index

K = 5


def fold_assign_current(n, k, seed):
    """Verbatim reimplementation of src/splits.py::_fold_assign."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    out = np.empty(n, dtype=int)
    for i, idx in enumerate(order):
        out[idx] = i % k
    return out


def load_pooled(cfg, arch, arm, pidx, folds=range(K)):
    """Every test row of `arm`, pooled across folds, with its archived probability."""
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    parts = []
    for f in folds:
        tag = f"lidc_binary_slice_{arm}_rep0_fold{f}"
        npz = os.path.join(O, "probs", f"{tag}_{arch}_none_seed42.npz")
        csv = os.path.join(O, "splits", f"{tag}_test.csv")
        if not (os.path.exists(npz) and os.path.exists(csv)):
            return None
        d = np.load(npz)
        s = pd.read_csv(csv)
        s["z_position"] = s["z_position"].round(2)
        m = s.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
        if len(m) != len(d["y_prob"]):
            raise RuntimeError(f"alignment failed {arm} f{f} {arch}")
        m["prob"] = d["y_prob"].astype(float)
        m["orig_fold"] = f
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def to_nodule(df):
    return df.groupby(["nodule_id"]).agg(patient_id=("patient_id", "first"),
                                         label=("label", "first"),
                                         prob=("prob", "mean")).reset_index()


def metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return None
    return (roc_auc_score(y, p), log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p))


def evaluate_partition(unit_df, fold_of_patient):
    """Per-fold metrics + per-fold positive rate, for one partition of the patient pool."""
    fold = unit_df["patient_id"].map(fold_of_patient).to_numpy()
    aucs, lls, brs, rates = [], [], [], []
    for k in range(K):
        sel = unit_df[fold == k]
        if len(sel) < 10:
            return None
        m = metrics(sel["label"], sel["prob"])
        if m is None:
            return None
        aucs.append(m[0]); lls.append(m[1]); brs.append(m[2])
        rates.append(float(np.mean(sel["label"])))
    return dict(auc=np.array(aucs), ll=np.array(lls), br=np.array(brs), rate=np.array(rates))


def ci_halfwidths(vals, rho):
    """naive t half-width and the Nadeau-Bengio corrected one (D6)."""
    s = vals.std(ddof=1)
    tc = float(tdist.ppf(0.975, K - 1))
    return tc * s / np.sqrt(K), tc * s * np.sqrt(1.0 / K + rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None,
                    help="comma-separated. DEFAULT: every architecture with runs on disk. A "
                         "hardcoded pair silently froze this projection at two architectures.")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--rho", type=float, default=0.2839, help="n_test/n_train, measured (D6)")
    ap.add_argument("--json", default="outputs/metrics/stratification_projection.json")
    args = ap.parse_args()
    from src.artifacts import resolve_archs
    import os as _os
    _probs = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           "outputs", "probs")
    _ARCHS, _ = resolve_archs(_probs, "lidc_binary", args.archs,
                              cells=(["slice"], ["patient", "random"], [0, 1, 2], [0, 1, 2, 3, 4]))
    args.archs = ",".join(_ARCHS)
    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")

    print(f"draws per scheme: {args.draws} · NB rho {args.rho:.4f}")
    print("predictions FIXED (archived); only the aggregation partition is re-drawn.")
    print("=> isolates the fold-COMPOSITION component of variance; retraining effects NOT captured.\n")

    report = {}
    for arch in args.archs.split(","):
        pat = load_pooled(cfg, arch, "patient", pidx)
        rnd = load_pooled(cfg, arch, "random", pidx)
        if pat is None or rnd is None:
            print(f"{arch}: artifacts missing - skipped"); continue

        for level, conv in (("nodule", to_nodule), ("slice", lambda d: d)):
            u = conv(pat).copy()
            rnd_u = conv(rnd)
            # random arm = constant comparator (its fold-to-fold SD is ~4x smaller than the
            # patient arm's, so the gap's fold variance is essentially the patient arm's).
            rm = metrics(rnd_u["label"], rnd_u["prob"])
            pats = u["patient_id"].unique()

            res = {}
            for scheme in ("current", "stratified"):
                acc = {k: [] for k in ("rate_range", "rate_sd", "auc_sd", "ll_sd", "br_sd",
                                       "auc_nb", "ll_nb", "br_nb", "auc_naive")}
                for d in range(args.draws):
                    if scheme == "current":
                        fa = fold_assign_current(len(pats), K, 1000 + d)
                        fmap = dict(zip(pats, fa))
                    else:
                        sgk = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=1000 + d)
                        fmap = {}
                        y = u["label"].to_numpy()
                        g = u["patient_id"].to_numpy()
                        for k, (_, te) in enumerate(sgk.split(np.zeros(len(u)), y, g)):
                            for pt in np.unique(g[te]):
                                fmap[pt] = k
                        if len(fmap) < len(pats):
                            continue
                    ev = evaluate_partition(u, fmap)
                    if ev is None:
                        continue
                    acc["rate_range"].append(ev["rate"].max() - ev["rate"].min())
                    acc["rate_sd"].append(ev["rate"].std(ddof=1))
                    for key, arr in (("auc", ev["auc"]), ("ll", ev["ll"]), ("br", ev["br"])):
                        acc[f"{key}_sd"].append(arr.std(ddof=1))
                        acc[f"{key}_nb"].append(ci_halfwidths(arr, args.rho)[1])
                    acc["auc_naive"].append(ci_halfwidths(ev["auc"], args.rho)[0])
                res[scheme] = {k: float(np.mean(v)) for k, v in acc.items() if v}
                res[scheme]["draws_used"] = len(acc["rate_sd"])

            if "current" not in res or "stratified" not in res:
                continue
            c, s = res["current"], res["stratified"]
            print(f"=== {arch} · {level}-level (patient arm; random-arm pooled "
                  f"AUC {rm[0]:.4f} treated as constant) ===")
            print(f"{'quantity':30}{'current':>12}{'stratified':>13}{'change':>12}")
            for key, lab in (("rate_range", "fold prevalence RANGE"), ("rate_sd", "fold prevalence SD"),
                             ("auc_sd", "fold AUC SD"), ("ll_sd", "fold log-loss SD"),
                             ("br_sd", "fold Brier SD"), ("auc_nb", "AUC NB CI half-width"),
                             ("ll_nb", "log-loss NB CI half-width"), ("br_nb", "Brier NB CI half-width")):
                ch = (s[key] - c[key]) / c[key] * 100 if c.get(key) else float("nan")
                print(f"  {lab:28}{c[key]:>12.4f}{s[key]:>13.4f}{ch:>11.1f}%")
            print()
            report[f"{arch}|{level}"] = res

    if args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(cfg["project"]["root"], args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(report, open(p, "w"), indent=1)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
