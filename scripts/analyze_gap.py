"""
CONFIRMATORY-LEVEL GAP ANALYSIS — the NODULE-AGGREGATED level (D17(i), D26).

!! LEVEL WARNING (was mislabelled "the paper's number" until 2026-07-23): this script ALWAYS
aggregates predictions to ONE VOTE PER NODULE (mean of slice probs). Under D17(i) --
pre-registered before any number existed -- the SLICE level is the PRIMARY statistically-powered
result and the nodule level is CONFIRMATORY robustness. The PRIMARY (slice-level) numbers come
from scripts/audit_controls.py, which reports BOTH levels; D26(iv) requires reporting both, always,
with the within-nodule correlation declared alongside the slice interval (D26(iii)).
`--sample-unit` selects WHICH TRAINED RUNS to load (slice-trained vs nodule-trained models); it
does NOT change the aggregation, which is per-nodule either way. The two scripts agree to ~0.0002
at the same level, by independent routes (this script RE-RUNS inference; audit_controls reads the
ARCHIVED probs) -- that agreement is the C1 provenance check, not a discrepancy.

For a given architecture, over reps x folds x {patient, random}, using the CANONICAL
checkpoint (best_val_loss) as the headline and the PROBE checkpoint (_final, max
memorization) alongside, compute the per-NODULE leakage-advantage gap in AUC, log-loss and
Brier (mean-of-slice-probs -> one vote per nodule):

    gap (leakage helps random):  AUC/acc = random - patient ; log-loss/Brier = patient - random

Reports per-fold gaps, mean, the naive t-CI AND the pre-registered Nadeau-Bengio CI (D6),
folds-positive, and Wilcoxon signed-rank with its attainable floor (n=15 -> min p = 0.0001).

    python scripts/analyze_gap.py --arch densenet121          # canonical only
    python scripts/analyze_gap.py --arch densenet121 --probe  # also print the _final probe

Reads only saved artifacts + splits. Skips folds whose model is missing (so it can be run
while the grid is still training). Nothing is fabricated.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.models import build_model, input_size_for
from src.datasets import load_processed_index, IMAGENET_MEAN, IMAGENET_STD
from src.stats import mean_ci, rho_from_splits, wilcoxon_paired


@torch.no_grad()
def nodule_preds(cfg, arch, arm, fold, suffix, pidx, mean, std, dev, sample_unit="slice", rep=0,
                 dataset="lidc_binary"):
    tag = f"{dataset}_{sample_unit}_{arm}_rep{rep}_fold{fold}"
    mp = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "models",
                      f"{tag}_{arch}_none_seed42{suffix}.pt")
    if not os.path.exists(mp):
        return None
    sp = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "splits", f"{tag}_test.csv")
    test = pd.read_csv(sp); test["z_position"] = test["z_position"].round(2)
    m = test.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
    model = build_model(arch, 2, pretrained=False).to(dev)
    model.load_state_dict(torch.load(mp, map_location=dev)); model.eval()
    # Match the size the model was TRAINED at (src/train.py resizes via input_size_for). The .npy
    # are 256x256; swin/maxvit are fixed at 224 and raise "Input height doesn't match model"
    # without this. Verified 2026-07-20: previously imported but never applied — this would have
    # crashed on the first transformer run of the grid.
    size = input_size_for(arch)
    pr = np.empty(len(m), dtype=np.float32)
    for i in range(0, len(m), 32):
        b = m["processed_path"].iloc[i:i+32].tolist()
        xs = torch.stack([(torch.from_numpy(np.load(p).astype(np.float32)).permute(2, 0, 1) - mean) / std
                          for p in b]).to(dev)
        if size != 256:
            xs = torch.nn.functional.interpolate(xs, size=(size, size), mode="bilinear",
                                                 align_corners=False, antialias=True)
        pr[i:i+len(b)] = torch.softmax(model(xs).float(), 1)[:, 1].cpu().numpy()
    m["prob"] = pr
    g = m.groupby("nodule_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    y, p = g["label"].to_numpy(), np.clip(g["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    return dict(auc=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), acc=accuracy_score(y, (p >= 0.5).astype(int)))


def analyze(cfg, arch, suffix, label, sample_unit="slice", reps=(0,), folds=range(5),
            dataset="lidc_binary"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pidx = load_processed_index(cfg, "none")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1); std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rows = []
    for rep in reps:
        for k in folds:
            mp = nodule_preds(cfg, arch, "patient", k, suffix, pidx, mean, std, dev, sample_unit, rep,
                              dataset)
            mr = nodule_preds(cfg, arch, "random", k, suffix, pidx, mean, std, dev, sample_unit, rep,
                              dataset)
            if mp is None or mr is None:
                print(f"  rep{rep} fold{k}: MISSING ({'patient' if mp is None else ''}{'/random' if mr is None else ''}) - skipped")
                continue
            # split sizes feed the Nadeau-Bengio rho (D6); read from the patient arm's own splits
            S = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "splits")
            tag = f"{dataset}_{sample_unit}_patient_rep{rep}_fold{k}"
            n_te = len(pd.read_csv(os.path.join(S, f"{tag}_test.csv")))
            n_tr = len(pd.read_csv(os.path.join(S, f"{tag}_train.csv")))
            rows.append(dict(rep=rep, fold=k, patient_auc=mp["auc"], random_auc=mr["auc"],
                             gap_auc=mr["auc"] - mp["auc"], gap_logloss=mp["logloss"] - mr["logloss"],
                             gap_brier=mp["brier"] - mr["brier"], n_test=n_te, n_train=n_tr))
    if not rows:
        print(f"  [{arch} {label}] no complete (rep,fold) points yet."); return
    df = pd.DataFrame(rows)
    print(f"\n=== {arch} | runs: {sample_unit}-trained | {label} | metric level: NODULE-AGGREGATED "
          f"| n={len(df)} (rep x fold) ===")
    if sample_unit == "slice":
        print("    NOTE: these are NOT slice-level numbers. The pre-registered PRIMARY level "
              "(D17(i)) is SLICE-level, from scripts/audit_controls.py.")
    print(df.round(4).to_string(index=False))
    print(f"  headroom (mean patient AUC): {df['patient_auc'].mean():.4f}")
    # Intervals come from src/stats.py — the single implementation (D27). Both the naive and the
    # pre-registered Nadeau-Bengio interval (D6) are printed: at n=5 the correction is the
    # difference between "excludes zero" and not, so showing only the naive one would overstate.
    rho = rho_from_splits(df["n_test"].to_numpy(), df["n_train"].to_numpy())
    for col, name in (("gap_auc", "AUC"), ("gap_logloss", "log-loss"), ("gap_brier", "Brier")):
        s = mean_ci(df[col].to_numpy(), rho=rho)
        line = (f"  gap {name:9}: mean {s['mean']:+.4f}  naive95 [{s['ci_naive'][0]:+.4f},"
                f"{s['ci_naive'][1]:+.4f}]  NB95 [{s['ci_nb'][0]:+.4f},{s['ci_nb'][1]:+.4f}]"
                f"  folds+ {s['folds_positive']}/{s['k']}")
        if not s["excludes_zero_nb"]:
            line += "  <- includes 0 under D6"
        print(line)
    w = wilcoxon_paired(df["gap_auc"].to_numpy())
    if w.get("p") is not None:
        print(f"  Wilcoxon (AUC gap): p={w['p']:.4f}  min attainable at n={w['n_nonzero']}: "
              f"{w['min_attainable_p']:.4f}" + ("  <- AT THE FLOOR: directional only, "
                                                "n is what would move it" if w["at_floor"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--sample-unit", default="slice", choices=["slice", "nodule"])
    ap.add_argument("--reps", default="0", help="comma-separated rep indices; grid S2 = 0,1,2 (n=15)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--probe", action="store_true", help="also print the _final (memorization) probe")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort, D37 -- never pooled with principal)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    reps = [int(x) for x in args.reps.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    analyze(cfg, args.arch, "", "CANONICAL (best_loss) - CONFIRMATORY", args.sample_unit, reps, folds,
            args.dataset)
    if args.probe:
        analyze(cfg, args.arch, "_final", "PROBE (final, max memorization) - mechanism only",
                args.sample_unit, reps, folds, args.dataset)


if __name__ == "__main__":
    main()
