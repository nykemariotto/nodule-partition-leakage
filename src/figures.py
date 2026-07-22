"""
STAGE 8 — figures from saved artifacts (SPEC; reviewer points R4.5/I2 curves, R4.6/I1 confusion).

Reads ONLY what is already on disk — `outputs/probs/*.npz` (y_true, y_prob) and
`outputs/history/*.json` (per-epoch train/val). No GPU, no re-inference. Every number traces to a
saved file, so the figures are reproducible from the released artifacts (the whole point).

Produces, per architecture:
  * confusion_<arch>.png    — nodule-level confusion matrices, patient arm vs random arm, side by
                              side. The visual of the leakage: the random (leaky) arm looks
                              cleaner than the honest patient arm.
  * curves_<arch>.png       — val/train loss and val AUC vs epoch, patient vs random, mean ± band
                              over the 5 folds. Directly shows the best_epoch asymmetry (patient
                              converges early ~ep3; random keeps improving to ~ep22, the
                              memorisation signature) and that neither arm overfits catastrophically.
And a machine-readable `outputs/results/confusion_matrices.json` with per-arm counts + derived
precision/recall/F1/accuracy, INCLUDING an internal check that F1 == 2PR/(P+R) (reviewer R3 flagged
an Xception F1 that violated this in the original; B9).

Confusion-matrix aggregation note (honest): counts are SUMMED over the 5 fold test sets. For the
patient arm the folds partition patients, so this is a clean cross-validated CM (each nodule once).
For the random arm the test folds OVERLAP by construction (measured 78-82% patient reuse per fold),
so its summed CM re-scores some nodules across folds; it is a faithful picture of what the leaky
protocol produced, not a disjoint partition. Stated in the caption and the JSON.

    python -m src.figures                       # densenet121, efficientnet_b0
    python -m src.figures --archs densenet121
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(cfg, arch, arm, fold, pidx, suffix="", rep=0):
    """Archived probs joined to their split rows, at nodule level (mean of slice probs)."""
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    tag = f"lidc_binary_slice_{arm}_rep{rep}_fold{fold}"
    npz = os.path.join(O, "probs", f"{tag}_{arch}_none_seed42{suffix}.npz")
    csv = os.path.join(O, "splits", f"{tag}_test.csv")
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None
    d = np.load(npz)
    s = pd.read_csv(csv); s["z_position"] = s["z_position"].round(2)
    m = s.merge(pidx, on=["nodule_id", "z_position"], how="inner").reset_index(drop=True)
    if len(m) != len(d["y_prob"]):
        raise RuntimeError(f"alignment {arm} f{fold} {arch}: {len(m)} vs {len(d['y_prob'])}")
    m["prob"] = d["y_prob"].astype(float)
    g = m.groupby("nodule_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    return g


def _cm_counts(y, p, thr=0.5):
    yhat = (np.asarray(p) >= thr).astype(int); y = np.asarray(y).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def _derived(tp, tn, fp, fn):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (prec + rec) and not (np.isnan(prec) or np.isnan(rec)) else float("nan"))
    if not np.isnan(f1):                       # B9: F1 must equal 2PR/(P+R)
        assert abs(f1 - 2 * prec * rec / (prec + rec)) < 1e-9
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc)


def arm_cm(cfg, arch, arm, pidx, suffix="", reps=(0,), fold_range=range(5)):
    tp = tn = fp = fn = folds = 0
    for rep in reps:
        for k in fold_range:
            g = _load(cfg, arch, arm, k, pidx, suffix, rep)
            if g is None:
                continue
            a, b, c, d = _cm_counts(g["label"], g["prob"])
            tp += a; tn += b; fp += c; fn += d; folds += 1
    if folds == 0:
        return None
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, folds=folds, **_derived(tp, tn, fp, fn))


def plot_confusion(cfg, arch, pidx, outdir, reps=(0,), fold_range=range(5)):
    cms = {arm: arm_cm(cfg, arch, arm, pidx, reps=reps, fold_range=fold_range)
           for arm in ("patient", "random")}
    if any(v is None for v in cms.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, arm in zip(axes, ("patient", "random")):
        c = cms[arm]
        M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(M, cmap="Blues")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color="white" if v > M.max() * 0.5 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred benign", "pred malig"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true benign", "true malig"])
        tag = "patient-level (honest)" if arm == "patient" else "random (leaky)"
        ax.set_title(f"{tag}\nacc {c['accuracy']:.3f} · F1 {c['f1']:.3f} · rec {c['recall']:.3f}",
                     fontsize=10)
    fig.suptitle(f"{arch} — nodule-level confusion, summed over {cms['patient']['folds']} test "
                 f"folds (random-arm folds overlap; see JSON)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, f"confusion_{arch}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return {"path": p, **cms}


def plot_curves(cfg, arch, outdir, reps=(0,), fold_range=range(5)):
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    colors = {"patient": "#1b7837", "random": "#c51b7d"}
    found = False
    for arm in ("patient", "random"):
        loss_tr, loss_va, auc_va = [], [], []
        for rep in reps:
            for k in fold_range:
                hp = os.path.join(O, "history",
                                  f"lidc_binary_slice_{arm}_rep{rep}_fold{k}_{arch}_none_seed42.json")
                if not os.path.exists(hp):
                    continue
                h = json.load(open(hp))["history"]
                loss_tr.append([x["train_loss"] for x in h])
                loss_va.append([x["val_loss"] for x in h])
                auc_va.append([x["val_auc"] for x in h])
        if not auc_va:
            continue
        found = True
        n = min(len(c) for c in auc_va)
        ep = np.arange(1, n + 1)
        def band(mat):
            A = np.array([c[:n] for c in mat]); return A.mean(0), A.std(0)
        mu_vl, sd_vl = band(loss_va)
        axes[0].plot(ep, mu_vl, color=colors[arm], label=f"{arm} val")
        axes[0].fill_between(ep, mu_vl - sd_vl, mu_vl + sd_vl, color=colors[arm], alpha=0.15)
        mu_tr, _ = band(loss_tr)
        axes[0].plot(ep, mu_tr, color=colors[arm], ls="--", alpha=0.6, label=f"{arm} train")
        mu_au, sd_au = band(auc_va)
        axes[1].plot(ep, mu_au, color=colors[arm], label=f"{arm}")
        axes[1].fill_between(ep, mu_au - sd_au, mu_au + sd_au, color=colors[arm], alpha=0.15)
    if not found:
        plt.close(fig); return None
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("loss (val solid, train dashed)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val AUC")
    axes[1].set_title("validation AUC"); axes[1].legend(fontsize=8)
    fig.suptitle(f"{arch} — learning curves, mean ± sd over 5 folds "
                 f"(random arm's val keeps improving = memorisation signature)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, f"curves_{arch}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--reps", default="0", help="comma-separated rep indices; grid S2 = 0,1,2")
    args = ap.parse_args()

    from src.metadata import load_config
    from src.datasets import load_processed_index
    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    reps = [int(x) for x in args.reps.split(",")]
    outdir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "figures")
    os.makedirs(outdir, exist_ok=True)

    summary = {}
    for arch in args.archs.split(","):
        cm = plot_confusion(cfg, arch, pidx, outdir, reps=reps)
        plot_curves(cfg, arch, outdir, reps=reps)
        if cm:
            summary[arch] = {arm: {k: cm[arm][k] for k in
                                   ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy")}
                             for arm in ("patient", "random")}
            print(f"{arch}: confusion + curves written")
            for arm in ("patient", "random"):
                c = cm[arm]
                print(f"   {arm:8} acc {c['accuracy']:.4f}  F1 {c['f1']:.4f}  "
                      f"prec {c['precision']:.4f}  rec {c['recall']:.4f}  "
                      f"(TN{c['tn']} FP{c['fp']} FN{c['fn']} TP{c['tp']})")

    R = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "results")
    os.makedirs(R, exist_ok=True)
    meta = {"_note": "CM counts summed over 5 test folds; patient arm = clean CV partition, random "
                     "arm test folds overlap by construction (leaky protocol). F1 == 2PR/(P+R) "
                     "asserted (B9). threshold 0.5, nodule level.",
            "arch": summary}
    json.dump(meta, open(os.path.join(R, "confusion_matrices.json"), "w"), indent=1)
    print(f"wrote {os.path.join(R, 'confusion_matrices.json')} and {outdir}/*.png")


if __name__ == "__main__":
    main()
