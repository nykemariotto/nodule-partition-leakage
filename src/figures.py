"""
STAGE 8 — figures from saved artifacts (SPEC; reviewer points R4.5/I2 curves, R4.6/I1 confusion).

Reads ONLY what is already on disk — `outputs/probs/*.npz` (y_true, y_prob) and
`outputs/history/*.json` (per-epoch train/val). No GPU, no re-inference. Every number traces to a
saved file, so the figures are reproducible from the released artifacts (the whole point).

Produces TWO figures, each covering every architecture DISCOVERED ON DISK (not one file per
architecture -- only one architecture's curves reached the submitted manuscript, and that is
easy to repeat). Restricting to a subset with --archs is allowed but renames the output, so a
partial figure can never occupy the filename the manuscript embeds:
  * confusion_matrices.png  — grid of nodule-level confusion matrices, architectures as rows and
                              partition conditions as columns, lettered panels. The visual of the
                              leakage: the random (leaky) condition looks cleaner than the honest
                              patient-grouped one.
  * curves_<units>.png      — val/train loss and val AUC vs epoch, patient vs random, mean ± band
                              over the runs; rows = sample unit, columns = architecture x metric.
                              Directly shows the best_epoch asymmetry (patient converges early ~ep3;
                              random keeps improving to ~ep22, the memorisation signature) and that
                              neither arm overfits catastrophically.
Neither figure carries a figure-level title: the IEEE Access template asks that captions not be part
of the figure. Everything descriptive lives in the LaTeX caption.
And a machine-readable `outputs/results/confusion_matrices.json` with per-arm counts + derived
precision/recall/F1/accuracy, INCLUDING an internal check that F1 == 2PR/(P+R) (reviewer R3 flagged
an Xception F1 that violated this in the original; B9).

Confusion-matrix aggregation note (honest): counts are SUMMED over the 5 fold test sets. For the
patient- and nodule-grouped conditions the folds partition their grouping unit, so those are clean
cross-validated CMs (each nodule scored once). For the random condition the test folds OVERLAP by
construction (measured 78-82% patient reuse per fold), so its summed CM re-scores some nodules
across folds; it is a faithful picture of what the leaky protocol produced, not a disjoint
partition. Stated in the caption and the JSON.

    python -m src.figures                 # every architecture found on disk
    python -m src.figures --archs densenet121    # -> a SUFFIXED filename, not the canonical one
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Split-file prefix, set from --dataset. "lidc_binary" = principal cohort; "lidc_binary_ge3" = the
# >=3-annotator sensitivity cohort (D37), which carries a distinct prefix so its runs can never be
# pooled with the principal ones. Was hardcoded until 2026-07-26.
DATASET = "lidc_binary"


def _load(cfg, arch, arm, fold, pidx, suffix="", rep=0):
    """Archived probs joined to their split rows, at nodule level (mean of slice probs)."""
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    tag = f"{DATASET}_slice_{arm}_rep{rep}_fold{fold}"
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


ARM_LABEL = {"patient": "patient-grouped", "random": "random", "nodule": "nodule-grouped"}


def run_counts(cfg, arch, arms, reps, folds, sus):
    """How many of the expected (sample_unit, arm, rep, fold) cells this architecture actually has."""
    import glob as _glob
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "probs")
    have = 0
    for su in sus:
        for arm in arms:
            for r in reps:
                for k in folds:
                    f = os.path.join(O, f"{DATASET}_{su}_{arm}_rep{r}_fold{k}_{arch}_none_seed42.npz")
                    if os.path.exists(f):
                        have += 1
    return have, len(sus) * len(arms) * len(reps) * len(folds)


def discover_archs(cfg):
    """Architectures that actually have runs on disk for DATASET, in a stable order.

    Globs are anchored on the sample unit: a bare "{DATASET}_*" also matches a LONGER dataset name,
    since "lidc_binary" is a prefix of "lidc_binary_ge3", and the sensitivity cohort's runs would be
    read as the principal cohort's.
    """
    import glob as _glob
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "probs")
    found = set()
    for su in ("slice", "nodule"):
        for f in _glob.glob(os.path.join(O, f"{DATASET}_{su}_*_seed42.npz")):
            b = os.path.basename(f)[:-4]
            if b.endswith("_final"):
                continue
            m = re.search(r"_fold\d+_(.+)_none_seed\d+$", b)
            if m:
                found.add(m.group(1))
    return sorted(found)


def name_for(stem, dataset, archs, all_archs):
    """Output filename. Keeps the canonical literal ONLY for the full arch set on the principal
    cohort, so the paths the manuscript embeds keep resolving; anything else is suffixed.

    The names used to be literals independent of both axes, so `--archs vit_small` or
    `--dataset lidc_binary_ge3` replaced the file the manuscript points at with different content
    under an unchanged name. That is the same defect fixed on the evaluate side, on the other axis.
    """
    canonical = (dataset == "lidc_binary") and (sorted(archs) == sorted(all_archs))
    if canonical:
        return stem
    parts = [stem]
    if dataset != "lidc_binary":
        parts.append(dataset)
    if sorted(archs) != sorted(all_archs):
        parts.append("+".join(sorted(archs)))
    return "__".join(parts)


def plot_confusion(cfg, archs, pidx, outdir, reps=(0,), fold_range=range(5),
                   arms=("patient", "nodule", "random"), stem="confusion_matrices"):
    """One figure: a grid of 2x2 confusion matrices, architectures as rows, conditions as columns.

    Form chosen to match published practice and the CLAIM checklist (item 37/2020, 39/2024), which
    asks for "a confusion matrix that shows tallies for predicted versus actual categories" -- i.e.
    raw counts in predicted x actual orientation, not a flat list of TN/FP/FN/TP. Surveyed
    medical-imaging DL papers present this as a grid/heatmap with lettered panels when several
    models or conditions are shown; a flat counts table is the least common form and evidently did
    not read as "providing the confusion matrix" to our reviewer.

    NO figure-level title: the IEEE Access template states "Please do not include captions as part
    of the figures". Everything descriptive belongs in the LaTeX caption.
    """
    archs = list(archs)
    grid = {}
    for a in archs:
        for arm in arms:
            grid[(a, arm)] = arm_cm(cfg, a, arm, pidx, reps=reps, fold_range=fold_range)
    if all(v is None for v in grid.values()):
        return None

    nr, nc = len(archs), len(arms)
    fig, axes = plt.subplots(nr, nc, figsize=(3.05 * nc, 3.25 * nr), squeeze=False)
    letters = string.ascii_lowercase
    k = 0
    for r, a in enumerate(archs):
        for c_i, arm in enumerate(arms):
            ax = axes[r][c_i]
            c = grid[(a, arm)]
            if c is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"({letters[k]}) not run", ha="center", va="center", fontsize=9)
                k += 1
                continue
            M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            ax.imshow(M / M.sum(), cmap="Blues", vmin=0, vmax=0.6)
            for (i, j), v in np.ndenumerate(M):
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12,
                        color="white" if v / M.sum() > 0.3 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "malignant"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "malignant"], fontsize=8)
            if r == nr - 1:
                ax.set_xlabel("Predicted", fontsize=9)
            if c_i == 0:
                ax.set_ylabel("Actual", fontsize=9)
            ax.set_title(f"({letters[k]}) {ARM_LABEL[arm]}\n"
                         f"acc {c['accuracy']:.3f} | F1 {c['f1']:.3f}", fontsize=9)
            k += 1
    # architecture as a bold row label, placed after layout so it clears the tick labels
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    for r, a in enumerate(archs):
        bb = axes[r][0].get_position()
        fig.text(0.012, (bb.y0 + bb.y1) / 2, a, rotation=90, ha="left", va="center",
                 fontsize=12, fontweight="bold")
    p = os.path.join(outdir, stem + ".png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return {"path": p, "grid": {f"{a}|{arm}": v for (a, arm), v in grid.items() if v}}


def _history_band(O, arch, arm, su, reps, fold_range):
    """mean +- s.d. trajectories over the runs of one (arch, arm, unit) cell, or None."""
    loss_tr, loss_va, auc_va = [], [], []
    for rep in reps:
        for k in fold_range:
            hp = os.path.join(O, "history",
                              f"{DATASET}_{su}_{arm}_rep{rep}_fold{k}_{arch}_none_seed42.json")
            if not os.path.exists(hp):
                continue
            h = json.load(open(hp))["history"]
            loss_tr.append([x["train_loss"] for x in h])
            loss_va.append([x["val_loss"] for x in h])
            auc_va.append([x["val_auc"] for x in h])
    if not auc_va:
        return None
    n = min(len(c) for c in auc_va)

    def band(mat):
        A = np.array([c[:n] for c in mat])
        return A.mean(0), A.std(0)

    return dict(ep=np.arange(1, n + 1), loss_va=band(loss_va), loss_tr=band(loss_tr),
                auc=band(auc_va), runs=len(auc_va))


def plot_curves(cfg, archs, outdir, reps=(0,), fold_range=range(5), sample_units=("slice",),
                stem=None):
    """Learning curves for BOTH architectures in ONE figure: rows = sample unit, columns =
    architecture x metric, lettered panels.

    Both architectures are shown because the claim is about a protocol, not about a model: a
    memorisation signature that appears in one backbone and not the other would be a property of
    that backbone. Showing both is the replication.

    With both units this figure IS the mechanism, visually (D41): at SLICE level the random arm's
    validation loss keeps falling with training loss (the leaked val split is memorised alongside
    it) while the patient arm's diverges upward — the honest generalisation gap. At NODULE level,
    where within-nodule slice redundancy is eliminated by construction, that asymmetry largely
    disappears, matching the best_epoch asymmetry (+21.5/+15.6 epochs at slice vs +5.0/+0.3 at
    nodule) and the ~0 nodule-axis gap.

    NO figure-level title (IEEE Access: "Please do not include captions as part of the figures").
    """
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    archs, sus = list(archs), list(sample_units)
    colors = {"patient": "#1b7837", "random": "#c51b7d"}
    nc = 2 * len(archs)
    fig, axes = plt.subplots(len(sus), nc, figsize=(min(4.9 * len(archs), 9.8), 3.4 * len(sus)),
                             squeeze=False)
    letters = string.ascii_lowercase
    any_found = False
    for r, su in enumerate(sus):
        populated = []
        for c_a, arch in enumerate(archs):
            ax_l, ax_a = axes[r][2 * c_a], axes[r][2 * c_a + 1]
            found = False
            for arm in ("patient", "random"):
                b = _history_band(O, arch, arm, su, reps, fold_range)
                if b is None:
                    continue
                found = any_found = True
                ep = b["ep"]
                mu, sd = b["loss_va"]
                ax_l.plot(ep, mu, color=colors[arm], label=f"{arm} val")
                ax_l.fill_between(ep, mu - sd, mu + sd, color=colors[arm], alpha=0.15)
                ax_l.plot(ep, b["loss_tr"][0], color=colors[arm], ls="--", alpha=0.6,
                          label=f"{arm} train")
                mu, sd = b["auc"]
                ax_a.plot(ep, mu, color=colors[arm], label=arm)
                ax_a.fill_between(ep, mu - sd, mu + sd, color=colors[arm], alpha=0.15)
            for j, (ax, lab) in enumerate(((ax_l, "loss"), (ax_a, "val AUC"))):
                ax.set_xlabel("epoch", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.set_title(f"({letters[r * nc + 2 * c_a + j]}) {arch}: {lab}", fontsize=9.5)
            # the panel title already names the quantity, so the y-label slot of the first column
            # carries the row label (sample unit) instead of repeating it
            if c_a == 0:
                ax_l.set_ylabel(f"{su} level", fontsize=11, fontweight="bold")
            if found:
                populated.append(2 * c_a + 1)
                if r == 0 and c_a == 0:          # one legend per figure; colours are consistent
                    ax_l.legend(fontsize=7); ax_a.legend(fontsize=7)
            else:
                for ax in (ax_l, ax_a):
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.5, 0.5, "not run", ha="center", va="center",
                            fontsize=9, transform=ax.transAxes)
        # Share the y-range across architectures for the AUC panels only, and ONLY over panels that
        # actually received data. An empty panel -- an architecture whose runs have not landed yet --
        # returns matplotlib's default (0.0, 1.0) from get_ylim(), which then wins the union and is
        # pushed onto every populated panel: an AUC band of 0.09 rendered on a 0-1 axis collapses
        # both arms into one line. Reproduced with two CNNs plus one empty transformer column.
        # NOT shared for loss: the backbones sit at different loss scales and a common range squashes
        # the flatter one into the bottom of its axis.
        if populated:
            lims = [axes[r][i].get_ylim() for i in populated]
            lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
            for i in populated:
                axes[r][i].set_ylim(lo, hi)
    if not any_found:
        plt.close(fig); return None
    fig.tight_layout()
    if stem is None:
        stem = f"curves_{'-'.join(sus)}"
    p = os.path.join(outdir, stem + ".png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archs", default=None,
                    help="comma-separated. DEFAULT: every architecture found on disk for this "
                         "dataset. It used to default to the two CNNs, which silently produced a "
                         "two-architecture figure once a third existed -- and the freshness gate "
                         "printed exactly that command as its remediation, so following the gate's "
                         "own instruction made it go green over a figure missing an architecture.")
    ap.add_argument("--reps", default="0", help="comma-separated rep indices; grid S2 = 0,1,2")
    ap.add_argument("--sample-units", default="slice",
                    help="comma-separated: slice,nodule. Passing both puts one ROW per unit in the "
                         "curves figure — the visual side-by-side of the memorisation asymmetry "
                         "(present at slice level, absent at nodule level).")
    ap.add_argument("--dataset", default="lidc_binary",
                    help="split-file prefix: lidc_binary (principal) or lidc_binary_ge3 "
                         "(>=3-annotator sensitivity cohort, D37 -- never pooled)")
    ap.add_argument("--arms", default="patient,nodule,random",
                    help="confusion-matrix columns, in display order (A, C, B)")
    args = ap.parse_args()
    global DATASET
    DATASET = args.dataset

    from src.metadata import load_config
    from src.datasets import load_processed_index
    cfg = load_config(args.config)
    pidx = load_processed_index(cfg, "none")
    reps = [int(x) for x in args.reps.split(",")]
    outdir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "figures")
    os.makedirs(outdir, exist_ok=True)

    present = discover_archs(cfg)
    if args.archs:
        archs = [a.strip() for a in args.archs.split(",") if a.strip()]
        missing = [a for a in present if a not in archs]
        if missing:
            print(f"[figures] NOTE: {', '.join(missing)} have runs on disk for {DATASET} and are "
                  f"EXCLUDED from this figure by an explicit --archs. The output name will say so.")
    else:
        archs = present
        if not archs:
            raise SystemExit(f"no runs found for dataset '{DATASET}' in outputs/probs/")
        print(f"[figures] architectures discovered on disk: {', '.join(archs)}")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    # ONE confusion figure and ONE curves figure covering every architecture, rather than one file
    # per architecture. Reviewer 4 asked for the confusion matrix and only one architecture's
    # curves reached the manuscript; a per-architecture file makes that omission easy to repeat.
    sus = [s.strip() for s in args.sample_units.split(",") if s.strip()]
    folds = list(range(5))

    # A PARTIAL architecture must never occupy the canonical filename. Discovery finds anything with
    # a single run on disk -- a timed probe is enough -- so without this check a figure carrying one
    # architecture's 2 probe runs next to another's 210 would be written under the exact name
    # manuscript.tex embeds. Completeness is counted, not assumed.
    incomplete = []
    for a in archs:
        have, want = run_counts(cfg, a, arms, reps, folds, sus)
        if have < want:
            incomplete.append((a, have, want))
    if incomplete:
        for a, have, want in incomplete:
            print(f"[figures] INCOMPLETE: {a} has {have} of {want} expected runs")
        print("[figures] -> writing under a NON-canonical name; the manuscript's figure is untouched.")

    canon_archs = present if not incomplete else None
    cm_stem = name_for("confusion_matrices", DATASET, archs, canon_archs or [None])
    cv_stem = name_for(f"curves_{'-'.join(sus)}", DATASET, archs, canon_archs or [None])
    cm = plot_confusion(cfg, archs, pidx, outdir, reps=reps, arms=arms, stem=cm_stem)
    cp = plot_curves(cfg, archs, outdir, reps=reps, sample_units=sus, stem=cv_stem)
    print(f"curves: {cp}")

    summary = {}
    if cm:
        print(f"confusion: {cm['path']}")
        for arch in archs:
            cells = {arm: cm["grid"][f"{arch}|{arm}"] for arm in arms
                     if f"{arch}|{arm}" in cm["grid"]}
            if not cells:
                continue
            summary[arch] = {arm: {k: c[k] for k in
                                   ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy")}
                             for arm, c in cells.items()}
            print(f"  {arch}")
            for arm, c in cells.items():
                print(f"   {ARM_LABEL[arm]:16} acc {c['accuracy']:.4f}  F1 {c['f1']:.4f}  "
                      f"prec {c['precision']:.4f}  rec {c['recall']:.4f}  "
                      f"(TN{c['tn']} FP{c['fp']} FN{c['fn']} TP{c['tp']})")

    R = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "results")
    os.makedirs(R, exist_ok=True)
    meta = {"_note": "CM counts summed over the test folds of each condition. patient- and "
                     "nodule-grouped folds partition their grouping unit, so each nodule is scored "
                     "once; the random condition's test folds OVERLAP by construction (leaky "
                     "protocol), so its summed CM re-scores some nodules. F1 == 2PR/(P+R) asserted "
                     "(B9). threshold 0.5, nodule level.",
            "arms": list(arms), "reps": reps, "dataset": DATASET,
            "archs": list(archs), "archs_present_on_disk": list(present),
            "arch": summary}
    json_name = cm_stem + ".json"
    json.dump(meta, open(os.path.join(R, json_name), "w"), indent=1)
    print(f"wrote {os.path.join(R, json_name)} and {outdir}/*.png")


if __name__ == "__main__":
    main()
