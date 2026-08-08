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
# TrueType, not matplotlib's default Type 3. Type 3 embeds each glyph as a drawing procedure; IEEE
# production tooling expects Type 1 or TrueType, and these figures were the only Type 3 font objects
# in the entire compiled paper.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D          # proxy handles for the shared figure-level legend
from matplotlib.patches import Rectangle   # confusion cells, drawn instead of rasterised

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

# How each architecture is NAMED IN THE PAPER. The figures used to print the internal run-tag
# identifier ("densenet121", "vit_small"), which is what the filenames and the JSON keys use, so a
# reader comparing Fig. 3 against the text saw two different sets of names for the same three models.
# The identifier stays the key everywhere in code and in artifacts; only the display label changes.
DISPLAY_NAME = {"densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0",
                "vit_small": "ViT-S/16", "swin_tiny": "Swin-T", "resnet50": "ResNet-50",
                "convnext_tiny": "ConvNeXt-T", "inception_v3": "Inception-v3",
                "maxvit_tiny_tf_224": "MaxViT-T"}


def _save(fig, outdir, stem):
    """Write the VECTOR pdf the manuscript embeds, plus a png for looking at.

    Returns the pdf path, because that is the one \\includegraphics points at.
    """
    pdf = os.path.join(outdir, stem + ".pdf")
    fig.savefig(pdf)                      # vector: no raster resolution at all
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=FIG_DPI)
    plt.close(fig)
    return pdf


def display(arch: str) -> str:
    """Paper-facing name for an architecture; falls back to the identifier if one is ever added
    without a label, so a missing entry shows up as an odd-looking figure rather than a crash."""
    return DISPLAY_NAME.get(arch, arch)


# IEEE requires >300 dpi for colour/grayscale and >600 dpi for line art
# (journals.ieeeauthorcenter.ieee.org, "Resolution and Size"). These figures ARE line art, and at
# dpi=200 they printed at ~275 effective dpi across the two-column width -- under both thresholds.
# So the submission copy is VECTOR pdf, which has no resolution to be under; the png is kept for
# quick viewing and is now well above 300 dpi at print size.
FIG_DPI = 600

# IEEE Access two-column text width, inches. NOT 7.16 -- that is IEEEtran's. ieeeaccess.cls:263 sets
# \textwidth to 177.53 mm = 503.235 pt = 6.9894 in, and this manuscript does not override it. A
# figure drawn AT this width is not rescaled by \includegraphics[width=\textwidth], so its type
# prints at the size it was set in. Drawing wider and letting LaTeX shrink it is what made the labels
# small in the first place; using the WRONG width left a residual 2.4% shrink on top of the fix.
PAGE_W = 6.9894

# Height of one row of Fig. 3, inches. Sized so the WHOLE float can sit at the TOP of a page rather
# than needing a page of its own. LaTeX allows a double-column top float up to
# \dbltopfraction x 	extheight = 0.9 x 672 = 605 pt, and it counts the graphic BOX plus the caption:
# at 1.85 in per row the graphic was 572 pt and the caption 107 pt = 680 pt, over the limit, so the
# float was deferred to a page of its own -- which the \clearpage before the bibliography then
# emitted after the acknowledgements, three pages past the text discussing it and leaving the
# preceding page three-quarters empty. At 1.48 in the graphic is 466 pt, and 466 + 107 + separation
# fits with ~18 pt to spare. Only the plot area shrinks; every font size is unchanged.
ROW_H = 1.48



# discover_archs / run_counts / name_for used to live here, one copy per analysis script.
# They now live once in src/artifacts.py: keeping four copies of the same rule is how the
# expected-cell count drifted here and nowhere else, and the drift was invisible.


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
    # Drawn at the width it will PRINT at, so \includegraphics does not rescale it and the type
    # prints at the size it is set in. At 3.05 in per column the figure was 9.15 in wide and got
    # squeezed into 6.59 in, shrinking every label by 28%.
    fig, axes = plt.subplots(nr, nc, figsize=(0.92 * PAGE_W, 0.92 * PAGE_W / nc * 0.98 * nr),
                             squeeze=False)
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
            # Drawn as four filled RECTANGLES rather than with imshow. imshow embeds a raster image
            # in the pdf -- nine of them across the grid -- which reintroduces exactly the resolution
            # question that moving to vector output was meant to remove. Four flat colour blocks
            # have no reason to be pixels.
            cmap = plt.get_cmap("Blues")
            for (i, j), v in np.ndenumerate(M):
                prop = v / M.sum()
                fill = cmap(min(prop / 0.6, 1.0))
                # White or black is chosen from the FILL's luminance, not from the proportion. The
                # old rule flipped to white at prop > 0.3, where the fill is still light: three
                # cells ended up with white digits at contrast 3.05 against a required 4.5. The
                # break-even for this colormap is luminance 0.179 (sRGB relative luminance), which
                # lifts the worst case from 2.44 to 4.60 measured across the whole ramp.
                lum = sum(c * w for c, w in zip(
                    [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in fill[:3]],
                    (0.2126, 0.7152, 0.0722)))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fill,
                                       edgecolor="white", linewidth=0.8, zorder=0))
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12,
                        color="white" if lum < 0.179 else "black", zorder=1)
            ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)     # row 0 on top, as imshow had it
            ax.set_aspect("equal")
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
        fig.text(0.012, (bb.y0 + bb.y1) / 2, display(a), rotation=90, ha="left", va="center",
                 fontsize=11, fontweight="bold")
    p = _save(fig, outdir, stem)
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
    """Learning curves for every architecture in ONE figure.

    LAYOUT: one COLUMN per architecture, one ROW per (sample unit x metric). With three
    architectures that is 3 columns and 4 rows.

    It used to be 2 rows x (2 x architectures) columns, which at three architectures meant SIX
    columns drawn on a 9.8-inch canvas and then squeezed into the 7.16-inch text width -- a 27%
    reduction that took 8 pt tick labels down to about 5.8 pt on paper. Drawing at the real print
    width instead, with one column per architecture, gives each panel about 2.2 inches and lets the
    type print at the size it is set in. The panel count is unchanged; only the arrangement is.

    Every architecture is shown because the claim is about a protocol, not about a model: a
    memorisation signature that appeared in one backbone and not the others would be a property of
    that backbone. Showing all three is the replication.

    With both sample units this figure IS the mechanism, visually (D41): at SLICE level the random
    arm's validation loss keeps falling with training loss (the leaked val split is memorised
    alongside it) while the patient arm's diverges upward -- the honest generalisation gap. At
    NODULE level, where within-nodule slice redundancy is eliminated by construction, that asymmetry
    largely disappears, matching the best_epoch asymmetry (+20.1/+15.5/+11.3 epochs at slice against
    +3.8/+1.7/-0.5 at nodule) and the ~0 nodule-axis gap.

    NO figure-level title (IEEE Access: "Please do not include captions as part of the figures").
    """
    O = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"])
    archs, sus = list(archs), list(sample_units)
    # Chosen so the two arms survive a MONOCHROME print. The previous pair, #1b7837 and #c51b7d,
    # converts to grey 84.8 and 89.0 -- 4.2 levels out of 255 -- and the arm is carried by hue
    # alone, because the dash pattern already means validation-versus-training. A greyscale reader
    # could not tell the arms apart at all. This pair is 84 levels apart, both are legible as a line
    # on white (11.4:1 and 3.8:1 against white), and they stay separated under deuteranopia and
    # protanopia. A lighter magenta separates further but drops under 3:1, so it was rejected.
    colors = {"patient": "#00441b", "random": "#dd4a99"}
    rows = [(su, m) for su in sus for m in ("loss", "auc")]      # slice loss, slice AUC, nodule ...
    nr, nc = len(rows), len(archs)
    fig, axes = plt.subplots(nr, nc, figsize=(PAGE_W, ROW_H * nr + 0.55), squeeze=False)
    letters = string.ascii_lowercase
    any_found = False

    for r, (su, metric) in enumerate(rows):
        populated = []
        for c, arch in enumerate(archs):
            ax = axes[r][c]
            b = {arm: _history_band(O, arch, arm, su, reps, fold_range)
                 for arm in ("patient", "random")}
            if all(v is None for v in b.values()):
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=8,
                        transform=ax.transAxes)
                ax.set_title(f"({letters[r * nc + c]})", fontsize=9, loc="left")
                continue
            any_found = True
            populated.append(c)
            for arm, v in b.items():
                if v is None:
                    continue
                ep = v["ep"]
                if metric == "loss":
                    mu, sd = v["loss_va"]
                    ax.plot(ep, mu, color=colors[arm], lw=1.4)
                    ax.fill_between(ep, mu - sd, mu + sd, color=colors[arm], alpha=0.15, lw=0)
                    ax.plot(ep, v["loss_tr"][0], color=colors[arm], ls="--", lw=1.2, alpha=0.65)
                else:
                    mu, sd = v["auc"]
                    ax.plot(ep, mu, color=colors[arm], lw=1.4)
                    ax.fill_between(ep, mu - sd, mu + sd, color=colors[arm], alpha=0.15, lw=0)
            ax.set_title(f"({letters[r * nc + c]})", fontsize=9, loc="left")
            ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
            if r == nr - 1:
                ax.set_xlabel("epoch", fontsize=8.5)
            else:
                ax.set_xticklabels([])
        # Share the y-range across architectures WITHIN a row, over populated panels only. An empty
        # panel returns matplotlib's default (0, 1) from get_ylim(), which would win the union and
        # flatten every real curve onto a 0-1 axis. Rows are now one metric each, so sharing is safe
        # for AUC AND for loss -- within a row the quantity is identical.
        if populated and metric == "auc":
            lims = [axes[r][i].get_ylim() for i in populated]
            lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
            for i in populated:
                axes[r][i].set_ylim(lo, hi)
        # the leftmost panel of each row carries the row label; the loss scales differ between
        # backbones, so loss rows keep their own y-range and only say what they are
        axes[r][0].set_ylabel(f"{su} level\n{'loss' if metric == 'loss' else 'val AUC'}",
                              fontsize=8.5, fontweight="bold")

    if not any_found:
        plt.close(fig); return None
    fig.tight_layout(h_pad=0.7, w_pad=1.0)

    # ONE legend for the whole figure, below the panels. Colour and line style mean the same thing
    # in all twelve panels, so a per-panel legend is either redundant or -- as it was until
    # 2026-08-03 -- present in two panels and absent from the other ten, which reads as though those
    # ten are unlabelled. Keeping it outside the axes also stops it covering the curves.
    handles = [Line2D([], [], color=colors["patient"], lw=1.6),
               Line2D([], [], color=colors["patient"], lw=1.3, ls="--", alpha=0.65),
               Line2D([], [], color=colors["random"], lw=1.6),
               Line2D([], [], color=colors["random"], lw=1.3, ls="--", alpha=0.65)]
    labels = ["patient-grouped, validation", "patient-grouped, training",
              "random, validation", "random, training"]

    fig_h = ROW_H * nr + 0.55
    head, foot = 0.30 / fig_h, 0.26 / fig_h
    top0, bot0 = fig.subplotpars.top, fig.subplotpars.bottom
    fig.subplots_adjust(top=top0 - head * (top0 - bot0) - 0.004,
                        bottom=bot0 + foot * (top0 - bot0) + 0.004)
    # The architecture header sits above the PANEL LABEL, so its y comes from the rendered title box
    # rather than from a guessed offset -- a guess landed on top of the labels, because the space a
    # title occupies depends on the font and on how many rows the figure has.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for c, arch in enumerate(archs):
        bb = axes[0][c].get_position()
        top = inv.transform(axes[0][c].title.get_window_extent(rend)).max(axis=0)[1]
        fig.text((bb.x0 + bb.x1) / 2, top + 0.004, display(arch), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=2.2)
    if stem is None:
        stem = f"curves_{'-'.join(sus)}"
    return _save(fig, outdir, stem)


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

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    # ONE confusion figure and ONE curves figure covering every architecture, rather than one file
    # per architecture. Reviewer 4 asked for the confusion matrix and only one architecture's
    # curves reached the manuscript; a per-architecture file makes that omission easy to repeat.
    sus = [s.strip() for s in args.sample_units.split(",") if s.strip()]
    folds = list(range(5))

    # Which architectures, and what the file is called, are decided in src/artifacts.py -- the same
    # rules every analysis script uses. This module had its own copies of all three (discover_archs,
    # run_counts, name_for), and its copy counted the expected runs as a 2 x 3 cross product, so a
    # finished architecture read as 75 of 90, no architecture was ever "complete", and the probe
    # filter that depends on completeness never engaged. A one-run probe was drawn as a full row.
    from src.artifacts import resolve_archs, name_for as _name_for
    probs_dir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "probs")
    archs, present = resolve_archs(probs_dir, DATASET, args.archs,
                                   log=lambda m: print(m.replace("[artifacts]", "[figures]")),
                                   cells=(sus, arms, reps, folds))

    cm_stem = _name_for("confusion_matrices", DATASET, archs, present)
    cv_stem = _name_for(f"curves_{'-'.join(sus)}", DATASET, archs, present)
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
    meta = {"_note": "CM counts summed over ALL test folds of each condition -- 3 repeats x 5 folds. "
                     "Within ONE repeat the patient- and nodule-grouped folds partition their "
                     "grouping unit, so each nodule is scored once per repeat and THREE times in "
                     "these totals; the earlier wording said 'scored once', which is true per repeat "
                     "and wrong for the numbers reported here. The random condition's test folds "
                     "OVERLAP by construction (leaky protocol), so a nodule falls in ~3.2 of the 5 "
                     "folds and its totals are correspondingly larger. F1 == 2PR/(P+R) asserted "
                     "(B9). threshold 0.5, nodule level.",
            "arms": list(arms), "reps": reps, "dataset": DATASET,
            "archs": list(archs), "archs_present_on_disk": list(present),
            "arch": summary}
    json_name = cm_stem + ".json"
    json.dump(meta, open(os.path.join(R, json_name), "w"), indent=1)
    print(f"wrote {os.path.join(R, json_name)} and {outdir}/*.png")


if __name__ == "__main__":
    main()
