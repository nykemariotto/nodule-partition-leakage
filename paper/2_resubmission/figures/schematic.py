"""
Method schematic for the controlled THREE-ARM leakage study (IEEE Access, Access-2026-27906).

DETERMINISTIC matplotlib figure — NO randomness, NO external assets, NO network, NO generative image
synthesis. Renders schematic.png (400 dpi) and schematic.pdf (vector). Run with the `nodules` env:
    C:\\ProgramData\\miniconda3\\envs\\nodules\\python.exe paper/2_resubmission/figures/schematic.py

SCOPE (D65): the MAIN design only. The >=3-annotator cohort and any additional backbone are
sensitivity analyses and belong in their own figure. Do not fold them in.

EVERY number in this figure is READ FROM AN ARTIFACT AT RENDER TIME, never typed here:
    cohort counts          <- outputs/metadata/dataset_accounting.json
    L_pat = fraction of test rows that share a PATIENT with the training fold
    L_nod = fraction of test rows that share a NODULE  with the training fold
            (the figure spells this out in words above the two rows: a bare label such as
             "Patient split 0.690" reads most naturally as "69% of patients were split", which is a
             DIFFERENT quantity -- a correct number a reader can carry away as a wrong fact)
                           <- computed from the committed split indices in outputs/splits/,
                              averaged over the 15 folds (3 repeats x 5), slice sample unit
and the values used are written next to the figure as `schematic_values.csv`, so the figure ships
with its own data (the project's figure gate requires it). If an artifact is missing the render
FAILS rather than falling back to a literal -- a hardcoded number is exactly how a figure silently
desynchronises from the data it claims to describe. No performance value is depicted.

DESIGN RULES (D66) — what makes a method schematic read as publication-grade is not the drawing
engine, so these constraints are deliberate and should survive future edits:
  * Type: ONE family (Arial, the journal standard), THREE sizes. Nothing else.
  * Colour: Okabe-Ito, colourblind-safe, and never the ONLY channel. One hue per arm, carried by
    that arm's rule alone -- headings stay black, because #cc79a7 converts to grey 151 and
    vanishes in monochrome. Vermillion means "this unit is split across training and test" and
    nothing else anywhere in the figure. Body text is never coloured. Colour never decorates.
  * No rounded corners, no drop shadows, no gradients, no 3-D, no outlined boxes. Grouping is by
    flat tint field and hairline, which is what current journal figures do.
  * One grid: the three arm columns share every x-offset, so the strips are pixel-aligned and the
    arms are comparable by eye rather than by reading.
  * Greyscale-safe, and CHECKED by converting the render to L and re-reading it: training/test
    differ in fill AND hatch; the arms differ in line style as well as hue; a split unit is
    marked by a 3x heavier bracket, not by vermillion alone (vermillion -> grey 119, the
    neutral -> grey 107, i.e. indistinguishable); and a non-zero rate is bold while a zero rate
    is regular grey, so the flagged value is the heavier one rather than the lighter one.
  * Panel letters (a)/(b)/(c) match the convention used in Figs. 2 and 3 of the same paper.
  * Hatches inherit the patch EDGE colour in matplotlib. A patch with edgecolor="none" has an
    invisible hatch -- set the edge colour and zero the line width instead.
"""
import csv
import glob
import json
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "hatch.linewidth": 0.6,
})
# The figure is placed at 	extwidth with NO reduction, so these are the sizes that print.
# 6.2 pt was the previous floor -- legible but at the low end of what production editors
# accept without comment, so the scale was lifted to a 7 pt floor and the layout re-checked.
FS_TITLE, FS_BODY, FS_TINY = 9.2, 7.7, 7.0

INK   = "#1a1a1a"
MUTED = "#6b6b6b"
HAIR  = "#c9c9c9"
FIELD = "#f2f4f6"
TRAIN = "#e6e6e6"
TEST  = "#8d8d8d"
LEAK  = "#d55e00"                                    # vermillion: ONLY "split across train/test"
ARM = {"A": "#0072b2", "C": "#009e73", "B": "#cc79a7"}
LS  = {"A": "-", "C": (0, (4, 1.5)), "B": (0, (1.2, 1.3))}

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUTS = os.path.join(REPO, "outputs")
DATASET = "lidc_binary"


def read_cohort():
    """Cohort counts, from the accounting artifact. No literals."""
    p = os.path.join(OUTPUTS, "metadata", "dataset_accounting.json")
    if not os.path.exists(p):
        raise SystemExit(f"FATAL: {p} missing -- refusing to render with hardcoded counts.")
    c = json.load(open(p))["principal_cohort_min1"]
    return {"patients": c["patients"], "nodules": c["nodule"]["n"], "slices": c["slice"]["n"]}


def measure_leakage(arm):
    """(L_pat, L_nod, n_folds) for one arm, from the committed split indices.

    L_pat = fraction of TEST ROWS whose patient also appears in TRAIN; L_nod likewise for nodule.
    Rows, not unique patients: it is the quantity the manuscript defines and the one that says how
    much of the evaluation is contaminated. Averaged over every (rep, fold) on disk.
    """
    lp, ln = [], []
    for tr_path in sorted(glob.glob(os.path.join(
            OUTPUTS, "splits", f"{DATASET}_slice_{arm}_rep*_fold*_train.csv"))):
        te_path = tr_path[: -len("_train.csv")] + "_test.csv"
        if not os.path.exists(te_path):
            continue
        tr = pd.read_csv(tr_path, usecols=["patient_id", "nodule_id"])
        te = pd.read_csv(te_path, usecols=["patient_id", "nodule_id"])
        lp.append(te.patient_id.isin(set(tr.patient_id)).mean())
        ln.append(te.nodule_id.isin(set(tr.nodule_id)).mean())
    if not lp:
        raise SystemExit(f"FATAL: no split indices found for arm '{arm}' in outputs/splits/.")
    return sum(lp) / len(lp), sum(ln) / len(ln), len(lp)

W, H = 140.0, 100.0

# vertical grid — every y in the figure comes from here, so the bands stay separated
Y_CAP, Y_PIPE, H_PIPE = 97.5, 78.0, 17.0
Y_BUS, Y_RULE = 69.5, 63.0
Y_STRIP, H_STRIP = 47.5, 4.6   # top of the strip must clear the sub-title above it
Y_RATE_HEAD = 33.0                    # the header that says what the two rates are a fraction OF
Y_RATE, D_RATE = 27.5, 5.4
Y_NOTE, Y_SEP, Y_CONTRAST = 15.0, 10.0, 6.0

# The two rates share a subject, so it is factored into a header and the rows complete the sentence.
# Without it "Patient split 0.690" reads most naturally as "69% of patients were split", which is a
# DIFFERENT quantity from the one measured -- a correct number that a reader can carry away as a
# wrong fact is worse than an unclear one.
RATE_HEAD = "Fraction of test rows that share a"
RATE_ROW = {"pat": "patient with the training fold", "nod": "nodule with the training fold"}


def field(ax, x, y, w, h, title, lines):
    """A flat tint field: no border, no rounding. Grouping by tone, not by outline."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=FIELD, edgecolor="none", zorder=1))
    ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="top", fontsize=FS_TITLE,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 8.6, "\n".join(lines), ha="center", va="top", fontsize=FS_BODY,
            color=MUTED, linespacing=1.45, zorder=3)


def bracket(ax, x0, x1, y, label, hot, depth):
    c = LEAK if hot else MUTED
    ax.plot([x0, x0, x1, x1], [y, y - depth, y - depth, y], color=c,
            lw=1.8 if hot else 0.6, zorder=3, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - depth - 0.8, label, ha="center", va="top", fontsize=FS_TINY,
            color=c, fontweight="bold" if hot else "normal", zorder=3)


def strip(ax, x, w, assign, nodules, patients, leak_nod, leak_pat):
    """Eight slice-cells = 2 patients x 2 nodules x 2 slices, bracketed twice.

    The two bracket levels ARE the figure: arm A splits neither unit, arm C splits the patient but
    not the nodule, arm B splits both. Everything else is scaffolding around that one contrast.
    """
    cw = w / len(assign)
    for i, a in enumerate(assign):
        test = a == "test"
        ax.add_patch(Rectangle((x + i * cw + 0.25, Y_STRIP), cw - 0.5, H_STRIP,
                               facecolor=TEST if test else TRAIN,
                               edgecolor="white" if test else "none",   # hatch takes the edge colour
                               linewidth=0, hatch="///" if test else "", zorder=2))
    for lab, i0, i1 in nodules:
        bracket(ax, x + i0 * cw + 0.25, x + i1 * cw - 0.5, Y_STRIP - 1.2, lab,
                lab in leak_nod, 1.3)
    for lab, i0, i1 in patients:
        bracket(ax, x + i0 * cw + 0.25, x + i1 * cw - 0.5, Y_STRIP - 7.6, lab,
                lab in leak_pat, 1.6)


def main():
    cohort = read_cohort()
    meas = {k: measure_leakage(a) for k, a in (("A", "patient"), ("C", "nodule"), ("B", "random"))}
    for k, (lp, ln, n) in meas.items():
        print(f"  arm {k}: L_pat {lp:.4f}  L_nod {ln:.4f}  (over {n} folds)")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    # ---- shared pipeline ----------------------------------------------------------------------
    pw, gap = 33.0, 9.0
    xs = [(W - (3 * pw + 2 * gap)) / 2 + i * (pw + gap) for i in range(3)]
    field(ax, xs[0], Y_PIPE, pw, H_PIPE, "Cohort",
          [f"{cohort['patients']:,} patients", f"{cohort['nodules']:,} nodules",
           f"{cohort['slices']:,} slices"])
    field(ax, xs[1], Y_PIPE, pw, H_PIPE, "Preprocessing",
          ["50% consensus ROI", "256 \u00d7 256 px", "identical in all arms"])
    field(ax, xs[2], Y_PIPE, pw, H_PIPE, "Backbone",
          ["ImageNet-pretrained", "DenseNet-121", "EfficientNet-B0"])
    for i in (0, 1):
        ax.add_patch(FancyArrow(xs[i] + pw + 2.0, Y_PIPE + H_PIPE / 2, gap - 4.0, 0, width=0.25,
                                head_width=1.5, head_length=1.6, length_includes_head=True,
                                color=MUTED, zorder=3))
    ax.text(W / 2, Y_CAP, "Cohort, labels, preprocessing, architecture and optimiser are identical "
            "across all three arms", ha="center", va="center", fontsize=FS_BODY, style="italic",
            color=MUTED)

    # ---- fork: one hairline tree, not three heavy arrows ---------------------------------------
    cw_col, cgap = 40.0, 7.0
    cx0 = (W - (3 * cw_col + 2 * cgap)) / 2
    cols = {k: cx0 + i * (cw_col + cgap) for i, k in enumerate(("A", "C", "B"))}
    ax.plot([W / 2, W / 2], [Y_PIPE, Y_BUS], color=HAIR, lw=0.8, zorder=1)
    ax.plot([cols["A"] + cw_col / 2, cols["B"] + cw_col / 2], [Y_BUS, Y_BUS],
            color=HAIR, lw=0.8, zorder=1)
    ax.text(W / 2, (Y_PIPE + Y_BUS) / 2, "the partition unit is the only thing that changes",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED, zorder=4,
            bbox=dict(boxstyle="square,pad=0.32", fc="white", ec="none"))

    # ---- the three arms, side by side so they are comparable by eye -----------------------------
    T, E = "train", "test"
    NOD = [("N1", 0, 2), ("N2", 2, 4), ("N3", 4, 6), ("N4", 6, 8)]
    PAT = [("Patient 1", 0, 4), ("Patient 2", 4, 8)]
    arms = [
        ("A", "(a)", "Arm A — Patient-grouped", "GroupKFold on patient",
         [T, T, T, T, E, E, E, E], "neither unit crosses the boundary"),
        ("C", "(b)", "Arm C — Nodule-grouped", "GroupKFold on nodule",
         [T, T, E, E, T, T, E, E], "patient route on, within-nodule route off"),
        ("B", "(c)", "Arm B — Random", "KFold, no grouping",
         [T, E, T, E, E, T, T, E], "both routes on"),
    ]
    for key, letter, name, rule, assign, note in arms:
        # which drawn units straddle is DERIVED from the drawn assignment, not asserted by hand,
        # and then checked against the measured rates -- an illustration that contradicted the
        # data it sits next to would be worse than no illustration
        lnod = {lab for lab, i0, i1 in NOD if len(set(assign[i0:i1])) > 1}
        lpat = {lab for lab, i0, i1 in PAT if len(set(assign[i0:i1])) > 1}
        for drawn, measured, unit in ((lnod, meas[key][1], "nodule"), (lpat, meas[key][0], "patient")):
            assert bool(drawn) == (measured > 0), (
                f"arm {key}: the drawn strip shows the {unit} unit "
                f"{'split' if drawn else 'intact'}, but the measured rate is {measured:.3f}")
        rates = ((RATE_ROW["pat"], f"{meas[key][0]:.3f}"),
                 (RATE_ROW["nod"], f"{meas[key][1]:.3f}"))
        x, c = cols[key], ARM[key]
        ax.plot([x, x + cw_col], [Y_RULE, Y_RULE], color=c, lw=1.5, ls=LS[key],
                zorder=3, solid_capstyle="butt")
        ax.plot([x + cw_col / 2, x + cw_col / 2], [Y_BUS, Y_RULE], color=HAIR, lw=0.8, zorder=1)
        ax.text(x, Y_RULE - 3.2, letter, ha="left", va="top", fontsize=FS_TITLE,
                fontweight="bold", color=INK)
        ax.text(x + 5.4, Y_RULE - 3.2, name, ha="left", va="top", fontsize=FS_TITLE,
                fontweight="bold", color=INK)
        ax.text(x + 5.4, Y_RULE - 7.8, rule, ha="left", va="top", fontsize=FS_TINY, color=MUTED)

        strip(ax, x + 1.5, cw_col - 3.0, assign, NOD, PAT, lnod, lpat)

        ax.text(x + 1.5, Y_RATE_HEAD, RATE_HEAD, ha="left", va="center", fontsize=FS_TINY,
                color=INK)
        y = Y_RATE
        for lab, val in rates:
            ax.text(x + 1.5, y, lab, ha="left", va="center", fontsize=FS_TINY, color=MUTED)
            hot = float(val) > 0
            ax.text(x + cw_col - 1.5, y, val, ha="right", va="center", fontsize=FS_BODY,
                    fontweight="bold" if hot else "normal",
                    color=LEAK if hot else MUTED)
            ax.plot([x + 1.5, x + cw_col - 1.5], [y - 2.4, y - 2.4], color=HAIR, lw=0.5)
            y -= D_RATE
        ax.text(x + 1.5, Y_NOTE, note, ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    # ---- what the three arms buy you ------------------------------------------------------------
    ax.plot([cx0, cx0 + 3 * cw_col + 2 * cgap], [Y_SEP, Y_SEP], color=HAIR, lw=0.5)
    # the tokens here are IDENTICAL to the Contrast column of Table 3, on purpose: a reader must be
    # able to map the figure onto the table without translating a second naming scheme
    for i, (lhs, rhs) in enumerate(((" B \u2212 A ", "total gap"),
                                    (" C \u2212 A ", "patient route"),
                                    (" B \u2212 C ", "within-nodule route"))):
        x = cx0 + i * (cw_col + cgap)
        ax.text(x + 1.5, Y_CONTRAST, lhs, ha="left", va="center", fontsize=FS_BODY,
                fontweight="bold", color=INK)
        ax.text(x + 11.0, Y_CONTRAST, rhs, ha="left", va="center", fontsize=FS_BODY, color=MUTED)

    # ---- legend, unframed, on its own band -------------------------------------------------------
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=TRAIN, edgecolor="none", label="training slice"),
        Rectangle((0, 0), 1, 1, facecolor=TEST, edgecolor="white", linewidth=0, hatch="///",
                  label="test slice"),
        Line2D([0], [0], color=LEAK, lw=1.2, label="unit split across training and test"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(cx0 / W, 0.0), ncol=3,
              frameon=False, fontsize=FS_TINY, handlelength=1.5, handleheight=0.9,
              columnspacing=2.0, labelcolor=MUTED)

    here = os.path.dirname(os.path.abspath(__file__))
    for ext, dpi in (("png", 400), ("pdf", None)):
        fig.savefig(os.path.join(here, f"schematic.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)

    # the figure ships with its own data (figure gate: "dado exportado junto")
    with open(os.path.join(here, "schematic_values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "arm", "value", "n_folds", "source"])
        for k, v in cohort.items():
            w.writerow([k, "all", v, "", "outputs/metadata/dataset_accounting.json"])
        for k, (lp, ln, n) in meas.items():
            w.writerow(["L_pat (patient split)", k, f"{lp:.4f}", n, "outputs/splits/*.csv"])
            w.writerow(["L_nod (nodule split)", k, f"{ln:.4f}", n, "outputs/splits/*.csv"])
    print("wrote schematic.png / schematic.pdf / schematic_values.csv to", here)


if __name__ == "__main__":
    main()
