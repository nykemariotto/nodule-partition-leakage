"""
Method schematic for the controlled THREE-ARM leakage study (IEEE Access, Access-2026-27906).

DETERMINISTIC matplotlib figure — NO randomness, NO external assets, NO network, NO generative image
synthesis. Renders schematic.png (400 dpi) and schematic.pdf (vector). Run with the `nodules` env:
    python paper/2_resubmission/figures/schematic.py        (environment: environment.yml)

SCOPE (D65, revised D73): the MAIN design only. The >=3-annotator cohort is a sensitivity analysis
and belongs in its own figure; do not fold it in. The three ARCHITECTURES, however, are all part of
the main design as of 2026-08-03 -- the transformer arm is not a sensitivity analysis -- so the
Backbone field lists all three. swin_tiny is a one-run timing probe and must never appear here.

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
# TrueType, not matplotlib's default Type 3 -- IEEE production tooling expects Type 1 or TrueType,
# and the three figures were the only Type 3 font objects in the whole compiled paper.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "hatch.linewidth": 0.6,
})
# The figure is placed at \textwidth with NO reduction, so these are the sizes that print.
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
# A MEASURED zero rate. It stays a numeral -- these zeros are the evidence that patient grouping and
# nodule grouping do what they claim, and a dash would say "not measured" instead of "measured and
# found to be zero". But it is set one step lighter than the labels so the eye reaches the flagged
# values first. The value is chosen by measurement, not taste: at 3.84:1 against white it has the
# SAME contrast as the vermillion flagged value (3.87:1), so neither number is harder to read than
# the other, and its greyscale value (130) is now LIGHTER than the flagged value's (119) instead of
# darker -- so in monochrome the flagged value is both bolder and darker, rather than relying on
# weight alone to invert an ordering the colours got backwards.
ZERO  = "#828282"
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

# IEEE Access two-column text width, inches: ieeeaccess.cls:263 sets \textwidth to 177.53 mm =
# 503.235 pt. NOT 7.16 in, which is IEEEtran's and is what this file used to assume.
PAGE_W = 6.9894

# Output stem. Renamed from "schematic" on 2026-08-04: replacing schematic.pdf in place on Overleaf
# repeatedly failed to take effect while the other two figures updated normally, and the compiled
# paper kept embedding an older render. A name that has never existed in that project cannot collide
# with a stale copy or a cached entry, so the next upload is unambiguous -- either the figure appears
# and is necessarily the new one, or it does not appear at all.
FIG_STEM = "fig1_design"

W, H = 140.0, 100.0

# vertical grid — every y in the figure comes from here, so the bands stay separated
Y_CAP = 97.5
# The pipeline row is anchored by its TOP, and its HEIGHT is computed from the longest field in the
# row (see field_h). It used to be a fixed height of 17.0 with the body text top-anchored inside it,
# which fitted exactly three lines and nothing more: adding ViT-S/16 as a fourth line to "Backbone"
# pushed that line straight out through the bottom of its tint box, and left the three-line boxes
# beside it looking top-heavy because their text hung from the top with the slack all below.
# Anchoring the top keeps the row clear of the caption line at Y_CAP; the growth goes downward,
# where there is slack before Y_BUS.
Y_PIPE_TOP = 95.0
LINE_H = 1.30                          # body line spacing, multiples of the font size
FIELD_HEAD = 7.2                       # box top -> first body line (top pad + title + gap)
FIELD_FOOT = 1.6                       # last body line -> box bottom
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


def _pt(size):
    """Points -> data units. The axes span H=100 units over 5.0 inches, so one unit is 3.6 pt."""
    return size / 3.6


def field_h(n_lines):
    """Height a tint field needs for `n_lines` of body text under its title, in data units."""
    return FIELD_HEAD + (n_lines - 1) * _pt(FS_BODY * LINE_H) + _pt(FS_BODY) + FIELD_FOOT


def field(ax, x, y, w, h, title, lines, size=None):
    """A flat tint field: no border, no rounding. Grouping by tone, not by outline.

    The body block is CENTRED in the space below the title rather than hung from the top, so fields
    with different line counts sit level with each other in a row. Top-anchoring made a three-line
    field look like it had drifted upward next to a four-line one.

    `size` overrides the body size for a field whose content is a LIST set on one line rather than a
    stack of short lines. It may only be one of the three sizes in the scale.
    """
    size = FS_BODY if size is None else size
    assert size in (FS_TITLE, FS_BODY, FS_TINY), "one family, three sizes — nothing else (D66)"
    ax.add_patch(Rectangle((x, y), w, h, facecolor=FIELD, edgecolor="none", zorder=1))
    ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=FS_TITLE,
            fontweight="bold", color=INK, zorder=3)
    zone_top, zone_bot = y + h - FIELD_HEAD + _pt(FS_BODY) / 2, y + FIELD_FOOT
    return ax.text(x + w / 2, (zone_top + zone_bot) / 2, "\n".join(lines), ha="center", va="center",
                   fontsize=size, color=MUTED, linespacing=LINE_H, zorder=3)


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

    # Drawn at the IEEE two-column text width, so \includegraphics[width=\textwidth] neither
    # enlarges nor shrinks it and the type prints at the size it is set in. Drawing wider and
    # letting LaTeX fit it to the column is what took Figs. 2 and 3 below the 7 pt floor (D76);
    # at 9.2 in it would put this figure's body text at 6.0 pt.
    fig, ax = plt.subplots(figsize=(PAGE_W, 5.0))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    # ---- shared pipeline ----------------------------------------------------------------------
    pw, gap = 39.0, 6.0
    xs = [(W - (3 * pw + 2 * gap)) / 2 + i * (pw + gap) for i in range(3)]
    # One height for the whole row, sized by its LONGEST field, so the three tint boxes stay level
    # and no field can overflow when a line is added to one of them.
    #
    # The three backbones are ONE line separated by middots, not a stack of three. A stack made the
    # Backbone field a line taller than its neighbours and grew the whole row with it; a list reads
    # as a list. It is set at FS_TINY because at FS_BODY the line needs 38.9 data units and would
    # leave the row 139.5 of 140 units wide, i.e. no margin at all. FS_TINY is already one of the
    # three sizes in the scale, so no new size is introduced.
    pipe = [("Cohort", [f"{cohort['patients']:,} patients", f"{cohort['nodules']:,} nodules",
                        f"{cohort['slices']:,} slices"], None),
            ("Preprocessing", ["50% consensus ROI", "256 \u00d7 256 px", "identical in all arms"], None),
            ("Backbone", ["ImageNet-pretrained",
                          "DenseNet-121 \u00b7 EfficientNet-B0 \u00b7 ViT-S/16"], FS_TINY)]
    h_pipe = field_h(max(len(lines) for _, lines, _ in pipe))
    y_pipe = Y_PIPE_TOP - h_pipe
    if y_pipe < Y_BUS + 4.0:
        raise SystemExit(f"FATAL: the pipeline row needs {h_pipe:.1f} units and would reach "
                         f"y={y_pipe:.1f}, colliding with the bus at {Y_BUS}. Shorten a field or "
                         f"move the bands.")
    body_texts = [field(ax, x, y_pipe, pw, h_pipe, title, lines, size)
                  for (title, lines, size), x in zip(pipe, xs)]
    # Does the text actually FIT the box it is in? Measured, not eyeballed. A line that overruns its
    # tint field is the exact failure that shipped once already (D78), and it is invisible in the
    # source: it depends on the font, the size and the string, none of which the geometry knows.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, (title, _, _) in zip(body_texts, pipe):
        w_units = t.get_window_extent(rend).width / fig.dpi * 72 / 72 / fig.get_figwidth() * W
        if w_units > pw - 2.0:
            raise SystemExit(f"FATAL: the '{title}' field's text is {w_units:.1f} data units wide "
                             f"and its box is {pw:.1f}. Widen the box, shorten the line, or drop it "
                             f"a size — do not ship a field that overruns its tint.")
        print(f"  field {title:14} text {w_units:5.1f} of {pw:.1f} units")
    for i in (0, 1):
        ax.add_patch(FancyArrow(xs[i] + pw + 1.2, y_pipe + h_pipe / 2, gap - 2.4, 0, width=0.25,
                                head_width=1.5, head_length=1.6, length_includes_head=True,
                                color=MUTED, zorder=3))
    ax.text(W / 2, Y_CAP, "Cohort, labels, preprocessing, architecture and optimiser are identical "
            "across all three arms", ha="center", va="center", fontsize=FS_BODY, style="italic",
            color=MUTED)

    # ---- fork: one hairline tree, not three heavy arrows ---------------------------------------
    cw_col, cgap = 40.0, 7.0
    cx0 = (W - (3 * cw_col + 2 * cgap)) / 2
    cols = {k: cx0 + i * (cw_col + cgap) for i, k in enumerate(("A", "C", "B"))}
    ax.plot([W / 2, W / 2], [y_pipe, Y_BUS], color=HAIR, lw=0.8, zorder=1)
    ax.plot([cols["A"] + cw_col / 2, cols["B"] + cw_col / 2], [Y_BUS, Y_BUS],
            color=HAIR, lw=0.8, zorder=1)
    ax.text(W / 2, (y_pipe + Y_BUS) / 2, "the partition unit is the only thing that changes",
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
        # the sub-label describes the COLUMN, so it sits on the column's left edge, not on the
        # title's indent -- the title is indented only because the panel letter precedes it
        ax.text(x, Y_RULE - 7.8, rule, ha="left", va="top", fontsize=FS_TINY, color=MUTED)

        strip(ax, x, cw_col, assign, NOD, PAT, lnod, lpat)

        ax.text(x, Y_RATE_HEAD, RATE_HEAD, ha="left", va="center", fontsize=FS_TINY,
                color=INK)
        y = Y_RATE
        for lab, val in rates:
            ax.text(x, y, lab, ha="left", va="center", fontsize=FS_TINY, color=MUTED)
            hot = float(val) > 0
            ax.text(x + cw_col, y, val, ha="right", va="center", fontsize=FS_BODY,
                    fontweight="bold" if hot else "normal",
                    color=LEAK if hot else ZERO)
            ax.plot([x, x + cw_col], [y - 2.4, y - 2.4], color=HAIR, lw=0.5)
            y -= D_RATE
        ax.text(x, Y_NOTE, note, ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    # ---- what the three arms buy you ------------------------------------------------------------
    ax.plot([cx0, cx0 + 3 * cw_col + 2 * cgap], [Y_SEP, Y_SEP], color=HAIR, lw=0.5)
    # the tokens here are IDENTICAL to the Contrast column of Table 3, on purpose: a reader must be
    # able to map the figure onto the table without translating a second naming scheme
    for i, (lhs, rhs) in enumerate(((" B \u2212 A ", "total gap"),
                                    (" C \u2212 A ", "patient route"),
                                    (" B \u2212 C ", "within-nodule route"))):
        x = cx0 + i * (cw_col + cgap)
        ax.text(x, Y_CONTRAST, lhs, ha="left", va="center", fontsize=FS_BODY,
                fontweight="bold", color=INK)
        ax.text(x + 9.5, Y_CONTRAST, rhs, ha="left", va="center", fontsize=FS_BODY, color=MUTED)

    # ---- legend, unframed, on its own band -------------------------------------------------------
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=TRAIN, edgecolor="none", label="training slice"),
        Rectangle((0, 0), 1, 1, facecolor=TEST, edgecolor="white", linewidth=0, hatch="///",
                  label="test slice"),
        Line2D([0], [0], color=LEAK, lw=1.8, label="unit split across training and test"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(cx0 / W, 0.0), ncol=3,
              frameon=False, fontsize=FS_TINY, handlelength=1.5, handleheight=0.9,
              columnspacing=2.0, labelcolor=MUTED)

    here = os.path.dirname(os.path.abspath(__file__))

    # The SAVED page must be exactly \textwidth, so \includegraphics[width=\textwidth] neither
    # enlarges nor shrinks it. bbox_inches="tight" crops to the ink, so the saved width is not the
    # figsize -- it came out 7.20 in from a 7.16 in figure, and against the real 6.9894 in text width
    # that printed the smallest type at 6.795 pt, under the 7 pt floor this figure's own rules set.
    # So: save, measure what tight cropping actually produced, correct the figsize by that delta,
    # and save again. Two passes, then assert. Solving it by raising the font sizes would not work --
    # a narrower canvas holds proportionally less type, which is the same squeeze from the other end.
    PAD = 0.02

    def _produced_width():
        """Width the saved page WILL have, in inches: the tight bbox plus the padding on each side.

        Asked of matplotlib rather than read back from the pdf, because this runs in the `nodules`
        env, which has no pdf reader -- and because get_tightbbox is what savefig itself uses, so it
        is the same number by construction rather than by agreement.
        """
        fig.canvas.draw()
        return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD

    produced = _produced_width()
    if abs(produced - PAGE_W) > 0.002:
        fig.set_size_inches(fig.get_figwidth() - (produced - PAGE_W), fig.get_figheight())
        produced = _produced_width()
    for ext, dpi in (("png", 400), ("pdf", None)):
        fig.savefig(os.path.join(here, f"{FIG_STEM}.{ext}"), dpi=dpi,
                    bbox_inches="tight", pad_inches=PAD)
    print(f"  page width {produced:.4f} in against \\textwidth {PAGE_W:.4f} in "
          f"-> placed scale {PAGE_W / produced:.5f}")
    assert abs(produced - PAGE_W) <= 0.01, (
        f"the saved page is {produced:.4f} in but \\textwidth is {PAGE_W:.4f} in; LaTeX would "
        f"rescale the figure by {PAGE_W / produced:.4f} and every font size with it")

    # the figure ships with its own data (figure gate: "dado exportado junto")
    with open(os.path.join(here, "schematic_values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "arm", "value", "n_folds", "source"])
        for k, v in cohort.items():
            w.writerow([k, "all", v, "", "outputs/metadata/dataset_accounting.json"])
        for k, (lp, ln, n) in meas.items():
            w.writerow(["L_pat (patient split)", k, f"{lp:.4f}", n, "outputs/splits/*.csv"])
            w.writerow(["L_nod (nodule split)", k, f"{ln:.4f}", n, "outputs/splits/*.csv"])
    print(f"wrote {FIG_STEM}.png / {FIG_STEM}.pdf / schematic_values.csv to", here)


if __name__ == "__main__":
    main()
