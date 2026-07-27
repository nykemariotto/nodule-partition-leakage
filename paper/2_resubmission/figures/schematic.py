"""
Method schematic for the controlled paired-design leakage study (IEEE Access, Access-2026-27906).

DETERMINISTIC matplotlib figure — NO randomness, NO external assets, NO network. Renders
paper/figures/schematic.png (300 dpi) and schematic.pdf. Run with the `nodules` env:
    C:\\ProgramData\\miniconda3\\envs\\nodules\\python.exe paper/figures/schematic.py

The ONLY numbers shown are real: cohort counts (740 patients / 1,695 nodules / 11,026 slices)
and the patient-leakage rates (L=0 for the patient-grouped arm, L~=0.99 for the random arm,
both properties of the split indices in outputs/splits/). No performance numbers are fabricated.
Grayscale-safe: train/test are distinguished by fill shade + hatch, and the two arms by solid vs
dashed borders; a single red accent marks the leaking (straddling) patient.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# --- palette -------------------------------------------------------------------------------------
INK      = "#1a1a1a"
BOX_FILL = "#f5f5f5"
BOX_EDGE = "#3f3f3f"
SHARE    = "#eef2f6"   # shared-pipeline fill (very light)
TRAIN    = "#dcdcdc"   # light gray = train
TEST     = "#767676"   # dark gray  = test
LEAK     = "#c1272d"   # single accent: the straddling patient in arm B
ARM_A    = "#22577a"   # patient arm (solid border)
ARM_B    = "#9c6a15"   # random arm (dashed border)


def box(ax, x, y, w, h, title, lines, edge=BOX_EDGE, lw=1.3, fill=BOX_FILL,
        fs_title=9.5, fs_body=8.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=fs_title,
            fontweight="bold", color=INK, zorder=3)
    if lines:
        ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top",
                fontsize=fs_body, color=INK, zorder=3, linespacing=1.4)


def arrow(ax, p0, p1, color=BOX_EDGE, lw=1.7, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1, shrinkA=1, shrinkB=1))


def fold_strip(ax, x, y, cw, ch, assignment, groups, leak_group=None):
    """A horizontal strip of slice-cells; assignment[i] in {'train','test'}; `groups` is a list of
    (label, i0, i1) patient spans drawn as brackets beneath. leak_group index -> red bracket."""
    for i, a in enumerate(assignment):
        fill = TRAIN if a == "train" else TEST
        hatch = "" if a == "train" else "////"
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.9, ch, facecolor=fill,
                               edgecolor="#2b2b2b", linewidth=0.6, hatch=hatch, zorder=2))
    yb = y - 0.14
    for gi, (label, i0, i1) in enumerate(groups):
        xL, xR = x + i0 * cw, x + (i1 - 1) * cw + cw * 0.9
        c = LEAK if gi == leak_group else "#555555"
        lw = 1.6 if gi == leak_group else 1.0
        ax.plot([xL, xL, xR, xR], [yb, yb - 0.10, yb - 0.10, yb], color=c, lw=lw, zorder=3)
        ax.text((xL + xR) / 2, yb - 0.30, label, ha="center", va="top", fontsize=6.8,
                color=c, fontweight=("bold" if gi == leak_group else "normal"), zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # ---- shared pipeline (left) -----------------------------------------------------------------
    yb, h, w = 3.55, 2.1, 3.2
    box(ax, 0.4, yb, w, h, "Cohort", ["740 patients", "1,695 nodules", "11,026 slices"], fill=SHARE)
    box(ax, 3.95, yb, w, h, "Preprocessing", ["50% consensus ROI", "256 x 256", "(both arms)"], fill=SHARE)
    box(ax, 7.5, yb, w, h, "Backbone", ["ImageNet-pretrained", "CNN / transformer"], fill=SHARE)
    arrow(ax, (0.4 + w, yb + h / 2), (3.95, yb + h / 2))
    arrow(ax, (3.95 + w, yb + h / 2), (7.5, yb + h / 2))

    # brace + label over the shared pipeline
    xL, xR, yb2 = 0.4, 7.5 + w, yb + h + 0.28
    ax.plot([xL, xL, xR, xR], [yb2, yb2 + 0.16, yb2 + 0.16, yb2], color="#666666", lw=1.1)
    ax.text((xL + xR) / 2, yb2 + 0.28, "Identical for both arms: cohort, labels, preprocessing, "
            "architecture, optimiser", ha="center", va="bottom", fontsize=8.2, style="italic",
            color="#444444")

    # ---- fork -----------------------------------------------------------------------------------
    fx = 7.5 + w              # right edge of Backbone (=10.7)
    fy = yb + h / 2
    ax_ax, ay = 12.4, 7.15    # arm A anchor (left-mid)
    bx, by = 12.4, 1.95       # arm B anchor (left-mid)
    arrow(ax, (fx, fy), (ax_ax, ay), color=ARM_A, lw=1.9)
    arrow(ax, (fx, fy), (bx, by), color=ARM_B, lw=1.9, ls=(0, (5, 2)))
    ax.text(11.55, 4.72, "Only the partition\nunit differs", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#999999", lw=0.8))

    # ---- Arm A: patient-grouped -----------------------------------------------------------------
    axx, ayy, aw, ah = 12.4, 5.6, 6.0, 3.1
    box(ax, axx, ayy, aw, ah, "Arm A  ·  Patient-grouped (GroupKFold)", [], edge=ARM_A, lw=1.8,
        fill="#ffffff")
    ax.text(axx + 0.30, ayy + ah - 0.62, "Each patient's slices lie entirely in one fold",
            ha="left", va="top", fontsize=7.6, color=INK)
    fold_strip(ax, axx + 0.35, ayy + 1.15, 0.40, 0.52,
               ["train"] * 6 + ["test"] * 3,
               [("P1", 0, 3), ("P2", 3, 6), ("P3", 6, 9)])
    ax.text(axx + 5.05, ayy + 1.45, r"$L = 0$", ha="center", va="center", fontsize=13,
            fontweight="bold", color=ARM_A)
    ax.text(axx + 5.05, ayy + 0.80, "no patient in\ntrain & test", ha="center", va="center",
            fontsize=6.8, color="#444444", linespacing=1.2)

    # ---- Arm B: random --------------------------------------------------------------------------
    bxx, byy, bw, bh = 12.4, 0.4, 6.0, 3.1
    box(ax, bxx, byy, bw, bh, "Arm B  ·  Random (KFold)", [], edge=ARM_B, lw=1.8, ls=(0, (5, 2)),
        fill="#ffffff")
    ax.text(bxx + 0.30, byy + bh - 0.62, "The same patient's slices straddle train & test",
            ha="left", va="top", fontsize=7.6, color=INK)
    fold_strip(ax, bxx + 0.35, byy + 1.15, 0.40, 0.52,
               ["train", "test", "train", "test", "train", "test", "test", "train", "test"],
               [("P1", 0, 3), ("P2", 3, 6), ("P3", 6, 9)], leak_group=1)
    ax.text(bxx + 5.05, byy + 1.45, r"$L \approx 0.99$", ha="center", va="center", fontsize=13,
            fontweight="bold", color=ARM_B)
    ax.text(bxx + 5.05, byy + 0.80, "patient leaks\nacross the split", ha="center", va="center",
            fontsize=6.8, color=LEAK, linespacing=1.2)

    # ---- output ---------------------------------------------------------------------------------
    ox, oy, ow, oh = 19.0, 3.35, 2.7, 2.55
    box(ax, ox, oy, ow, oh, "Leakage gap", [r"$\Delta_M =$", r"$M(\pi_B) - M(\pi_A)$",
        "per architecture", "& sample-unit"], edge="#222222", lw=1.5, fill="#f0efe9",
        fs_title=9.0, fs_body=8.0)
    arrow(ax, (axx + aw, ayy + 1.4), (ox, oy + oh - 0.5), color=ARM_A, lw=1.7)
    arrow(ax, (bxx + bw, byy + 1.4), (ox, oy + 0.5), color=ARM_B, lw=1.7, ls=(0, (5, 2)))

    # ---- legend ---------------------------------------------------------------------------------
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=TRAIN, edgecolor="#2b2b2b", label="train slice"),
        Rectangle((0, 0), 1, 1, facecolor=TEST, edgecolor="#2b2b2b", hatch="////", label="test slice"),
        Line2D([0], [0], color=ARM_A, lw=2, label="patient arm (A)"),
        Line2D([0], [0], color=ARM_B, lw=2, ls=(0, (5, 2)), label="random arm (B)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=4,
              frameon=False, fontsize=7.6, handlelength=1.6, columnspacing=1.4)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.02)
    here = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(here, "schematic.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(here, "schematic.pdf"), bbox_inches="tight")
    print("wrote schematic.png / schematic.pdf to", here)


if __name__ == "__main__":
    main()
