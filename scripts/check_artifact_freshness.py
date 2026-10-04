"""
Fail if any analysis artifact is OLDER than the run outputs it was computed from.

Why this exists: a results table was once written from an analysis file computed before five
runs were retrained. Nothing failed, nothing warned -- the numbers were simply wrong by up to
0.0025. A timestamp comparison would have caught it in one second.

So: every artifact derived from `outputs/probs/**`, `outputs/history/**`, `outputs/splits/**` or
`outputs/metadata/**` must be newer than the newest input file it depends on. Run it before quoting
any number in the manuscript.

What it cannot see: an artifact computed by an older version of its producer from unchanged inputs
looks fresh. Rerun the producer after changing its code.

    python scripts/check_artifact_freshness.py
    python scripts/check_artifact_freshness.py --dataset lidc_binary_ge3

Exit 0 = every artifact is at least as new as its inputs. Exit 1 = at least one is stale; re-run the
producer named in the output before trusting anything downstream.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The artifact list is PER COHORT, and deliberately so. Two ways of getting this wrong, both silent:
#
#   1. Input globs must be anchored on the sample unit. A bare "{ds}_*" matches a LONGER dataset
#      name -- "lidc_binary_*" sweeps in "lidc_binary_ge3_*" -- so finishing the sensitivity grid
#      made every principal-cohort artifact report as stale. A gate that cries wolf gets ignored,
#      which is worse than no gate.
#   2. A single shared list meant that running with --dataset lidc_binary_ge3 checked the PRINCIPAL
#      cohort's confusion matrices, curves and route decomposition against the sensitivity cohort's
#      runs. Those artifacts do not exist for that cohort at all -- it has no nodule-grouped arm and
#      no figures of its own -- so the comparison was meaningless as well as failing.
#
# artifact -> (input globs relative to outputs/, command that regenerates it). The inputs include
# the split files as well as the runs: every analysis aligns the archived predictions to them.
_SL = "probs/{ds}_slice_*_seed42.npz;splits/{ds}_slice_*.csv"
_NOD = "probs/{ds}_nodule_*_seed42.npz;splits/{ds}_nodule_*.csv"
_SLICE_AB = ("probs/{ds}_slice_patient_*.npz;probs/{ds}_slice_random_*.npz;"
             "splits/{ds}_slice_patient_*.csv;splits/{ds}_slice_random_*.csv")
_PROBS = _SL + ";" + _NOD
_HIST = "history/{ds}_slice_*_seed42.json;history/{ds}_nodule_*_seed42.json"
_FIG = "python -m src.figures --sample-units slice,nodule"

ARTIFACTS = {
    "lidc_binary": {
        "results/per_model_metrics.json": (_SL, "python -m src.evaluate"),
        "results/confusion_matrices.json": (_SL, _FIG),
        # the manuscript embeds the VECTOR pdf; the png is kept only for looking at
        "figures/confusion_matrices.pdf": (_SL, _FIG),
        "figures/curves_slice-nodule.pdf": (_HIST, _FIG),
        "figures/confusion_matrices.png": (_SL, _FIG),
        "figures/curves_slice-nodule.png": (_HIST, _FIG),
        "metrics/audit_controls.json": (_SLICE_AB, "python scripts/audit_controls.py"),
        "_analysis/decomposition.json": (_SL, "python scripts/decompose_routes.py"),
        # the paired test the paper reports, and the nodule level with one prediction per nodule
        "_analysis/fold_tests.json": (_SL, "python scripts/fold_tests.py"),
        "results/nodule_oof.json": (_SL, "python scripts/nodule_oof.py"),
        # the gap with the nodule as the sample unit (one representative slice per nodule)
        "_analysis/nodule_unit_gap.json": (_NOD, "python scripts/nodule_unit_gap.py"),
        # the Discussion's standardised gap and the stratification / balancer projections
        "metrics/standardised_gap.json": (_SLICE_AB, "python scripts/standardised_gap.py"),
        "metrics/stratification_projection.json": (_SLICE_AB, "python scripts/project_stratification.py"),
        "metrics/balancer_projection_740.json": (_SLICE_AB, "python scripts/project_balancer.py"),
        # read from the committed indices and metadata, not from runs
        "metrics/leakage_assertion.json": ("splits/*.csv", "python scripts/verify_leakage.py"),
        "metrics/leakage_assertion.txt": ("splits/*.csv", "python scripts/verify_leakage.py"),
        "literature/search_2026-10-02/survey_counts.json":
            ("literature/search_2026-10-02/fulltext_extraction.csv;literature/search_2026-10-02/search_log.json",
             "python scripts/literature_survey_counts.py"),
        "_analysis/cohort_summary.json":
            ("metadata/*.csv;splits/{ds}_slice_patient_rep0_fold0_*.csv", "python scripts/cohort_summary.py"),
        # Fig. 1 and its data, drawn from the indices and the cohort accounting (outside outputs/)
        "../figures/fig1_design.pdf":
            ("splits/{ds}_slice_*.csv;metadata/dataset_accounting.json", "python figures/schematic.py"),
        "../figures/schematic_values.csv":
            ("splits/{ds}_slice_*.csv;metadata/dataset_accounting.json", "python figures/schematic.py"),
    },
    # The sensitivity cohort has two arms and no figures of its own, so it has two artifacts.
    "lidc_binary_ge3": {
        "results/per_model_metrics__lidc_binary_ge3.json":
            (_SL, "python -m src.evaluate --dataset lidc_binary_ge3"),
        "metrics/audit_controls__lidc_binary_ge3.json":
            (_SLICE_AB, "python scripts/audit_controls.py --dataset lidc_binary_ge3"),
    },
}


def _newest(outputs, patterns, ds):
    newest, src = 0.0, None
    for pat in patterns.split(";"):
        for f in glob.glob(os.path.join(outputs, pat.format(ds=ds))):
            t = os.path.getmtime(f)
            if t > newest:
                newest, src = t, f
    return newest, src


# The manuscript compiles from its OWN copy of each figure, because the build directory has to hold
# every file it embeds. Nothing synchronises the two, so regenerating a figure and forgetting the
# copy leaves the manuscript compiling an older picture than the one the numbers were taken from --
# the same staleness this gate exists to catch, one directory over. That build directory is not part
# of this repository; when it is absent, as in a fresh clone, the check has nothing to compare and
# is skipped.
FIGURE_COPIES = ("outputs/figures/confusion_matrices.pdf", "outputs/figures/curves_slice-nodule.pdf",
                 "outputs/figures/confusion_matrices.png", "outputs/figures/curves_slice-nodule.png",
                 "figures/fig1_design.pdf", "figures/fig1_design.png")
MANUSCRIPT_FIGS = os.environ.get("MANUSCRIPT_FIGS")   # set this to the build directory to enable the check


def _checked_figs():
    """Whether the figure-copy comparison actually ran.

    A summary that reports a comparison it skipped is the same defect as a stale figure,
    one line further down.
    """
    return bool(MANUSCRIPT_FIGS) and os.path.isdir(os.path.join(ROOT, MANUSCRIPT_FIGS))


def _check_figure_copies(outputs):
    out = []
    if not MANUSCRIPT_FIGS or not os.path.isdir(os.path.join(ROOT, MANUSCRIPT_FIGS)):
        # Say so rather than return quietly: a check that skips in silence reads as a check that
        # passed, and this one is the difference between the figure in the manuscript and the
        # figure the numbers came from.
        print("note   figure-copy check skipped: set MANUSCRIPT_FIGS to the manuscript build "
              "directory to enable it")
        return out
    for rel in FIGURE_COPIES:
        f = os.path.basename(rel)
        src, dst = os.path.join(ROOT, rel), os.path.join(ROOT, MANUSCRIPT_FIGS, f)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            out.append((f, None, os.path.getmtime(src)))
        elif os.path.getmtime(dst) < os.path.getmtime(src):
            out.append((f, os.path.getmtime(dst), os.path.getmtime(src)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lidc_binary")
    args = ap.parse_args()
    outputs = os.path.join(ROOT, "outputs")

    stale, missing = [], []
    table = ARTIFACTS.get(args.dataset)
    if table is None:
        raise SystemExit(f"no artifact list registered for dataset '{args.dataset}'. Add one "
                         f"rather than letting it fall through to another cohort's artifacts.")
    for art, (patterns, producer) in table.items():
        p = os.path.join(outputs, art)
        if not os.path.exists(p):
            missing.append(art)
            continue
        newest, src = _newest(outputs, patterns, args.dataset)
        if newest == 0.0:
            continue                                   # no inputs for this dataset yet
        if os.path.getmtime(p) < newest:
            stale.append((art, p, newest, src, producer))

    def stamp(t):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))

    for art, p, newest, src, producer in stale:
        print(f"STALE  {art}")
        print(f"         written {stamp(os.path.getmtime(p))}, but input is "
              f"{stamp(newest)}  ({os.path.basename(src)})")
        print(f"         regenerate: {producer}")
    for art in missing:
        print(f"ABSENT {art}  -- regenerate: {table[art][1]}")

    copies = _check_figure_copies(outputs) if args.dataset == "lidc_binary" else []
    for f, dst_t, src_t in copies:
        where = "MISSING" if dst_t is None else f"written {stamp(dst_t)}"
        print(f"STALE  {MANUSCRIPT_FIGS}/{f}")
        print(f"         {where}, but its source {f} is {stamp(src_t)}")
        print(f"         regenerate: copy the source of {f} into {MANUSCRIPT_FIGS}/")

    n = len(table)
    if stale or missing or copies:
        # A registered artifact that is absent fails too: in a clone it means the result the
        # paper cites is not there, and reporting success would hide exactly that.
        print(f"\nFAIL: {len(stale)} of {n} artifacts are older than their inputs, {len(missing)} "
              f"absent"
              + (f", and {len(copies)} manuscript figure copies are behind outputs/figures" if copies else "")
              + ". Any manuscript number taken from them is not what the code on disk now produces.")
        raise SystemExit(1)
    print(f"\nOK: all {n} artifacts are present and at least as new as their inputs"
          + (", and the manuscript's figure copies match outputs/figures."
             if args.dataset == "lidc_binary" and _checked_figs() else "."))


if __name__ == "__main__":
    main()
