"""
Fail if any analysis artifact is OLDER than the run outputs it was computed from.

Why this exists (D65): `outputs/results/per_model_metrics.json` was written at 00:34 on 2026-07-24,
five runs of the D47/D49 patience remediation finished between 02:09 and 04:22 the same night, and
the manuscript was then written from the stale file. Nothing failed, nothing warned -- the numbers
were simply wrong by up to 0.0025, and the random-arm McNemar counts had b and c the wrong way
round. A timestamp comparison would have caught it in one second.

So: every artifact derived from `outputs/probs/**` or `outputs/history/**` must be newer than the
newest input file it depends on. Run it before quoting any number in the manuscript.

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

# The artifact list is PER COHORT, and deliberately so. Two ways of getting this wrong were found on
# 2026-08-01, both silent:
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
# artifact -> (input glob relative to outputs/, command that regenerates it). A producer of None
# marks an artifact that is knowingly superseded: reported, never fatal.
_PROBS = "probs/{ds}_slice_*_seed42.npz;probs/{ds}_nodule_*_seed42.npz"
_HIST = "history/{ds}_slice_*_seed42.json;history/{ds}_nodule_*_seed42.json"
_FIG = "python -m src.figures --reps 0,1,2 --sample-units slice,nodule"

ARTIFACTS = {
    "lidc_binary": {
        "results/per_model_metrics.json":
            (_PROBS, "python -m src.evaluate --sample-unit slice --reps 0,1,2 "
                     "--arms patient,random,nodule"),
        "results/confusion_matrices.json": (_PROBS, _FIG),
        # the manuscript embeds the VECTOR pdf; the png is kept only for looking at
        "figures/confusion_matrices.pdf": (_PROBS, _FIG),
        "figures/curves_slice-nodule.pdf": (_HIST, _FIG),
        "figures/confusion_matrices.png": (_PROBS, _FIG),
        "figures/curves_slice-nodule.png": (_HIST, _FIG),
        "metrics/audit_controls.json":
            ("probs/{ds}_slice_patient_*.npz;probs/{ds}_slice_random_*.npz",
             "python scripts/audit_controls.py --reps 0,1,2"),
        "_analysis/decomposition.json":
            ("probs/{ds}_slice_*_seed42.npz", "python scripts/decompose_routes.py --reps 0,1,2"),
        # superseded on purpose
        "results/stats_nodule.json": (_PROBS, None),
        "_analysis/audit_controls.json": (_PROBS, None),
        "_analysis/audit_controls_AFTER.json": (_PROBS, None),
        "_analysis/decomposition_final.json": (_PROBS, None),
    },
    # The sensitivity cohort has two arms and no figures of its own, so it has two artifacts.
    "lidc_binary_ge3": {
        "results/per_model_metrics__lidc_binary_ge3.json":
            (_PROBS, "python -m src.evaluate --sample-unit slice --reps 0,1,2 "
                     "--dataset lidc_binary_ge3"),
        "metrics/audit_controls__lidc_binary_ge3.json":
            (_PROBS, "python scripts/audit_controls.py --dataset lidc_binary_ge3 --reps 0,1,2"),
        # written before src/artifacts.py fixed the naming; kept as history, never read
        "results/per_model_metrics_lidc_binary_ge3.json": (_PROBS, None),
        "metrics/audit_controls_lidc_binary_ge3.json": (_PROBS, None),
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


# The manuscript compiles from its OWN copy of each figure (graphicspath points at
# paper/2_resubmission/figures, and must, or the Overleaf upload breaks). Nothing synchronises the
# two, so regenerating a figure and forgetting the copy leaves the paper compiling an older picture
# than the one the numbers were taken from -- the same staleness this gate exists to catch, one
# directory over.
FIGURE_COPIES = ("confusion_matrices.pdf", "curves_slice-nodule.pdf",
                 "confusion_matrices.png", "curves_slice-nodule.png")
PAPER_FIGS = os.path.join("paper", "2_resubmission", "figures")


def _check_figure_copies(outputs):
    out = []
    for f in FIGURE_COPIES:
        src, dst = os.path.join(outputs, "figures", f), os.path.join(ROOT, PAPER_FIGS, f)
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

    stale, missing, superseded = [], [], []
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
            (superseded if producer is None else stale).append((art, p, newest, src, producer))

    def stamp(t):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))

    for art, p, newest, src, producer in stale:
        print(f"STALE  {art}")
        print(f"         written {stamp(os.path.getmtime(p))}, but input is "
              f"{stamp(newest)}  ({os.path.basename(src)})")
        print(f"         regenerate: {producer}")
    for art, p, newest, src, _ in superseded:
        print(f"note   {art} is older than its inputs but is superseded by a newer artifact "
              f"(written {stamp(os.path.getmtime(p))})")
    for art in missing:
        print(f"absent {art} (not yet produced)")

    copies = _check_figure_copies(outputs) if args.dataset == "lidc_binary" else []
    for f, dst_t, src_t in copies:
        where = "MISSING" if dst_t is None else f"written {stamp(dst_t)}"
        print(f"STALE  {PAPER_FIGS}/{f}")
        print(f"         {where}, but outputs/figures/{f} is {stamp(src_t)}")
        print(f"         regenerate: copy outputs/figures/{f} into {PAPER_FIGS}/")

    n = len(table) - len(missing)
    if stale or copies:
        print(f"\nFAIL: {len(stale)} of {n} artifacts are older than their inputs"
              + (f", and {len(copies)} manuscript figure copies are behind outputs/figures" if copies else "")
              + ". Any manuscript number taken from them is not what the code on disk now produces.")
        raise SystemExit(1)
    print(f"\nOK: all {n - len(superseded)} live artifacts are at least as new as their inputs"
          + (", and the manuscript's figure copies match outputs/figures." if args.dataset == "lidc_binary"
             else "."))


if __name__ == "__main__":
    main()
