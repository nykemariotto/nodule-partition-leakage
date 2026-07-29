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

# artifact -> (input glob relative to outputs/, command that regenerates it). Artifacts that are
# knowingly superseded are listed with `None` as the producer and only warned about.
ARTIFACTS = {
    "results/per_model_metrics.json":
        ("probs/{ds}_*_seed42.npz", "python -m src.evaluate --sample-unit slice --reps 0,1,2 "
                                    "--arms patient,random,nodule"),
    "results/confusion_matrices.json":
        ("probs/{ds}_*_seed42.npz", "python -m src.figures --reps 0,1,2 --sample-units slice,nodule"),
    "figures/confusion_matrices.png":
        ("probs/{ds}_*_seed42.npz", "python -m src.figures --reps 0,1,2 --sample-units slice,nodule"),
    "figures/curves_slice-nodule.png":
        ("history/{ds}_*_seed42.json", "python -m src.figures --reps 0,1,2 "
                                       "--sample-units slice,nodule"),
    "_analysis/audit_controls_AFTER.json":
        ("probs/{ds}_slice_patient_*.npz;probs/{ds}_slice_random_*.npz",
         "python scripts/audit_controls.py"),
    "_analysis/decomposition_final.json":
        ("probs/{ds}_slice_*_seed42.npz", "python scripts/decompose_routes.py"),
    # superseded on purpose -- reported, never fatal
    "results/stats_nodule.json": ("probs/{ds}_*_seed42.npz", None),
    "_analysis/audit_controls.json": ("probs/{ds}_*_seed42.npz", None),
}


def _newest(outputs, patterns, ds):
    newest, src = 0.0, None
    for pat in patterns.split(";"):
        for f in glob.glob(os.path.join(outputs, pat.format(ds=ds))):
            t = os.path.getmtime(f)
            if t > newest:
                newest, src = t, f
    return newest, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lidc_binary")
    args = ap.parse_args()
    outputs = os.path.join(ROOT, "outputs")

    stale, missing, superseded = [], [], []
    for art, (patterns, producer) in ARTIFACTS.items():
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

    n = len(ARTIFACTS) - len(missing)
    if stale:
        print(f"\nFAIL: {len(stale)} of {n} artifacts are older than their inputs. Any manuscript "
              f"number taken from them is not what the code on disk now produces.")
        raise SystemExit(1)
    print(f"\nOK: all {n - len(superseded)} live artifacts are at least as new as their inputs.")


if __name__ == "__main__":
    main()
