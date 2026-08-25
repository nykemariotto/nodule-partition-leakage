"""Shared rules for WHICH runs an analysis covers and WHAT its output is called.

Every analysis script needs the same two answers, and each one having its own copy is how the
project kept shipping the same defect. The history:

  * D62 -- the dataset axis was parameterised on the reading side of six scripts and left hardcoded
    on the writing side, so analysing the sensitivity cohort would have overwritten the principal
    cohort's file under the principal cohort's name.
  * 2026-08-01 -- the same shape on the architecture axis: `--archs` defaulted to the two CNNs in
    every script, so once a third architecture existed the default invocation silently produced a
    two-architecture result. The freshness gate then printed exactly that invocation as its
    remediation, so following the gate's own instruction made it go GREEN over an artifact missing
    an entire architecture.

So the rules live here, once:

  discover_archs  -- what is actually on disk, rather than what someone hardcoded.
  completeness    -- how many of the expected runs each architecture really has.
  name_for        -- the canonical filename is reserved for the canonical run; anything partial or
                     restricted gets a name that says so, and therefore cannot occupy the path the
                     manuscript embeds.

Globs are anchored on the sample unit throughout. A bare "{dataset}_*" also matches a LONGER dataset
name -- "lidc_binary" is a prefix of "lidc_binary_ge3" -- which silently mixed cohorts twice before.
"""
from __future__ import annotations

import glob
import os
import re

CANONICAL_DATASET = "lidc_binary"
_ARCH_RE = re.compile(r"_fold\d+_(.+)_none_seed\d+$")

# The design has FIVE legitimate (sample unit, arm) combinations, not the 2 x 3 cross product.
# There is no nodule-sample-unit / nodule-grouped cell: with one sample per nodule, grouping folds by
# nodule gives groups of size one, which is exactly arm B. Running it would duplicate the random arm
# under a third arm's name.
#
# Counting the cross product instead makes a COMPLETE architecture (75 runs) look like 75 of 90, so
# every architecture reads as partial, the canonical filename is refused, and -- worse -- a probe
# with a single run stops being distinguishable from a finished grid. That is what happened on
# 2026-08-03: the figure was written with swin_tiny's one run as a fourth row.
VALID_CELLS = frozenset({("slice", "patient"), ("slice", "random"), ("slice", "nodule"),
                         ("nodule", "patient"), ("nodule", "random")})


def discover_archs(probs_dir: str, dataset: str) -> list[str]:
    """Architectures with at least one canonical run on disk for `dataset`, sorted."""
    found = set()
    for su in ("slice", "nodule"):
        for f in glob.glob(os.path.join(probs_dir, f"{dataset}_{su}_*_seed42.npz")):
            base = os.path.basename(f)[:-4]
            if base.endswith("_final"):          # the memorisation probe, never a reported number
                continue
            m = _ARCH_RE.search(base)
            if m:
                found.add(m.group(1))
    return sorted(found)


def completeness(probs_dir: str, dataset: str, arch: str, sample_units, arms, reps, folds):
    """(have, want) canonical runs for one architecture over the requested cells."""
    want = have = 0
    for su in sample_units:
        for arm in arms:
            if (su, arm) not in VALID_CELLS:      # never run by design -- see VALID_CELLS
                continue
            for r in reps:
                for k in folds:
                    want += 1
                    f = os.path.join(probs_dir,
                                     f"{dataset}_{su}_{arm}_rep{r}_fold{k}_{arch}_none_seed42.npz")
                    if os.path.exists(f):
                        have += 1
    return have, want


def name_for(stem: str, dataset: str, archs, all_archs) -> str:
    """Output name. The bare `stem` is reserved for the principal cohort with the FULL architecture
    set; every other combination is suffixed so it cannot be mistaken for -- or overwrite -- the
    artifact the manuscript points at.

    Pass all_archs=None to force a suffix, which is what a caller does when some architecture is
    incomplete: a half-finished figure or table is not the canonical one no matter which
    architectures it nominally covers.
    """
    if all_archs is not None and dataset == CANONICAL_DATASET and sorted(archs) == sorted(all_archs):
        return stem
    parts = [stem]
    if dataset != CANONICAL_DATASET:
        parts.append(dataset)
    if all_archs is None or sorted(archs) != sorted(all_archs):
        parts.append("+".join(sorted(archs)))
    return "__".join(parts)


def resolve_archs(probs_dir: str, dataset: str, requested: str | None, log=print,
                  cells=None):
    """(archs_to_use, archs_in_the_experiment). `requested` None means: everything on disk.

    `cells` is (sample_units, arms, reps, folds). When given, an architecture counts as part of the
    EXPERIMENT only if it has every one of those runs. This matters: a timed probe leaves a single
    artifact on disk, and without the check a one-run architecture would be treated as a member of
    the full set -- which both corrupts the "is this the canonical set?" test and would put an
    almost-empty row into every figure. Probes are reported and then set aside, not silently kept.

    Announces what it found and, when a caller restricts the set, says plainly which architectures
    with runs on disk are being left out. Silence there is what made the earlier defect invisible.
    """
    on_disk = discover_archs(probs_dir, dataset)
    if not on_disk:
        raise SystemExit(f"no canonical runs found for dataset '{dataset}' in {probs_dir}")

    if cells is None:
        present = on_disk
    else:
        sus, arms, reps, folds = cells
        present, partial = [], []
        for a in on_disk:
            have, want = completeness(probs_dir, dataset, a, sus, arms, reps, folds)
            if have == want:
                present.append(a)
            else:
                partial.append((a, have, want))
        for a, have, want in partial:
            log(f"[artifacts] {a}: {have} of {want} runs -- treated as a PROBE, not part of the "
                f"experiment for {dataset}. Pass --archs explicitly to include it anyway.")
        if not present:
            raise SystemExit(f"no architecture has a complete run set for '{dataset}'")
    if requested:
        archs = [a.strip() for a in requested.split(",") if a.strip()]
        unknown = [a for a in archs if a not in on_disk]
        if unknown:
            raise SystemExit(f"requested architecture(s) with no runs for '{dataset}': "
                             f"{', '.join(unknown)}. On disk: {', '.join(on_disk)}")
        # An explicit --archs may include an incomplete one -- the probe line above says so -- but
        # then the result is not the canonical artifact, and returning all_archs=None makes
        # name_for suffix it. Validating against `present` instead would have made that
        # instruction impossible to follow.
        if any(a not in present for a in archs):
            log(f"[artifacts] partial architecture(s) included by explicit request -- the output "
                f"name will be suffixed so it cannot occupy the canonical path.")
            return archs, None
        missing = [a for a in present if a not in archs]
        if missing:
            log(f"[artifacts] NOTE: {', '.join(missing)} have runs on disk for {dataset} and are "
                f"EXCLUDED by an explicit --archs. The output name will say so.")
    else:
        archs = present
        log(f"[artifacts] architectures discovered for {dataset}: {', '.join(archs)}")
    return archs, present
