"""Shared rules for WHICH runs an analysis covers and WHAT its output is called.

Every analysis script needs the same two answers, and each one having its own copy is how the
project kept shipping the same defect. The history:

  * the dataset axis was parameterised on the reading side of six scripts and left hardcoded
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

import pandas as pd

CANONICAL_DATASET = "lidc_binary"
_ARCH_RE = re.compile(r"_fold\d+_(.+)_none_seed\d+$")

# The repository this file lives in. The analysis reads and writes the artifacts NEXT TO THE CODE,
# not under the machine path in config.yaml (project.root): the configuration is hashed into every
# run, so that path cannot be edited, and a clone anywhere else must recompute from its own files.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The experiment the paper reports. Declared, not inferred from what is on disk: an architecture of
# this set that is incomplete stops a canonical write, and any other architecture with runs on disk
# (the one-run Swin-T timing check) is reported as a probe and left out. Only a run set covering
# all of FULL_REPS x FULL_FOLDS writes under a canonical name (see name_for).
REPORTED_ARCHS = ("densenet121", "efficientnet_b0", "vit_small")
FULL_REPS = (0, 1, 2)
FULL_FOLDS = (0, 1, 2, 3, 4)


def outputs_dir(cfg) -> str:
    """outputs/ of this checkout (see REPO)."""
    return os.path.join(REPO, cfg["paths"]["outputs"])


def read_split(path: str) -> pd.DataFrame:
    """A committed split file, in the row order its runs were evaluated in.

    Training aligned each split with the preprocessed-image index by an inner merge, which keeps
    the split's own row order, and the coverage gate refused to train unless every row had an image.
    The archived predictions are therefore in the split file's order, and the analysis needs no
    preprocessing output: only the committed splits and probabilities. Callers check the alignment
    against the length and the labels stored with the predictions.
    """
    s = pd.read_csv(path)
    s["z_position"] = s["z_position"].round(2)
    return s


def check_aligned(split: pd.DataFrame, y_true, y_prob, tag: str) -> None:
    """Refuse predictions that do not line up with the split rows they are said to score."""
    if len(split) != len(y_prob):
        raise RuntimeError(f"ALIGNMENT {tag}: {len(split)} split rows vs {len(y_prob)} predictions")
    if (split["label"].to_numpy().astype(int) != y_true.astype(int)).any():
        raise RuntimeError(f"LABEL MISMATCH {tag}: the split's labels are not the ones stored with "
                           f"its predictions")


def arms_on_disk(probs_dir: str, dataset: str, sample_unit: str) -> list[str]:
    """Arms with at least one canonical run for this dataset and sample unit, in design order."""
    found = []
    for arm in ("patient", "random", "nodule"):
        if (sample_unit, arm) in VALID_CELLS and glob.glob(os.path.join(
                probs_dir, f"{dataset}_{sample_unit}_{arm}_rep*_fold*_*_seed42.npz")):
            found.append(arm)
    return found

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


def name_for(stem: str, dataset: str, archs, all_archs, reps=FULL_REPS, folds=FULL_FOLDS,
             arms=None, all_arms=None, extra=()) -> str:
    """Output name. The bare `stem` is reserved for the principal cohort with the FULL architecture
    set, every repeat and fold of the reported grid, and every arm the cell has; every other
    combination is suffixed so it cannot be mistaken for -- or overwrite -- the artifact the
    manuscript points at.

    Pass all_archs=None to force a suffix, which is what a caller does when some architecture is
    incomplete: a half-finished figure or table is not the canonical one no matter which
    architectures it nominally covers. `extra` adds suffixes a caller needs beyond these (e.g. a
    sample unit other than the one the canonical file holds).
    """
    parts = [stem]
    if dataset != CANONICAL_DATASET:
        parts.append(dataset)
    if all_archs is None or sorted(archs) != sorted(all_archs):
        parts.append("+".join(sorted(archs)))
    if sorted(reps) != list(FULL_REPS):
        parts.append("reps" + "-".join(str(r) for r in sorted(reps)))
    if sorted(folds) != list(FULL_FOLDS):
        parts.append("folds" + "-".join(str(k) for k in sorted(folds)))
    if arms is not None and all_arms is not None and sorted(arms) != sorted(all_arms):
        parts.append("+".join(sorted(arms)))
    parts += list(extra)
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
            if a in REPORTED_ARCHS and not requested:
                # A member of the reported experiment with a missing run is a broken grid, not a
                # probe: dropping it would publish a two-architecture result under the canonical
                # name.
                raise SystemExit(f"[artifacts] {a} belongs to the reported experiment but has "
                                 f"{have} of {want} runs for {dataset}: refusing to write a result "
                                 f"without it. Pass --archs to run on a subset; the output name "
                                 f"will say so.")
            log(f"[artifacts] {a}: {have} of {want} runs -- treated as a PROBE, not part of the "
                f"experiment for {dataset}. Pass --archs explicitly to include it anyway.")
        extra = [a for a in present if a not in REPORTED_ARCHS]
        if extra:
            log(f"[artifacts] {', '.join(extra)}: complete but not part of the reported experiment "
                f"-- left out unless requested with --archs.")
            present = [a for a in present if a in REPORTED_ARCHS]
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
