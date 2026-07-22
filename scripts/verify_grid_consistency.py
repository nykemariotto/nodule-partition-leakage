"""
GRID INTERNAL-CONSISTENCY GATE — the whole grid must be ONE experiment (D34 §internal consistency).

A grid runs for days over hundreds of runs, across crashes and resumes. Averaging a metric over
runs that were trained under different settings would silently mix experiments. This gate asserts
that every canonical run present on disk agrees on the three things that define "the same
experiment":

  * config_sha256    — the parsed-config hash stamped by src/train.py (D25). Catches any config
                       value change mid-grid.
  * max_epochs       — the ACTUAL epoch budget used. Stamped separately because run_grid.ps1 can
                       override config.train.max_epochs via --max-epochs, which the config hash
                       alone would NOT catch (the file is unchanged). This is the exact hole the
                       740-convergence ceiling amendment could open: half the grid at 30, half at
                       60. Here it is caught.
  * patience         — same reasoning for early-stopping patience.

Exit 0 = all present runs share one (config_sha256, max_epochs, patience). Exit 1 = the grid is
NOT a single experiment; do not average across it or report it until reconciled.

    python scripts/verify_grid_consistency.py
    python scripts/verify_grid_consistency.py --glob "lidc_binary_slice_*rep*"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--glob", default="lidc_binary_*", help="run-name glob (canonical runs only)")
    ap.add_argument("--require-stamp", action="store_true",
                    help="fail if any run lacks a config_sha256 (use once the grid is stamped)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hdir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "history")

    groups = defaultdict(list)
    unstamped = []
    for hp in sorted(glob.glob(os.path.join(hdir, args.glob + ".json"))):
        run = os.path.basename(hp)[:-5]
        if run.endswith("_final"):            # probe checkpoints are not the grid average
            continue
        h = json.load(open(hp))
        stamp = h.get("config_sha256")
        if stamp is None:
            unstamped.append(run)
            if not args.require_stamp:
                continue
        key = (stamp, h.get("max_epochs"), h.get("patience"))
        groups[key].append(run)

    print(f"canonical runs inspected: {sum(len(v) for v in groups.values())} "
          f"| unstamped (pre-D25): {len(unstamped)}")
    for (stamp, me, pat), runs in groups.items():
        s = stamp[:12] if stamp else "NONE"
        print(f"  group config {s} · max_epochs {me} · patience {pat}: {len(runs)} runs")

    if args.require_stamp and unstamped:
        print(f"\nFAIL: {len(unstamped)} run(s) have no config stamp: {unstamped[:5]}")
        return 1
    stamped_groups = {k: v for k, v in groups.items() if k[0] is not None}
    if len(stamped_groups) > 1:
        print("\nFAIL: the grid spans MORE THAN ONE (config, max_epochs, patience) — not one "
              "experiment. Reconcile before averaging/reporting:")
        for (stamp, me, pat), runs in stamped_groups.items():
            print(f"  {stamp[:12]} / me={me} / pat={pat}: {sorted(runs)[:3]} ...({len(runs)})")
        return 1
    print("\nOK — every stamped canonical run shares one (config, max_epochs, patience).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
