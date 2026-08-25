"""
GRID INTERNAL-CONSISTENCY GATE — the whole grid must be ONE experiment (D34 §internal consistency),
AND that one experiment must be the one the config DECLARES (D48).

A grid runs for days over hundreds of runs, across crashes and resumes. Averaging a metric over runs
trained under different settings would silently mix experiments. But mutual agreement is not enough:
three times now (`--max-epochs 8` in the sanity stage, hardcoded `channels_last`, `--patience 30` in
the grid) every run agreed with every other run and all of them disagreed with `config.yaml`. A gate
that only compares runs TO EACH OTHER passes all three. So this gate checks both directions:

  * config_sha256   — the parsed-config hash stamped by src/train.py (D25); one value across the grid.
  * max_epochs      — stamped value must EQUAL config.train.max_epochs (not merely be uniform).
                      CLI overrides do not change the config file, so the hash alone cannot see them.
  * checkpoint      — CHECKPOINT EQUIVALENCE, not stamp equality (D47). A run is valid iff the
    equivalence     checkpoint it actually selected is the one the pre-registered protocol
                      (config.train.early_stopping_patience) would have selected. This is recomputed
                      here from each run's stored val-loss trajectory, so it does not trust the
                      stamped `patience`/`best_epoch` at all. A run trained with a larger patience is
                      still valid IF its selected checkpoint lies inside the pre-registered window —
                      which is exactly how 111 of the 120 runs of the first grid qualified.

Exit 0 = the grid is one experiment and it is the declared one. Exit 1 = do not average or report.

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

TOL = 1e-4


def canonical_epoch(history, patience, tol=TOL):
    """(selected_epoch, stop_epoch) under the pre-registered protocol.

    selected_epoch = argmin(val_loss) among epochs up to the point where `patience` consecutive
    non-improvements would have stopped training; stop_epoch = that stopping point (None if the
    run would never have stopped inside its budget). Recomputed from the raw trajectory so the
    check is independent of whatever `patience`/`best_epoch` the run stamped.
    """
    sel_val, sel_ep, bad, stop_ep = float("inf"), None, 0, None
    for h in history:
        if stop_ep is not None:
            break
        vl = h["val_loss"]
        if vl < sel_val - tol:
            sel_val, sel_ep, bad = vl, h["epoch"], 0
        else:
            bad += 1
            if bad >= patience:
                stop_ep = h["epoch"]
    return sel_ep, stop_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--glob", default="lidc_binary_*", help="run-name glob (canonical runs only)")
    ap.add_argument("--require-stamp", action="store_true",
                    help="fail if any run lacks a config_sha256 (use once the grid is stamped)")
    ap.add_argument("--json", default=None,
                    help="write the offending runs (parsed into su/arm/rep/fold/arch) to this path. "
                         "Lets a repair script derive its work list from THIS gate -- the source of "
                         "truth -- instead of a hardcoded list that silently goes stale (D49).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hdir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "history")
    cfg_me = int(cfg["train"]["max_epochs"])
    cfg_pat = int(cfg["train"]["early_stopping_patience"])

    groups = defaultdict(list)
    unstamped, bad_epochs, bad_ckpt = [], [], []
    n = 0
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
        n += 1
        groups[stamp].append(run)

        # --- against the CONFIG, not merely against each other (D48) ---
        if h.get("max_epochs") != cfg_me:
            bad_epochs.append((run, h.get("max_epochs")))

        # --- checkpoint equivalence, recomputed from the trajectory (D47) ---
        hist = h.get("history") or []
        if hist:
            sel_ep, stop_ep = canonical_epoch(hist, cfg_pat)
            if sel_ep is not None and h.get("best_epoch") != sel_ep:
                bad_ckpt.append((run, h.get("best_epoch"), sel_ep, stop_ep))

    print(f"canonical runs inspected: {n} | unstamped (pre-D25): {len(unstamped)}")
    for stamp, runs in groups.items():
        print(f"  group config {stamp[:12] if stamp else 'NONE'}: {len(runs)} runs")
    print(f"config declares: max_epochs {cfg_me} · early_stopping_patience {cfg_pat}")

    failed = False
    if args.require_stamp and unstamped:
        print(f"\nFAIL: {len(unstamped)} run(s) have no config stamp: {unstamped[:5]}")
        failed = True
    stamped = {k: v for k, v in groups.items() if k is not None}
    if len(stamped) > 1:
        print("\nFAIL: the grid spans MORE THAN ONE config_sha256 — not one experiment:")
        for stamp, runs in stamped.items():
            print(f"  {stamp[:12]}: {sorted(runs)[:3]} ...({len(runs)})")
        failed = True
    if bad_epochs:
        print(f"\nFAIL: {len(bad_epochs)} run(s) stamped a max_epochs that DISAGREES with "
              f"config.train.max_epochs={cfg_me} (a CLI override the config hash cannot see):")
        for run, me in bad_epochs[:8]:
            print(f"  {run}: stamped {me}")
        failed = True
    if bad_ckpt:
        print(f"\nFAIL: {len(bad_ckpt)} run(s) selected a checkpoint the pre-registered protocol "
              f"(patience {cfg_pat}) would NOT have selected — retrain them (D47):")
        for run, be, sel, stop in bad_ckpt[:12]:
            print(f"  {run}: selected epoch {be}, protocol would select {sel} "
                  f"(window closes at {stop})")
        failed = True

    if args.json:
        import re
        # NOTE: the dataset prefix may itself contain underscores (lidc_binary_ge3), so it is matched
        # non-greedily up to the sample-unit token rather than assumed to be two words.
        pat = re.compile(r"^(?P<dataset>.+?)_(?P<su>slice|nodule)_(?P<arm>patient|random|nodule)"
                         r"_rep(?P<rep>\d+)_fold(?P<fold>\d+)_(?P<arch>.+?)_none_seed(?P<seed>\d+)$")
        items = []
        for run, be, sel, stop in bad_ckpt:
            m = pat.match(run)
            it = {"run": run, "selected_epoch": be, "protocol_epoch": sel, "window_closes": stop}
            if m:
                it.update({k: (int(v) if k in ("rep", "fold", "seed") else v)
                           for k, v in m.groupdict().items()})
            items.append(it)
        with open(args.json, "w") as f:
            json.dump({"bad_checkpoint": items,
                       "bad_max_epochs": [{"run": r, "stamped": me} for r, me in bad_epochs],
                       "unstamped": unstamped, "inspected": n,
                       "config": {"max_epochs": cfg_me, "early_stopping_patience": cfg_pat}}, f, indent=2)
        print(f"wrote offender list -> {args.json}")

    if failed:
        return 1
    print(f"\nOK — one config; max_epochs matches the config; and all {n} canonical checkpoints are "
          f"the ones the pre-registered patience-{cfg_pat} protocol would select.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
