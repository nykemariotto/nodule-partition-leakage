"""
PROTOCOL-FIDELITY AUDIT — is the saved CANONICAL checkpoint the one the pre-registered
protocol would have selected?

The canonical protocol (DECISIONS D22) is: 30 epochs + early stopping patience 10 +
checkpoint = best_val_loss. But `run_grid.ps1` deliberately passes `-Patience 30`, so the
training loop never breaks — that is REQUIRED by the dual-checkpoint design (D22 ii: one
training pass, two readings; the `_final` memorization PROBE cannot exist if the loop stops
early).

Side effect: with the loop running to epoch 30, `best_val_loss` is the argmin over ALL 30
epochs. Early stopping at patience 10 would have selected the argmin over only the epochs it
survived to see. These coincide in most runs — but not necessarily all. Where they differ, the
saved "canonical" model got MORE training than the protocol allows. If that happens in the
`random` arm it inflates the reported leakage gap, i.e. it biases in favour of the hypothesis.

This script replays each saved history and reports every mismatch. It reads only
`outputs/history/*.json` — no GPU, no retraining.

    python scripts/check_checkpoint_fidelity.py
    python scripts/check_checkpoint_fidelity.py --patience 10 --probe

Exit code 1 if any run's canonical checkpoint differs from the early-stopped selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config


def best_state_epoch(history, upto=None):
    """Replay train.py's best-state rule (`vl < best - 1e-4`) — NOT plain argmin."""
    best, ep = float("inf"), None
    for h in history:
        if upto is not None and h["epoch"] > upto:
            break
        if h["val_loss"] < best - 1e-4:
            best, ep = h["val_loss"], h["epoch"]
    return ep


def early_stop_epoch(history, patience):
    """Epoch at which patience would have fired (None = never)."""
    best, bad = float("inf"), 0
    for h in history:
        if h["val_loss"] < best - 1e-4:
            best, bad = h["val_loss"], 0
        else:
            bad += 1
            if bad >= patience:
                return h["epoch"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Audit canonical-checkpoint protocol fidelity.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--patience", type=int, default=0, help="0 = config train.early_stopping_patience")
    ap.add_argument("--archs", default="densenet121,efficientnet_b0")
    ap.add_argument("--arms", default="patient,random")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enh", default="none")
    args = ap.parse_args()

    cfg = load_config(args.config)
    patience = args.patience or cfg["train"]["early_stopping_patience"]
    hdir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "history")

    print(f"protocol: max_epochs {cfg['train']['max_epochs']} · early_stopping_patience {patience} "
          f"· checkpoint {cfg['train']['checkpoint_selection']}")
    print(f"{'arch':17}{'arm':9}{'fold':6}{'ES_pick':9}{'stop@':7}{'saved':7}  verdict")

    mismatches = []
    for arch in args.archs.split(","):
        for arm in args.arms.split(","):
            for fold in args.folds.split(","):
                run = f"lidc_binary_slice_{arm}_rep0_fold{fold}_{arch}_{args.enh}_seed{args.seed}"
                path = os.path.join(hdir, f"{run}.json")
                if not os.path.exists(path):
                    print(f"{arch:17}{arm:9}{fold:<6}{'-':9}{'-':7}{'-':7}  MISSING")
                    continue
                hist = json.load(open(path))["history"]
                stop = early_stop_epoch(hist, patience)
                es_pick = best_state_epoch(hist, upto=stop)
                saved = best_state_epoch(hist)
                ok = es_pick == saved
                if not ok:
                    d = {h["epoch"]: h for h in hist}
                    mismatches.append(dict(
                        arch=arch, arm=arm, fold=int(fold), es_pick=es_pick, saved=saved,
                        val_loss_gain=d[es_pick]["val_loss"] - d[saved]["val_loss"],
                        val_auc_gain=d[saved]["val_auc"] - d[es_pick]["val_auc"]))
                print(f"{arch:17}{arm:9}{fold:<6}{str(es_pick):9}{str(stop):7}{str(saved):7}  "
                      f"{'match' if ok else '*** DIFFERS ***'}")

    print()
    if not mismatches:
        print("OK — every canonical checkpoint equals the early-stopped selection.")
        return 0
    print(f"{len(mismatches)} run(s) where the saved canonical checkpoint got MORE training than "
          f"patience-{patience} early stopping allows:")
    for m in mismatches:
        print(f"  {m['arch']} {m['arm']} fold{m['fold']}: ES ep{m['es_pick']} -> saved ep{m['saved']}"
              f" · val_loss {m['val_loss_gain']:+.5f} · val_auc {m['val_auc_gain']:+.4f} (val, not test)")
    arms = {m["arm"] for m in mismatches}
    if arms == {"random"}:
        print("  WARNING: all mismatches are in the RANDOM arm — the bias favours the hypothesis.")
    print("  FIX: retrain these runs with --patience {p} (preserve the existing _final PROBE "
          "artifacts first — train.py rewrites both checkpoints).".format(p=patience))
    return 1


if __name__ == "__main__":
    sys.exit(main())
