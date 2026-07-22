"""
CONTENT-VALIDATED SKIP CHECK (not existence-based).

A run counts as "done" only if ALL of its artifacts exist AND are internally consistent:
  * model .pt loads and holds finite weights
  * probs .npz loads, y_true/y_prob aligned, no NaN, y_prob in [0,1], y_true binary
  * probs length == the test-split length it claims to come from  (catches STALE runs
    produced before a data fix, and truncated writes)
  * history .json loads with test metrics

Exit 0 = valid, safe to SKIP.  Exit 1 = missing/invalid, MUST retrain.

    python scripts/run_is_valid.py lidc_binary_slice_random_rep0_fold2_densenet121_none_seed42

This exists because a zombie job wrote a stale model (probs 642 vs split 657) after a data
fix, and an existence-only skip would have folded it into the canonical average (2026-07-18).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

O = r"E:\NODULES\outputs"
_CFG = r"E:\NODULES\config.yaml"


def _current_config_hash():
    """Hash of the config in effect NOW (D25). None if unavailable (never blocks on that)."""
    try:
        from src.metadata import load_config
        from src.config_contract import config_hash
        return config_hash(load_config(_CFG))
    except Exception:
        return None


def is_valid(run: str, verbose: bool = False) -> bool:
    why = []
    mp = os.path.join(O, "models", f"{run}.pt")
    pp = os.path.join(O, "probs", f"{run}.npz")
    hp = os.path.join(O, "history", f"{run}.json")

    for p, name in ((mp, "model"), (pp, "probs"), (hp, "history")):
        if not os.path.exists(p):
            why.append(f"{name} missing")
    if why:
        if verbose: print(f"[invalid] {run}: " + "; ".join(why))
        return False

    try:
        sd = torch.load(mp, map_location="cpu")
        if not isinstance(sd, dict) or not sd:
            why.append("model empty")
        else:
            t = next(iter(sd.values()))
            if hasattr(t, "numel") and not torch.isfinite(t.float()).all():
                why.append("model has non-finite weights")
    except Exception as e:
        why.append(f"model unreadable ({type(e).__name__})")

    n_prob = None
    try:
        d = np.load(pp)
        yt, yp = d["y_true"], d["y_prob"]
        n_prob = len(yp)
        if len(yt) != len(yp) or n_prob == 0:
            why.append("probs length mismatch/empty")
        if np.isnan(yp).any() or np.isnan(yt).any():
            why.append("NaN in probs")
        if yp.size and (yp.min() < 0 or yp.max() > 1):
            why.append("y_prob outside [0,1]")
        if yt.size and not set(np.unique(yt)).issubset({0, 1}):
            why.append("y_true not binary")
    except Exception as e:
        why.append(f"probs unreadable ({type(e).__name__})")

    try:
        h = json.load(open(hp))
        if not h.get("history") or "auc" not in h.get("test", {}):
            why.append("history incomplete")
        # D25 CONFIG-STALENESS: a run trained under a different config must not be folded into
        # the grid average. If the run carries a stamp and it differs from the config in effect
        # now, it is stale. Runs with no stamp (pre-2026-07-20) skip this check (backward compat).
        stamp = h.get("config_sha256")
        cur = _current_config_hash()
        if stamp and cur and stamp != cur:
            why.append(f"STALE CONFIG: run config {stamp[:12]} != current {cur[:12]}")
    except Exception as e:
        why.append(f"history unreadable ({type(e).__name__})")

    # STALENESS: probs must match the test split this run claims to come from
    base = run.replace("_final", "")
    tag = "_".join(base.split("_")[:6])            # lidc_binary_slice_{arm}_rep0_fold{k}
    sp = os.path.join(O, "splits", f"{tag}_test.csv")
    if n_prob is not None and os.path.exists(sp):
        n_split = len(pd.read_csv(sp))
        if n_prob != n_split:
            why.append(f"STALE: probs n={n_prob} != test split n={n_split}")

    if why:
        if verbose: print(f"[invalid] {run}: " + "; ".join(why))
        return False
    if verbose: print(f"[valid] {run}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_is_valid.py <run_name>"); sys.exit(2)
    sys.exit(0 if is_valid(sys.argv[1], verbose="-v" in sys.argv) else 1)
