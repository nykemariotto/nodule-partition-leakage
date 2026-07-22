"""
STATIC AUDIT of the config-code contract (DECISIONS D27 / D23).

`src/config_contract.assert_contract` runs at every training start, but it can only check that
each config key is DECLARED. It cannot check whether a key declared `consumed` is still read by
anything — delete the line that reads it and the declaration silently becomes a lie. That is the
same failure mode as the rejected submission, one level up.

This script closes that hole by checking the declarations against the source:

  1. every config key is declared, and every declaration still has a config key (same as runtime);
  2. every key declared `consumed` actually appears as a quoted string in src/ or scripts/;
  3. no key declared `hardcoded` / `documentation` / `deferred` is in fact being read
     (it should be re-declared `consumed` — a stale status is drift too);
  4. reports what is `deferred`, because those must never be described as done in the manuscript.

Run it as a gate before any batch of training:

    python scripts/verify_config_contract.py        # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metadata import load_config
from src.config_contract import CONTRACT, assert_contract, config_keys


def source_text(root, dirs=("src", "scripts")):
    out = {}
    for d in dirs:
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".py"):
                out[os.path.join(d, f)] = open(os.path.join(p, f), encoding="utf-8").read()
    return out


def is_read(leaf, sources, skip=("config_contract.py",)):
    """Does any source file index a key by this leaf name? Returns the files that do."""
    pat = re.compile(r'["\']' + re.escape(leaf) + r'["\']')
    return [f for f, t in sources.items()
            if not f.endswith(skip) and pat.search(t)]


def is_read_pathaware(key, sources, skip=("config_contract.py",)):
    """Like is_read, but for a dotted key requires BOTH the leaf AND its parent segment to
    appear (quoted) in the same file. Kills false positives on generic leaf names: e.g. the
    config key `ensemble.method` must not be flagged merely because `src/evaluate.py` accesses
    a McNemar result field `mcn[arm]['method']` — that file never mentions `ensemble`. Config
    access always names the parent (`cfg["ensemble"]["method"]` or via an `ensemble` alias)."""
    segs = key.split(".")
    leaf = segs[-1]
    parent = segs[-2] if len(segs) >= 2 else None
    leaf_pat = re.compile(r'["\']' + re.escape(leaf) + r'["\']')
    files = [f for f, t in sources.items() if not f.endswith(skip) and leaf_pat.search(t)]
    if parent is None:
        return files
    parent_pat = re.compile(r'["\']' + re.escape(parent) + r'["\']')
    return [f for f in files if parent_pat.search(sources[f])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(args.config)

    # --- 1. declaration completeness (same check the runtime does) ---
    try:
        summary = assert_contract(cfg)
    except RuntimeError as e:
        print(f"FAIL (declaration): {e}")
        return 1
    print(f"[1/4] declarations complete: {summary}")

    sources = source_text(root)
    keys = config_keys(cfg)
    problems = []

    # --- 2. 'consumed' must really be consumed ---
    ghosts = []
    for k in keys:
        status, _ = CONTRACT[k]
        if status != "consumed":
            continue
        if not is_read(k.split(".")[-1], sources):
            ghosts.append(k)
    if ghosts:
        problems.append(f"{len(ghosts)} key(s) declared 'consumed' that NO source reads: {ghosts}")
    print(f"[2/4] consumed keys verified against source: "
          f"{'OK' if not ghosts else f'{len(ghosts)} GHOST(S)'}")

    # --- 3. stale non-consumed statuses ---
    stale = []
    for k in keys:
        status, _ = CONTRACT[k]
        if status == "consumed":
            continue
        hits = is_read_pathaware(k, sources)   # parent+leaf: avoids generic-word collisions
        if hits:
            stale.append((k, status, hits[:3]))
    if stale:
        problems.append(f"{len(stale)} key(s) whose status is stale (source reads them): "
                        + "; ".join(f"{k} declared '{s}' but read in {h}" for k, s, h in stale))
    print(f"[3/4] non-consumed statuses: {'OK' if not stale else f'{len(stale)} STALE'}")

    # --- 4. deferred inventory (reporting, not a failure) ---
    deferred = sorted(k for k in keys if CONTRACT[k][0] == "deferred")
    print(f"[4/4] deferred (NOT implemented — never describe as done in the manuscript): "
          f"{len(deferred)}")
    for k in deferred:
        print(f"        {k}  — {CONTRACT[k][1]}")

    if problems:
        print("\nFAIL — config-code contract drifted:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("\nOK — config.yaml and the code agree on every key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
