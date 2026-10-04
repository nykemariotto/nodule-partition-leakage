"""
Re-verify the partition rules on the COMMITTED split indices, and export the result.

src/splits.py asserts the rules at the moment it writes a split. This script checks the files as
they stand in outputs/splits/, so the claim can be audited from the repository without
regenerating anything, and it exports what it measured:

  every arm      no row (slice of a nodule) is in two of train / val / test in any fold, every fold
                 covers the same rows, and the five test folds of a repeat partition those rows,
                 so each row is tested exactly once per repeat;
  arm A patient  no patient identifier is in two of train / val / test in any fold;
  arm C nodule   no nodule identifier is in two of train / val / test in any fold, and patients DO
                 cross the train/test boundary (otherwise the arm would not test the patient route);
  arm B random   patients DO cross the train/test boundary.

It also exports the leakage rates read from the same files -- L(patient) and L(nodule), the
fraction of test rows whose patient / nodule also appears in the training fold -- and the training
rows per training patient, each averaged over the folds on disk.

    python scripts/verify_leakage.py

Writes outputs/metrics/leakage_assertion.txt (one line per fold) and leakage_assertion.json (per
fold, plus the averages). Exit 1 if any rule fails.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "outputs", "splits")
OUT = os.path.join(ROOT, "outputs", "metrics")

NAME = re.compile(r"^(?P<ds>lidc_binary(?:_ge3)?)_(?P<su>slice|nodule)_(?P<arm>patient|random|nodule)"
                  r"_rep(?P<rep>\d+)_fold(?P<fold>\d+)_test\.csv$")
ANY_PART = re.compile(r"^(lidc_binary(?:_ge3)?)_(slice|nodule)_(patient|random|nodule)_rep\d+_fold\d+"
                      r"_(train|val|test)\.csv$")
COLS = ["patient_id", "nodule_id", "slice_id"]


def _rows(df):
    return set(zip(df.nodule_id, df.slice_id))


def check_fold(tr, va, te, arm):
    """Overlap counts and rates for one fold, and the list of rules it breaks."""
    p = {k: set(d.patient_id) for k, d in (("tr", tr), ("va", va), ("te", te))}
    n = {k: set(d.nodule_id) for k, d in (("tr", tr), ("va", va), ("te", te))}
    r = {k: _rows(d) for k, d in (("tr", tr), ("va", va), ("te", te))}
    rec = {
        "rows_train": len(tr), "rows_val": len(va), "rows_test": len(te),
        "pat_tr_te": len(p["tr"] & p["te"]), "pat_tr_va": len(p["tr"] & p["va"]),
        "pat_va_te": len(p["va"] & p["te"]),
        "nod_tr_te": len(n["tr"] & n["te"]), "nod_tr_va": len(n["tr"] & n["va"]),
        "nod_va_te": len(n["va"] & n["te"]),
        "row_tr_te": len(r["tr"] & r["te"]), "row_tr_va": len(r["tr"] & r["va"]),
        "row_va_te": len(r["va"] & r["te"]),
        "L_patient": float(te.patient_id.isin(p["tr"]).mean()),
        "L_nodule": float(te.nodule_id.isin(n["tr"]).mean()),
        "train_rows_per_patient": len(tr) / len(p["tr"]),
    }
    broken = []
    if rec["row_tr_te"] or rec["row_tr_va"] or rec["row_va_te"]:
        broken.append("a row is in two partitions")
    if len(tr) != len(r["tr"]) or len(va) != len(r["va"]) or len(te) != len(r["te"]):
        broken.append("a row is repeated within one partition")
    if arm == "patient" and (rec["pat_tr_te"] or rec["pat_tr_va"] or rec["pat_va_te"]):
        broken.append("patient arm: a patient is in two partitions")
    if arm == "nodule":
        if rec["nod_tr_te"] or rec["nod_tr_va"] or rec["nod_va_te"]:
            broken.append("nodule arm: a nodule is in two partitions")
        if not rec["pat_tr_te"]:
            broken.append("nodule arm: no patient crosses train/test, so the patient route is off")
    if arm == "random" and not rec["pat_tr_te"]:
        broken.append("random arm: no patient crosses train/test")
    return rec, broken, r["tr"] | r["va"] | r["te"], r["te"]


def main():
    tests = sorted(f for f in os.listdir(SPLITS) if NAME.match(f))
    if not tests:
        sys.exit(f"FATAL: no split indices in {SPLITS}")
    folds, failures = [], []
    # A split file this script does not recognise would otherwise be skipped in silence, and a
    # rule broken inside it would never be seen. Every CSV must be a known partition of a known
    # fold, and every fold must have all three partitions.
    for f in sorted(os.listdir(SPLITS)):
        if f.endswith(".csv") and not ANY_PART.match(f):
            failures.append(f"{f}: not a recognised split file, so it was not checked")
    for f in tests:
        for part in ("train", "val"):
            if not os.path.exists(os.path.join(SPLITS, f.replace("_test.csv", f"_{part}.csv"))):
                sys.exit(f"FATAL: {f} has no {part} partition")
    by_repeat = {}
    for f in tests:
        m = NAME.match(f).groupdict()
        stem = os.path.join(SPLITS, f[: -len("_test.csv")])
        tr, va, te = (pd.read_csv(f"{stem}_{s}.csv", usecols=COLS) for s in ("train", "val", "test"))
        rec, broken, universe, test_rows = check_fold(tr, va, te, m["arm"])
        rec.update(dataset=m["ds"], sample_unit=m["su"], arm=m["arm"], rep=int(m["rep"]),
                   fold=int(m["fold"]), ok=not broken, broken=broken)
        folds.append(rec)
        failures += [f"{f[: -len('_test.csv')]}: {b}" for b in broken]
        key = (m["ds"], m["su"], m["arm"], int(m["rep"]))
        by_repeat.setdefault(key, []).append((universe, test_rows))

    # every fold of a repeat covers the same rows, and the test folds partition them
    for key, items in sorted(by_repeat.items()):
        tag = "{}_{}_{}_rep{}".format(*key)
        universe = items[0][0]
        if any(u != universe for u, _ in items):
            failures.append(f"{tag}: the folds of this repeat do not cover the same rows")
        tested = [t for _, t in items]
        if sum(len(t) for t in tested) != len(universe) or set().union(*tested) != universe:
            failures.append(f"{tag}: the test folds do not partition the rows "
                            f"({sum(len(t) for t in tested)} tested, {len(universe)} rows)")

    summary = {}
    for rec in folds:
        k = f"{rec['dataset']}|{rec['sample_unit']}|{rec['arm']}"
        summary.setdefault(k, []).append(rec)
    summary = {k: {"n_folds": len(v),
                   "L_patient": sum(r["L_patient"] for r in v) / len(v),
                   "L_nodule": sum(r["L_nodule"] for r in v) / len(v),
                   "train_rows_per_patient": sum(r["train_rows_per_patient"] for r in v) / len(v),
                   "patients_shared_train_test_max": max(r["pat_tr_te"] for r in v)}
               for k, v in summary.items()}

    os.makedirs(OUT, exist_ok=True)
    hdr = (f"{'dataset':16}{'unit':8}{'arm':9}{'rep':>4}{'fold':>5}{'pat tr/te':>10}{'pat tr/va':>10}"
           f"{'pat va/te':>10}{'nod tr/te':>10}{'row tr/te':>10}{'L(pat)':>8}{'L(nod)':>8}"
           f"{'rows/pat':>9}  verdict")
    lines = [hdr]
    for r in folds:
        lines.append(f"{r['dataset']:16}{r['sample_unit']:8}{r['arm']:9}{r['rep']:>4}{r['fold']:>5}"
                     f"{r['pat_tr_te']:>10}{r['pat_tr_va']:>10}{r['pat_va_te']:>10}{r['nod_tr_te']:>10}"
                     f"{r['row_tr_te']:>10}{r['L_patient']:>8.3f}{r['L_nodule']:>8.3f}"
                     f"{r['train_rows_per_patient']:>9.2f}  {'ok' if r['ok'] else 'FAIL'}")
    lines.append("")
    lines.append("averages over the folds on disk")
    for k, s in summary.items():
        lines.append(f"  {k:36} n={s['n_folds']:<3} L(patient) {s['L_patient']:.4f}  "
                     f"L(nodule) {s['L_nodule']:.4f}  training rows per patient "
                     f"{s['train_rows_per_patient']:.2f}")
    lines.append("")
    lines.append("RESULT: " + ("every rule holds in every fold" if not failures
                               else f"{len(failures)} rule(s) broken:"))
    lines += [f"  {x}" for x in failures]
    open(os.path.join(OUT, "leakage_assertion.txt"), "w", encoding="utf-8", newline="\n").write(
        "\n".join(lines) + "\n")
    json.dump({"_about": "rules re-verified on the committed split indices; L(patient) and "
                         "L(nodule) are the fraction of test rows whose patient / nodule also "
                         "appears in the training fold",
               "ok": not failures, "failures": failures, "summary": summary, "folds": folds},
              open(os.path.join(OUT, "leakage_assertion.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(lines[-(len(summary) + 4 + len(failures)):]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
