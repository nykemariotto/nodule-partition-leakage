"""
The counts the article's Related Work reports from the structured 2025-2026 literature search,
computed from the released tables in outputs/literature/<search>/:

  search_log.json            records found per source and after deduplication
  fulltext_extraction.csv    one row per deduplicated record: the title/abstract screening decision,
                             whether the full text was read, eligibility, and for each included study
                             the partition unit, the sentence it was classified on, and the accuracy
                             and AUC it reports

The flow (identified, screened, sought, not retrieved, read, excluded, included), the tally of
partition units among the included studies (also without the studies flagged as borderline), and
how many included studies report an accuracy above 95% or an AUC above 0.98, by partition unit.

    python scripts/literature_survey_counts.py

Writes outputs/literature/<search>/survey_counts.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="search_2026-10-02")
    args = ap.parse_args()
    d = os.path.join(ROOT, "outputs", "literature", args.search)
    rows = list(csv.DictReader(open(os.path.join(d, "fulltext_extraction.csv"), encoding="utf-8")))
    log = json.load(open(os.path.join(d, "search_log.json"), encoding="utf-8"))

    inc = [r for r in rows if r["eligible"] == "yes"]
    borderline = sorted(int(r["index"]) for r in inc if r["flag"].startswith("eligibility borderline"))
    hi_acc = [r for r in inc if r["accuracy_reported"] and float(r["accuracy_reported"]) > 95]
    hi_auc = [r for r in inc if r["auc_reported"] and float(r["auc_reported"]) > 0.98]
    out = dict(
        flow=dict(identified=log["counts"], deduplicated=len(rows),
                  excluded_title_abstract=sum(r["screening"] == "exclude" for r in rows),
                  sought_full_text=sum(r["screening"] == "full text" for r in rows),
                  not_retrieved=dict(Counter(r["full_text"] for r in rows
                                             if r["full_text"].startswith("not retrieved"))),
                  read=sum(r["full_text"] == "read" for r in rows),
                  excluded_full_text=sum(r["eligible"] == "no" for r in rows),
                  included=len(inc)),
        tally=dict(Counter(r["partition_unit"] for r in inc)),
        borderline=borderline,
        tally_without_borderline=dict(Counter(r["partition_unit"] for r in inc
                                              if int(r["index"]) not in borderline)),
        reported_accuracy=sum(bool(r["accuracy_reported"]) for r in inc),
        reported_auc=sum(bool(r["auc_reported"]) for r in inc),
        accuracy_above_95=len(hi_acc),
        accuracy_above_95_by_unit=dict(Counter(r["partition_unit"] for r in hi_acc)),
        auc_above_098=len(hi_auc),
        auc_above_098_by_unit=dict(Counter(r["partition_unit"] for r in hi_auc)),
        auc_above_099=sum(bool(r["auc_reported"]) and float(r["auc_reported"]) > 0.99 for r in inc),
    )
    f = out["flow"]
    if f["deduplicated"] != f["excluded_title_abstract"] + f["sought_full_text"]:
        raise SystemExit("FATAL: screening decisions do not account for every record")
    if f["sought_full_text"] != sum(f["not_retrieved"].values()) + f["read"]:
        raise SystemExit("FATAL: full-text status does not account for every record sought")
    if f["read"] != f["excluded_full_text"] + f["included"]:
        raise SystemExit("FATAL: eligibility does not account for every record read")
    p = os.path.join(d, "survey_counts.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    t = out["tally"]
    below = t.get("nodule", 0) + t.get("image, slice or patch", 0)
    unstated = t.get("not stated", 0) + t.get("no split described", 0)
    print(f"{f['deduplicated']} records; {f['read']} read; {f['included']} included")
    print("partition unit:", ", ".join(f"{k} {v}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))
    rest = {k: v for k, v in t.items()
            if k not in ("nodule", "image, slice or patch", "not stated", "no split described")}
    print(f"below the patient or unstated: {below + unstated} of {f['included']} "
          f"({below} below the patient, {unstated} unstated or no split); the other "
          f"{sum(rest.values())}: " + ", ".join(f"{k} {v}" for k, v in sorted(rest.items(), key=lambda kv: -kv[1])))
    print(f"accuracy > 95%: {out['accuracy_above_95']} ({out['accuracy_above_95_by_unit'].get('patient', 0)} "
          f"split by patient); AUC > 0.98: {out['auc_above_098']} "
          f"({out['auc_above_098_by_unit'].get('patient', 0)} split by patient)")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
