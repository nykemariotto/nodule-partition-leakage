"""Title-and-abstract screening decisions for the 2025-2026 search (criteria fixed beforehand).

A record is excluded here only when the reason is clear from its title and abstract; anything else
goes to full-text reading. Indices are the 1-based row numbers of records_deduplicated.csv.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {
    2: "segmentation only",
    5: "review",
    7: "segmentation only",
    14: "data not from LIDC-IDRI (BIR Lung CT dataset)",
    28: "segmentation only",
    31: "segmentation only",
    47: "detection only",
    53: "detection only",
    55: "not peer reviewed (repository record)",
    56: "not peer reviewed (repository record; duplicate version of the record above)",
    70: "detection only",
    77: "detection only",
    80: "segmentation only",
    99: "detection only",
}
NOTES = {
    3: "same method name (CCDC-HNN) as record 12: possible duplicate publication, checked at full text",
    12: "same method name (CCDC-HNN) as record 3: possible duplicate publication, checked at full text",
    26: "issue year 2025 as indexed; first published online in 2023",
    41: "already cited in the article",
    69: "already cited in the article",
    60: "vision-language model: eligible only if a model is trained on the data",
}

rows = list(csv.DictReader(open(os.path.join(HERE, "records_deduplicated.csv"), encoding="utf-8")))
with open(os.path.join(HERE, "screening_title_abstract.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "doi", "title", "year", "venue", "decision", "reason", "note"])
    for i, r in enumerate(rows, 1):
        dec = "exclude" if i in EXCLUDE else "full text"
        w.writerow([i, r["doi"], r["title"], r["year"], r["venue"], dec, EXCLUDE.get(i, ""), NOTES.get(i, "")])
print(f"{len(rows)} records: {len(EXCLUDE)} excluded at title/abstract, {len(rows) - len(EXCLUDE)} to full text")
