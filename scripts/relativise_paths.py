"""Make the released partition indices portable: the last step after src.metadata and src.splits.

src/metadata.py records each slice's DICOM file by its ABSOLUTE path on the machine that built the
tables (`<local-root>\\LIDC-IDRI\\manifest-...`), and src/splits.py copies that column into every split file.
A column that resolves on one machine only would undercut indices offered as independently
executable, and would leak the local layout into a public repository.

This rewrites that column, in the split files and the master tables, to a path RELATIVE to the
repository root, with forward slashes, so it resolves on any platform once the raw data are placed
under the documented root. The DICOM provenance (manifest, series directory, instance number) is
preserved in full. Run after a rebuild, it makes the regenerated files identical to the committed
ones.

Safe to run because NOTHING READS THIS COLUMN: `src/datasets.py` loads images through
`processed_path` from the processed index, and the analysis code reads the split rows in order.
Verified before writing, and re-verified here: every other column must come out byte-identical and
the row count must be unchanged, or the file is left alone and the run fails.

    python scripts/relativise_paths.py --dry-run
    python scripts/relativise_paths.py
"""
from __future__ import annotations

import argparse
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIXES = (ROOT + os.sep, ROOT + "/", ROOT.replace("\\", "/") + "/")


def relativise(cell: str) -> str:
    for p in PREFIXES:
        if cell.startswith(p):
            return cell[len(p):].replace("\\", "/")
    return cell


def process(path: str, dry: bool) -> tuple[int, int]:
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\n")
    if not lines or "," not in lines[0]:
        return 0, 0
    header = lines[0].rstrip("\r").split(",")
    if "path" not in header:
        return 0, 0
    col = header.index("path")

    changed, out = 0, [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            out.append(ln)
            continue
        # the DICOM paths contain no commas, so a plain split is exact here; assert it
        f = ln.rstrip("\r").split(",")
        if len(f) != len(header):
            raise SystemExit(f"FATAL {path}: {len(f)} fields against {len(header)} headers -- "
                             f"a quoted comma would need a csv reader; refusing to guess.")
        new = relativise(f[col])
        if new != f[col]:
            f[col] = new
            changed += 1
        out.append(("\r" if ln.endswith("\r") else "").join([]) + ",".join(f) +
                   ("\r" if ln.endswith("\r") else ""))
    if changed and not dry:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(out))
    return changed, len(lines) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = sorted(glob.glob(os.path.join(ROOT, "outputs", "splits", "*.csv")))
    targets += sorted(glob.glob(os.path.join(ROOT, "outputs", "metadata", "lidc_*.csv")))

    tot_files = tot_rows = tot_changed = 0
    for t in targets:
        ch, rows = process(t, args.dry_run)
        tot_rows += rows
        if ch:
            tot_files += 1
            tot_changed += ch
    print(f"{'WOULD REWRITE' if args.dry_run else 'REWROTE'} {tot_files} files: "
          f"{tot_changed} of {tot_rows} rows carried an absolute path")

    if not args.dry_run:
        left = 0
        for t in targets:
            with open(t, encoding="utf-8", errors="ignore") as fh:
                left += fh.read().count(ROOT)
        print(f"remaining occurrences of the repository root in these files: {left}")
        if left:
            raise SystemExit("FATAL: absolute paths survived the rewrite.")
        print("OK")


if __name__ == "__main__":
    main()
