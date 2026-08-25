"""
Data-completeness verifier (read-only). Currently covers LIDC-IDRI.

For every scan in the pylidc DB it checks that (1) the DICOM series resolves on disk,
(2) the on-disk slice count equals the count in TCIA `metadata.csv` (no truncation),
and (3) every annotated slice is physically present. XML-on-disk is intentionally
NOT required — pylidc supplies annotations from its bundled database.

    python scripts/verify_download.py --config config.yaml

Exits 0 if complete, 1 otherwise. Writes outputs/metadata/verify_lidc.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pydicom
import yaml
import pylidc as pl


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_lidc(cfg) -> dict:
    dicom_root = cfg["paths"]["lidc_dicom_root"]
    meta_csv = cfg["paths"]["lidc_metadata_csv"]
    workers = int(cfg["metadata"]["io_workers"])

    expected = {}
    with open(meta_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Modality") == "CT":
                try:
                    expected[row["Series UID"]] = int(row["Number of Images"])
                except Exception:
                    pass

    # disk map: series_uid -> (folder, dcm_count, has_xml)
    series_dirs = []
    for pat in os.scandir(dicom_root):
        if pat.is_dir():
            for study in os.scandir(pat.path):
                if study.is_dir():
                    for series in os.scandir(study.path):
                        if series.is_dir():
                            series_dirs.append(series.path)

    def read_series(sd):
        dcms = glob.glob(os.path.join(sd, "*.dcm"))
        if not dcms:
            return None
        has_xml = len(glob.glob(os.path.join(sd, "*.xml"))) > 0
        try:
            d = pydicom.dcmread(dcms[0], stop_before_pixels=True, force=True,
                                specific_tags=["SeriesInstanceUID"])
            suid = str(d.get("SeriesInstanceUID", ""))
        except Exception:
            suid = ""
        return (suid, sd, len(dcms), has_xml) if suid else None

    disk = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_series, series_dirs):
            if r:
                disk[r[0]] = r[1:]

    scans = pl.query(pl.Scan).all()
    scan_info = []
    for s in scans:
        zs = set()
        n = 0
        for a in s.annotations:
            n += 1
            for c in a.contours:
                zs.add(round(float(c.image_z_position), 2))
        scan_info.append({"pid": s.patient_id, "suid": s.series_instance_uid,
                          "nann": n, "zs": zs})

    missing_dicom, count_mismatch, ok = [], [], 0
    for info in scan_info:
        suid = info["suid"]
        if suid not in disk:
            missing_dicom.append(info["pid"])
            continue
        folder, cnt, has_xml = disk[suid]
        exp = expected.get(suid)
        if exp is not None and cnt != exp:
            count_mismatch.append((info["pid"], f"disk={cnt} meta={exp}"))
        else:
            ok += 1

    def check_slices(info):
        if info["nann"] == 0 or info["suid"] not in disk:
            return (info["pid"], 0)
        folder = disk[info["suid"]][0]
        zset = set()
        for f in glob.glob(os.path.join(folder, "*.dcm")):
            try:
                d = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                                    specific_tags=["ImagePositionPatient"])
                ipp = d.get("ImagePositionPatient", None)
                if ipp is not None:
                    zset.add(round(float(ipp[2]), 2))
            except Exception:
                pass
        miss = sum(1 for z in info["zs"] if not any(abs(z - zd) < 0.05 for zd in zset))
        return (info["pid"], miss)

    annotated = [i for i in scan_info if i["nann"] > 0]
    slice_problems = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, miss in ex.map(check_slices, annotated):
            if miss > 0:
                slice_problems.append((pid, miss))

    complete = not missing_dicom and not count_mismatch and not slice_problems
    return {
        "ct_series_metadata": len(expected),
        "pylidc_scans": len(scan_info),
        "dicom_resolved": len(scan_info) - len(missing_dicom),
        "count_ok": ok,
        "missing_dicom": missing_dicom,
        "count_mismatch": count_mismatch,
        "annotated_scans": len(annotated),
        "slice_problems": slice_problems,
        "verdict_complete": complete,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = verify_lidc(cfg)

    out_dir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "metadata")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verify_lidc.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("LIDC verification:")
    print(f"  scans resolved on disk : {res['dicom_resolved']}/{res['pylidc_scans']}")
    print(f"  slice-count OK         : {res['count_ok']}")
    print(f"  missing DICOM          : {len(res['missing_dicom'])}")
    print(f"  count mismatch         : {len(res['count_mismatch'])}")
    print(f"  annotated-slice gaps   : {len(res['slice_problems'])}")
    print(f"  COMPLETE               : {res['verdict_complete']}")
    sys.exit(0 if res["verdict_complete"] else 1)


if __name__ == "__main__":
    main()
