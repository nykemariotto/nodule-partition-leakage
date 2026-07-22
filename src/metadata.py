"""
STAGE 1 — Master metadata table for the LIDC-IDRI study (SPEC §6.1, v2).

Builds, from the pylidc annotation database + on-disk DICOM:
  * a per-sample master table  (patient_id, nodule_id, slice_id, label, path, ...)
  * a per-nodule table         (consensus label, #annotators, diameter, flags, ...)
  * a dataset-accounting report answering Reviewer 3's nodule-vs-annotation counting
    question explicitly (both counts), plus the exclusion funnel and the >4-annotator
    over-merge flag report (Decision 3).

The master table is a SUPERSET (every labelled nodule, min_annotators>=1). Two flags let
downstream stages select cohorts without rebuilding (SPEC v2):
  * `cohort_main`             = n_annotators >= metadata.min_annotators_principal (=3).
  * `is_representative_slice` = the nodule's largest-cross-section slice (sample_unit=nodule).

Unit of analysis (R3 / I4):
  * ANNOTATION = one radiologist's reading of a >=3 mm nodule (one malignancy score).
  * NODULE     = a physical lesion = pylidc cluster of annotations. Labelling/grouping
                 unit is the NODULE; BOTH counts are reported.

Label (config `label`): per-nodule consensus = median of the malignancy scores.
    median > 3 -> malignant (1);  median < 3 -> benign (0);  median == 3 -> excluded.
<3 mm markings, non-nodules and malignancy=0 are not labelled (excluded by construction).

Over-merge guard (Decision 3): nodules with > over_merge_flag_threshold (=4) annotators
are FLAGGED (`flag_over_merge`), never split; count + label breakdown go in the accounting.

No split or training logic here. Deterministic; no randomness.
Run:  python -m src.metadata --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import glob
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import pydicom
import yaml
import pylidc as pl


# ----------------------------------------------------------------------------- helpers
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _shoelace_area(coords) -> float:
    """Absolute polygon area (pixel^2) from an Nx2 array of (i, j) contour points."""
    if coords is None or len(coords) < 3:
        return 0.0
    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _consensus_label(scores, cfg_label):
    """(label_name, encoded) from malignancy scores via the median rule.
    label_name in {'malignant','benign','ambiguous','no_score'}; encoded in {1,0,None}."""
    if not scores:
        return "no_score", None
    med = median(scores)
    if med > 3:
        return "malignant", cfg_label["encoding"]["malignant"]
    if med < 3:
        return "benign", cfg_label["encoding"]["benign"]
    return "ambiguous", None  # median == 3 -> excluded


def build_series_to_folder(dicom_root: str, workers: int) -> dict:
    """SeriesInstanceUID -> (folder, dcm_count) by reading one header/folder.
    Robust to TCIA file renaming; avoids pylidc's per-scan directory walk."""
    series_dirs = []
    for pat in os.scandir(dicom_root):
        if not pat.is_dir():
            continue
        for study in os.scandir(pat.path):
            if not study.is_dir():
                continue
            for series in os.scandir(study.path):
                if series.is_dir():
                    series_dirs.append(series.path)

    def read_one(sd):
        dcms = glob.glob(os.path.join(sd, "*.dcm"))
        if not dcms:
            return None
        try:
            d = pydicom.dcmread(dcms[0], stop_before_pixels=True, force=True,
                                specific_tags=["SeriesInstanceUID"])
            suid = str(d.get("SeriesInstanceUID", ""))
        except Exception:
            return None
        return (suid, sd, len(dcms)) if suid else None

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(read_one, series_dirs):
            if r:
                out[r[0]] = (r[1], r[2])
    return out


def _natkey(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def _read_hdr(f):
    """(z_rounded, sop_uid, abspath) or None — header only (ImagePositionPatient + SOP)."""
    try:
        d = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                            specific_tags=["ImagePositionPatient", "SOPInstanceUID"])
        ipp = d.get("ImagePositionPatient", None)
        if ipp is None:
            return None
        return (round(float(ipp[2]), 2), str(d.get("SOPInstanceUID", "")), str(Path(f).resolve()))
    except Exception:
        return None


def build_zmap(folder: str, needed_zs=None) -> dict:
    """z(2dp) -> (SOPInstanceUID, abspath). If needed_zs is given, read ONLY the files that a
    linear z<->index model maps to those z's (+ endpoints) — ~5x fewer header reads — with a
    full-scan fallback if the model is inconsistent. Otherwise read every slice."""
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")), key=_natkey)
    if not files:
        return {}

    def full_scan():
        zmap = {}
        for f in files:
            r = _read_hdr(f)
            if r:
                zmap[r[0]] = (r[1], r[2])
        return zmap

    if not needed_zs or len(files) < 3:
        return full_scan()
    a, b = _read_hdr(files[0]), _read_hdr(files[-1])
    if not a or not b or a[0] == b[0]:
        return full_scan()
    z0, span, n = a[0], (b[0] - a[0]), len(files)
    zmap = {a[0]: (a[1], a[2]), b[0]: (b[1], b[2])}
    for z in needed_zs:
        idx = int(round((z - z0) / span * (n - 1)))
        if idx < 0 or idx >= n:
            return full_scan()
        r = _read_hdr(files[idx])
        if not r or abs(r[0] - z) > 0.05:      # model broke (non-uniform/unsorted) -> be safe
            return full_scan()
        zmap[r[0]] = (r[1], r[2])
    return zmap


def _resolve_slice(z, zmap):
    """z -> (sop_uid, path); exact match on rounded z, else nearest within 0.05 mm."""
    key = round(float(z), 2)
    if key in zmap:
        return zmap[key]
    if not zmap:
        return None, None
    best = min(zmap.keys(), key=lambda k: abs(k - key))
    return zmap[best] if abs(best - key) <= 0.05 else (None, None)


# ------------------------------------------------------------------- Phase A: pylidc read
def extract_scan_records(cfg) -> list:
    """Single-threaded pylidc access (sqlite-safe). Returns plain dicts per scan."""
    cl = cfg["metadata"]["clustering"]
    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()
    records = []
    for s in scans:
        clusters = s.cluster_annotations(metric=cl["metric"], tol=cl["tol"],
                                         factor=cl["factor"], min_tol=cl["min_tol"],
                                         verbose=False)
        nodules = []
        for c_ in clusters:
            scores = [int(a.malignancy) for a in c_ if 1 <= int(a.malignancy) <= 5]
            diameters = [float(a.diameter) for a in c_]
            slices = defaultdict(lambda: {"area": 0.0, "annots": set()})
            for ai, a in enumerate(c_):
                for ct in a.contours:
                    z = round(float(ct.image_z_position), 2)
                    try:
                        coords = ct.to_matrix(include_k=False)
                    except Exception:
                        coords = None
                    slices[z]["area"] += _shoelace_area(coords)
                    slices[z]["annots"].add(ai)
            nodules.append({
                "scores": scores,
                "n_annotators": len(c_),
                "diameters": diameters,
                "slices": {z: {"area": v["area"], "n_annot": len(v["annots"])}
                           for z, v in slices.items()},
            })
        records.append({
            "patient_id": s.patient_id,
            "series_uid": s.series_instance_uid,
            "n_annotations": len(s.annotations),
            "nodules": nodules,
        })
    return records


# ------------------------------------------------------------------- assembly (superset)
def assemble(records, series_map, cfg):
    """Build per-nodule + per-sample (slice superset) tables. No min-annotator filtering:
    every LABELLED nodule contributes ALL its slices; flags mark cohort/representative."""
    label_cfg = cfg["label"]
    principal = int(cfg["metadata"]["min_annotators_principal"])
    over_thr = int(cfg["metadata"]["over_merge_flag_threshold"])

    # Only series with >=1 labelled nodule need a zmap, and only their nodules' slice z's
    # (the linear z<->index model in build_zmap reads ~5x fewer headers this way).
    needed_z_by_series = defaultdict(set)
    for rec in records:
        if rec["series_uid"] not in series_map:
            continue
        for nod in rec["nodules"]:
            if _consensus_label(nod["scores"], label_cfg)[1] is None:
                continue
            needed_z_by_series[rec["series_uid"]].update(nod["slices"].keys())
    needed = {su: series_map[su][0] for su in needed_z_by_series}

    # Persistent slice->file (zmap) cache: the DICOM header reads are the slow part; cache
    # them so re-runs are instant (keyed by SeriesInstanceUID; paths are absolute).
    cache_path = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"],
                              "metadata", "_zmap_cache.pkl")
    zmaps = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                zmaps = pickle.load(f)
        except Exception:
            zmaps = {}
    missing = [su for su in needed if su not in zmaps]
    if missing:
        with ThreadPoolExecutor(max_workers=int(cfg["metadata"]["io_workers"])) as ex:
            for su, zmap in zip(missing,
                                ex.map(lambda s: build_zmap(needed[s], needed_z_by_series[s]), missing)):
                zmaps[su] = zmap
        with open(cache_path, "wb") as f:
            pickle.dump(zmaps, f)

    nodule_rows, sample_rows = [], []
    for rec in records:
        pid, suid = rec["patient_id"], rec["series_uid"]
        on_disk = suid in series_map
        zmap = zmaps.get(suid, {})
        series_short = suid[-8:]
        for ni, nod in enumerate(rec["nodules"], start=1):
            scores, n_ann = nod["scores"], nod["n_annotators"]
            label_name, encoded = _consensus_label(scores, label_cfg)
            med = median(scores) if scores else None
            diam = float(np.median(nod["diameters"])) if nod["diameters"] else None
            nodule_id = f"{pid}_{series_short}_N{ni:02d}"
            slices = nod["slices"]
            cohort_main = n_ann >= principal
            flag_over_merge = n_ann > over_thr

            nodule_rows.append({
                "patient_id": pid, "nodule_id": nodule_id, "series_uid": suid,
                "n_annotators": n_ann,
                "malignancy_scores": "|".join(map(str, scores)),
                "median_malignancy": med, "label_name": label_name, "label": encoded,
                "nodule_diameter_mm": diam, "n_slices": len(slices),
                "on_disk": on_disk, "cohort_main": cohort_main,
                "flag_over_merge": flag_over_merge,
            })

            if encoded is None:          # ambiguous / no-score nodules yield no samples
                continue

            rep_z = max(slices.keys(),
                        key=lambda z: (slices[z]["area"], slices[z]["n_annot"], -abs(z)))
            for z in sorted(slices.keys()):
                sop, path = _resolve_slice(z, zmap)
                sample_rows.append({
                    "patient_id": pid, "nodule_id": nodule_id,
                    "slice_id": sop if sop else f"{series_short}_{z:.2f}",
                    "label": encoded, "path": path, "label_name": label_name,
                    "series_uid": suid, "z_position": z, "sop_uid": sop,
                    "n_annotators": n_ann,
                    "n_annotators_slice": slices[z]["n_annot"],
                    "malignancy_scores": "|".join(map(str, scores)),
                    "median_malignancy": med, "nodule_diameter_mm": diam,
                    "cohort_main": cohort_main,
                    "is_representative_slice": (z == rep_z),
                    "on_disk": on_disk and path is not None,
                })

    return pd.DataFrame(nodule_rows), pd.DataFrame(sample_rows)


# ------------------------------------------------------------------- accounting
def _cohort_counts(nodules_df, thr):
    labelled = nodules_df[nodules_df["label"].notna()]
    kept = labelled[labelled["n_annotators"] >= thr]
    return {
        "kept_nodules": int(len(kept)),
        "malignant": int((kept["label"] == 1).sum()),
        "benign": int((kept["label"] == 0).sum()),
        "patients": int(kept["patient_id"].nunique()),
    }


def _sample_counts(samples_df, mask):
    sub = samples_df[mask]
    slc = sub
    nod = sub[sub["is_representative_slice"]]
    return {
        "slice": {"n": int(len(slc)), "malignant": int((slc["label"] == 1).sum()),
                  "benign": int((slc["label"] == 0).sum())},
        "nodule": {"n": int(len(nod)), "malignant": int((nod["label"] == 1).sum()),
                   "benign": int((nod["label"] == 0).sum())},
    }


def write_accounting(records, nodules_df, samples_df, series_map, cfg, out_dir):
    principal = int(cfg["metadata"]["min_annotators_principal"])
    over_thr = int(cfg["metadata"]["over_merge_flag_threshold"])
    total_scans = len(records)
    total_patients = len({r["patient_id"] for r in records})
    total_annotations = sum(r["n_annotations"] for r in records)
    scans_with_ann = sum(1 for r in records if r["n_annotations"] > 0)
    total_nodules = int(len(nodules_df))
    ambiguous = int((nodules_df["label_name"] == "ambiguous").sum())
    labelled = int(nodules_df["label"].notna().sum())
    ann_per_nodule = Counter(nodules_df["n_annotators"])

    # over-merge report (Decision 3)
    over = nodules_df[nodules_df["flag_over_merge"]]
    over_report = {
        "threshold": over_thr,
        "n_flagged": int(len(over)),
        "label_breakdown": {k: int(v) for k, v in over["label_name"].value_counts().items()},
        "note": "flagged, NOT split; labels are the median-consensus over all their annotations",
    }

    sens = {str(t): _cohort_counts(nodules_df, t)
            for t in cfg["metadata"]["min_annotators_sensitivity"]}
    principal_samples = _sample_counts(samples_df, samples_df["cohort_main"])
    superset_samples = _sample_counts(samples_df, samples_df["label"].notna())

    # per-patient correlation (principal cohort, slice unit) — motivates patient-level split
    main_slices = samples_df[samples_df["cohort_main"]]
    if len(main_slices):
        pp = main_slices.groupby("patient_id").size()
        per_patient = {"min": int(pp.min()), "median": int(pp.median()), "max": int(pp.max())}
    else:
        per_patient = {}

    data = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": "pylidc " + pl.__version__,
        "config": {"min_annotators_principal": principal,
                   "consensus": cfg["label"]["consensus"],
                   "clustering": cfg["metadata"]["clustering"]},
        "source": {"ct_scans_in_pylidc_db": total_scans, "distinct_patients": total_patients,
                   "ct_series_folders_on_disk": len(series_map)},
        "annotations": {"total_ge3mm_annotations": total_annotations,
                        "scans_with_ge1_annotation": scans_with_ann},
        "nodules": {"total_physical_nodules_clustered": total_nodules,
                    "labelled_nodules_binary": labelled,
                    "ambiguous_median3_excluded": ambiguous,
                    "annotators_per_nodule_distribution":
                        {str(k): int(v) for k, v in sorted(ann_per_nodule.items())}},
        "over_merge_flag": over_report,
        "sensitivity_by_min_annotators": sens,
        "principal_cohort": {"min_annotators": principal, "samples": principal_samples,
                             "slices_per_patient": per_patient},
        "superset_cohort": {"min_annotators": 1, "samples": superset_samples},
    }
    with open(os.path.join(out_dir, "dataset_accounting.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    md = []
    md.append("# LIDC-IDRI dataset accounting (STAGE 1, v2)\n")
    md.append(f"_Generated {data['generated_utc']} · {data['engine']} · "
              f"principal cohort = min_annotators >= {principal}_\n")
    md.append("## Unit of analysis (Reviewer 3 / I4)\n")
    md.append("- **Annotation** = one radiologist's reading of a >=3 mm nodule "
              "(one malignancy score + contours); a physical nodule carries 1-4 annotations.\n"
              "- **Nodule** = a physical lesion (pylidc clustering). Labelling/grouping unit = "
              "the **nodule**. Both counts reported.\n")
    md.append("## Counting funnel\n")
    md.append(f"- CT scans (pylidc DB): **{total_scans}** across **{total_patients}** patients\n")
    md.append(f"- >=3 mm **annotations** (per-radiologist): **{total_annotations}**\n")
    md.append(f"- Physical **nodules** (clustered): **{total_nodules}**\n")
    md.append(f"- Labelled binary nodules (median != 3): **{labelled}**\n")
    md.append(f"- Ambiguous excluded (median == 3): **{ambiguous}**\n")
    md.append("### Annotators per nodule\n")
    for k, v in sorted(ann_per_nodule.items()):
        md.append(f"- {k}: {int(v)} nodules\n")
    md.append(f"## Over-merge flag (Decision 3 — >{over_thr} annotators, flagged not split)\n")
    md.append(f"- Flagged nodules: **{over_report['n_flagged']}** "
              f"({over_report['n_flagged']*100.0/max(total_nodules,1):.1f}% of nodules); "
              f"label breakdown: {over_report['label_breakdown']}\n")
    md.append("## Sensitivity to min-annotators\n")
    md.append("| min_annotators | kept nodules | malignant | benign | patients |\n")
    md.append("|---|---|---|---|---|\n")
    for t, c in sens.items():
        md.append(f"| {t} | {c['kept_nodules']} | {c['malignant']} | {c['benign']} | {c['patients']} |\n")
    md.append(f"## Principal cohort (min_annotators >= {principal}) — sample counts\n")
    for unit in ("slice", "nodule"):
        s = principal_samples[unit]
        md.append(f"- **{unit}-level**: {s['n']} samples "
                  f"(malignant {s['malignant']}, benign {s['benign']})\n")
    if per_patient:
        md.append(f"- Samples per patient (slice, principal): min {per_patient['min']}, "
                  f"median {per_patient['median']}, max {per_patient['max']} "
                  f"(within-patient correlation motivates patient-level splitting)\n")
    md.append("## Superset cohort (min_annotators >= 1, =1 sensitivity/ablation)\n")
    for unit in ("slice", "nodule"):
        s = superset_samples[unit]
        md.append(f"- **{unit}-level**: {s['n']} samples "
                  f"(malignant {s['malignant']}, benign {s['benign']})\n")
    with open(os.path.join(out_dir, "dataset_accounting.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    return data


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="STAGE 1 — build LIDC master metadata table.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.join(cfg["project"]["root"], cfg["paths"]["outputs"], "metadata")
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] mapping SeriesInstanceUID -> on-disk folder ...")
    series_map = build_series_to_folder(cfg["paths"]["lidc_dicom_root"],
                                        int(cfg["metadata"]["io_workers"]))
    print(f"      {len(series_map)} series folders resolved on disk.")

    print("[2/4] reading pylidc annotations + clustering nodules ...")
    records = extract_scan_records(cfg)
    print(f"      {len(records)} scans processed.")

    print("[3/4] assembling nodule + sample tables (superset; resolving slice paths) ...")
    nodules_df, samples_df = assemble(records, series_map, cfg)
    nodules_df.to_csv(os.path.join(out_dir, "lidc_nodules.csv"), index=False)
    samples_df.to_csv(os.path.join(out_dir, "lidc_master.csv"), index=False)
    print(f"      {len(nodules_df)} nodules, {len(samples_df)} slice-samples written.")

    print("[4/4] writing dataset accounting ...")
    data = write_accounting(records, nodules_df, samples_df, series_map, cfg, out_dir)

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "scans": data["source"]["ct_scans_in_pylidc_db"],
        "annotations_ge3mm": data["annotations"]["total_ge3mm_annotations"],
        "physical_nodules": data["nodules"]["total_physical_nodules_clustered"],
        "labelled_binary": data["nodules"]["labelled_nodules_binary"],
        "ambiguous_excluded": data["nodules"]["ambiguous_median3_excluded"],
        "over_merge_flagged": data["over_merge_flag"]["n_flagged"],
        "principal_cohort": data["principal_cohort"],
        "superset_cohort": data["superset_cohort"],
    }, indent=2))
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
