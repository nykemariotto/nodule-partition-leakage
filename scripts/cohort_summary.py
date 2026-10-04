"""
Cohort and acquisition summary (Table I of the paper), and the clustering of the label within
patients that the Discussion uses to explain why stratified folds do not help.

Reads only committed metadata:
  outputs/metadata/lidc_master.csv         one row per slice sample of the principal cohort
  outputs/metadata/lidc_nodules.csv        one row per nodule (diameter)
  outputs/metadata/acquisition_params.csv  one DICOM header per series

and checks that the slice table is the cohort the experiment ran on: the same rows as the three
partitions of one fold of the patient arm. Continuous values are reported as median [IQR] (range),
with pandas' default (linear) quantiles.

Label clustering, on the nodule labels of the 740 patients:
  ICC(1)         one-way random-effects ANOVA intraclass correlation of the binary label within
                 patient, with the n0 correction for unequal group sizes;
  design effect  1 + (m_w - 1) x ICC, with m_w = sum(n_i^2) / sum(n_i), the size-weighted mean
                 number of nodules per patient, which allows for unequal cluster sizes (with equal
                 sizes it is the plain mean; that equal-size value is reported beside it);
  label-pure     share of patients whose nodules all carry the same label, among all patients
                 (a patient with a single nodule is pure by definition) and among the patients with
                 more than one nodule, which is where label purity constrains a stratified split.

    python scripts/cohort_summary.py

Writes outputs/_analysis/cohort_summary.json and prints the table.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "outputs", "metadata")
SPLITS = os.path.join(ROOT, "outputs", "splits")
OUT = os.path.join(ROOT, "outputs", "_analysis", "cohort_summary.json")


def describe(x):
    x = pd.Series(x, dtype=float)
    return {"median": float(x.median()), "q1": float(x.quantile(0.25)), "q3": float(x.quantile(0.75)),
            "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


def label_clustering(nodules):
    """ICC(1) of the binary label within patient, design effect, and label purity."""
    g = nodules.groupby("patient_id")["label"]
    n_i = g.size().to_numpy(float)
    k, n = len(n_i), n_i.sum()
    grand = nodules["label"].mean()
    msb = (n_i * (g.mean().to_numpy() - grand) ** 2).sum() / (k - 1)
    msw = ((nodules["label"] - nodules["patient_id"].map(g.mean())) ** 2).sum() / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    mean_size = n / k
    return {"icc1": float(icc), "n0": float(n0), "mean_nodules_per_patient": float(mean_size),
            "design_effect_equal_sizes": float(1 + (mean_size - 1) * icc),
            "design_effect": float(1 + ((n_i ** 2).sum() / n - 1) * icc),
            "label_pure_share": float(g.nunique().eq(1).mean()),
            "single_nodule_share": float((g.size() == 1).mean()),
            "patients_multi_nodule": int((g.size() > 1).sum()),
            "label_pure_share_multi_nodule": float(g.nunique().eq(1)[g.size() > 1].mean())}


def main():
    master = pd.read_csv(os.path.join(META, "lidc_master.csv"))
    nodtab = pd.read_csv(os.path.join(META, "lidc_nodules.csv"))
    acq = pd.read_csv(os.path.join(META, "acquisition_params.csv"))

    # the slice table must be exactly the rows the experiment partitioned
    split = pd.concat(pd.read_csv(os.path.join(SPLITS, f"lidc_binary_slice_patient_rep0_fold0_{p}.csv"),
                                  usecols=["nodule_id", "slice_id"]) for p in ("train", "val", "test"))
    if set(zip(master.nodule_id, master.slice_id)) != set(zip(split.nodule_id, split.slice_id)):
        sys.exit("FATAL: lidc_master.csv is not the set of rows partitioned in outputs/splits/")

    nodules = master.drop_duplicates("nodule_id")[["patient_id", "nodule_id", "label"]]
    diam = nodtab.set_index("nodule_id").loc[nodules.nodule_id, "nodule_diameter_mm"]
    series = acq.drop_duplicates("series_uid")
    if set(series.series_uid) != set(master.series_uid):
        sys.exit("FATAL: acquisition_params.csv does not hold exactly the cohort's CT series")

    def shares(col, top=None):
        s = series[col].value_counts(normalize=True)
        return {"n_distinct": int(series[col].nunique()),
                "share": {str(k): float(v) for k, v in (s.head(top) if top else s).items()}}

    out = {
        "cohort": {
            "patients": int(master.patient_id.nunique()),
            "nodules": int(len(nodules)),
            "nodules_malignant": int((nodules.label == 1).sum()),
            "nodules_benign": int((nodules.label == 0).sum()),
            "slices": int(len(master)),
            "slices_malignant": int((master.label == 1).sum()),
            "slices_benign": int((master.label == 0).sum()),
            "slices_per_patient": describe(master.groupby("patient_id").size()),
            "nodules_per_patient": describe(nodules.groupby("patient_id").size()),
            "nodule_diameter_mm": describe(diam),
        },
        "acquisition": {
            "series": int(len(series)),
            "manufacturer": shares("manufacturer"),
            "kernel": shares("kernel", top=3),
            "slice_thickness_mm": {**describe(series.slice_thickness),
                                   "n_distinct": int(series.slice_thickness.nunique())},
            "in_plane_spacing_mm": describe(series.pixel_spacing),
            "tube_voltage_kvp": describe(series.kvp),
            "tube_current_ma": describe(series.tube_current),
        },
        "label_clustering": label_clustering(nodules),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    c, a, lc = out["cohort"], out["acquisition"], out["label_clustering"]

    def fmt(d, nd):
        f = f"{{:.{nd}f}}"
        return (f"{f.format(d['median'])} [{f.format(d['q1'])}-{f.format(d['q3'])}] "
                f"({f.format(d['min'])}-{f.format(d['max'])})")

    print(f"patients {c['patients']} | nodules {c['nodules']} ({c['nodules_malignant']} malignant / "
          f"{c['nodules_benign']} benign) | slices {c['slices']} ({c['slices_malignant']} / "
          f"{c['slices_benign']})")
    print(f"slices per patient   {fmt(c['slices_per_patient'], 0)}")
    print(f"nodules per patient  {fmt(c['nodules_per_patient'], 0)}")
    print(f"nodule diameter (mm) {fmt(c['nodule_diameter_mm'], 1)}")
    print(f"series {a['series']} | manufacturer "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in a["manufacturer"]["share"].items()))
    print(f"kernel {a['kernel']['n_distinct']} distinct; "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in a["kernel"]["share"].items()))
    print(f"slice thickness (mm) {fmt(a['slice_thickness_mm'], 2)}, "
          f"{a['slice_thickness_mm']['n_distinct']} distinct")
    print(f"in-plane spacing (mm) {fmt(a['in_plane_spacing_mm'], 2)}")
    print(f"tube voltage (kVp) {fmt(a['tube_voltage_kvp'], 0)}")
    print(f"tube current (mA) {fmt(a['tube_current_ma'], 1)}")
    print(f"label within patient: ICC {lc['icc1']:.3f}, design effect {lc['design_effect']:.2f} "
          f"({lc['design_effect_equal_sizes']:.2f} if all patients had the mean number of nodules); "
          f"label-pure: {100 * lc['label_pure_share']:.0f}% of all patients, "
          f"{100 * lc['label_pure_share_multi_nodule']:.0f}% of the {lc['patients_multi_nodule']} "
          f"with more than one nodule")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
