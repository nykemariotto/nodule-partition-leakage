# LIDC-IDRI dataset accounting (STAGE 1, v2)
_Generated 2026-07-17T16:40:21Z · PRINCIPAL cohort = min_annotators >= 1 (D16 reversal)_

## Unit of analysis (R3/I4): annotation vs nodule — both reported
- Annotation = one radiologist's >=3 mm reading; Nodule = pylidc cluster. Labelling unit = nodule.

## Counting funnel
- CT scans (pylidc DB): **1018** / **1010** patients · >=3mm annotations: **6859**
- Physical nodules (clustered): **2651** · labelled binary: **1695** · ambiguous excluded (median==3): **956**

## Over-merge guard (>4 annotators, FLAGGED not split)
- Flagged: **14** (0.5%); labels {'malignant': 9, 'ambiguous': 4, 'benign': 1}. Flag is label-neutral (0 relabelled).

## PRINCIPAL cohort (min_annotators >= 1) — literature-comparable + statistically viable
- **slice-level (PRIMARY power experiment)**: 11026 slices (malignant 5504, benign 5522) · 740 patients
- **nodule-level (CONFIRMATORY, wider CI — D17)**: 1695 nodules (malignant 545, benign 1150)
- slices/patient: min 1, median 11, max 97

## SENSITIVITY cohort (min_annotators >= 3) — high-agreement label-noise robustness (R4.4)
- slice-level: 6973 slices (malignant 4450, benign 2523) · 535 patients
- nodule-level: 879 nodules (malignant 430, benign 449)
