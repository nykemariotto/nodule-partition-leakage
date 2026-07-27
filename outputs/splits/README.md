# Cross-validation partition indices

These files **are** the experiment. Every result in the paper is a comparison between models trained
on the partitions defined here, so they are committed rather than regenerated — a reviewer can check
the partitioning claim without running anything, and a rerun cannot silently produce different folds.

## Naming

```
{dataset}_{sample_unit}_{arm}_rep{R}_fold{K}_{train|val|test}.csv
```

| field | values | meaning |
|---|---|---|
| `dataset` | `lidc_binary` | principal cohort: every labelled nodule, ≥1 annotator (740 patients) |
| | `lidc_binary_ge3` | high-agreement sensitivity cohort, ≥3 annotators (535 patients). **Never pooled with the principal cohort** — the distinct prefix is what makes pooling structurally impossible |
| `sample_unit` | `slice` | one row per axial slice the nodule spans (the primary axis) |
| | `nodule` | one row per nodule, its largest cross-section (confirmatory axis) |
| `arm` | `patient` | **arm A** — folds grouped by patient |
| | `random` | **arm B** — rows assigned independently of patient and nodule |
| | `nodule` | **arm C** — folds grouped by nodule, but nodules assigned independently of patient |
| `rep` | 0, 1, 2 | repeat; seeds 42, 123, 2024 from `config.repetition.seed_list` |
| `fold` | 0–4 | 5-fold |

## The grouping unit, stated unambiguously

This matters more here than in most studies, because the paper is *about* the partition unit.

- **arm A groups on `patient_id`.** No patient identifier appears in more than one of
  train/val/test in any fold. Asserted at split time and re-verified on these committed files.
- **arm B groups on nothing.** Rows are assigned independently, so a patient's slices — and a
  nodule's slices — straddle the boundary. This is the condition under audit, not a bug.
- **arm C groups on `nodule_id`.** All slices of a nodule stay on one side, but a patient's other
  nodules may fall across it. This isolates the patient route from the within-nodule route.

Validation is carved from the training part **using the same unit as the test split**, so the
train/val boundary cannot reintroduce a route the train/test boundary excludes.

## Measured leakage rates (fraction of test rows whose patient / nodule also appears in training)

| arm | L(patient) | L(nodule) | train slices per patient |
|---|---|---|---|
| A `patient` | 0.000 | 0.000 | 14.9 |
| B `random` | 0.995 | 0.973 | 10.5 |
| C `nodule` | 0.690 | 0.000 | 12.4 |

Arm C carrying a high patient-leakage rate *at full sample density* is the point: a null result for
the patient route cannot then be dismissed as too few samples per patient to learn a signature.

## Regenerating

```
python -m src.splits --config config.yaml --dataset lidc_binary \
    --sample-unit slice --arm patient --repeats 3 --folds 5
```
`--arm {patient,random,nodule}`, `--cohort {principal,ge3}`. Generation is seeded and deterministic:
regenerating overwrites these files with identical content. The leakage assertions run at generation
time and fail loudly rather than warning.

## Known limitation

Each row carries the full sample metadata, including an **absolute path** from the machine that
generated it (`E:\NODULES\LIDC-IDRI\...`). Only `patient_id`, `nodule_id`, `slice_id`, `z_position`
and `label` are needed to reconstruct a partition; the rest is redundant and makes these files
roughly an order of magnitude larger than the identifier lists other repositories ship. The format is
kept as-is because the analysis code aligns archived predictions to these rows by
`nodule_id` + `z_position`, and changing it would invalidate that alignment for runs already on disk.
