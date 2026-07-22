# Patient-Level vs Nodule-Level Partitioning in Pulmonary-Nodule Classification

Controlled **paired experiment** measuring how the data-partition unit
(patient-level vs nodule/slice-level) inflates reported performance on
pulmonary-nodule classification. Rebuilt for resubmission to *IEEE Access*
(Access-2026-27906). **`SPEC.md` is the methodology contract and overrides any
assumption.**

The central claim is *tested by design*: for each `(dataset, architecture, seed)`
two arms are run that are byte-for-byte identical except the split unit
(Arm A = `GroupShuffleSplit` on `patient_id`; Arm B = random nodule/slice split).
The gap between arms, with a 95% CI and a paired test, is the
partitioning-induced inflation. Flagship setting: **S2 — LIDC-IDRI binary
malignancy (4-5 vs 1-2, exclude 3)**.

## Hard rules (SPEC §0)
- Code == manuscript: every reported number is reproducible by a script here.
- Patient-level = grouped by `patient_id` (never `flow_from_directory(validation_split=)`).
- Every run persists `y_true`, `y_prob`, and split indices to disk.
- All hyperparameters live in `config.yaml`.
- Seed everything; split CSVs exported to `outputs/splits/`.

## Environment
```
conda create -n nodules python=3.10 -y
conda activate nodules
pip install -r requirements.txt
# create pylidc.conf pointing at the LIDC DICOM root — see data/README.md
```
GPU target: RTX 4060 Ti 8 GB (AMP + grad-accum effective batch 32, channels_last).

## Build order (SPEC §6)
```
1  python scripts/verify_download.py --config config.yaml   # data completeness
2  python -m src.metadata  --config config.yaml             # STAGE 1  (this stage)
3  python -m src.preprocess --config config.yaml            # STAGE 2
4  python -m src.splits     --config config.yaml            # STAGE 3  (exports split CSVs)
5  python -m src.train ...                                  # STAGE 4
6  evaluate → ensemble → stats → figures → paper            # STAGE 5-9
```

## Repository layout
See `SPEC.md §3`. Data (`LIDC-IDRI/`, `LNDb/`) and model weights are git-ignored;
`outputs/splits/`, `outputs/metadata/`, `outputs/probs/` are committed for
reproducibility (Reviewer 3).

## Status
- [x] STAGE 1 — `src/metadata.py`: master table + nodule table + dataset accounting.
- [ ] STAGE 2-9 — see SPEC §6.
