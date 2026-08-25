# Partition-unit survey — the evidence behind Related Work

Ten high-performance LIDC-IDRI malignancy classifiers, with the partition unit each one declares and
the sentence from its own methods section on which we classified it. The manuscript's Related Work
reports the tally; this file is the evidence, so every case can be checked against its source rather
than taken on trust.

**How it was assembled.** We read the methods section of each study and recorded the unit it states,
verbatim. Where a paper states no unit, that is itself the finding, and it is recorded as such —
those are absence claims, so for them the whole methods or experimental-setup section was read, not
just the sentence naming cross-validation. Papers we could not access were excluded from the tally
rather than guessed at.

**What it is not.** A purposive, non-systematic sample, assembled to characterise how the partition
unit is reported. The counts describe the studies we read; they are not an estimate of a field-wide
rate. What they establish is that partitioning below the patient, and not reporting the unit at all,
are both common enough to find immediately among the best-performing papers.

**Verification.** Assembled 2026-07-23. On 2026-08-05 an author opened each of the ten studies and
checked both the quote and the classification against the paper's own text: 10/10 quotes verbatim,
10/10 classifications upheld, tally unchanged.

**Tally: 3 patient · 3 nodule · 3 unspecified · 1 no split described.**

---

## Partitioned by patient (3)

**Afshar et al. 2020 — 3D-MCN.** *Sci. Rep.* 10:7948. [10.1038/s41598-020-64824-5](https://doi.org/10.1038/s41598-020-64824-5)
> "there were no shared patients between the two … sets"

**Nasrullah et al. 2019 — CMixNet.** *Sensors* 19(17):3722. [10.3390/s19173722](https://doi.org/10.3390/s19173722)
> "LUNA16's dataset split principle of 10-fold cross-validation of patient level data"

**Saha & Prakash 2025 — Multi-attention stacked ensemble.** [arXiv:2507.20221](https://arxiv.org/abs/2507.20221)
> "Our patient-level stratified split yields 5187 training samples …"

## Partitioned by nodule (3)

**Al-Shabi, Lee & Tan 2019 — Gated-dilated networks.** *IEEE Access* 7:178827–178838. [10.1109/ACCESS.2019.2958663](https://doi.org/10.1109/ACCESS.2019.2958663)
> "the sum of the 406 malignant and 442 benign nodules where randomly divided into 10 exclusive partitions"

The `where` is in the original. This is the reference pipeline reused by several later studies, and
the division is deliberate rather than incidental: the same passage describes balancing the
benign/malignant ratio across folds, so the authors considered how to divide and did not group by
patient.

**Wang et al. 2024 — ContrastDiagnosis.** [arXiv:2403.05280](https://arxiv.org/abs/2403.05280)
> "1,226 lung nodules … randomly divided into two subsets for training (980 nodules) and independent testing (246 nodules)"

**Mamun et al. 2025 — LMLCC-Net.** [arXiv:2505.06370](https://arxiv.org/abs/2505.06370) (v2)
> "We divided the data into test, train, and validation sets based on nodule ID"

The cohort here is LUNA16, which ships an official patient-level split; splitting by nodule ID
overrides it.

## k-fold reported, unit not stated (3)

These are absence claims, so each was checked across the full methods or experimental-setup section
rather than on the sentence naming cross-validation alone.

**Al-Shabi et al. 2019 — Local-Global networks.** *Int. J. Comput. Assist. Radiol. Surg.* 14(10):1815–1819. [10.1007/s11548-019-01981-7](https://doi.org/10.1007/s11548-019-01981-7)
> "10-fold cross-validation", with no partition-unit statement

Verified in §2.2 "Experimental Setup", which gives dataset, optimiser, epochs, batch size, loss,
framework, hardware and the AUC tool, and refers to "all 10 folds of the 10-fold cross-validation
experiment" without saying what the folds are formed on. The one procedure it imports from another
paper is scoped in its own words to "data pre-processing", so the cross-reference does not supply a
partition unit by inheritance. Checked on the preprint of the cited version; the published article
is shorter, so an omission there does not become a statement.

**Al-Shabi, Shak & Tan 2022 — ProCAN.** *Pattern Recognit.* 122:108309. [10.1016/j.patcog.2021.108309](https://doi.org/10.1016/j.patcog.2021.108309)
> "a 10-fold cross-validation method, in which 9 folds were used for training and one fold for testing"

**Xiao et al. 2020 — Ensemble classification.** *Oncol. Lett.* 20(1):401–408. [10.3892/ol.2020.11576](https://doi.org/10.3892/ol.2020.11576)
> "10-fold cross-validation was used to obtain the Acc"

## No split procedure described (1)

**Wang et al. 2024 — Attention pyramid pooling.** *PLoS ONE* 19(5):e0302641. [10.1371/journal.pone.0302641](https://doi.org/10.1371/journal.pone.0302641)

§3.1 is titled "Dataset preprocessing and splitting", but the splitting it describes divides the
cohort into malignancy-rating subsets, with the counts given in the paper's Table 2. No train/test
procedure and no partition unit is stated anywhere.

---

## A note on how these were classified

Each study is classified on **what it declares about itself**, never on what can be inferred from a
citation or from a shared pipeline. That rule is conservative in both directions: it is why
Al-Shabi et al. 2019 (Local-Global) sits under "unit not stated" rather than under "nodule", even
though the study whose preprocessing it adopts partitions by nodule. Reading such cross-references
broadly would move entries into the nodule column and make the pattern look stronger than the
evidence supports.
