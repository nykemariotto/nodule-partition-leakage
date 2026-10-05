# Partition-unit survey — the evidence behind Related Work

Ten high-performance LIDC-IDRI malignancy classifiers, with the partition unit each one declares and
the sentence from its own methods section on which we classified it. The manuscript's Related Work
reports the tally; this file is the evidence, so every case can be checked against its source rather
than taken on trust.

The second part, *Structured search, 2025–2026*, does the same for the reproducible search that the
article reports beside these ten: every record found, the screening decision on each, and, for every
study read, the partition unit it declares with the passage behind the classification.

**How it was assembled.** We read the methods section of each study and recorded the unit it states,
verbatim. Where a paper states no unit, that is itself the finding, and it is recorded as such —
those are absence claims, so for them the whole methods or experimental-setup section was read, not
just the sentence naming cross-validation. Papers we could not access were excluded from the tally
rather than guessed at.

**What it is not.** A purposive, non-systematic sample, assembled to characterise how the partition
unit is reported. The counts describe the studies we read; they are not an estimate of a field-wide
rate. What they show is that partitioning below the patient, and not reporting the unit at all, both
occur among high-performing studies; how often they occur is what the structured search in the
article's Related Work measures.

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

---

## Structured search, 2025–2026

The ten studies above were chosen by hand, so they can show that a practice exists but not how common it is.
To check them against a reproducible set, we searched three databases with a procedure fixed before any query
was run.

**Search.** PubMed (E-utilities), Europe PMC (REST) and OpenAlex, which also indexes IEEE Xplore and most
conference proceedings, on 2 October 2026, for records published from 1 January 2025 that name LIDC-IDRI, LIDC,
the Lung Image Database Consortium or LUNA16, malignancy or benignity, and a deep-learning term in the title or
abstract. Scopus and Web of Science were not available to us. The exact strings and the raw counts are in
`outputs/literature/search_2026-10-02/search_log.json`, and `scripts/literature_search.py` runs them again.

**Eligibility.** Peer-reviewed primary studies that train and evaluate a deep-learning classifier (convolutional,
transformer, hybrid, or deep features fed to a classifier) of nodule malignancy on LIDC-IDRI, or on a set derived
from it, and report a classification metric. Reviews, detection- or segmentation-only work, other targets, other
datasets, work without deep learning, preprints, papers not in English, errata and duplicates were excluded. A
record was excluded on title and abstract only when the reason was clear from them; every other record went to
full text.

**Full text.** As fixed before the search, full texts were retrieved by script from open sources (PubMed
Central, Europe PMC, publisher open access and open repositories). Because that left nearly half of the studies
unread, an author then downloaded by hand the full texts that the scripts could not retrieve and that were
available to us, open access or through an institutional subscription. This step was added after the search but
before any of those studies was classified, and they were read and coded with the same rules; the released
table gives the route of each full text. A study whose full text could not be obtained was not classified.

**Flow.** 157 records retrieved (PubMed 32, Europe PMC 31, OpenAlex 94); 99 after deduplication by DOI, or by title when there was no DOI; 14 excluded on title and abstract; 85 sought in full text, of which 66 were read (45 retrieved by script, 9 open access and 12 by subscription downloaded by an author) and 19 could not be retrieved (15 available only by subscription, 2 open access and 1 free to read in PubMed Central but not obtained, 1 removed by the publisher); of those read, 7 excluded and **59 included**.

**Coding.** Each study is classified on what it declares about itself, as for the ten above, into categories
fixed before the search. An official patient-level split counts as patient level (LUNA16's ten subsets were
split on a patient level by design). A split stated over images, slices or patches is one category, whether or
not the paper says how many images each nodule contributes; where the samples are nodules, one per nodule, a
split of the samples is a split by nodule, and likewise for one sample per scan. "Not stated" is an absence
claim, so the whole methods and experimental-setup sections were read before it was used. A study that fits
none of the categories is coded "other" and described as found.

**Tally (59 studies): 23 patient · 3 scan · 14 nodule · 9 image, slice or patch · 6 not stated · 3 no split described · 1 other.** Nine inclusions are borderline on eligibility (records 9, 15, 42, 65, 67, 71, 86, 91, 92, marked below); without them the tally is 22 · 1 · 14 · 8 · 3 · 2 · 0. As with the ten above, the counts describe the studies read; the 19 that could not be read could change them.

**Reported performance** was extracted only to check what the manuscript says about the range of published
results, never to compare or rank studies: 26 of the 59 studies report an
accuracy above 95% and 11 an AUC above 0.98, and one an AUC above 0.99 (record 15, a borderline inclusion). Of those above 95% accuracy, 8 declared a patient-level split. The values are in the
released table, where a reported value that the paper itself defines as something other than an AUC is
noted and not counted as one.

**Verification.** Every quoted passage below was checked by script against the full text it was taken from.
An author checked each screening decision and each classification against the source study on 4 October 2026.

### Partitioned by patient (23)

| # | Study | Passage | Basis |
|---|---|---|---|
| 1 | 3D-DHA-MAFNet: A Multi-View Attention-Enhanced Deep Learning System for Lung Nodule Detection and Classification (2025). [10.31449/inf.v49i25.11184](https://doi.org/10.31449/inf.v49i25.11184) | “at a ratio of 7:2:1 at the patient level to ensure no overlap” | declared by the authors |
| 21 | Deep learning-based lung cancer classification of CT images (2025). [10.1186/s12885-025-14320-8](https://doi.org/10.1186/s12885-025-14320-8) | “the dataset was split at the patient level” | declared by the authors |
| 22 | Deep learning driven image-based cancer diagnosis (2025). [10.30574/wjaets.2025.16.2.1311](https://doi.org/10.30574/wjaets.2025.16.2.1311) | “We maintained patient-level splitting to prevent any overlap between training and testing” | declared by the authors |
| 23 | Deep learning framework for early diagnosis of lung cancer using multi- modal medical imaging (2025). [10.3934/math.20251310](https://doi.org/10.3934/math.20251310) | “All data partitioning was done at the patient level” | declared by the authors |
| 38 | MSTD: A Multi-Scale Transformer-Based Method to Diagnose Benign and Malignant Lung Nodules (2025). [10.1109/access.2025.3531001](https://doi.org/10.1109/access.2025.3531001) | “The LUNA16 dataset was divided into 10 subsets” | LUNA16's ten official subsets, which were split on a patient level |
| 40 | Multi-phase deep learning framework with Multiscale Adaptive Swin Transformer and embedding attention for precision lung nodule detection and classification (2025). [10.1038/s41598-025-31147-2](https://doi.org/10.1038/s41598-025-31147-2) | “all slices and corresponding nodules belonging to a single patient were grouped together” | declared by the authors |
| 51 | Timely Detection of Lung Nodule Malignancy Using 3D Convolutional Neural Networks (2025). [10.7759/s44389-025-06619-1](https://doi.org/10.7759/s44389-025-06619-1) | “Patient-wise division of the dataset into the train (70%), validation (15%), and test (15%)” | declared by the authors |
| 57 | A novel heterogeneous 3D fusion model with image restoration for enhanced pulmonary nodule classification on CT scans (2026). [10.1016/j.bspc.2026.111313](https://doi.org/10.1016/j.bspc.2026.111313) | “this study adopts a patient-level splitting strategy” | declared by the authors |
| 58 | A Parallel Convolutional–Hypergraph Feature learning Network for pulmonary nodule classification (2026). [10.1016/j.engappai.2026.116157](https://doi.org/10.1016/j.engappai.2026.116157) | “randomly split into five balanced subsets grouped by patient IDs” | declared by the authors |
| 59 | A unified imaging-to-pharmacology AI framework for pulmonary nodule detection, malignancy classification, and mutation-stratified drug validation from low-dose CT scans (2026). [10.36922/jctr026260059](https://doi.org/10.36922/jctr026260059) | “Patient-stratified 10-fold cross-validation ensured no patient-level data leakage” | declared by the authors |
| 62 | Adaptive Mixture-of-Experts Vision Transformer Ensemble with 3D Nodule Reasoning and Clinical Calibration for Lung Cancer Diagnosis (2026). [10.1007/s42979-026-05082-6](https://doi.org/10.1007/s42979-026-05082-6) | “all nodules originating from the same patient were assigned to a single split” | declared by the authors |
| 64 | An end-to-end hybrid 3D CNN–transformer architecture for comprehensive lung cancer assessment from chest CT scans (2026). [10.1038/s41598-026-67423-y](https://doi.org/10.1038/s41598-026-67423-y) | “partitioned on a patient-wise basis into 686 training scans (70%)” | declared by the authors |
| 65 | An intelligence-based approach to lung CT image classification: A PSO-optimized deep learning framework for slice-level malignancy detection (2026). [10.1016/j.ibmed.2026.100436](https://doi.org/10.1016/j.ibmed.2026.100436) | “partitioned strictly at the patient level into a 70% training subset” | declared by the authors; eligibility borderline: malignant vs normal or benign slices, LIDC-IDRI pooled with another source; the source of the labels is not stated |
| 66 | Attention-Based Deep Learning Framework for Lung Nodule Classification in CT Images (2026). [10.3390/sym18030431](https://doi.org/10.3390/sym18030431) | “All slices from a single patient were assigned to the same fold” | declared by the authors |
| 76 | Deep Learning Method for Early Detection of Lung Cancer using Deep Learning Algorithms (2026). [10.5815/ijem.2026.02.11](https://doi.org/10.5815/ijem.2026.02.11) | “the patient level was used as the basis for how the dataset was cut” | declared by the authors |
| 78 | Dual-Scale Hybrid Concept Bottleneck Network for Explainable 3D Lung Nodule Malignancy Classification in CT Imaging (2026). [10.3390/jimaging12090455](https://doi.org/10.3390/jimaging12090455) | “all nodules from a patient were assigned to one subset” | declared by the authors |
| 81 | Graph-Aware and Sequence-Aware Multimodal Deep Learning Framework for Cancer Detection and Risk Analysis from Medical Imaging (2026). [10.3390/jimaging12090431](https://doi.org/10.3390/jimaging12090431) | “The dataset split was performed at the patient level” | declared by the authors |
| 84 | Integrative Deep Diagnostic Architecture: A Convolutional Variational Attention Transformer for Personalized and Multi-Strategy Lung Nodule Detection (2026). [10.1109/access.2026.3679217](https://doi.org/10.1109/access.2026.3679217) | “patient-independent data splits to prevent correlation bias” | declared by the authors |
| 87 | LungDxFormer: a transformer-CNN hybrid model with dynamic spatial attention for accurate lung cancer detection and classification (2026). [10.1038/s41598-026-50449-7](https://doi.org/10.1038/s41598-026-50449-7) | “a patient-level split with a 70:15:15 ratio” | declared by the authors |
| 89 | LungMate: Calibrated malignancy risk estimation for pulmonary nodules from chest CT under dataset shift (2026). [10.1016/j.aej.2026.04.028](https://doi.org/10.1016/j.aej.2026.04.028) | “We employ a strict patient-level split protocol to prevent data leakage.” | declared by the authors |
| 94 | Omni-Swin3D: A radiomics-aware architecture for robust lung cancer diagnosis (2026). [10.1016/j.aej.2026.04.007](https://doi.org/10.1016/j.aej.2026.04.007) | “patient-level 10-fold cross-validation with the official LUNA16 patient partitioning identifiers” | declared by the authors |
| 95 | Refined Graph-Guided Fusion Network for Explainable Multimodal Lung Cancer Classification Using CT Imaging and Semantic Features (2026). [10.3390/info17090839](https://doi.org/10.3390/info17090839) | “we partitioned the publicly available LIDC-IDRI dataset at the patient level” | declared by the authors |
| 97 | Segmentation-Guided Hybrid Deep Learning for Pulmonary Nodule Detection and Risk Prediction from Multi-Cohort CT Images (2026). [10.3390/diseases14010021](https://doi.org/10.3390/diseases14010021) | “strict patient-level splits were enforced across training, validation, and testing sets” | declared by the authors |

### Partitioned by scan (3)

| # | Study | Passage | Basis |
|---|---|---|---|
| 9 | AI-Driven Bayesian Deep Learning for Lung Cancer Prediction: Precision Decision Support in Big Data Health Informatics (2025). [10.3390/biomedinformatics5030039](https://doi.org/10.3390/biomedinformatics5030039) | “our dataset was divided into five parts” · “condenses each CT volume into a 1 × 20 feature vector” | one feature vector per CT scan; folds formed over those samples; eligibility borderline: a fully connected network on tabular features predicts patient-level risk classes; the source of the labels is not stated |
| 61 | Adaptive geometric-attention network for two-stage lung nodule segmentation and malignancy classification in federated healthcare IoT edge environments (2026). [10.1371/journal.pone.0341096](https://doi.org/10.1371/journal.pone.0341096) |  | Table 2 gives the LIDC-IDRI training/validation/test split in scans (712/203/103 of 1,018) |
| 92 | Multi‐Stage Deep Learning Model for the Detection and Classification of Lung Nodules From CT Scan Images (2026). [10.1049/ipr2.70415](https://doi.org/10.1049/ipr2.70415) | “710 CT scans were used for training, 128 CT scans for validation” | split stated over CT scans (710/128/50), kept through the whole pipeline; eligibility borderline: cancer vs non-cancer labels assigned per CT scan; the source of the labels is not stated |

### Partitioned by nodule (14)

| # | Study | Passage | Basis |
|---|---|---|---|
| 4 | A Novel Automatic Lung Nodule Classification Scheme using Fusion Ghost Convolution and Hybrid Normalization in Chest CTs (2025). [10.2174/0115734056330120250310053454](https://doi.org/10.2174/0115734056330120250310053454) | “In the training set, there were a total of 809 lung nodules” | one cube per nodule; the sets are counted in nodules |
| 19 | Cross-ViT based benign and malignant classification of pulmonary nodules (2025). [10.1371/journal.pone.0318670](https://doi.org/10.1371/journal.pone.0318670) | “The dataset is randomly divided into two sets according to the ratio of 8:2” | one block per nodule; nodule-level folds confirmed in the published author response |
| 20 | DCAI: a dual cross-attention integration framework for benign-malignant classification of pulmonary nodules (2025). [10.3389/fmed.2025.1636008](https://doi.org/10.3389/fmed.2025.1636008) | “The dataset is randomly partitioned into five distinct subsets” | one sample per nodule, split at random |
| 32 | Low-Rank Adaptation of Pre-Trained Large Vision Models for Improved Lung Nodule Malignancy Classification (2025). [10.1109/ojemb.2025.3530841](https://doi.org/10.1109/ojemb.2025.3530841) | “The resulting dataset comprised 647 nodules” · “Models underwent 5-fold cross-validation” | one image per nodule; folds formed over those samples |
| 43 | Optimized theory-guided convolutional neural network for lung cancer classification using CT images with advanced FPGA implementation (2025). [10.1016/j.bspc.2025.107719](https://doi.org/10.1016/j.bspc.2025.107719) | “2985 lung nodules, comprising 1478 benign, 1507 malignant nodules, were gathered” · “15 % of them were chosen at random to serve as test, validation set” | one sample per nodule; nodules split at random (85/15) |
| 46 | Prior knowledge-based multi-task learning network for pulmonary nodule classification (2025). [10.1016/j.compmedimag.2025.102511](https://doi.org/10.1016/j.compmedimag.2025.102511) | “a list of 1736 pulmonary nodules” · “we employed five-fold cross-validation for all experiments on LIDC-IDRI dataset” | one crop per nodule; folds formed over those samples |
| 49 | Squeeze-and-Excitation Vision Transformer for Lung Nodule Classification (2025). [10.1109/access.2025.3529127](https://doi.org/10.1109/access.2025.3529127) | “374 benign lung nodules and 326 malignant lung nodules are divided into 10 equal parts” | one crop per nodule; nodules divided into ten parts |
| 50 | Texture-embedded Generative Adversarial Nets for the synthesis of 3D pulmonary nodules computed tomography images (2025). [10.1016/j.eswa.2025.126860](https://doi.org/10.1016/j.eswa.2025.126860) | “883 benign nodules and 492 malignant nodules from LIDC-IDRI dataset were randomly divided” | one volume per nodule; nodules split at random |
| 54 | 3D-NoduleNet: A comprehensive framework for benign and malignant pulmonary nodule classification (2026). [10.1016/j.asoc.2026.114645](https://doi.org/10.1016/j.asoc.2026.114645) | “under 10-fold cross-validation with T = 1004 nodules” | one volume per nodule; folds formed over those samples |
| 69 | Binary classification of lung cancer using vision transformer models on CT images (2026). [10.1007/s10791-026-09931-z](https://doi.org/10.1007/s10791-026-09931-z) | “315 lung nodule patches were extracted based on radiologist markings” · “All augmentation was done after the train-test split” | one patch per nodule; split before augmentation |
| 73 | Collaborative deep learning framework based on adaptive feature fusion for malignancy prediction of lung nodules (2026). [10.1007/s10489-025-07073-1](https://doi.org/10.1007/s10489-025-07073-1) | “832 lung nodule samples are randomly divided into 5 equally sized subsets” | one sample per nodule; nodules split at random into five folds |
| 90 | M3Net: A macro → meso → micro clinical-inspired hierarchical 3D network for pulmonary nodule classification (2026). [10.1016/j.inffus.2026.104334](https://doi.org/10.1016/j.inffus.2026.104334) | “2,625 pulmonary nodules (1,992 benign and 633 malignant), was randomly divided into 70% for training” | one cube per clustered nodule; nodules split at random |
| 93 | Multi-view attention-based deep learning for benign-malignant classification of pulmonary nodules (2026). [10.1371/journal.pone.0357963](https://doi.org/10.1371/journal.pone.0357963) | “The 869 nodules were divided at the nodule level into training, validation, and test sets” | split at the nodule level, as the authors state |
| 96 | Retrieval-augmented multimodal deep learning for interpretable lung nodule malignancy prediction (2026). [10.1016/j.displa.2026.103525](https://doi.org/10.1016/j.displa.2026.103525) | “Following prior work, we perform five-fold cross-validation” · “a cubic volume centered at the annotated nodule location is extracted” | one volume per nodule; folds formed over those samples |

### Split over images, slices or patches (9)

| # | Study | Passage | Basis |
|---|---|---|---|
| 12 | An Innovative Hybrid Deep Learning Technique Based on Neural Networks for Early Detection of Lung Cancer (2025). [10.47001/irjiet/2025.inspire47](https://doi.org/10.47001/irjiet/2025.inspire47) | “designating 80% of the images for training purposes and 20% for testing” | split stated over images; images per nodule not stated |
| 18 | Convolutional Neural Network Architecture-Inception (GoogLeNet) For Deep Architected Learning-Assisted Lung Cancer Classification in Computed Tomography Images (2025). [10.17485/ijst/v18i7.3086](https://doi.org/10.17485/ijst/v18i7.3086) | “divides 70% of the images for training and the remaining 30% for testing” | split stated over images; images per nodule not stated |
| 35 | LungCT-NET: An explainable transfer learning-based robust ensemble model for lung cancer diagnosis (2025). [10.1016/j.knosys.2025.113854](https://doi.org/10.1016/j.knosys.2025.113854) | “partitioned into training and testing subsets with an 80:20 ratio” · “comprising 8905, 2227, and 2784 images, respectively” | split stated over images (8,905/2,227/2,784); images per nodule not stated |
| 41 | Multitask Swin Transformer for classification and characterization of pulmonary nodules in CT images (2025). [10.21037/qims-24-1619](https://doi.org/10.21037/qims-24-1619) | “added enhanced data to the training set” · “1,600 pulmonary nodule images were selected as the test set” | augmented copies added before the training/validation split; the test set is selected as images |
| 44 | Optimizing lung nodule classification through the integration of image processing and transfer learning techniques (2025). [10.1007/s10586-025-05408-8](https://doi.org/10.1007/s10586-025-05408-8) | “210 segmented nodule images were utilized for feature extraction” · “170 allocated for training and 40 reserved for testing” | split stated over images (170/40); images per nodule not stated |
| 52 | Towards Reliable Lung Cancer Diagnosis: A Novel CNN-RNN Hybrid Model For CT-Based Detection (2025). [10.52783/jisem.v10i42s.7905](https://doi.org/10.52783/jisem.v10i42s.7905) | “After applying data augmentation techniques, the dataset size increased to 14,488 images” | augmented images split 60:20:20 |
| 63 | An attention aided wavelet convolutional neural network for lung nodule characterization (2026). [10.1016/j.ijmedinf.2025.106118](https://doi.org/10.1016/j.ijmedinf.2025.106118) | “an 80:20 split for training and internal validation” · “#Slices Used for Training” | split stated over slices (4,000/1,000); slices per nodule not stated |
| 83 | Improving Early Detection and Diagnosis of Lung Cancer Using an Enhanced Ensemble Deep Learning Model on CT Images (2026). [10.18502/fbt.v13i3.22752](https://doi.org/10.18502/fbt.v13i3.22752) | “They are then divided into an 80:20 train-test ratio” | images split 80:20; the table counts include augmented images |
| 91 | Modified ShuffleNet trained on gradient pattern and shape-based features for lung cancer classification with improved M-SegNet segmentation (2026). [10.1038/s41598-026-39492-6](https://doi.org/10.1038/s41598-026-39492-6) | “For training (90%), the training image used is 2102, and the testing image is 234” | split stated over images; images per nodule not stated; eligibility borderline: cancer vs non-cancer slices; the source of the labels is not stated |

### Split described, unit not stated (6)

| # | Study | Passage | Basis |
|---|---|---|---|
| 3 | A new approach to early Diagnosis of Lung Cancer Employing Neural Networks Utilising Hybrid Deep Learning (2025). [10.29070/0ntgfm39](https://doi.org/10.29070/0ntgfm39) | “The prepared data is then divided into training and testing sets.” | a split is described without its unit |
| 8 | AI-Assisted Early Screening of Lung Cancer: Evaluation of Deep Learning Models based on CT Images (2025). [10.54097/dmwrfd35](https://doi.org/10.54097/dmwrfd35) | “The data set was divided by stratified sampling” | a split is described without its unit |
| 16 | Big Data-Driven Approach for Lung Cancer Identification via Advanced Deep Transfer Learning Models (2025). [10.47672/ejt.2730](https://doi.org/10.47672/ejt.2730) | “a training set (80%) and a validation set (20%)” | a split is described without its unit |
| 42 | Optimized Lung Nodule Classification Using CLAHE-Enhanced CT Imaging and Swin Transformer-Based Deep Feature Extraction (2025). [10.3390/jimaging11100346](https://doi.org/10.3390/jimaging11100346) | “stratified train–test splits were employed to preserve class distributions” | a split is described without its unit; eligibility borderline: slices labelled by scan (malignant if any nodule is), from the LIDC-IDRI ratings |
| 67 | Automated lung cancer classification using intensity-driven RoI selection and transfer learning (2026). [10.1186/s12911-026-03519-9](https://doi.org/10.1186/s12911-026-03519-9) | “a primary ratio of testing and training is 60:40 ratios” | a split is described without its unit; eligibility borderline: three classes labelled by the authors on LIDC-IDRI pooled with other sources |
| 71 | CCSA-RF enhanced lightweight multi-scale CNN framework for robust lung cancer classification (2026). [10.1016/j.ibmed.2026.100363](https://doi.org/10.1016/j.ibmed.2026.100363) | “The data were divided into 80/20 percent training and testing, respectively.” | a split is described without its unit; eligibility borderline: cancerous vs non-cancerous tissue; the source of the labels is not stated |

### No split described (3)

| # | Study | Passage | Basis |
|---|---|---|---|
| 15 | BCM-FSS-MHA: attention-enhanced few-shot learning approach for solitary pulmonary nodule detection in low-dose CT imaging (2025). [10.1007/s41870-025-02956-5](https://doi.org/10.1007/s41870-025-02956-5) |  | no separation of training and test data is described; eligibility borderline: LUNA16 pooled with another source; the source of the malignancy labels is not stated |
| 25 | Explainable artificial intelligence for detecting lung cancer (2025). [10.35784/iapgos.6626](https://doi.org/10.35784/iapgos.6626) |  | no separation of training and test data is described |
| 45 | Performance of VGG-16 Convolutional Neural Network Model Based Lung Cancer Classification on Computed Tomography (2025). [10.55041/ijsrem47387](https://doi.org/10.55041/ijsrem47387) |  | no separation of training and test data is described |

### Other, described as found (1)

| # | Study | Passage | Basis |
|---|---|---|---|
| 86 | LNA-Net: Enhancing detection and classification of benign and malignant pulmonary nodules in CT scans (2026). [10.1016/j.bspc.2026.109505](https://doi.org/10.1016/j.bspc.2026.109505) | “The dataset contains 6691 scans” · “The split ratio for the classification set is 60:40.” | split 60:40 over 6,691 items the paper calls scans; its source set has 888 scans and 1,186 nodules; eligibility borderline: labels from a third-party set derived from LUNA16; their derivation is not stated |

### Excluded at full text (7)

| # | Study | Reason |
|---|---|---|
| 10 | An Effective Multi-Architecture Approach for Lung Cancer Detection (2025). [10.18280/mmep.121224](https://doi.org/10.18280/mmep.121224) | not nodule malignancy classification: a lung-cancer risk classifier on CT, genomic and clinical data |
| 11 | An Efficient Hybrid Framework U-Net-based deep learning Technique of Early Detection of Pulmonary Nodules using LIDC-IDRI Dataset (2025). [10.21928/uhdjst.v9n2y2025.pp266-275](https://doi.org/10.21928/uhdjst.v9n2y2025.pp266-275) | the target classes are not benign and malignant; the labels described are nodule size classes |
| 33 | LUNG CANCER DETECTION USING MACHINE LEARNING TECHNIQUES (2025). [10.52783/jisem.v10i47s.9265](https://doi.org/10.52783/jisem.v10i47s.9265) | no trained and evaluated model is described; the task named is histological subtype |
| 36 | LungDetectNet: a multi-task deep learning framework with enhanced detection and descriptive capabilities (2025). [10.1016/j.bspc.2025.109158](https://doi.org/10.1016/j.bspc.2025.109158) | no classification metric: malignancy is regressed as a score and evaluated by its error |
| 48 | R-GAN-NET: A Novel Deep Convolution Neural Network with Augmentation to Detect Lounge Nodules (2025). [10.58346/jowua.2025.i1.029](https://doi.org/10.58346/jowua.2025.i1.029) | not malignancy classification: the classes are no nodule, nodule under 3 mm and nodule of 3 mm or more |
| 88 | LungGuard: An IoT-Integrated Wireless Sensor and Hybrid CNN–Vision Transformer Framework for Early Lung Cancer Detection and Risk Assessment (2026). [10.53365/nrfhh.2513](https://doi.org/10.53365/nrfhh.2513) | no trained and evaluated model: the result tables are stated to be placeholders |
| 98 | Voxel Normalization in LDCT Imaging: Its Significance in Texture Feature Selection for Pulmonary Nodule Malignancy Classification: Insights from Two Centers (2026). [10.3390/diagnostics16020186](https://doi.org/10.3390/diagnostics16020186) | the malignancy data are two in-house cohorts; no model is trained or tested on LIDC-IDRI |

### Full text not retrieved (19)

| # | Study | Reason |
|---|---|---|
| 6 | A teacherless lightweight classification framework for benign and malignant pulmonary nodules based on GAS (2025). [10.1088/2057-1976/ae268a](https://doi.org/10.1088/2057-1976/ae268a) | subscription only |
| 13 | Analysis on grading of lung nodule images with segmentation using u-net and classification with Convolutional Neural Network Fish Swarm Optimization (2025). [10.1016/j.bbe.2024.12.002](https://doi.org/10.1016/j.bbe.2024.12.002) | subscription only |
| 17 | Computer Aided Diagnosis of lung cancer using deep learning techniques (2025). [10.1016/j.bspc.2025.107994](https://doi.org/10.1016/j.bspc.2025.107994) | subscription only |
| 24 | Domain-Specific Data Augmentation for Lung Nodule Malignancy Classification (2025). [10.1109/embc58623.2025.11254063](https://doi.org/10.1109/embc58623.2025.11254063) | subscription only |
| 26 | Explainable Classification of Benign-Malignant Pulmonary Nodules With Neural Networks and Information Bottleneck (2025). [10.1109/tnnls.2023.3303395](https://doi.org/10.1109/tnnls.2023.3303395) | subscription only |
| 27 | Explainable deep learning model with the internet of medical devices for early lung abnormality detection (2025). [10.1016/j.engappai.2025.110961](https://doi.org/10.1016/j.engappai.2025.110961) | subscription only |
| 29 | Fusing radiomics and deep learning features for automated classification of multi-type pulmonary nodule (2025). [10.1002/mp.17901](https://doi.org/10.1002/mp.17901) | subscription only |
| 30 | Gradient-driven pixel connectivity convolutional neural networks classification based on U-Net lung nodule segmentation (2025). [10.1016/j.medengphy.2025.104376](https://doi.org/10.1016/j.medengphy.2025.104376) | subscription only |
| 34 | Lung Nodule Malignancy Classification On 3D CT Images Using a Cosine Similarity-Enhanced Graph Attention Network (2025). [10.1109/embc58623.2025.11253663](https://doi.org/10.1109/embc58623.2025.11253663) | subscription only |
| 37 | MS-MambaNet: A lightweight Multi-Scale Mamba architecture for lung cancer classification with explainability (2025). [10.1016/j.bspc.2025.109131](https://doi.org/10.1016/j.bspc.2025.109131) | subscription only |
| 39 | Multi-level deep imaging features fusion for lung nodule malignancy prediction (2025). [10.1016/j.bspc.2025.108530](https://doi.org/10.1016/j.bspc.2025.108530) | subscription only |
| 60 | A vision-language model-based approach for lung cancer diagnosis using lossless 3D CT images: evaluation of GPT-4.1 and GPT-4o for patient-level malignancy assessment (2026). [10.1007/s13755-025-00417-8](https://doi.org/10.1007/s13755-025-00417-8) | free to read in PubMed Central, not retrievable by script |
| 68 | Bayesian Probabilistic Knowledge From Diameter Prior for Decision Fusion to Detect Lung Nodule Heterogeneity (2026). [10.1109/tai.2026.3665484](https://doi.org/10.1109/tai.2026.3665484) | subscription only |
| 72 | Classification of lung nodules in CT images based upon a multiplane dense inception network (2026). [10.1002/mp.70316](https://doi.org/10.1002/mp.70316) | open access, not retrievable by script |
| 74 | Cross Fusion‐ CapNets : Interpretable Modeling for Benign‐Malignant Classification of Lung Nodules (2026). [10.1002/ima.70426](https://doi.org/10.1002/ima.70426) | subscription only |
| 75 | Decision Support System for Explainable Assessment of Lung Nodule Malignancy in Computed Tomography Images (2026). [10.1134/s1054661826700057](https://doi.org/10.1134/s1054661826700057) | subscription only |
| 79 | Effective lung nodule segmentation and classification by employing a SPPUNet model and global context attention-based InceptionV3 (2026). [10.1080/17538157.2025.2611342](https://doi.org/10.1080/17538157.2025.2611342) | subscription only |
| 82 | Hybrid CNN–Vision Transformer Framework with Self-Attention for Lung Nodule Segmentation and Cancer Detection from CT Scans (2026). [10.64388/irev9i11-1717646](https://doi.org/10.64388/irev9i11-1717646) | removed by the publisher |
| 85 | Linear Nodule Growth Pattern Prediction (LNGP) Metric for Lung Pulmonary Malignancy Classification (2026). no DOI | open access, not retrievable by script |

The 14 records excluded on title and abstract, each with its reason, are in
`outputs/literature/search_2026-10-02/screening_title_abstract.csv`.
