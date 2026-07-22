# DRAFT — request to the LNDb authors (NOT SENT)

**Status: draft for the researcher to review and send.** Not sent by Claude — sending to real
researchers is an outward-facing action, and I do not have a *verified* recipient address (I will
not fabricate one). Recipient must be confirmed from a verified source before sending:
- the LNDb challenge site contact (lndb.grand-challenge.org), or
- the corresponding author of Pedrosa et al., "LNDb: A Lung Nodule Database on Computed
  Tomography" (arXiv:1911.08434) / the Med. Image Anal. 2021 challenge paper.

Purpose (D29): obtain the **exam→patient mapping**, or failing that any signal that lets us
*estimate* how many CT scans share a patient — so the S1 residual-leak can be **quantified**, not
merely declared.

---

**Subject:** LNDb — request for exam-to-patient mapping (or duplicate-patient indicators) for a
data-partitioning methodology study

Dear LNDb authors,

We are using the LNDb dataset in a methodological study on how the **data-partition unit**
(patient-level vs random splitting) affects reported pulmonary-nodule classification performance,
as a replication cohort alongside LIDC-IDRI. Thank you for releasing the dataset — the annotations
and challenge materials have been directly useful.

Our study's core requirement is **patient-level grouping**: every scan from a given patient must
fall entirely within one cross-validation fold, otherwise the "grouped" arm leaks. In the public
release we can identify each CT scan by `LNDbID`, but we have not found a field that links
multiple scans to the same **patient**. The README notes that anonymisation preserved patient
birth year and gender, but these do not appear in the distributed CSVs or the `.mhd` headers.

Could you help us with either of the following?

1. **An exam→patient mapping** — a table linking `LNDbID` values that belong to the same patient
   (even a de-identified group index would suffice); or

2. If that cannot be shared, **any indicator that lets us estimate patient overlap** — e.g. the
   preserved birth-year + gender per `LNDbID`, or simply the total number of *unique patients*
   behind the 294 scans and whether any patient contributed more than one scan.

Either would let us either (a) perform genuine patient-level grouping, or (b) quantify and report
the residual leakage if we must group at the exam level. We will of course cite the LNDb dataset
paper and acknowledge your assistance.

Thank you very much for your time.

Best regards,
Nycolas Mariotto
[affiliation / role]
n.mariotto@unesp.br

---

_Note to self (not part of the email): even a plain "294 scans came from N unique patients, of
which M contributed >1 scan" turns the D29 lower-bound argument from qualitative into quantified._
