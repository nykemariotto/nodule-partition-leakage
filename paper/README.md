# `paper/` — two packages, deliberately separated

Split on 2026-07-26 to make it structurally impossible to confuse the manuscript that was
**rejected** with the one being **rebuilt**. They share a title lineage and a template, and nothing
else: different scope, different cohort, different numbers, different conclusions.

```
paper/
├── reviewer_comments.md      <- SHARED. Verbatim reviews of the original. The bridge between the two.
├── 1_original_submission/    <- FROZEN. Never edit.
└── 2_resubmission/           <- ACTIVE. All current work happens here.
```

## `1_original_submission/` — FROZEN, read-only by convention

Exactly what was submitted as **Access-2026-27906** (June 2026) and what the four reviewers read,
extracted verbatim from the submission ZIP (kept alongside). **Do not edit anything in here.** Its
only purposes are:

- to check what the reviewers actually saw when interpreting a comment;
- to copy the house-format machinery (author block, `\history`, `\corresp`, funding line);
- to preserve the record of what was claimed, so the rebuild can be compared against it.

`manuscript.tex` here describes the **OLD three-scenario study** (LNDb multiclass + LIDC binary +
end-to-end segmentation pipeline, 10 architectures, ensembles). That scope was **removed** under
DECISIONS D34 — the resubmission is the controlled partition-unit experiment only. Numbers in this
file (ensemble accuracies 94.0 / 88.1 / 79.9 %) belong to the old study and **must never be reused**.

`fig1.jpg`, `fig1.png`, `fig2.png`, `fig3.png`, `fig4.png` are the original figures. They were
deliberately **NOT** carried into the resubmission: the figures are being rebuilt manually, and
`fig4.png` is the one whose provenance is tied to the AI-disclosure question. Two of them also have
mismatched extensions (`fig1.jpg` is actually a PNG; `fig2/fig3.png` are actually JPEG), so anything
reused from here must be renamed to its real format first.

## `2_resubmission/` — ACTIVE

The rebuilt study. `manuscript.tex` here is the only manuscript under development.

- `manuscript.tex` — ported to `ieeeaccess.cls` on 2026-07-26; carries a pre-submission checklist at
  the end of the file.
- `references.bib` — 24 keys, every one source-verified at its published version; the author-list
  policy is documented at the foot of the file.
- `figures/` — **all figures must live here** (`\graphicspath{{figures/}}`). Do not point at
  `../../outputs/figures`: paths outside the manuscript directory break on Overleaf and in the
  submission package.
- `response_to_reviewers.md` — the reply being written against `../reviewer_comments.md`.
- template machinery (`ieeeaccess.cls`, `IEEEtran.cls/.bst`, `spotcolor.sty`, logos, `t1-*` fonts)
  copied from the original package so the project compiles standalone.

## Compiling

Overleaf, `pdfLaTeX`, main document `manuscript.tex`. Upload **only** `2_resubmission/` — the
original package must never end up in the same project, or two files named `manuscript.tex` will be
in play at once, which is precisely the failure this split exists to prevent.

## SYNC RULE — local is the source of truth (set 2026-07-26)

`2_resubmission/manuscript.tex` **on disk** is authoritative. Overleaf is used **only to compile and
read the PDF**.

- Edits are made here, then the single changed file is uploaded to Overleaf (Upload → Overwrite).
- Do **not** edit in the Overleaf editor while this mode is in force: the two copies diverge
  silently and reconciling `.tex` by hand is painful.
- If this ever flips — e.g. when the advisor and coauthors start reviewing in Overleaf — download
  the `.tex` from Overleaf and overwrite the local copy **before** any further local edit, and
  record the flip here.

First successful compile: 2026-07-26, 10 pages, 0 errors, 0 undefined references.
