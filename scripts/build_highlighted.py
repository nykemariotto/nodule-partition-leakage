"""Produce the 'manuscript with changes highlighted' file required by the IEEE Access
resubmission checklist (item 7).

The usual form of this file marks edited sentences. That form is useless here: the manuscript was
rebuilt, and 91% of its sentences (253 of 278) do not appear in the original at all. Highlighting
everything conveys nothing.

So this builds the artifact that actually helps a reviewer: a cover page stating the rebuild and
quantifying it, followed by the manuscript with the passages that ANSWER SPECIFIC CONCERNS
highlighted and annotated with the concern number. The reviewer can then jump straight to their own
comment instead of re-reading a document that is new throughout.

Run with the `geral` env (PyMuPDF).
"""
import fitz

SRC = r"E:\NODULES\paper\2_resubmission\overleaf_upload.pdf"
OUT = r"E:\NODULES\paper\2_resubmission\Manuscript_highlighted-changes.pdf"

# anchor phrase -> which reviewer concern(s) it answers. Anchors are short and distinctive so the
# search does not straddle a line break; PyMuPDF matches across lines but short anchors are safer.
ANCHORS = [
    ("The experiment is a matched comparison of three arms",      "R1.1, R3.1 - controlled experiment replaces the literature comparison"),
    ("Why the third arm",                                          "R3.1 - isolates which route produces the inflation"),
    ("Matched-rows control",                                       "R3.1 - the gap is not a test-set artefact"),
    ("design is repeated over R = 3 repeats",                     "R1.2 - repeated cross-validation, several seeds"),
    ("Nadeau–Bengio correction",                              "R1.3 - corrected confidence intervals"),
    ("exact attainable minimum",                                   "R1.3 - significance test with its floor stated"),
    ("Preprocessing is identical across all",                      "R1.4 - preprocessing cannot confound the contrast"),
    ("Architecture dependence",                                    "R1.5 - scope of architecture coverage stated explicitly"),
    ("external, multi-institutional validation",                   "R1.6 - external validation declared out of scope"),
    ("All metrics reported in this paper are computed on the",     "R1.7, R3.2 - no result is reported on a validation split"),
    ("Why the partition unit, and not stratification",             "R1.8 - variance is driven by case composition, not class balance"),
    ("Related work",                                               "R2 - the missing related-work section"),
    ("Formalising the leakage",                                    "R2 - the mechanism stated as formulae"),
    ("Acquisition characteristics",                                "R2, R3.2 - dataset detail and the acquisition channel"),
    ("Counting (annotation vs. nodule)",                           "R3.2 - annotation and nodule counts reported separately"),
    ("Patient grouping",                                           "R3.2 - zero patient overlap asserted and exported"),
    ("Label definition",                                           "R3.2 - how multiple scores become one label"),
    ("Reproducibility and integrity safeguards",                   "R3.2 - numerical inconsistencies cannot recur"),
    ("Which route produces the inflation",                         "R3.1 - the three-arm result"),
    ("2D backbones on volumetric CT",                              "R4.3 - the 2D-for-3D limitation"),
    ("high-agreement subcohort",                                   "R4.4 - label-noise sensitivity analysis"),
    ("Learning dynamics",                                          "R4.5 - training and validation curves"),
    ("nodule-level confusion matrices",                            "R4.6 - confusion matrices provided"),
]

doc = fitz.open(SRC)

# ---- cover page ---------------------------------------------------------------------------
cover = doc.new_page(0, width=doc[1].rect.width, height=doc[1].rect.height)
y = 70
BOX = 400          # generous height; the ACTUAL height used is measured from the return value


def line(txt, size=10.5, bold=False, gap=8, colour=(0, 0, 0)):
    """Write a block and advance y by the height it actually occupied, plus `gap`.

    The previous version advanced by a fixed amount, so a title that wrapped to three lines was
    overwritten by the block beneath it -- visible on the cover as two collided lines. `insert_textbox`
    returns the unused height of the rectangle, which gives the used height exactly.
    """
    global y
    r = fitz.Rect(60, y, cover.rect.width - 60, y + BOX)
    left = cover.insert_textbox(r, txt, fontname="hebo" if bold else "helv", fontsize=size,
                                color=colour, align=0)
    y += (BOX - left) + gap


# Straight quotes on purpose: the base-14 "helv"/"hebo" fonts render U+201C/U+201D as "?" here, and a
# cover page full of question marks is worse than typographically plain quotation marks.
line("Manuscript with changes highlighted", 15, True, 16)
line("Original Manuscript ID: Access-2026-27906", 10.5, True, 8)
line('Original title: "Patient-Level Versus Nodule-Level Data Partitioning in Pulmonary Nodule '
     'Classification: An Empirical Analysis of Performance Bias Across LNDb and LIDC-IDRI Datasets"',
     10, False, 8)
line('New title: "Slice-Level Data Partitioning Inflates Pulmonary Nodule Classification '
     'Performance: A Controlled Three-Arm Study"', 10, True, 20)

body = (
"This file is submitted in fulfilment of checklist item 7. It departs from the usual form of a "
"highlighted-changes document, and we explain why rather than leave the reviewer to infer it.\n\n"
"The manuscript was not edited; it was rebuilt. Reviewer 3 observed that the central claim was not "
"tested by the experimental design, and we agreed. The three-scenario comparison was removed, a "
"controlled experiment was built in its place, and the analysis, the statistics and the "
"reproducibility apparatus were written from scratch around it.\n\n"
"To quantify that rather than assert it, we compared the two compiled manuscripts sentence by "
"sentence, excluding the reference list, counting a sentence as unchanged only when it matches the "
"original exactly after normalising case and punctuation. On that measure 284 of the 290 sentences "
"in this manuscript (98%) do not appear in the submitted version in any form, and 6 are unchanged. "
"Highlighting every changed sentence would therefore highlight almost the entire document and would "
"tell the reviewer nothing.\n\n"
"We have instead highlighted the passages that answer specific concerns, each annotated with the "
"concern it addresses. Hovering or clicking a highlight shows the note. Everything not highlighted "
"should also be read as new.\n\n"
"The point-by-point response is in the accompanying Response to Reviewers document."
)
cover.insert_textbox(fitz.Rect(60, y, cover.rect.width - 60, cover.rect.height - 70),
                     body, fontname="helv", fontsize=10.5, align=0)

# ---- highlight the anchors ----------------------------------------------------------------
hits, misses = 0, []
for anchor, note in ANCHORS:
    found = False
    for pno in range(1, doc.page_count):          # page 0 is the cover we just inserted
        page = doc[pno]
        areas = page.search_for(anchor, quads=False)
        if areas:
            a = page.add_highlight_annot(areas[0])
            a.set_info(title="Response to reviewers", content=note)
            a.set_colors(stroke=(1, 0.92, 0.35))
            a.update()
            hits += 1
            found = True
            break
    if not found:
        misses.append(anchor)

doc.save(OUT, garbage=3, deflate=True)
print(f"wrote {OUT}")
print(f"highlighted {hits}/{len(ANCHORS)} anchors, {doc.page_count} pages")
if misses:
    print("NOT FOUND (fix the anchor or the text moved):")
    for m in misses:
        print("   -", m)
