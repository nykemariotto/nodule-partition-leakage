"""Produce the 'manuscript with changes highlighted' file required by the IEEE Access
resubmission checklist (item 7).

The usual form of this file marks edited sentences. That form is useless here: the manuscript was
rebuilt, and 98% of its sentences (291 of 297) do not appear in the original at all. The cover
page recomputes that figure, so this docstring is a summary rather than the source. Highlighting
everything conveys nothing.

So this builds the artifact that actually helps a reviewer: a cover page stating the rebuild and
quantifying it, followed by the manuscript with the passages that ANSWER SPECIFIC CONCERNS
highlighted and annotated with the concern number. The reviewer can then jump straight to their own
comment instead of re-reading a document that is new throughout.

Run with the `geral` env (PyMuPDF).
"""
import re

import fitz

SRC = r"E:\NODULES\paper\2_resubmission\overleaf_upload.pdf"
OLD = r"E:\NODULES\paper\1_original_submission\submitted_manuscript.pdf"
OUT = r"E:\NODULES\paper\2_resubmission\Manuscript_highlighted-changes.pdf"


def _prose(path):
    """Normalised sentences of a compiled manuscript, references and biographies excluded.

    Both are cut because they are not prose of the paper and would distort the count: the reference
    list is bibliographic data and the biographies were carried over unchanged from the original.
    Fragments under 40 characters are dropped -- headings, caption labels and page furniture.
    """
    txt = "\n".join(p.get_text() for p in fitz.open(path))
    m = re.search(r"\nREFERENCES\s*\n", txt)
    if m:
        txt = txt[:m.start()]
    txt = re.sub(r"\s+", " ", txt.replace("-\n", ""))       # join hyphenation, collapse whitespace
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", txt):
        if len(s.strip()) < 40:
            continue
        norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()
        if norm:
            out.append(norm)
    return out


def rebuild_extent():
    """(n_changed, n_total, pct, n_same) -- COMPUTED, because this figure goes to the reviewer.

    It used to be four literals in the cover text. They were written once and never revisited, so
    every later edit to the manuscript made the statement on the cover page a little less true --
    the exact defect class the project's traceability rule exists to prevent (D35).
    """
    new, old = _prose(SRC), set(_prose(OLD))
    same = sum(1 for s in new if s in old)
    return len(new) - same, len(new), round(100 * (len(new) - same) / len(new)), same

# anchor phrase -> which reviewer concern(s) it answers. Anchors are short and distinctive so the
# search does not straddle a line break; PyMuPDF matches across lines but short anchors are safer.
ANCHORS = [
    # Labels are written out in full and use the same wording as the response document
    # ("Reviewer #N, Concern # M"), because "R2.1" is our shorthand and a reviewer has no reason to
    # decode it. Each one names the item it answers, so the reader can go straight to it there.
    ("The experiment is a matched comparison of three arms",      "Reviewer 1 Concern 1; Reviewer 3 Concern 1 - controlled experiment replaces the literature comparison"),
    ("Why the third arm",                                          "Reviewer 3 Concern 1 - isolates which route produces the inflation"),
    ("Matched-rows control",                                       "Reviewer 3 Concern 1 - the gap is not a test-set artifact"),
    ("design is repeated over R = 3 repeats",                     "Reviewer 1 Concern 2 - repeated cross-validation, several seeds"),
    # Anchor was "Nadeau-Bengio correction" and it stopped matching on the 2026-08-05 recompile:
    # the line broke inside the compound ("Nadeau-\nBengio correction"), and locate() cannot rescue
    # a two-word anchor because its shortening floor is min(3, len(words)) -- there is nothing to
    # trim. Anchored on a longer, hyphen-free run in the same sentence instead.
    ("Because cross-validation fold estimates",                "Reviewer 1 Concern 3 - corrected confidence intervals"),
    ("exact attainable minimum",                                   "Reviewer 1 Concern 3 - significance test with its floor stated"),
    ("Preprocessing is identical across all",                      "Reviewer 1 Concern 4 - preprocessing cannot confound the contrast"),
    ("Architecture dependence",                                    "Reviewer 1 Concern 5 - scope of architecture coverage stated explicitly"),
    ("external, multi-institutional validation",                   "Reviewer 1 Concern 6 - external validation declared out of scope"),
    ("All metrics reported in this paper are computed on the",     "Reviewer 1 Concern 7; Reviewer 3 Concern 2 - no result is reported on a validation split"),
    ("Why the partition unit, and not stratification",             "Reviewer 1 Concern 8 - variance is driven by case composition, not class balance"),
    ("Related work",                                               "Reviewer 2 Concern 4 - the missing related-work section"),
    ("Formalizing the leakage",                                    "Reviewer 2 Concern 5 - the mechanism stated as formulae"),
    ("Acquisition characteristics",                                "Reviewer 2 Concern 9; Reviewer 3 Concern 2 - dataset detail and the acquisition channel"),
    ("Counting (annotation vs. nodule)",                           "Reviewer 3 Concern 2 - annotation and nodule counts reported separately"),
    ("Patient grouping",                                           "Reviewer 3 Concern 2 - zero patient overlap asserted and exported"),
    ("Label definition",                                           "Reviewer 3 Concern 2 - how multiple scores become one label"),
    ("Reproducibility and integrity safeguards",                   "Reviewer 3 Concern 2 - numerical inconsistencies cannot recur"),
    ("Which route produces the inflation",                         "Reviewer 3 Concern 1 - the three-arm result"),
    # The three declared changes of scope. They answer no single concern -- they are consequences of
    # the rebuild that we state rather than let a reviewer discover -- but they are the most
    # consequential edits in the document and were not marked at all until 2026-08-11.
    ("removed the three-scenario comparison",                      "Scope: the three scenarios of the earlier version, including the end-to-end pipeline, were removed rather than repaired"),
    ("LNDb entered the earlier version",                           "Scope: why LNDb is no longer used - it carried a second task, which is the confound Reviewer 3 identified"),
    ("The title has been changed",                                 "Scope: why the title changed - the earlier one named both cohorts and a two-way contrast the three-arm design supersedes"),
    ("the conclusion of this work is not the conclusion",          "Scope: the conclusion of this version is not the conclusion of the earlier one, and we say so rather than let it be noticed"),
    ("2D backbones on volumetric CT",                              "Reviewer 4 Concern 3 - the 2D-for-3D limitation"),
    ("high-agreement subcohort",                                   "Reviewer 4 Concern 4 - label-noise sensitivity analysis"),
    ("Learning dynamics",                                          "Reviewer 4 Concern 5 - training and validation curves"),
    ("nodule-level confusion matrices",                            "Reviewer 4 Concern 6 - confusion matrices provided"),
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

n_changed, n_total, pct, n_same = rebuild_extent()
body = (
"This file is submitted in fulfilment of checklist item 7. It departs from the usual form of a "
"highlighted-changes document, and we explain why rather than leave the reviewer to infer it.\n\n"
"The manuscript was not edited; it was rebuilt. Reviewer 3 observed that the central claim was not "
"tested by the experimental design, and we agreed. The three-scenario comparison was removed, a "
"controlled experiment was built in its place, and the analysis, the statistics and the "
"reproducibility apparatus were written from scratch around it.\n\n"
"To quantify that rather than assert it, we compared the two compiled manuscripts sentence by "
"sentence, excluding the reference list, counting a sentence as unchanged only when it matches the "
f"original exactly after normalising case and punctuation. On that measure {n_changed} of the "
f"{n_total} sentences in this manuscript ({pct}%) are not present in the submitted version, and "
f"{n_same} are unchanged.\n\n"
"Marking every individual change would therefore mean marking almost every line of the article. "
"The result would be extremely cluttered and difficult to follow, which would defeat the purpose "
"of the file. We have instead highlighted the topics where the changes are, so that the reading is "
"easier: each highlight sits at the passage that answers a particular concern and is annotated "
"with the reviewer and concern number it addresses, so a reviewer can go straight to their own "
"comment. Hovering or clicking a highlight shows the note. Everything not highlighted should also "
"be read as new.\n\n"
"The point-by-point response is in the accompanying Response to Reviewers document."
)
cover.insert_textbox(fitz.Rect(60, y, cover.rect.width - 60, cover.rect.height - 70),
                     body, fontname="helv", fontsize=10.5, align=0)

# ---- highlight the anchors ----------------------------------------------------------------
def locate(anchor):
    """(page_no, rect, phrase_used) for an anchor, shortening it word by word if needed.

    LaTeX hyphenates across line breaks, and PyMuPDF's search matches across lines but not across a
    hyphen: "2D backbones on volumetric CT" is unfindable once the pdf holds "volumet-\\nric CT".
    That silently dropped one highlight, and which anchor breaks depends on where the line falls,
    so it changes with every recompile. Shortening from the right until a match is found keeps the
    highlight; the floor of three words stops a stub matching somewhere unintended, and the phrase
    actually used is reported so a suspicious shortening is visible rather than silent.
    """
    words = anchor.split()
    # Never shorten below three words, but ALWAYS try the anchor at its own length: many anchors here
    # are two-word section names, and a floor written as range(len(words), 2, -1) is EMPTY for those,
    # so nine of them were silently never searched.
    floor = min(3, len(words))
    for n in range(len(words), floor - 1, -1):
        phrase = " ".join(words[:n])
        for pno in range(1, doc.page_count):      # page 0 is the cover we just inserted
            areas = doc[pno].search_for(phrase, quads=False)
            if areas:
                return pno, areas[0], phrase
    return None, None, None


hits, misses, shortened = 0, [], []
for anchor, note in ANCHORS:
    pno, rect, phrase = locate(anchor)
    if pno is None:
        misses.append(anchor)
        continue
    # Bind the Page to a name. `doc[pno].add_highlight_annot(...)` lets the temporary Page be
    # collected while the annotation still refers to it, and PyMuPDF then raises "annotation not
    # bound to any page" on the next call against it.
    page = doc[pno]
    a = page.add_highlight_annot(rect)
    a.set_info(title="Response to reviewers", content=note)
    a.set_colors(stroke=(1, 0.92, 0.35))
    a.update()
    hits += 1
    if phrase != anchor:
        shortened.append((anchor, phrase))

if misses:
    # Check BEFORE writing. The abort used to run after doc.save, which left an incomplete file on
    # disk for someone to pick up -- the failure mode the abort exists to prevent, one step later.
    print("NOT FOUND (fix the anchor or the text moved):")
    for m in misses:
        print("   -", m)
    raise SystemExit(f"ABORT: {len(misses)} anchor(s) unmatched; nothing was written.")

doc.save(OUT, garbage=3, deflate=True)
print(f"wrote {OUT}")
print(f"rebuild extent (computed): {n_changed}/{n_total} sentences new ({pct}%), {n_same} unchanged")
print(f"highlighted {hits}/{len(ANCHORS)} anchors, {doc.page_count} pages")
for anchor, phrase in shortened:
    print(f"   shortened to match across a hyphenated break: {anchor!r} -> {phrase!r}")
if misses:
    print("NOT FOUND (fix the anchor or the text moved):")
    for m in misses:
        print("   -", m)
    # Exit non-zero. A missing anchor means a reviewer concern has no highlight pointing at its
    # answer, in the one file whose entire purpose is to point at the answers -- and the loss is
    # invisible in the PDF itself. It has to be impossible to ship this without noticing.
    raise SystemExit(f"ABORT: {len(misses)} anchor(s) unmatched; the highlighted PDF is incomplete.")
