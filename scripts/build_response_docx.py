"""Transpose response_to_reviewers.md into the IEEE Access response-to-reviewers .docx template.

The template ships fixed slots (Reviewer#1 Concern #1-3, Reviewer#2 Concern #1-3). We have four
reviewers with 8/8/2/7 concerns, so the slots are regenerated rather than filled -- the STRUCTURE
the template asks for is preserved: (a) the reviewer's concern, (b) author response, (c) author
action. Run with the `geral` env (python-docx).
"""
import re
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = r"E:\NODULES\paper\2_resubmission\response_to_reviewers.md"
OUT = r"E:\NODULES\paper\2_resubmission\Response-to-Reviewers.docx"

ORIG_ID = "Access-2026-27906"
ORIG_TITLE = ("Patient-Level Versus Nodule-Level Data Partitioning in Pulmonary Nodule "
              "Classification: An Empirical Analysis of Performance Bias Across LNDb and "
              "LIDC-IDRI Datasets")

md = open(SRC, encoding="utf-8").read()

# ---------------------------------------------------------------- parse the markdown
def clean(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"\*(.+?)\*", r"\1", t)          # italics
    t = t.replace("§", "Section ")
    return t.strip()

def bullets_to_text(block):
    """Flatten a markdown bullet list into numbered sentences, preserving every sub-point."""
    out = []
    for b in re.findall(r"\n\s*- (.*?)(?=\n\s*- |\Z)", "\n" + block, re.S):
        t = clean(b).replace("→", ":")
        if t:
            out.append(t)
    return "  ".join(f"({i}) {t}" for i, t in enumerate(out, 1))


items = []          # (reviewer, n_within, concern, response, action)
# numbered items: **N.M —** (a) "..."  (b) ...  (c) ...
blocks = re.split(r"\n(?=\*\*\d+\.\d+ —)", md)
for b in blocks:
    m = re.match(r"\*\*(\d+)\.(\d+) —\*\*", b)
    if not m:
        continue
    rev = int(m.group(1))
    body = b[m.end():]
    # Most items have separate (b) and (c). Item 3.2 -- the largest, and the one that matters most
    # since Reviewer 3 is the primary target -- writes "(b/c) Each sub-point:" followed by a bullet
    # list. The first version of this parser matched only "(b)" and "(c)" and silently dropped that
    # whole response, which is how it reached the .docx with an empty Author action.
    pa = re.search(r"\(a\)(.*?)(?=\n\(b)", body, re.S)
    combined = re.search(r"\n\(b/c\)(.*)", body, re.S)
    if combined:
        resp, act = "", bullets_to_text(combined.group(1))
    else:
        pb = re.search(r"\n\(b\)(.*?)(?=\n\(c\)|\Z)", body, re.S)
        pc = re.search(r"\n\(c\)(.*?)(?=\n\n|\Z)", body, re.S)
        resp = clean(pb.group(1)) if pb else ""
        act = clean(pc.group(1)) if pc else ""
    items.append((rev, int(m.group(2)), clean(pa.group(1)) if pa else "", resp, act))

# Reviewer 2 is written as a bullet list -> convert each bullet into a numbered concern
r2 = re.search(r"## Reviewer 2\n(.*?)\n## Reviewer 3", md, re.S)
if r2:
    bullets = re.findall(r"\n- (.*?)(?=\n- |\Z)", "\n" + r2.group(1), re.S)
    for i, bl in enumerate(bullets, start=1):
        parts = re.split(r"→|->", bl, maxsplit=1)
        concern = clean(parts[0])
        action = clean(parts[1]) if len(parts) > 1 else ""
        items.append((2, i, concern, "", action))

items.sort(key=lambda x: (x[0], x[1]))
print("parsed concerns:", {r: sum(1 for i in items if i[0] == r) for r in (1, 2, 3, 4)})

# ---------------------------------------------------------------- build the document
d = docx.Document()
st = d.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)


def para(text="", bold=False, italic=False, space_after=6, colour=None):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if colour:
        r.font.color.rgb = colour
    p.paragraph_format.space_after = Pt(space_after)
    return p


RED = RGBColor(0xC0, 0x00, 0x00)

para(f"Original Manuscript ID: {ORIG_ID}", bold=True, space_after=2)
para(f"Original Article Title: \u201c{ORIG_TITLE}\u201d", bold=True, space_after=2)
para("New Article Title: “Slice-Level Data Partitioning Inflates Pulmonary Nodule "
     "Classification Performance: A Controlled Three-Arm Study”", bold=True, space_after=10)

para("To: IEEE Access Editor", space_after=2)
para("Re: Response to reviewers", space_after=10)

para("Dear Editor,", space_after=6)

para("Thank you for allowing a resubmission of our manuscript, with an opportunity to address the "
     "reviewers' comments.", space_after=6)

# --- the title-change declaration (D63) -------------------------------------------------
para("We note at the outset that the title of this resubmission differs from the original. The "
     "change follows directly from Reviewer 3's central criticism, which observed that \u201cthe "
     "title, abstract, discussion, and conclusion attribute a reported 6\u201314 percentage-point "
     "difference to patient-level versus nodule-level partitioning\u201d while \u201cno matched "
     "nodule-level experiment is performed using the same cohort, labels, preprocessing, "
     "architecture, training procedure, and evaluation set.\u201d We have now performed that "
     "matched experiment. It shows that the contrast named in the original title is not where the "
     "inflation arises, so retaining that title would repeat the very mismatch between claim and "
     "evidence that Reviewer 3 identified. The study, the cohort, the research question and the "
     "author list are unchanged.", space_after=8)

# --- opening letter, carried over from the markdown -------------------------------------
op = re.search(r"## Opening letter.*?\n\n(.*?)\n\n---", md, re.S)
if op:
    for chunk in op.group(1).split("\n\n"):
        para(clean(chunk), space_after=6)

para("We are uploading (a) our point-by-point response to the comments (below), (b) an updated "
     "manuscript with the changes highlighted, and (c) a clean updated manuscript.", space_after=6)
para("Best regards,", space_after=2)
para("N. Mariotto et al.", space_after=12)

para("\u2014" * 30, space_after=10)

cur = None
for rev, n, concern, resp, action in items:
    if rev != cur:
        cur = rev
        h = d.add_paragraph()
        hr = h.add_run(f"Reviewer #{rev}")
        hr.bold = True
        hr.font.size = Pt(13)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)

    para(f"Reviewer #{rev}, Concern # {n}:", bold=True, space_after=2)
    para(concern, italic=True, space_after=4)
    if resp:
        p = d.add_paragraph()
        p.add_run("Author response: ").bold = True
        p.add_run(resp)
        p.paragraph_format.space_after = Pt(4)
    if action:
        p = d.add_paragraph()
        p.add_run("Author action: ").bold = True
        p.add_run(action)
        p.paragraph_format.space_after = Pt(10)

d.add_page_break()
para("NOTE FOR THE AUTHORS \u2014 DELETE BEFORE SUBMITTING", bold=True, colour=RED, space_after=6)
para("This file was generated from response_to_reviewers.md. Everything above is factual and "
     "traceable to artifacts in the repository. Before submitting:", space_after=6)
for t in [
    "2. Rewrite the four passages that still assume a patient-level conclusion (title, abstract "
    "ending, Conclusion, and the Discussion's \u201cMethodological implication\u201d). They must be "
    "changed together, or the paper contradicts itself.",
    "3. Resolve the transformer scope: run it, run it including the nodule-grouped condition, or "
    "decline it explicitly. Concern 1.5 and Reviewer 2's SOTA item are both marked pending.",
    "4. Run the >=3-annotator sensitivity cohort, or state that it is not included.",
    "5. Revise the CRediT statement with the coauthors \u2014 it still describes the original study.",
    "6. Re-read for grammar, and confirm references are in order of citation.",
]:
    para(t, space_after=4)

d.save(OUT)
print("wrote", OUT)
print("concerns transposed:", len(items))
