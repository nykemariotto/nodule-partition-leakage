# Maintaining `manuscript.tex`

Notes for anyone editing or recompiling the manuscript source. These were LaTeX comments until
2026-08-05; they were moved here because the source is published and travels to the publisher, and
a comment is a poor container for anything worth keeping — it ages, and nobody reviews it.

The source itself is kept comment-free. `scripts/strip_comments.py` performs the removal and
refuses to write unless the stripped file produces the identical TeX token stream, so stripping
cannot change the compiled output.

---

## Template and house format

Ported to `ieeeaccess.cls` on 2026-07-26. The preamble, author block, running head,
corresponding-author line and funding line are taken from the originally submitted package (kept
under `paper/1_original_submission/`) so the resubmission matches the accepted house format
exactly. The template machinery — `ieeeaccess.cls`, `IEEEtran.cls/.bst`, `spotcolor.sty`, logos and
the `t1-*` font descriptors — lives beside the `.tex` and is deliberately not tracked.

## `\graphicspath` — do not point outside this directory

Figures live in `paper/2_resubmission/figures/`, next to the `.tex`. **Never** point at
`../outputs/figures`: a relative path leaving the manuscript directory breaks on Overleaf and in
the submission package, where the source is uploaded as one flat project.

Both `figures/` and the project root are declared, in that order. IEEE Access accepts a flat
package — every source file in one directory — so the figures may legitimately end up beside the
`.tex` at packaging time, and declaring both means the same manuscript compiles either way. The
order matters: `figures/` is searched first, so if a stale duplicate is ever left at the root, the
current file still wins.

## `url.sty`

Loaded for the Data and Code Availability statement. That URL is 52 characters and overruns an
88 mm IEEE Access column as `\texttt`, and it cannot be hyphenated by hand without corrupting it.
`url.sty` breaks at `/` and `.` and is safe to load beside whatever the class provides.

## `\nmath{...}` — uppercase Greek inside a caption

IEEE Access captions are bold, so math inside them switches to the bold math version, whose
operators font is `T1/times`. T1 is a **text** encoding and has no Greek, so `\Delta` silently
renders as an acute accent — the Fig. 1 caption once printed `'M = M(pi_B) - M(pi_A)`. Wrap caption
math containing uppercase Greek in `\nmath`, which typesets in the normal math version.

## `\vcite`

Marks a citation verified at its published version (author list, year, venue, DOI checked at
Crossref or the publisher page). It is a plain alias for `\cite`; it exists so the verification
state is visible in the source.

## Float geometry — the 605 pt budget

A double-column float (`figure*`) can only be set at the top of a page or on a page of its own. The
binding constraint is `\dbltopfraction` × `\textheight` = 0.9 × 672 = **605 pt**, and LaTeX weighs
the graphic **box plus its caption**, not the ink inside it.

Fig. 3 was 572 + 107 = 680 pt, so no parameter relaxation could ever place it at the top of a page —
it could only get a page to itself, which is why it kept landing in the back matter. It was
**resized** (`ROW_H` in `src/figures.py`) rather than accommodated.

Fig. 3 is also declared early — right after Fig. 2, not at the head of the section that discusses
it. Declared at its point of discussion it was still queued when the body ended, so the
`\clearpage` before the bibliography put it on its own page *after* Author Contributions, three
pages past the text referring to it.

A `\clearpage` before the bibliography was used as a backstop from 2026-08-03 to 2026-08-04 and
then **removed**: once the figures were sized to place naturally, nothing was ever pending there,
and the forced break left a page holding one line of Author Contributions. The protection it gave
is now a *check* rather than a page break — a post-compile script reads the PDF and reports the
page of every float against the page of REFERENCES. A check costs nothing when there is no failure;
a `\clearpage` costs a page either way. If a future edit makes a float too tall to place, resize it
or shorten its caption; do not reinstate the break.

## Fig. 3 caption ↔ panel coupling (silent failure)

Panel letters are assigned row-major over a grid of (sample units × 2 metrics) rows by
(architectures) columns. With three architectures and two sample units the rows are slice-loss
(a–c), slice-AUC (d–f), nodule-loss (g–i), nodule-AUC (j–l).

**Adding or removing an architecture, or a sample unit, silently relabels every panel, and no gate
detects it.** The caption phrases naming panel letters, plateau values and backbone counts must be
updated in the *same edit* that regenerates the figure. Fig. 2's caption does not have this
coupling — it names rows and columns rather than letters.

## Architecture scope

Three architectures are run: `densenet121`, `efficientnet_b0` and `vit_small`
(`vit_small_patch16_224.augreg_in21k_ft_in1k`). `swin_tiny` exists on disk as a one-run timing
probe and is **not** part of the experiment; `src/artifacts.py` enforces that no artifact can
silently include it.

If a fourth architecture is ever added, three places must change together: the Architectures
sentence in Methods, the count in the Abstract, and the Discussion paragraph on architecture
dependence.

## The bibliography is inlined — do not edit `references.bib` and expect a change

Since 2026-08-10 the manuscript carries a `thebibliography` block generated by `IEEEtran.bst`,
pasted in where `\bibliographystyle` and `\bibliography{references}` used to be. It is no longer
read from the `.bib` at compile time.

**Why.** IEEE wants the LaTeX source alongside the PDF, and Overleaf's *Download Source* exports
only the project files, not the `.bbl`, which is build output. A source that says
`\bibliography{references}` therefore arrives at the publisher with no bibliography unless whoever
compiles it also runs BibTeX — and if they do not, all 27 citations render as `[?]`. The inlined
block removes that dependency.

**Consequence.** Changing a reference is no longer a matter of editing `references.bib`. Either
edit the inlined block directly, or run the cycle again: edit the `.bib`, compile with BibTeX,
take the generated `.bbl`, and paste it back over the block
(`scratchpad/inline_bbl.py` does the paste and refuses unless the number of `\bibitem` entries
matches the number of distinct cited keys).

`references.bib` stays in the repository. It is where the entries were verified, and it is what a
reader would check; it is simply not what the document compiles from.

The one surviving `%` line in the source is the `IEEEtran.bst` version stamp at the head of the
generated block. It is left in place deliberately: it records which style file produced the
bibliography.

**Underscores in DOIs must be escaped as `\_`.** Springer DOIs carry them
(`10.1007/978-3-030-87602-9_19`), and an unescaped one does **not** raise a LaTeX error here — the
class makes `_` harmless in text mode, so the document compiles and the DOI silently prints as
`...-919`, which does not resolve. The failure is invisible in the log and visible only in the
printed reference, so check the PDF, not the source, whenever a DOI is added.

## Equation layout

Two equations are split across lines to fit the single column — they were 87 pt and 112 pt overfull
as one-liners. The underbraces are kept because they carry the route names that the surrounding
prose refers to.
