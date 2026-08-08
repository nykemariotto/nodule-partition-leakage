"""Strip comments from the manuscript source, and prove the stripped file compiles identically.

WHY. The repository is public and the LaTeX source also travels to the publisher in the submission
package, so every comment in it is read by strangers. Comments are the wrong container for anything
worth publishing -- a comment ages and nobody reviews it, which is how three notes saying "numbers
are placeholders" survived into a manuscript whose numbers are final. Technical warnings worth
keeping belong in a maintenance document; coordination between authors belongs outside the
repository altogether.

WHY NOT JUST RECOMPILE AND DIFF. There is no local LaTeX toolchain, and an empirical comparison
would be weaker anyway. This checks TOKEN EQUIVALENCE: it derives, for both the original and the
stripped file, the exact sequence of characters TeX would see, and refuses to write unless the two
are identical. That is a proof rather than a sample.

THE TRAP THIS AVOIDS. A `%` at the end of a line is not decoration -- TeX discards from `%` to the
end of the line INCLUDING the newline, so `\includegraphics{a}%` deliberately suppresses a space.
Deleting that `%` changes the output. Inline comments therefore keep their `%` and lose only the
text after it; only whole-comment lines are deleted, because those contribute nothing at all.

Usage:  python scripts/strip_comments.py <file> [<file> ...] [--apply]
Without --apply it reports and writes nothing.
"""
import io
import os
import re
import sys


def _comment_at(line):
    """Index of the first unescaped % in `line`, or -1.

    A percent is a comment marker only when the run of backslashes before it is even -- `\\%` is a
    literal percent, `\\\\%` is an escaped backslash followed by a comment marker.
    """
    for m in re.finditer("%", line):
        i = m.start()
        back = len(line[:i]) - len(line[:i].rstrip("\\"))
        if back % 2 == 0:
            return i
    return -1


def tex_token_stream(text):
    """What TeX actually sees: a list of (kind, content) per line.

    A whole-comment line yields nothing -- TeX consumes it and its newline. A line with an inline
    comment yields its content with the newline SUPPRESSED. Trailing spaces are dropped because TeX
    drops them at end of line.
    """
    out = []
    for line in text.split("\n"):
        i = _comment_at(line)
        if i == -1:
            out.append(("NL", line.rstrip()))
        elif line[:i].strip() == "":
            continue                       # whole-comment line: contributes nothing
        else:
            out.append(("NONL", line[:i].rstrip()))
    return out


def strip_tex(text):
    kept = []
    for line in text.split("\n"):
        i = _comment_at(line)
        if i == -1:
            kept.append(line)
        elif line[:i].strip() == "":
            continue                       # drop the line entirely
        else:
            kept.append(line[:i].rstrip() + " %")   # keep the marker, drop the note
    return "\n".join(kept)


def strip_bib(text):
    """Drop whole-comment lines that sit OUTSIDE an entry.

    BibTeX ignores anything between entries, which is why `%` works there by convention. Inside an
    entry a percent is ordinary text and must survive, so brace depth is tracked and only depth-0
    comment lines are removed.
    """
    kept, depth = [], 0
    for line in text.split("\n"):
        is_comment = line.lstrip().startswith("%")
        if is_comment and depth == 0:
            continue
        if not is_comment:
            depth += line.count("{") - line.count("}")
            depth = max(depth, 0)
        kept.append(line)
    return "\n".join(kept)


def bib_token_stream(text):
    """BibTeX-visible content: every line that is not a depth-0 comment."""
    out, depth = [], 0
    for line in text.split("\n"):
        is_comment = line.lstrip().startswith("%")
        if is_comment and depth == 0:
            continue
        if not is_comment:
            depth += line.count("{") - line.count("}")
            depth = max(depth, 0)
        out.append(line.rstrip())
    return out


def process(path, apply):
    raw = io.open(path, encoding="utf-8", errors="strict", newline="").read()
    text = raw.replace("\r\n", "\n")
    is_bib = path.lower().endswith(".bib")

    stripped = strip_bib(text) if is_bib else strip_tex(text)
    before = bib_token_stream(text) if is_bib else tex_token_stream(text)
    after = bib_token_stream(stripped) if is_bib else tex_token_stream(stripped)

    n_before = len(text.split("\n"))
    n_after = len(stripped.split("\n"))
    name = os.path.basename(path)

    if before != after:
        # Show the first divergence rather than a bare failure: the point of the check is to be
        # actionable when it fires.
        for k, (a, b) in enumerate(zip(before, after)):
            if a != b:
                print(f"  {name}: *** DIVERGENCIA no token {k}\n      antes: {a!r}\n      depois: {b!r}")
                break
        else:
            print(f"  {name}: *** DIVERGENCIA de comprimento {len(before)} -> {len(after)}")
        return False

    print(f"  {name}: {n_before} -> {n_after} linhas "
          f"(-{n_before - n_after}), {len(raw)} -> {len(stripped)} bytes. "
          f"Equivalencia de tokens: OK ({len(before)} tokens identicos).")

    if apply:
        io.open(path + ".commented", "w", encoding="utf-8", newline="\n").write(text)
        io.open(path, "w", encoding="utf-8", newline="\n").write(stripped)
        print(f"      escrito. original preservado em {name}.commented")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    if not args:
        raise SystemExit(__doc__)
    print("APLICANDO" if apply else "SIMULACAO (use --apply para escrever)")
    ok = all([process(p, apply) for p in args])
    raise SystemExit(0 if ok else 1)
