"""Flatten the revised manuscript into the single .tex file PLOS requires, and check it.

  python scripts/flatten_manuscript.py

Reads revision/manuscript_main.tex, expands every \\input (sections, generated tables), replaces the
number macros of generated/numbers.tex by their values, and writes revision/manuscript_revised.tex.
Checks: every \\ref has a \\label, every \\cite key is in references.bib, no \\tbd placeholder is left;
\\authorcheck notes (questions to the authors) are listed and must be resolved before submission.
The bibliography still has to be compiled once with BibTeX and the .bbl pasted in (PLOS rule).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[2] / "revision"


def expand(text: str, base: Path) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        path = base / (name if name.endswith(".tex") else f"{name}.tex")
        return expand(path.read_text(encoding="utf-8").rstrip("\n"), base)
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def main() -> int:
    text = expand((REV / "manuscript_main.tex").read_text(encoding="utf-8"), REV)
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (REV / "generated" / "numbers.tex").read_text(encoding="utf-8")))
    text = re.sub(r"^\\newcommand\{\\\w+\}\{.*\}\n?", "", text, flags=re.M)  # drop the expanded macro file
    for name in sorted(macros, key=len, reverse=True):
        text = re.sub(rf"\\{name}(?![A-Za-z])", lambda _: macros[name], text)
    text = text.replace("% Revision helpers (removed by flatten.py)\n", "")
    problems = []
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    for ref in sorted(set(re.findall(r"\\ref\{([^}]+)\}", text)) - labels):
        problems.append(f"undefined reference: {ref}")
    bib = set(re.findall(r"@\w+\{([^,]+),", (REV / "references.bib").read_text(encoding="utf-8")))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for k in group.split(",")}
    for key in sorted(cited - bib):
        problems.append(f"missing bibliography entry: {key}")
    for m in re.findall(r"\\tbd\{([^}]*)\}", text):
        problems.append(f"placeholder left: {m}")
    leftover = [n for n in re.findall(r"\\([a-z]+[A-Z]\w*)(?![A-Za-z])", text) if n in macros]
    if leftover:
        problems.append(f"unexpanded number macros: {sorted(set(leftover))}")
    notes = re.findall(r"\\authorcheck\{([^}]*)\}", text)
    out = REV / "manuscript_revised.tex"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.split())} words incl. markup)")
    for n in notes:
        print("AUTHORS:", n)
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
