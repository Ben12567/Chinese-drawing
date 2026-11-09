"""Convert the response letter (Markdown subset) to .docx for the PLOS submission system.

Supports: # / ## / ### headings, paragraphs, '- ' bullets (two levels), '1. ' numbered items,
'> ' quoted reviewer comments, **bold**, *italic*, `code` (kept, highlighted when it is an
<AUTHORS: ...> placeholder) and '---' rules (ignored).
Usage: python scripts/letter_to_docx.py ../revision/response_letter.md out.docx
"""
from __future__ import annotations

import re
import sys

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Pt

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_runs(par, text: str, italic: bool = False) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_runs_plain(par, part[2:-2], bold=True, italic=italic)
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            run.italic = italic
            if "AUTHORS" in part:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            add_runs_plain(par, part[1:-1], italic=True)
        else:
            add_runs_plain(par, part, italic=italic)


def add_runs_plain(par, text: str, bold: bool = False, italic: bool = False) -> None:
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic


def main(src: str, dst: str) -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    lines = open(src, encoding="utf-8").read().splitlines()
    buf: list[str] = []
    table_rows: list[list[str]] = []
    in_code = False

    def flush():
        if buf:
            add_runs(doc.add_paragraph(), " ".join(buf))
            buf.clear()
        if table_rows:
            ncol = max(len(r) for r in table_rows)
            t = doc.add_table(rows=0, cols=ncol)
            t.style = "Table Grid"
            for r in table_rows:
                cells = t.add_row().cells
                for c, text in zip(cells, r):
                    add_runs(c.paragraphs[0], text)
            table_rows.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            run = doc.add_paragraph().add_run(raw)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                continue
            if buf:
                add_runs(doc.add_paragraph(), " ".join(buf))
                buf.clear()
            table_rows.append(cells)
            continue
        if table_rows:
            flush()
        if not line.strip():
            flush()
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            h = doc.add_heading(level=level)
            add_runs(h, m.group(2))
            continue
        if line.strip() == "---":
            flush()
            continue
        if line.startswith(">"):
            flush()
            p = doc.add_paragraph(style="Quote")
            add_runs(p, line.lstrip("> ").strip(), italic=True)
            continue
        m = re.match(r"^(\s*)- (.*)", line)
        if m:
            flush()
            style = "List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"
            add_runs(doc.add_paragraph(style=style), m.group(2))
            continue
        m = re.match(r"^(\d+)\. (.*)", line)
        if m:
            flush()
            p = doc.add_paragraph()
            add_runs(p, f"{m.group(1)}. {m.group(2)}")
            continue
        buf.append(line.strip())
    flush()
    doc.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
