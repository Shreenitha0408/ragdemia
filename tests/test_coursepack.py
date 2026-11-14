"""Word/PowerPoint-exported course packs: heading levels by size, table cells never headings."""

from __future__ import annotations

import pymupdf

from ragdemia.parsing import parse_pdf

TEXT = "Tokenization splits raw text into units that a model can process. " * 5


def make_course_pack(path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 200), "LARGE LANGUAGE MODELS", fontsize=36, fontname="hebo")
    # page 2: an administrative page with a ruled table whose header cells are bold
    p = doc.new_page()
    p.insert_text((72, 90), "Lecture Plan", fontsize=32, fontname="hebo")
    x = [72, 200, 330, 460]
    ys = [130, 160, 190, 220, 250]
    for y in ys:
        p.draw_line((x[0], y), (x[-1], y))
    for xx in x:
        p.draw_line((xx, ys[0]), (xx, ys[-1]))
    # two bold header rows, like the real lecture plan ("PROPOSED / ACTUAL PERIOD")
    for r, row in enumerate([["TOPIC", "PERIODS", "MODE"], ["Proposed", "Actual", "Delivery"]]):
        for i, h in enumerate(row):
            p.insert_text((x[i] + 5, ys[r] + 20), h, fontsize=11, fontname="hebo")
    for r, row in enumerate([["Intro", "1", "ICT"], ["BPE", "2", "Board"]], start=2):
        for i, cell in enumerate(row):
            p.insert_text((x[i] + 5, ys[r] + 20), cell, fontsize=11, fontname="helv")
    p.insert_textbox(pymupdf.Rect(72, 270, 520, 430), TEXT, fontsize=10.5, fontname="helv")
    # page 3: content with lost multi-level numbering - chapter 14pt, subsections 12pt
    p = doc.new_page()
    y = 72
    for heading, size in [
        ("1. Introduction to NLP", 14),
        ("1. What is NLP?", 12),
        ("2. A Brief History", 12),
        ("2. Words and Tokens", 14),
        ("1. What is a Word?", 12),
    ]:
        p.insert_text((72, y), heading, fontsize=size, fontname="hebo")
        p.insert_textbox(
            pymupdf.Rect(72, y + 8, 520, y + 120), TEXT, fontsize=10.5, fontname="helv"
        )
        y += 135
    doc.save(path)
    return path


def test_size_ranked_levels_and_table_cells(tmp_path):
    d = parse_pdf(make_course_pack(tmp_path / "pack.pdf"))
    paths = [s.path for s in d.sections]
    assert "1. Introduction to NLP > 1. What is NLP?" in paths
    assert "1. Introduction to NLP > 2. A Brief History" in paths
    assert "2. Words and Tokens > 1. What is a Word?" in paths
    # bold table header cells are not headings
    assert not any("TOPIC" in s.title or "Proposed" in s.title for s in d.sections)
    # the admin page title does not become the parent of the content chapters
    assert all(not p.startswith("Lecture Plan >") for p in paths if "NLP" in p)
