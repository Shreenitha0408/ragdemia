"""PDF parsing: turn a PDF into a list of *sections*, each holding page-tagged paragraphs.

Why not just ``page.get_text()``?  Plain text extraction loses the two things a
citation-aware RAG system needs most: **where** a sentence is (page) and **what part of
the document** it belongs to (section).  So we work one level lower, on PyMuPDF's
``rawdict`` output, which gives every character with its font, size and bounding box.

Pipeline (all heuristics are font statistics, no ML):

1. **Line reconstruction** - rebuild each line from characters, inserting a space wherever
   the horizontal gap between glyphs exceeds a fraction of the font size.  Some PDFs
   (e.g. IEEE templates) position words without emitting space glyphs, which otherwise
   yields "RetrievalAugmentedGeneration".
2. **Noise removal** - drop rotated text (the arXiv sidebar), text much smaller than the
   body font (figure labels, axis ticks), bare page numbers, and running headers/footers
   that repeat on many pages.
3. **Body font estimation** - the font size that covers the most characters is the body
   text size; headings are recognised *relative* to it.
4. **Heading detection** - a short block that does not end like a sentence and is larger,
   bold, or numbered ("3.2 Results") is a heading.  Section numbers that PDF producers
   emit as a separate block ("2.1" then "Models") are re-attached.
5. **Sectioning** - headings open sections; sections such as *References* are dropped
   because bibliography text matches almost every query lexically and wastes context.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# A space is inserted when the gap between two glyphs exceeds this fraction of font size.
SPACE_GAP_RATIO = 0.12
# Text whose (normalised) content repeats on at least this share of pages is a header/footer.
REPEAT_PAGE_RATIO = 0.5
HEADING_MAX_WORDS = 14
HEADING_MAX_CHARS = 120

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>(?:\d+(?:\.\d+)*\.?)|(?:[IVXL]+\.)|(?:[A-H](?:\.\d+)*\.?))\s+(?P<title>[A-Z].*)$"
)
SECTION_NUMBER_ONLY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVXL]+\.|[A-H](?:\.\d+)*\.?)$")
CAPTION_RE = re.compile(r"^(fig(ure)?|table|algorithm|listing|eq(uation)?)\.?\s*\d", re.I)


@dataclass
class Line:
    text: str
    size: float  # character-weighted mean font size
    bold_ratio: float  # share of characters set in a bold font
    n_chars: int


@dataclass
class Block:
    text: str
    page: int  # 1-based page number, as a human would cite it
    size: float
    bold_ratio: float
    n_lines: int
    y0: float
    y1: float
    page_height: float
    n_chars: int


@dataclass
class Paragraph:
    text: str
    page: int


@dataclass
class Section:
    title: str
    path: str  # breadcrumb, e.g. "2 Methods > 2.1 Models"
    level: int
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def page_start(self) -> int | None:
        return self.paragraphs[0].page if self.paragraphs else None


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_path: str
    num_pages: int
    sections: list[Section]
    body_font_size: float
    empty_pages: list[int]
    sha256: str
    skipped_sections: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for s in self.sections for p in s.paragraphs)


# --------------------------------------------------------------------------- helpers


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_doc_id(path: Path, root: Path | None = None) -> str:
    """Stable, human-readable id derived from the file path (relative to the corpus root)."""
    rel = path.relative_to(root) if root and path.is_relative_to(root) else Path(path.name)
    stem = rel.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9._/-]+", "_", stem)


def _is_bold(span: dict) -> bool:
    font = span.get("font", "")
    return bool(span.get("flags", 0) & 16) or any(
        k in font for k in ("Bold", "CMBX", "Black", "Heavy")
    )


def _line_from_spans(spans: list[dict]) -> Line | None:
    """Rebuild one visual line from glyphs, inserting spaces at large horizontal gaps."""
    out: list[str] = []
    prev = None
    size_weight = 0.0
    bold_chars = 0
    n = 0
    for span in spans:
        bold = _is_bold(span)
        for ch in span.get("chars", []):
            c = ch["c"]
            if prev is not None and c != " " and prev["c"] != " ":
                gap = ch["bbox"][0] - prev["bbox"][2]
                if gap > SPACE_GAP_RATIO * span["size"]:
                    out.append(" ")
            out.append(c)
            prev = ch
            if not c.isspace():
                n += 1
                size_weight += span["size"]
                bold_chars += int(bold)
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    if not text or n == 0:
        return None
    return Line(text=text, size=size_weight / n, bold_ratio=bold_chars / n, n_chars=n)


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines, undoing end-of-line hyphenation ("impres-" + "sive")."""
    text = ""
    for line in lines:
        if not text:
            text = line
        elif re.search(r"[a-z]-$", text) and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}"
    return text


def _normalise_for_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


# --------------------------------------------------------------------------- extraction


def _extract_blocks(doc: pymupdf.Document) -> tuple[list[Block], list[int]]:
    blocks: list[Block] = []
    empty_pages: list[int] = []
    for pno, page in enumerate(doc, start=1):
        raw = page.get_text("rawdict")
        page_has_text = False
        for b in raw["blocks"]:
            if b.get("type") != 0:  # 1 = image block
                continue
            lines: list[Line] = []
            for ln in b["lines"]:
                dx, dy = ln.get("dir", (1, 0))
                if abs(dy) > 0.1 or dx < 0:  # rotated / vertical text (e.g. arXiv sidebar)
                    continue
                line = _line_from_spans(ln["spans"])
                if line:
                    lines.append(line)
            if not lines:
                continue
            page_has_text = True
            n_chars = sum(line.n_chars for line in lines)
            blocks.append(
                Block(
                    text=_join_lines([line.text for line in lines]),
                    page=pno,
                    size=sum(line.size * line.n_chars for line in lines) / n_chars,
                    bold_ratio=sum(line.bold_ratio * line.n_chars for line in lines) / n_chars,
                    n_lines=len(lines),
                    y0=b["bbox"][1],
                    y1=b["bbox"][3],
                    page_height=page.rect.height,
                    n_chars=n_chars,
                )
            )
        if not page_has_text:
            empty_pages.append(pno)
    return blocks, empty_pages


def _body_font_size(blocks: list[Block]) -> float:
    counter: Counter[float] = Counter()
    for b in blocks:
        counter[round(b.size * 2) / 2] += b.n_chars  # bucket to 0.5pt
    return counter.most_common(1)[0][0] if counter else 10.0


def _remove_noise(blocks: list[Block], n_pages: int, body: float, min_ratio: float) -> list[Block]:
    # Running headers / footers: same text (digits ignored) in the top or bottom margin
    # of many pages.
    margin_texts: Counter[str] = Counter()
    for b in blocks:
        in_margin = b.y1 < 0.1 * b.page_height or b.y0 > 0.9 * b.page_height
        if in_margin:
            margin_texts[_normalise_for_repeat(b.text)] += 1
    repeated = {
        t for t, c in margin_texts.items() if n_pages >= 3 and c >= REPEAT_PAGE_RATIO * n_pages
    }

    kept = []
    for b in blocks:
        if b.size < min_ratio * body:
            continue
        # Bare page numbers like "7", "- 7 -", "iii" - but keep section numbers like "2.1"
        # (unless they sit in the footer), which the heading merger needs.
        is_number_like = re.fullmatch(r"[\divxlc\s.\-–/]+", b.text.lower()) and len(b.text) <= 8
        if is_number_like and (
            not SECTION_NUMBER_ONLY_RE.match(b.text) or b.y0 > 0.9 * b.page_height
        ):
            continue
        if _normalise_for_repeat(b.text) in repeated:
            continue
        kept.append(b)
    return kept


def _plausible_number(num: str) -> bool:
    """Section numbers are small ("3", "4.2", "B.1"); figure text like "20 Total ..." is not."""
    first = num.rstrip(".").split(".")[0]
    return not first.isdigit() or 1 <= int(first) <= 15


def _looks_like_heading(b: Block, body: float, skip_keys: set[str]) -> bool:
    text = b.text.strip()
    words = text.split()
    if not words or len(words) > HEADING_MAX_WORDS or len(text) > HEADING_MAX_CHARS:
        return False
    # Sections we explicitly exclude ("REFERENCES" in small caps) must always be caught,
    # whatever their styling, otherwise the bibliography leaks into the index.
    if _heading_key(text) in skip_keys and b.n_lines == 1:
        return True
    if b.n_lines > 3 or text[-1] in ".,;:" or CAPTION_RE.match(text) or "@" in text:
        return False
    if re.search(r"[=∼~|∑∈≤≥†∗‡§]", text) or re.search(r"\(\d+\)$", text):
        return False  # equations, author affiliations
    if re.search(r"\s\d+$", text):
        return False  # table-of-contents entry ending in a page number
    m = NUMBERED_HEADING_RE.match(text)
    if m and not _plausible_number(m.group("num")):
        return False
    core = m.group("title") if m else text  # judge the title, not its section number
    letters = sum(c.isalpha() for c in core)
    if letters < 3 or letters / len(core) < 0.6:
        return False
    numbered = bool(m) and len(words) <= 10
    # Small-caps headings ("II. OVERVIEW") are set in smaller glyphs, so the size floor
    # only applies to un-numbered candidates - which is where figure labels live.
    if not numbered and b.size < body * 0.95:
        return False
    larger = b.size >= body * 1.12
    bold = b.bold_ratio >= 0.8
    return larger or bold or numbered


def _heading_level(text: str, size: float, body: float, roman_style: bool) -> int:
    m = NUMBERED_HEADING_RE.match(text)
    if m:
        num = m.group("num").rstrip(".")
        if re.fullmatch(r"[IVXL]+", num):
            return 1
        if roman_style and re.fullmatch(r"[A-H]", num):
            return 2  # IEEE style: "II. OVERVIEW" > "A. Naive RAG"
        return num.count(".") + 1
    if size >= body * 1.4:
        return 1
    return 2


def _merge_split_numbers(blocks: list[Block]) -> list[Block]:
    """Re-attach section numbers that the PDF emitted as their own block ("2.1" / "Models")."""
    merged: list[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if (
            SECTION_NUMBER_ONLY_RE.match(b.text.strip())
            and i + 1 < len(blocks)
            and blocks[i + 1].page == b.page
            and blocks[i + 1].text[:1].isupper()
        ):
            nxt = blocks[i + 1]
            merged.append(
                Block(
                    text=f"{b.text.strip()} {nxt.text}",
                    page=nxt.page,
                    size=max(b.size, nxt.size),
                    bold_ratio=min(b.bold_ratio, nxt.bold_ratio),
                    n_lines=nxt.n_lines,
                    y0=b.y0,
                    y1=nxt.y1,
                    page_height=nxt.page_height,
                    n_chars=b.n_chars + nxt.n_chars,
                )
            )
            i += 2
            continue
        merged.append(b)
        i += 1
    return merged


def _extract_title(doc: pymupdf.Document, blocks: list[Block], fallback: str) -> str:
    meta_title = (doc.metadata or {}).get("title", "") or ""
    if (
        meta_title
        and len(meta_title) > 8
        and not re.search(r"microsoft|untitled|\.docx?|\.pdf", meta_title, re.I)
    ):
        return meta_title.strip()
    first_page = [b for b in blocks if b.page == 1 and len(b.text) > 3]
    if not first_page:
        return fallback
    top = max(b.size for b in first_page)
    title_blocks = [b.text for b in first_page if abs(b.size - top) < 0.5][:3]
    title = " ".join(title_blocks).strip()
    return title if 3 < len(title) < 250 else fallback


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_key(text: str) -> str:
    """'7 References' -> 'references' (used for skip-section matching)."""
    m = NUMBERED_HEADING_RE.match(text)
    core = m.group("title") if m else text
    return re.sub(r"[^a-z]", "", core.lower())


# --------------------------------------------------------------------------- public API


def parse_pdf(
    path: str | Path,
    *,
    root: str | Path | None = None,
    min_font_ratio: float = 0.7,
    skip_sections: list[str] | None = None,
) -> ParsedDocument:
    """Parse one PDF into sections of page-tagged paragraphs."""
    path = Path(path)
    skip = {re.sub(r"[^a-z]", "", s.lower()) for s in (skip_sections or [])}
    doc = pymupdf.open(path)
    try:
        blocks, empty_pages = _extract_blocks(doc)
        body = _body_font_size(blocks)
        blocks = _remove_noise(blocks, len(doc), body, min_font_ratio)
        blocks = _merge_split_numbers(blocks)
        title = _extract_title(doc, blocks, fallback=path.stem)
        n_pages = len(doc)
    finally:
        doc.close()

    sections: list[Section] = [Section(title="Front matter", path="Front matter", level=1)]
    stack: list[tuple[int, str]] = []  # (level, heading) for the breadcrumb
    skipped: list[str] = []
    skipping = False
    title_norm = re.sub(r"\W", "", title.lower())
    roman_style = any(re.match(r"^(I|II|III)\.\s+[A-Z]", b.text) for b in blocks)
    in_front_matter = True  # author lists etc. before the abstract / first numbered section

    for b in blocks:
        text = b.text.strip()
        if b.page == 1 and title_norm and re.sub(r"\W", "", text.lower()) in title_norm:
            continue  # the title itself is document metadata, not content
        if _looks_like_heading(b, body, skip):
            heading = _clean_heading(text)
            is_anchor = _heading_key(heading) in {"abstract", "introduction"} or bool(
                NUMBERED_HEADING_RE.match(heading)
            )
            if in_front_matter and not is_anchor:
                sections[-1].paragraphs.append(Paragraph(text=text, page=b.page))
                continue
            in_front_matter = False
            level = _heading_level(heading, b.size, body, roman_style)
            skipping = _heading_key(heading) in skip
            if skipping:
                skipped.append(heading)
                continue
            stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
            sections.append(
                Section(title=heading, path=" > ".join(h for _, h in stack), level=level)
            )
            continue
        if skipping:
            continue
        sections[-1].paragraphs.append(Paragraph(text=text, page=b.page))

    sections = [s for s in sections if s.paragraphs]
    for s in sections:
        # Drop caps: "I. INTRODUCTION L" + "arge language models..." -> move the letter back.
        m = re.search(r"([A-Z]{3,}) ([A-Z])$", s.title)
        first_word = s.paragraphs[0].text.split(" ", 1)[0]
        if m and (first_word[:1].islower() or (len(first_word) >= 2 and first_word.isupper())):
            s.title = s.title[:-2]
            s.path = s.path[:-2]
            s.paragraphs[0].text = m.group(2) + s.paragraphs[0].text
    return ParsedDocument(
        doc_id=make_doc_id(path, Path(root) if root else None),
        title=title,
        source_path=str(path),
        num_pages=n_pages,
        sections=sections,
        body_font_size=body,
        empty_pages=empty_pages,
        sha256=file_sha256(path),
        skipped_sections=skipped,
    )


def discover_pdfs(raw_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() == ".pdf")
