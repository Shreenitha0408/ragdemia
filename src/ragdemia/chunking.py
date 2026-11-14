"""Structure-aware chunking with sentence-level overlap.

Design choices (each one is a deliberate trade-off you can defend):

* **Chunks never cross a section boundary.**  A chunk that starts in "Methods" and ends in
  "Results" has a blurred embedding and an ambiguous citation.  Keeping chunks inside one
  section makes every chunk *about* one thing and citable as "§3.2, p. 4".
* **Sentences are the atomic unit.**  We pack whole sentences until the token budget is hit,
  so no chunk starts or ends mid-sentence (fixed-size character windows do).
* **Overlap is measured in sentences, bounded by tokens.**  The last few sentences of chunk
  *n* are repeated at the start of chunk *n+1*, so a fact that straddles the boundary is
  fully present in at least one chunk.
* **Token budget uses the embedding model's own tokenizer.**  BGE truncates at 512 tokens;
  counting words instead would silently cut off the end of long chunks.
* **Contextual header.**  Each chunk is embedded as ``"<doc title> | <section path>\\n<text>"``.
  A paragraph that says "our method improves recall by 9%" is useless on its own; with the
  header the embedding knows *which* method and paper it belongs to.
* **Tiny sections are merged forward** so we do not index 10-token fragments.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .parsing import ParsedDocument, Section

TokenCounter = Callable[[str], int]

# Abbreviations that end in "." but do not end a sentence.
_ABBREVIATIONS = {
    "e.g",
    "i.e",
    "et al",
    "etc",
    "fig",
    "figs",
    "eq",
    "eqs",
    "sec",
    "tab",
    "vs",
    "cf",
    "resp",
    "approx",
    "no",
    "dr",
    "prof",
    "mr",
    "mrs",
    "ms",
    "al",
    "ref",
    "refs",
    "appx",
    "ch",
}
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tuned for academic prose.

    Splits after ``. ! ?`` followed by whitespace and an upper-case letter / digit, unless the
    word before the period is a known abbreviation ("et al.", "Fig.") or a single initial
    ("J. Smith").
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        candidate = text[start : m.start()].rstrip()
        last_word = re.findall(r"([A-Za-z][A-Za-z.]*)\.?[\"')\]]*$", candidate)
        lw = last_word[0].lower().rstrip(".") if last_word else ""
        if lw in _ABBREVIATIONS or (len(lw) == 1 and lw.isalpha()):
            continue
        sentences.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def whitespace_token_counter(text: str) -> int:
    """Cheap fallback: subword tokenizers produce ~1.3 tokens per English word."""
    return int(len(text.split()) * 1.3) + 1


def hf_token_counter(model_name: str) -> TokenCounter:
    """Count tokens with the embedding model's own (fast, Rust) tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 1_000_000  # we only *count* here; truncation happens at embed time

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    section: str  # breadcrumb path, e.g. "2 Methods > 2.1 Models"
    page_start: int
    page_end: int
    text: str  # what the LLM sees and what we show the user
    n_tokens: int

    @property
    def embed_text(self) -> str:
        """What the embedding model sees: text plus a contextual header."""
        return f"{self.doc_title} | {self.section}\n{self.text}"

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else (f"pp. {self.page_start}-{self.page_end}")
        )
        return f"{self.doc_title} — {self.section}, {pages}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


@dataclass
class _Sentence:
    text: str
    page: int
    n_tokens: int


def _section_sentences(section: Section, count: TokenCounter) -> list[_Sentence]:
    out: list[_Sentence] = []
    for para in section.paragraphs:
        for s in split_sentences(para.text):
            out.append(_Sentence(s, para.page, count(s)))
    return out


def _split_long_sentence(s: _Sentence, max_tokens: int, count: TokenCounter) -> list[_Sentence]:
    """A 'sentence' longer than the budget (tables, run-on lists) is split on word windows."""
    words = s.text.split()
    pieces: list[_Sentence] = []
    current: list[str] = []
    for w in words:
        current.append(w)
        if count(" ".join(current)) >= max_tokens:
            current.pop()
            if current:
                pieces.append(_Sentence(" ".join(current), s.page, count(" ".join(current))))
            current = [w]
    if current:
        pieces.append(_Sentence(" ".join(current), s.page, count(" ".join(current))))
    return pieces


def _merge_small_sections(sections: list[Section], count: TokenCounter, min_tokens: int):
    """Yield (path, sentences) groups; sections under ``min_tokens`` join the next one."""
    pending_path: list[str] = []
    pending: list[_Sentence] = []
    for sec in sections:
        sents = _section_sentences(sec, count)
        pending.extend(sents)
        pending_path.append(sec.path)
        if sum(s.n_tokens for s in pending) >= min_tokens:
            yield _shared_path(pending_path), pending
            pending, pending_path = [], []
    if pending:
        yield _shared_path(pending_path), pending


def _shared_path(paths: list[str]) -> str:
    if len(paths) == 1:
        return paths[0]
    # ["2 Methods", "2 Methods > 2.1 Models"]          -> "2 Methods > 2.1 Models"
    # ["C Results > C.2 Human", "C Results > C.3 Qual"] -> "C Results > C.2 Human; C.3 Qual"
    parts = [p.split(" > ") for p in dict.fromkeys(paths)]
    common: list[str] = []
    for level in zip(*parts, strict=False):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    leaves = [" > ".join(p[len(common) :]) for p in parts if len(p) > len(common)]
    return " > ".join(common + (["; ".join(leaves)] if leaves else []))


def _make_chunk(
    doc: ParsedDocument, path: str, window: list[_Sentence], n_tokens: int, index: int
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{index:04d}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        source_path=doc.source_path,
        section=path,
        page_start=min(s.page for s in window),
        page_end=max(s.page for s in window),
        text=" ".join(s.text for s in window),
        n_tokens=n_tokens,
    )


def chunk_document(
    doc: ParsedDocument,
    *,
    max_tokens: int = 320,
    overlap_tokens: int = 64,
    min_tokens: int = 40,
    count_tokens: TokenCounter = whitespace_token_counter,
) -> list[Chunk]:
    """Split a parsed document into overlapping, section-bounded chunks."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[Chunk] = []
    header_budget = count_tokens(f"{doc.title} | ") + 24  # reserve room for the context header
    budget = max(32, max_tokens - header_budget)

    for path, sentences in _merge_small_sections(doc.sections, count_tokens, min_tokens):
        units: list[_Sentence] = []
        for s in sentences:
            units.extend(
                _split_long_sentence(s, budget, count_tokens) if s.n_tokens > budget else [s]
            )

        window: list[_Sentence] = []
        window_tokens = 0
        new_since_emit = 0  # sentences added since the last emitted chunk

        for unit in units:
            if window and window_tokens + unit.n_tokens > budget:
                chunks.append(_make_chunk(doc, path, window, window_tokens, len(chunks)))
                # Carry the tail of the window over as overlap.
                carry: list[_Sentence] = []
                carry_tokens = 0
                for s in reversed(window):
                    if carry_tokens + s.n_tokens > overlap_tokens:
                        break
                    carry.insert(0, s)
                    carry_tokens += s.n_tokens
                window, window_tokens, new_since_emit = carry, carry_tokens, 0
            window.append(unit)
            window_tokens += unit.n_tokens
            new_since_emit += 1
        if window and new_since_emit > 0:
            chunks.append(_make_chunk(doc, path, window, window_tokens, len(chunks)))
    return chunks
