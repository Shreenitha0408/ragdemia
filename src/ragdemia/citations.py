"""Citation verification and post-hoc attribution.

The LLM is *asked* to cite; this module *checks* what it actually did, without a second
model call:

1. **Explicit citations** - parse ``[2]``, ``[1][3]``, ``[1, 3]``, ``[1-3]``; flag numbers that
   point at no source (a hallucination signal).
2. **Sentence-level attribution** - for every answer sentence, measure how much of its
   content is lexically present in each source (content-word recall).  This gives:

   * a *support* map ``sentence -> best source``, used to auto-attribute sentences the model
     forgot to cite (shown to the user as "auto" citations, never disguised as the model's);
   * a cheap **groundedness** score: the share of sentences whose content words are mostly
     found in the retrieved sources.  A sentence with no lexical support at all is a
     candidate hallucination.

Lexical overlap is a deliberate choice over an NLI model: it is transparent, instant and
deterministic.  Its known blind spot is paraphrase (a correct but reworded sentence scores
low), so the score is a *lower bound* on true groundedness - documented in the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bm25 import tokenize
from .chunking import split_sentences

_CITE_RE = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")

ABSTENTION_PATTERNS = re.compile(
    r"couldn['’]?t find|could not find|can(?:no|')t find|not (?:found|mentioned|provided|stated)"
    r" in|do(?:es)? not (?:contain|mention|provide)|don['’]?t (?:know|have)|do not (?:know|have)"
    r"|no information|not enough information|unable to (?:find|answer)"
    r"|i(?: am|'m) not (?:sure|confident)",
    re.I,
)

SUPPORT_THRESHOLD = 0.6  # >= 60% of a sentence's content words appear in one source


@dataclass
class SentenceAttribution:
    sentence: str
    cited: list[int]  # sources the model cited for this sentence
    best_source: int | None  # source with highest content-word recall (1-based)
    support: float  # that recall, 0..1

    @property
    def supported(self) -> bool:
        return self.support >= SUPPORT_THRESHOLD


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)  # model-written citations (1-based, valid)
    invalid: list[int] = field(default_factory=list)  # citations to non-existent sources
    auto_attributed: list[int] = field(default_factory=list)  # added by post-hoc attribution
    sentences: list[SentenceAttribution] = field(default_factory=list)
    abstained: bool = False

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def coverage(self) -> float:
        """Share of sentences carrying a model-written citation."""
        if not self.sentences:
            return 0.0
        return sum(1 for s in self.sentences if s.cited) / len(self.sentences)

    @property
    def groundedness(self) -> float:
        """Share of sentences lexically supported by at least one retrieved source."""
        if not self.sentences:
            return 0.0
        return sum(1 for s in self.sentences if s.supported) / len(self.sentences)

    @property
    def all_sources(self) -> list[int]:
        return self.cited + [n for n in self.auto_attributed if n not in self.cited]


def _expand(group: str) -> list[int]:
    nums: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[-–]", part):
            a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            nums.extend(range(min(a, b), max(a, b) + 1) if abs(b - a) < 20 else [a, b])
        elif part.strip().isdigit():
            nums.append(int(part))
    return nums


def extract_citations(text: str) -> list[int]:
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        for n in _expand(m.group(1)):
            if n not in out:
                out.append(n)
    return out


def is_abstention(text: str) -> bool:
    t = text.strip()
    return bool(ABSTENTION_PATTERNS.search(t[:300])) and len(t) < 600


def answer_sentences(answer: str) -> list[str]:
    # Models often write "... text. [2]" - move trailing citations inside the sentence
    # ("... text [2].") so the splitter keeps them attached to the claim they support.
    text = re.sub(r"\*\*|__", "", answer)  # markdown bold
    text = re.sub(
        r"([.!?])\s*((?:\[[\d,\s\-–]+\]\s*)+)",
        lambda m: f" {m.group(2).strip()}{m.group(1)} ",
        text,
    )
    lines = [ln.strip(" -*•\t") for ln in text.splitlines() if ln.strip()]
    sents = [s for ln in lines for s in split_sentences(ln)]
    return [s for s in sents if len(tokenize(_CITE_RE.sub("", s))) >= 3]


def _recall(sentence_tokens: set[str], source_tokens: set[str]) -> float:
    return len(sentence_tokens & source_tokens) / len(sentence_tokens) if sentence_tokens else 0.0


def check_citations(answer: str, sources: list[str]) -> CitationReport:
    """Verify citations in ``answer`` against the ``sources`` texts that were shown to the LLM."""
    n_sources = len(sources)
    cited_all = extract_citations(answer)
    report = CitationReport(
        cited=[c for c in cited_all if 1 <= c <= n_sources],
        invalid=[c for c in cited_all if not 1 <= c <= n_sources],
        abstained=is_abstention(answer),
    )
    if report.abstained:
        return report

    source_tokens = [set(tokenize(s)) for s in sources]
    for sent in answer_sentences(answer):
        toks = set(tokenize(_CITE_RE.sub("", sent)))
        recalls = [_recall(toks, st) for st in source_tokens]
        best = max(range(n_sources), key=lambda i: recalls[i]) if n_sources else None
        attribution = SentenceAttribution(
            sentence=sent,
            cited=[c for c in extract_citations(sent) if 1 <= c <= n_sources],
            best_source=(best + 1) if best is not None else None,
            support=recalls[best] if best is not None else 0.0,
        )
        report.sentences.append(attribution)
        if (
            not attribution.cited
            and attribution.supported
            and attribution.best_source
            and attribution.best_source not in report.cited + report.auto_attributed
        ):
            report.auto_attributed.append(attribution.best_source)
    return report
