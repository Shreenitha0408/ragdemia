"""Okapi BM25, implemented from scratch (~60 lines) so every term in the formula is visible.

Why have a lexical retriever next to the dense one?  Dense embeddings capture *meaning* but
are weak on exact strings: dataset names ("TriviaQA"), acronyms ("HyDE"), numbers, and the
rare technical terms that academic questions are full of.  BM25 is the opposite.  Fusing the
two (see ``retrieval.py``) is consistently stronger than either alone.

Score of document *d* for query *q*::

    score(d, q) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

    IDF(t) = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )

* ``tf``  - term frequency in the document; saturates (controlled by ``k1``) so the 10th
  occurrence of a word adds much less than the 1st.
* ``b``   - length normalisation; long chunks are not rewarded just for being long.
* ``IDF`` - rare terms ("Jeopardy") weigh far more than common ones ("model").
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_STOPWORDS = frozenset(
    (
        "a an and are as at be but by can did do does for from had has have how i if in into is "
        "it its of on or our over so such than that the their them then there these they this "
        "to was we were what when where which while who whom why will with you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self.idf: dict[str, float] = {}

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    def fit(self, texts: list[str]) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for idx, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, freq in tf.items():
                postings[term].append((idx, freq))
        self.postings = dict(postings)
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(doc_idx, score)`` pairs, best first.  Uses the inverted index, so only
        documents containing at least one query term are scored."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for idx, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # -- persistence (plain JSON: inspectable, no pickle security issues)
    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "doc_len": self.doc_len, "postings": self.postings}

    @classmethod
    def from_dict(cls, d: dict) -> BM25Index:
        obj = cls(k1=d["k1"], b=d["b"])
        obj.doc_len = d["doc_len"]
        obj.postings = {t: [tuple(x) for x in p] for t, p in d["postings"].items()}
        n = len(obj.doc_len)
        obj.avgdl = sum(obj.doc_len) / n if n else 0.0
        obj.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in obj.postings.items()
        }
        return obj
