"""Retrieval: dense (FAISS), lexical (BM25), hybrid (Reciprocal Rank Fusion), optional reranking.

**Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009)::

    RRF(d) = Σ_{r ∈ retrievers}  1 / (k + rank_r(d))

Why fuse *ranks* and not *scores*?  BM25 scores are unbounded (0 .. ~30) while cosine
similarities live in [-1, 1]; adding them requires a fragile normalisation.  Ranks are
already on a common scale.  ``k`` (60 by convention) damps the advantage of rank 1 over
rank 2, so a document that both retrievers rank moderately well beats one that a single
retriever ranks first.

**Cross-encoder reranking** (optional): the bi-encoder embeds query and passage
*separately*, which is fast but cannot model word-level interactions.  A cross-encoder reads
``[query, passage]`` jointly through one transformer and outputs a relevance score - much
more accurate, but O(candidates) forward passes per query, so we only apply it to the
fused top-N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .chunking import Chunk
from .embeddings import Embedder
from .store import IndexStore

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # final score used for ranking (RRF, cosine, BM25 or reranker)
    rank: int
    scores: dict[str, float] = field(default_factory=dict)  # per-retriever diagnostics


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float, dict[str, float]]]:
    """Fuse several ranked lists of ``(item_id, score)``; returns ``(id, rrf, diagnostics)``."""
    fused: dict[int, float] = {}
    diag: dict[int, dict[str, float]] = {}
    for name, ranking in rankings.items():
        for rank, (item, score) in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
            diag.setdefault(item, {})[f"{name}_rank"] = rank
            diag[item][f"{name}_score"] = round(float(score), 4)
    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(i, s, diag[i]) for i, s in order]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder

        from .embeddings import resolve_device

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=resolve_device(device))

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return items
        scores = self._model.predict([(query, it.chunk.embed_text) for it in items])
        for it, s in zip(items, scores, strict=True):
            it.scores["rerank_score"] = round(float(s), 4)
            it.score = float(s)
        items.sort(key=lambda it: it.score, reverse=True)
        for r, it in enumerate(items, start=1):
            it.rank = r
        return items


class Retriever:
    def __init__(
        self,
        store: IndexStore,
        embedder: Embedder,
        *,
        mode: Mode = "hybrid",
        candidates: int = 30,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.reranker = reranker

    def retrieve(
        self, query: str, top_k: int = 5, mode: Mode | None = None
    ) -> list[RetrievedChunk]:
        mode = mode or self.mode
        n = max(top_k, self.candidates)
        rankings: dict[str, list[tuple[int, float]]] = {}
        if mode in ("dense", "hybrid"):
            rankings["dense"] = self.store.dense_search(self.embedder.encode_query(query), n)
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self.store.bm25.search(query, n)

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        else:
            ((name, ranking),) = rankings.items()
            fused = [
                (i, s, {f"{name}_rank": r, f"{name}_score": round(s, 4)})
                for r, (i, s) in enumerate(ranking, start=1)
            ]

        pool = top_k * 3 if self.reranker else top_k
        results = [
            RetrievedChunk(chunk=self.store.chunks[i], score=s, rank=r, scores=d)
            for r, (i, s, d) in enumerate(fused[:pool], start=1)
        ]
        if self.reranker:
            results = self.reranker.rerank(query, results)[:top_k]
        return results
