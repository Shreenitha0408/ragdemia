"""The RAG pipeline: question -> retrieve -> build prompt -> generate -> verify citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .citations import CitationReport, check_citations
from .config import Config, load_config
from .llm import LLM
from .prompts import clean_rewrite, closed_book_messages, rag_messages, rewrite_messages
from .retrieval import Mode, RetrievedChunk, Retriever


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    citations: CitationReport = field(default_factory=CitationReport)
    mode: str = "rag"  # "rag" | "closed_book"
    timings: dict[str, float] = field(default_factory=dict)
    search_query: str | None = None  # standalone rewrite of a follow-up question, if any

    @property
    def cited_sources(self) -> list[tuple[int, RetrievedChunk]]:
        return [
            (n, self.sources[n - 1]) for n in self.citations.all_sources if n <= len(self.sources)
        ]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "search_query": self.search_query,
            "answer": self.text,
            "mode": self.mode,
            "sources": [
                {
                    "n": i,
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "doc_title": r.chunk.doc_title,
                    "section": r.chunk.section,
                    "page_start": r.chunk.page_start,
                    "page_end": r.chunk.page_end,
                    "citation": r.chunk.citation,
                    "score": round(r.score, 4),
                    "scores": r.scores,
                    "cited": i in self.citations.cited,
                    "auto_attributed": i in self.citations.auto_attributed,
                    "text": r.chunk.text,
                }
                for i, r in enumerate(self.sources, start=1)
            ],
            "citations": {
                "cited": self.citations.cited,
                "auto_attributed": self.citations.auto_attributed,
                "invalid": self.citations.invalid,
                "coverage": round(self.citations.coverage, 3),
                "groundedness": round(self.citations.groundedness, 3),
                "abstained": self.citations.abstained,
                "sentences": [
                    {
                        "sentence": a.sentence,
                        "cited": a.cited,
                        "best_source": a.best_source,
                        "support": round(a.support, 3),
                    }
                    for a in self.citations.sentences
                ],
            },
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


History = list[tuple[str, str]]  # [(question, answer), ...] oldest first


class RAGPipeline:
    def __init__(
        self, retriever: Retriever, llm: LLM, top_k: int = 5, index_dir: str = "data/index"
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.index_dir = index_dir

    @classmethod
    def from_config(cls, cfg: Config | None = None, llm: LLM | None = None) -> RAGPipeline:
        """Wire up all components from config (loads models - takes a few seconds)."""
        from .embeddings import SentenceTransformerEmbedder
        from .llm import build_llm
        from .retrieval import CrossEncoderReranker
        from .store import IndexStore

        cfg = cfg or load_config()
        store = IndexStore.load(cfg.paths.index_dir)
        model = store.manifest.get("embedding_model", cfg.embedding.model)
        if model != cfg.embedding.model:
            # Querying with a different model than the index was built with gives garbage.
            raise ValueError(
                f"Index was built with '{model}' but config uses '{cfg.embedding.model}'. "
                "Re-run `ragdemia ingest`."
            )
        embedder = SentenceTransformerEmbedder(
            cfg.embedding.model,
            cfg.embedding.query_instruction,
            cfg.embedding.batch_size,
            cfg.embedding.device,
        )
        reranker = (
            CrossEncoderReranker(cfg.retrieval.reranker_model, cfg.embedding.device)
            if cfg.retrieval.use_reranker
            else None
        )
        retriever = Retriever(
            store,
            embedder,
            mode=cfg.retrieval.mode,
            candidates=cfg.retrieval.candidates,
            rrf_k=cfg.retrieval.rrf_k,
            reranker=reranker,
        )
        return cls(
            retriever,
            llm or build_llm(cfg.llm),
            top_k=cfg.retrieval.top_k,
            index_dir=cfg.paths.index_dir,
        )

    # ------------------------------------------------------------------ RAG
    def retrieve(self, question: str, top_k: int | None = None, mode: Mode | None = None):
        return self.retriever.retrieve(question, top_k or self.top_k, mode)

    def reload_index(self) -> None:
        """Pick up a re-built index (e.g. after uploading a document) without reloading models."""
        from .store import IndexStore

        self.retriever.store = IndexStore.load(self.index_dir)

    def condense(self, question: str, history: History | None) -> str:
        """Rewrite a follow-up into a standalone search query (no-op without history)."""
        if not history:
            return question
        raw = self.llm.generate(rewrite_messages(question, history))
        return clean_rewrite(raw, question)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        mode: Mode | None = None,
        history: History | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        query = self.condense(question, history)
        t1 = time.perf_counter()
        sources = self.retrieve(query, top_k, mode)
        t2 = time.perf_counter()
        # The answer prompt gets the standalone question, so it is self-contained and the
        # answer stays grounded only in the sources.
        text = self.llm.generate(rag_messages(query, sources)).strip()
        t3 = time.perf_counter()
        timings = {"retrieval_s": t2 - t1, "generation_s": t3 - t2}
        if history:
            timings = {"rewrite_s": t1 - t0, **timings}
        return Answer(
            question=question,
            text=text,
            sources=sources,
            citations=check_citations(text, [s.chunk.text for s in sources]),
            mode="rag",
            timings=timings,
            search_query=query if query != question else None,
        )

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        mode: Mode | None = None,
        history: History | None = None,
    ) -> tuple[str, list[RetrievedChunk], Iterator[str]]:
        """Condense + retrieve synchronously, then return ``(query, sources, token stream)``."""
        query = self.condense(question, history)
        sources = self.retrieve(query, top_k, mode)
        return query, sources, self.llm.stream(rag_messages(query, sources))

    def finalize(
        self, question: str, text: str, sources: list[RetrievedChunk], query: str | None = None
    ) -> Answer:
        texts = [s.chunk.text for s in sources]
        return Answer(
            question,
            text.strip(),
            sources,
            check_citations(text, texts),
            "rag",
            search_query=query if query and query != question else None,
        )

    # ------------------------------------------------------------------ baseline
    def ask_closed_book(self, question: str) -> Answer:
        t0 = time.perf_counter()
        text = self.llm.generate(closed_book_messages(question)).strip()
        return Answer(
            question=question,
            text=text,
            citations=check_citations(text, []),
            mode="closed_book",
            timings={"generation_s": time.perf_counter() - t0},
        )
