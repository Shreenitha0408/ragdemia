"""REST API (FastAPI).  Interactive docs at http://localhost:8000/docs.

The pipeline (embedding model, reranker, FAISS index) is loaded once at startup and shared
across requests; loading per request would add seconds of latency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .llm import LLMError
from .pipeline import RAGPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = RAGPipeline.from_config(load_config())
    yield
    _state.clear()


app = FastAPI(
    title="RAGDemia API",
    version="0.1.0",
    description="Retrieval-augmented Q&A over your academic documents, with section-level citations.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    compare: bool = Field(False, description="Also return a closed-book (no retrieval) answer")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] | None = None


def _pipe() -> RAGPipeline:
    if "pipeline" not in _state:
        raise HTTPException(503, "Pipeline not loaded")
    return _state["pipeline"]


@app.get("/health")
def health() -> dict:
    p = _pipe()
    m = p.retriever.store.manifest
    return {
        "status": "ok",
        "llm": p.llm.name,
        "embedding_model": m.get("embedding_model"),
        "reranker": p.retriever.reranker.model_name if p.retriever.reranker else None,
        "n_documents": len(m.get("documents", [])),
        "n_chunks": m.get("n_chunks"),
        "index_built_at": m.get("built_at"),
    }


@app.get("/documents")
def documents() -> list[dict]:
    return _pipe().retriever.store.manifest.get("documents", [])


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> list[dict]:
    results = _pipe().retrieve(req.query, req.top_k, req.mode)
    return [
        {
            "rank": r.rank,
            "score": r.score,
            "scores": r.scores,
            "citation": r.chunk.citation,
            **r.chunk.to_dict(),
        }
        for r in results
    ]


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    p = _pipe()
    try:
        out = {"rag": p.ask(req.question, req.top_k, req.mode).to_dict()}
        if req.compare:
            out["closed_book"] = p.ask_closed_book(req.question).to_dict()
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    return out
