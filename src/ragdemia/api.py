"""REST API (FastAPI).  Interactive docs at http://localhost:8000/docs.

The pipeline (embedding model, reranker, FAISS index) is loaded once at startup and shared
across requests; loading per request would add seconds of latency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .chunking import hf_token_counter
from .config import load_config
from .library import Library, LibraryError
from .llm import LLMError
from .pipeline import RAGPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["pipeline"] = RAGPipeline.from_config(cfg)
    _state["library"] = Library(cfg, _state["pipeline"], hf_token_counter(cfg.embedding.model))
    yield
    _state.clear()


app = FastAPI(
    title="RAGDemia API",
    version="0.1.0",
    description="Retrieval-augmented Q&A over your academic documents, with section-level citations.",
    lifespan=lifespan,
)


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    history: list[Turn] = Field(
        default_factory=list,
        description="Previous turns, oldest first; follow-ups are rewritten into standalone queries",
    )
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
    _pipe()
    return _state["library"].documents


def _report(report) -> dict:
    return {
        "added": report.added,
        "updated": report.updated,
        "removed": report.removed,
        "n_chunks": report.n_chunks,
        "n_embedded": report.n_embedded,
        "seconds": round(report.seconds, 2),
        "failures": [{"path": p, "error": e} for p, e in report.failures],
    }


@app.post("/documents", status_code=201)
async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    """Upload one or more PDFs; they are indexed incrementally and searchable immediately."""
    _pipe()
    lib: Library = _state["library"]
    saved = []
    try:
        for f in files:
            saved.append(str(lib.add(f.filename or "document.pdf", await f.read())))
    except LibraryError as exc:
        for path in saved:  # all-or-nothing
            Path(path).unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    return {"saved": saved, **_report(lib.sync())}


@app.delete("/documents/{doc_id:path}")
def delete_document(doc_id: str) -> dict:
    """Remove a document that was uploaded through the API or UI."""
    _pipe()
    lib: Library = _state["library"]
    try:
        lib.remove(doc_id)
    except LibraryError as exc:
        raise HTTPException(403 if "on disk" in str(exc) else 404, str(exc)) from exc
    return _report(lib.sync())


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
        history = [(t.question, t.answer) for t in req.history]
        out = {"rag": p.ask(req.question, req.top_k, req.mode, history=history).to_dict()}
        if req.compare:
            out["closed_book"] = p.ask_closed_book(req.question).to_dict()
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    return out
